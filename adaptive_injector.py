import re
import time
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from .ir_models import GrainLevel, Grain
from .encoded_chunk_store import EncodedChunkStore
from .encoder_base import ContentDecoder
from .execution_profile import (
    ExecutionProfile, PROFILE_DEFAULTS, PROFILE_UPGRADE_STRATEGY,
)


@dataclass
class UpgradeRequest:
    chunk_id: str
    target_level: GrainLevel
    raw_marker: str
    position: int


@dataclass
class DowngradeRequest:
    chunk_id: str
    raw_marker: str
    position: int


@dataclass
class InjectionAuditEntry:
    session_id: str
    chunk_id: str
    profile: str
    grain_requested: GrainLevel
    grain_actual: GrainLevel
    grain_trigger: str
    tokens_injected: int
    triggered_upgrade: bool
    latency_ms: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "chunk_id": self.chunk_id,
            "profile": self.profile,
            "grain_requested": self.grain_requested.value,
            "grain_actual": self.grain_actual.value,
            "grain_trigger": self.grain_trigger,
            "tokens_injected": self.tokens_injected,
            "triggered_upgrade": self.triggered_upgrade,
            "latency_ms": round(self.latency_ms, 2),
            "timestamp": self.timestamp,
        }


class AdaptiveInjector:

    NEED_FULL_PATTERN = re.compile(r"\[NEED_CHUNK_FULL:([A-Za-z0-9_\-]+)\]")
    NEED_DETAIL_PATTERN = re.compile(r"\[NEED_CHUNK_DETAIL:([A-Za-z0-9_\-]+)\]")
    GRAIN_SUFFICIENT_PATTERN = re.compile(r"\[GRAIN_SUFFICIENT:([A-Za-z0-9_\-]+)\]")

    DEFAULT_COOLDOWN_ROUNDS = 3

    def __init__(
        self,
        encoded_store: EncodedChunkStore,
        decoder: Optional[ContentDecoder] = None,
        default_grain: Optional[GrainLevel] = None,
        profile: ExecutionProfile = ExecutionProfile.LOCAL_CONSTRAINED,
        cooldown_rounds: int = DEFAULT_COOLDOWN_ROUNDS,
    ):
        self._store = encoded_store
        self._decoder = decoder or ContentDecoder()
        self._profile = profile
        self._default_grain = default_grain or PROFILE_DEFAULTS[profile]
        self._cooldown_rounds = cooldown_rounds

        self._grain_upgrade_map: Dict[str, Tuple[GrainLevel, int]] = {}
        self._lock = threading.RLock()

        self._audit_log: List[InjectionAuditEntry] = []
        self._injection_counts: Dict[str, int] = {}
        self._upgrade_counts: Dict[str, int] = {}

        self._session_id: str = ""

    def set_session(self, session_id: str) -> None:
        self._session_id = session_id

    def set_profile(self, profile: ExecutionProfile) -> None:
        self._profile = profile
        self._default_grain = PROFILE_DEFAULTS[profile]

    @property
    def profile(self) -> ExecutionProfile:
        return self._profile

    @property
    def default_grain(self) -> GrainLevel:
        return self._default_grain

    def inject(
        self,
        messages: List[Dict[str, str]],
        chunk_id: str,
        available_tokens: int = 4000,
        min_level: GrainLevel = GrainLevel.KEYWORDS,
    ) -> Tuple[List[Dict[str, str]], GrainLevel]:
        t0 = time.perf_counter()
        encoded = self._store.get_encoded(chunk_id)
        if not encoded:
            messages.append({
                "role": "system",
                "content": f"[LCM: chunk '{chunk_id}' not found]",
            })
            return messages, GrainLevel.KEYWORDS

        effective_min = self._resolve_effective_min_level(chunk_id, min_level)

        if min_level == GrainLevel.KEYWORDS and effective_min == GrainLevel.KEYWORDS:
            preferred_grain = encoded.ir.grains.get(self._default_grain)
            if preferred_grain and preferred_grain.tokens <= available_tokens:
                content, selected_level = preferred_grain.content, self._default_grain
            else:
                content, selected_level = self._decoder.decode(
                    encoded.ir, available_tokens, min_level=GrainLevel.KEYWORDS,
                )
        else:
            content, selected_level = self._decoder.decode(
                encoded.ir, available_tokens, min_level=effective_min,
            )

        grain = encoded.ir.grains.get(selected_level)
        actual_tokens = grain.tokens if grain else 0

        is_upgrade = self._is_upgrade(chunk_id, selected_level)
        trigger = self._determine_trigger(chunk_id, selected_level, min_level)

        messages.append({
            "role": "system",
            "content": (
                f"[chunk {chunk_id} | grain:{selected_level.value} | "
                f"{actual_tokens}tok]\n{content}"
            ),
        })

        if selected_level != GrainLevel.FULL:
            hint = self._build_upgrade_hint(chunk_id, selected_level)
            messages.append({"role": "system", "content": hint})

        latency_ms = (time.perf_counter() - t0) * 1000
        self._record_audit(chunk_id, min_level, selected_level, trigger, actual_tokens, is_upgrade, latency_ms)

        return messages, selected_level

    def inject_full(
        self,
        messages: List[Dict[str, str]],
        chunk_id: str,
    ) -> List[Dict[str, str]]:
        encoded = self._store.get_encoded(chunk_id)
        if not encoded:
            messages.append({
                "role": "system",
                "content": f"[LCM: chunk '{chunk_id}' not found]",
            })
            return messages

        full_grain = encoded.ir.grains.get(GrainLevel.FULL)
        content = full_grain.content if full_grain else encoded.original.content
        tokens = full_grain.tokens if full_grain else encoded.original.tokens

        messages.append({
            "role": "system",
            "content": (
                f"[chunk {chunk_id} | grain:full | {tokens}tok]\n{content}"
            ),
        })
        return messages

    def detect_upgrade_requests(self, text: str) -> List[UpgradeRequest]:
        requests = []
        for m in self.NEED_FULL_PATTERN.finditer(text):
            requests.append(UpgradeRequest(
                chunk_id=m.group(1), target_level=GrainLevel.FULL,
                raw_marker=m.group(0), position=m.start(),
            ))
        for m in self.NEED_DETAIL_PATTERN.finditer(text):
            requests.append(UpgradeRequest(
                chunk_id=m.group(1), target_level=GrainLevel.DETAIL,
                raw_marker=m.group(0), position=m.start(),
            ))
        return requests

    def detect_downgrade_requests(self, text: str) -> List[DowngradeRequest]:
        requests = []
        for m in self.GRAIN_SUFFICIENT_PATTERN.finditer(text):
            requests.append(DowngradeRequest(
                chunk_id=m.group(1), raw_marker=m.group(0), position=m.start(),
            ))
        return requests

    def process_sentinels(self, response_text: str) -> None:
        for m in self.NEED_DETAIL_PATTERN.finditer(response_text):
            cid = m.group(1)
            with self._lock:
                self._grain_upgrade_map[cid] = (GrainLevel.DETAIL, self._cooldown_rounds)
                self._upgrade_counts[cid] = self._upgrade_counts.get(cid, 0) + 1

        for m in self.NEED_FULL_PATTERN.finditer(response_text):
            cid = m.group(1)
            with self._lock:
                self._grain_upgrade_map[cid] = (GrainLevel.FULL, self._cooldown_rounds)
                self._upgrade_counts[cid] = self._upgrade_counts.get(cid, 0) + 1

        for m in self.GRAIN_SUFFICIENT_PATTERN.finditer(response_text):
            cid = m.group(1)
            with self._lock:
                self._grain_upgrade_map.pop(cid, None)

    def tick_cooldown(self) -> None:
        with self._lock:
            to_delete = []
            for cid, (level, cooldown) in self._grain_upgrade_map.items():
                if cooldown - 1 <= 0:
                    to_delete.append(cid)
                else:
                    self._grain_upgrade_map[cid] = (level, cooldown - 1)
            for cid in to_delete:
                del self._grain_upgrade_map[cid]

    def get_upgrade_map(self) -> Dict[str, Tuple[GrainLevel, int]]:
        with self._lock:
            return dict(self._grain_upgrade_map)

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._audit_log[-limit:]]

    def geturr_stats(self) -> Dict[str, float]:
        with self._lock:
            all_chunks = set(list(self._injection_counts.keys()) + list(self._upgrade_counts.keys()))
            result = {}
            for cid in all_chunks:
                total = self._injection_counts.get(cid, 0)
                upgrades = self._upgrade_counts.get(cid, 0)
                result[cid] = upgrades / max(total, 1)
            return result

    def get_stats(self) -> Dict[str, Any]:
        return {
            "profile": self._profile.value,
            "default_grain": self._default_grain.value,
            "active_upgrades": len(self._grain_upgrade_map),
            "total_injections": sum(self._injection_counts.values()),
            "total_upgrades": sum(self._upgrade_counts.values()),
            "urr_stats": self.geturr_stats(),
            "audit_entries": len(self._audit_log),
        }

    def reset_session(self) -> None:
        with self._lock:
            self._grain_upgrade_map.clear()
        self._session_id = ""

    def _resolve_effective_min_level(
        self, chunk_id: str, requested_min: GrainLevel,
    ) -> GrainLevel:
        with self._lock:
            if chunk_id in self._grain_upgrade_map:
                upgraded_level, _ = self._grain_upgrade_map[chunk_id]
                if upgraded_level.finer_or_equal(requested_min):
                    return upgraded_level
                return requested_min
        return requested_min

    def _is_upgrade(self, chunk_id: str, selected_level: GrainLevel) -> bool:
        return selected_level.finer_or_equal(self._default_grain) and selected_level != self._default_grain

    def _determine_trigger(
        self, chunk_id: str, selected_level: GrainLevel, requested_min: GrainLevel,
    ) -> str:
        with self._lock:
            if chunk_id in self._grain_upgrade_map:
                return "UPGRADE"
        if selected_level == self._default_grain:
            return "DEFAULT"
        if requested_min != GrainLevel.KEYWORDS:
            return "EXPLICIT"
        return "FALLBACK"

    def _record_audit(
        self,
        chunk_id: str,
        grain_requested: GrainLevel,
        grain_actual: GrainLevel,
        trigger: str,
        tokens: int,
        is_upgrade: bool,
        latency_ms: float,
    ) -> None:
        self._injection_counts[chunk_id] = self._injection_counts.get(chunk_id, 0) + 1

        entry = InjectionAuditEntry(
            session_id=self._session_id,
            chunk_id=chunk_id,
            profile=self._profile.value,
            grain_requested=grain_requested,
            grain_actual=grain_actual,
            grain_trigger=trigger,
            tokens_injected=tokens,
            triggered_upgrade=is_upgrade,
            latency_ms=latency_ms,
        )
        self._audit_log.append(entry)

    def _build_upgrade_hint(self, chunk_id: str, current: GrainLevel) -> str:
        hints = {
            GrainLevel.KEYWORDS: (
                f"[hint: chunk '{chunk_id}' at keywords. "
                f"need summary->[NEED_CHUNK:{chunk_id}] "
                f"need detail->[NEED_CHUNK_DETAIL:{chunk_id}] "
                f"need full->[NEED_CHUNK_FULL:{chunk_id}]]"
            ),
            GrainLevel.SUMMARY: (
                f"[hint: chunk '{chunk_id}' at summary. "
                f"need detail->[NEED_CHUNK_DETAIL:{chunk_id}] "
                f"need full->[NEED_CHUNK_FULL:{chunk_id}]]"
            ),
            GrainLevel.DETAIL: (
                f"[hint: chunk '{chunk_id}' at detail. "
                f"need full->[NEED_CHUNK_FULL:{chunk_id}]]"
            ),
        }
        return hints.get(current, "")
