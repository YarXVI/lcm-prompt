import time
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path

from .ir_models import GrainLevel


@dataclass
class ABTestConfig:
    test_name: str
    control_group: str = "static_ir"
    experiment_group: str = "dynamic_template"
    traffic_split: float = 0.5
    min_samples: int = 100
    metrics: List[str] = field(default_factory=lambda: [
        "tokens_injected", "tokens_saved", "upgrade_count", "latency_ms",
    ])


@dataclass
class ABTestResult:
    test_name: str
    group: str
    chunk_id: str
    session_id: str
    tokens_injected: int
    tokens_saved: int
    upgrade_count: int
    latency_ms: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "group": self.group,
            "chunk_id": self.chunk_id,
            "session_id": self.session_id,
            "tokens_injected": self.tokens_injected,
            "tokens_saved": self.tokens_saved,
            "upgrade_count": self.upgrade_count,
            "latency_ms": round(self.latency_ms, 2),
            "timestamp": self.timestamp,
        }


class ABTestRouter:

    def __init__(self, storage_dir: Optional[str] = None):
        self._configs: Dict[str, ABTestConfig] = {}
        self._results: Dict[str, List[ABTestResult]] = {}
        self._storage_dir = Path(storage_dir) if storage_dir else None

    def create_test(self, config: ABTestConfig) -> None:
        self._configs[config.test_name] = config
        self._results[config.test_name] = []

    def get_group(self, test_name: str, session_id: str) -> str:
        config = self._configs.get(test_name)
        if not config:
            return "control"

        hash_val = int(hashlib.md5(f"{test_name}:{session_id}".encode()).hexdigest(), 16)
        ratio = (hash_val % 1000) / 1000.0

        if ratio < config.traffic_split:
            return "experiment"
        return "control"

    def record_result(self, result: ABTestResult) -> None:
        if result.test_name not in self._results:
            self._results[result.test_name] = []
        self._results[result.test_name].append(result)
        self._persist_result(result)

    def get_analysis(self, test_name: str) -> Dict[str, Any]:
        results = self._results.get(test_name, [])
        if not results:
            return {"test_name": test_name, "status": "no_data"}

        control = [r for r in results if r.group == "control"]
        experiment = [r for r in results if r.group == "experiment"]

        def avg_metric(group: List[ABTestResult], metric: str) -> float:
            vals = [getattr(r, metric) for r in group]
            return sum(vals) / max(len(vals), 1)

        config = self._configs.get(test_name)
        metrics = config.metrics if config else ["tokens_injected", "tokens_saved"]

        analysis = {
            "test_name": test_name,
            "status": "running" if len(results) < (config.min_samples if config else 100) else "complete",
            "total_samples": len(results),
            "control_samples": len(control),
            "experiment_samples": len(experiment),
            "control": {m: round(avg_metric(control, m), 2) for m in metrics},
            "experiment": {m: round(avg_metric(experiment, m), 2) for m in metrics},
        }

        if control and experiment:
            improvements = {}
            for m in metrics:
                c_val = avg_metric(control, m)
                e_val = avg_metric(experiment, m)
                if c_val != 0:
                    improvements[m] = round((e_val - c_val) / abs(c_val) * 100, 2)
                else:
                    improvements[m] = 0.0
            analysis["improvement_pct"] = improvements

            tokens_saved_imp = improvements.get("tokens_saved", 0)
            analysis["experiment_wins"] = tokens_saved_imp > 0

        return analysis

    def list_tests(self) -> List[Dict[str, Any]]:
        return [
            {"test_name": name, "config": {
                "control": c.control_group,
                "experiment": c.experiment_group,
                "traffic_split": c.traffic_split,
                "min_samples": c.min_samples,
            }}
            for name, c in self._configs.items()
        ]

    def _persist_result(self, result: ABTestResult) -> None:
        if not self._storage_dir:
            return
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        path = self._storage_dir / f"ab_{result.test_name}_{ts}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
