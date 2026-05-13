from enum import Enum
from typing import Dict

from .ir_models import GrainLevel


class ExecutionProfile(str, Enum):
    LOCAL_CONSTRAINED = "local_constrained"
    CLOUD_TOKEN_BILLED = "cloud_token_billed"
    CLOUD_CALL_BILLED = "cloud_call_billed"
    CLOUD_DYNAMIC = "cloud_dynamic"


PROFILE_DEFAULTS: Dict[ExecutionProfile, GrainLevel] = {
    ExecutionProfile.LOCAL_CONSTRAINED: GrainLevel.DETAIL,
    ExecutionProfile.CLOUD_TOKEN_BILLED: GrainLevel.SUMMARY,
    ExecutionProfile.CLOUD_CALL_BILLED: GrainLevel.DETAIL,
    ExecutionProfile.CLOUD_DYNAMIC: GrainLevel.KEYWORDS,
}

PROFILE_UPGRADE_STRATEGY = {
    ExecutionProfile.LOCAL_CONSTRAINED: "explicit",
    ExecutionProfile.CLOUD_TOKEN_BILLED: "explicit_with_audit",
    ExecutionProfile.CLOUD_CALL_BILLED: "passive",
    ExecutionProfile.CLOUD_DYNAMIC: "dynamic_template",
}

PROFILE_PROMPT_CACHING = {
    ExecutionProfile.LOCAL_CONSTRAINED: False,
    ExecutionProfile.CLOUD_TOKEN_BILLED: True,
    ExecutionProfile.CLOUD_CALL_BILLED: False,
    ExecutionProfile.CLOUD_DYNAMIC: True,
}

PROFILE_DYNAMIC_RENDERING = {
    ExecutionProfile.LOCAL_CONSTRAINED: False,
    ExecutionProfile.CLOUD_TOKEN_BILLED: False,
    ExecutionProfile.CLOUD_CALL_BILLED: False,
    ExecutionProfile.CLOUD_DYNAMIC: True,
}
