"""Algorithm entrypoints.

Training is routed through the official `external/on-policy` implementation.
"""

from .onpolicy_baseline import build_onpolicy_smac_args, run_onpolicy_smac

__all__ = ["build_onpolicy_smac_args", "run_onpolicy_smac"]
