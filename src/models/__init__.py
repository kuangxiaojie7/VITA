from src.vita import VITAAgent, VITAAgentConfig
from .recurrent_policy import RecurrentMAPPOPolicy

AgentConfig = VITAAgentConfig

__all__ = ["VITAAgent", "AgentConfig", "RecurrentMAPPOPolicy"]
