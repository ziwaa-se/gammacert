"""gammacert: what a silent (or nearly silent) monitoring record rules out when the monitored AI system can see the monitoring design.

Companion code for a paper by Zihua She and Xiao Wang (under review).
"""
from .api import (campaign_escape, certificate, certificate_curve, coverage, equal_coverage_design,  # noqa: F401
                  floor_certificate, hidden_floor, lower_tail, minimax_design, randomized_monitor,
                  recall_lower_bounds, selection_coverage, single_action_certified)

__version__ = "0.1.0"
__all__ = [n for n in dir() if not n.startswith("_") and n not in {"api", "core", "certlib"}]
