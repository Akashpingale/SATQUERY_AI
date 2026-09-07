"""Agentic orchestration.

Pipeline: input inspection -> query understanding -> task classification ->
compatibility validation -> model/tool selection -> workflow planning -> specialist
execution -> evidence -> confidence -> answer synthesis -> execution trace.

The agent may only use tools from the model registry, and only with parameters those
tools declare. Tasks whose specialists are not yet implemented are reported as
unavailable - never answered with fabricated output.
"""

from .orchestrator import build_input_configuration, run_analysis
from .planner import TASK_TOOLCHAIN, build_plan, planned_tools
from .types import (
    AnalysisOutcome,
    AnalysisPlan,
    AnalysisStatus,
    ConfidenceAssessment,
    ConfidenceLevel,
    EvidenceItem,
    InputConfiguration,
    InputImage,
    PlanStep,
    QueryIntent,
    SpecialistResult,
    TraceEntry,
)
from .understanding import classify, sanitize_query, understand
from .validation import PlanRejected, validate_parameters, validate_plan

__all__ = [
    "AnalysisOutcome",
    "AnalysisPlan",
    "AnalysisStatus",
    "ConfidenceAssessment",
    "ConfidenceLevel",
    "EvidenceItem",
    "InputConfiguration",
    "InputImage",
    "PlanRejected",
    "PlanStep",
    "QueryIntent",
    "SpecialistResult",
    "TASK_TOOLCHAIN",
    "TraceEntry",
    "build_input_configuration",
    "build_plan",
    "classify",
    "planned_tools",
    "run_analysis",
    "sanitize_query",
    "understand",
    "validate_parameters",
    "validate_plan",
]
