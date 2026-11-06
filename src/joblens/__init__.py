"""JobLens - explainable resume-to-job matching with skill gap analysis."""

from joblens.matcher import MatchResult, MatchWeights, match
from joblens.sections import JobRequirements, split_requirements
from joblens.skills import SkillExtractor, default_extractor

__version__ = "0.1.0"

__all__ = [
    "JobRequirements",
    "MatchResult",
    "MatchWeights",
    "SkillExtractor",
    "default_extractor",
    "match",
    "split_requirements",
    "__version__",
]
