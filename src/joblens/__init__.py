"""JobLens - explainable resume-to-job matching with skill gap analysis."""

from joblens.matcher import MatchResult, MatchWeights, match
from joblens.ranker import Ranking, rank_jobs
from joblens.sections import JobRequirements, split_requirements
from joblens.similarity import EmbeddingSimilarity, TfidfSimilarity, get_backend
from joblens.skills import SkillExtractor, default_extractor

__version__ = "0.4.0"

__all__ = [
    "EmbeddingSimilarity",
    "JobRequirements",
    "MatchResult",
    "MatchWeights",
    "Ranking",
    "SkillExtractor",
    "TfidfSimilarity",
    "default_extractor",
    "get_backend",
    "match",
    "rank_jobs",
    "split_requirements",
    "__version__",
]
