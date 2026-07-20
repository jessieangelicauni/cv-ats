from src.utils.skill_normalizer import compute_skill_matches
from src.models.schemas import SkillRequirement, SkillEntry


def _req(skill: str) -> SkillRequirement:
    return SkillRequirement(skill=skill, level="expert", is_mandatory=True)


def test_matches_skill_within_comma_separated_blob():
    # Extraction often bundles an entire resume skills section into one
    # SkillEntry instead of one entry per skill (see cv:7e290569883c795bd
    # in .cache/extractions.db for a real example).
    blob = SkillEntry(
        raw_mention=(
            "Python, Julia, LangChain, OpenAI API, Polars, Weights & Biases, "
            "Airflow, Neo4j, Pinecone, Weaviate, Qdrant, Hugging Face Endpoints, "
            "Bedrock, Vector Databases, MLOps, A/B Testing"
        ),
        proficiency="intermediate",
        evidence_quote="Seasoned software professional with deep expertise in Python, Julia, and LangChain.",
    )

    matches = compute_skill_matches([_req("Python")], [blob])

    assert matches[0].best_match is not None
