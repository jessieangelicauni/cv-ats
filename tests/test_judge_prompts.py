from src.prompts.judge import SYSTEM, retry_human


def test_system_prompt_forbids_general_knowledge():
    assert "general knowledge" in SYSTEM


def test_retry_human_lists_dimension_claim_and_quote():
    prompt = retry_human([
        ("Technical Skills Fit", "Strong Python skills.", "Python, Git, Docker"),
    ])
    assert "Technical Skills Fit" in prompt
    assert "Strong Python skills." in prompt
    assert "Python, Git, Docker" in prompt


def test_retry_human_forbids_not_found_in_cv_escape():
    prompt = retry_human([
        ("Technical Skills Fit", "Strong Python skills.", "Python, Git, Docker"),
    ])
    assert "Do not respond with" in prompt
    assert "NOT FOUND IN CV" in prompt


def test_retry_human_lists_multiple_failed_items():
    prompt = retry_human([
        ("Technical Skills Fit", "Claim A", "Quote A"),
        ("Education & Credentials", "Claim B", "Quote B"),
    ])
    assert "Claim A" in prompt
    assert "Quote A" in prompt
    assert "Claim B" in prompt
    assert "Quote B" in prompt
