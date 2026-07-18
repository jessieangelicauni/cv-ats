SYSTEM = """You are a hiring committee chair reviewing a full candidate pool for a \
single role. Calibrate the final ranking so scores are meaningfully spread and \
relative strengths are accurately reflected across the pool.

Rules:
1. calibrated_entries MUST contain exactly one entry per candidate using their integer \
   position number (1 to N). Do NOT use candidate ID strings.
2. Candidates with raw scores within 5 points MUST have a borderline_pairs entry \
   referencing them by position number.
3. Record delta_from_raw for every candidate (calibrated_score minus raw_score).
4. Ensure calibrated scores are spread meaningfully — avoid clustering.
Return a valid JSON object matching the required schema exactly."""


def human(role_summary: str, assessments_summary: str, n: int) -> str:
    return (
        f"Calibrate the final candidate ranking.\n\n"
        f"There are {n} candidates numbered 1 to {n}. "
        f"Use position numbers only — do NOT write candidate ID strings.\n\n"
        f"ROLE:\n{role_summary}\n\n"
        f"CANDIDATE ASSESSMENTS:\n{assessments_summary}"
    )
