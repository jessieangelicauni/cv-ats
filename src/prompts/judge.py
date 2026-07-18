SYSTEM = """You are an experienced senior technical recruiter with 15 years in IT hiring.
You reason like a human expert: leadership and measurable impact matter more than job \
titles; relevant tenure increases confidence; career changes that represent growth are \
not penalized.

When a candidate has worked at a Tier 1 multinational company, treat this as a \
meaningful positive signal — it indicates exposure to large-scale systems, professional \
engineering culture, and organisational breadth. Weight this in your holistic \
assessment, particularly under Career Trajectory and Experience Depth.

EVIDENCE RULE: Every evidence_chain item MUST include an evidence_quote containing \
exact text from the candidate profile. If supporting text is not found, set \
evidence_quote to "NOT FOUND IN CV" and lower the dimension_score accordingly. \
Never state a fact not traceable to the profile.
Return a valid JSON object matching the required schema exactly."""


def human(
    jd_json: str,
    profile_json: str,
    skill_matches: list | None = None,
) -> str:
    parts: list[str] = [
        f"Assess the following candidate against the job requirements.\n\n"
        f"JOB REQUIREMENTS:\n{jd_json}\n\n"
        f"CANDIDATE PROFILE:\n{profile_json}\n\n"
    ]

    if skill_matches:
        rows = [
            "SKILL COVERAGE (pre-computed — use as grounding for Technical Skills Fit):",
            "| JD Skill | Candidate Match | Score | Required |",
            "|---|---|---|---|",
        ]
        for m in skill_matches:
            match_str = m.best_match if m.best_match else "(no match)"
            req_str = "Yes" if m.is_required else "No"
            rows.append(f"| {m.jd_skill} | {match_str} | {m.score:.2f} | {req_str} |")
        parts.append("\n".join(rows) + "\n\n")

    parts.append(
        "Assess across these dimensions:\n"
        "1. Technical Skills Fit\n"
        "2. Experience Depth\n"
        "3. Career Trajectory\n"
        "4. Leadership & Impact\n"
        "5. Education & Credentials\n\n"
        "For each: write assessment, cite evidence_quote (exact text or "
        '"NOT FOUND IN CV"), score 0-10.\n\n'
        "Then produce raw_score (0-100) as your HOLISTIC judgment — not a formula. "
        "Weigh dimensions contextually based on what this specific role requires."
    )

    return "".join(parts)
