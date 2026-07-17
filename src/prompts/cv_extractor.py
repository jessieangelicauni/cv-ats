SYSTEM_2A = """Extract structured data from this resume. Rules:
1. Extract basic personal info (name, email, phone, location, LinkedIn, current title) \
from the header or contact section. Set fields to null if not present — do not fabricate.
2. All achievement strings must be VERBATIM QUOTES from the resume text.
3. Compute tenure_months from dates given. If dates are missing, set null — never estimate.
4. Never merge two separate jobs into one WorkEntry.
5. Skills must appear exactly as written. Do not infer skills from job titles.
6. Do not normalize skill names — capture them as-is in raw_mention. \
Set canonical_skill equal to raw_mention for now.
7. Set total_experience_months as the sum of all non-null tenure_months.
Return a valid JSON object matching the required schema exactly."""


def human_2a(cv_text: str, candidate_id: str) -> str:
    return (
        f"Extract structured profile from the following resume.\n\n"
        f"CANDIDATE_ID: {candidate_id}\n\n"
        f"RESUME TEXT:\n{cv_text}"
    )


SYSTEM_2B = """You are a technology skills normalizer.
Given a list of skill mentions, return the canonical industry-standard name for each one.

Rules:
1. Map all variants of the same technology to one canonical name.
   Examples: "postgre", "postgres", "postgresql" → "PostgreSQL"
             "node js", "nodejs", "node.js" → "Node.js"
2. Preserve distinctions between DIFFERENT technologies even if they serve \
similar purposes. "vue js" → "Vue.js" and "reactjs" → "React" are NOT the same.
3. Apply correct industry capitalisation: "javascript" → "JavaScript", \
"aws" → "AWS", "k8s" → "Kubernetes".
4. If a mention is genuinely ambiguous, keep the raw mention as the canonical name.
5. Return JSON with a single key "mappings": {raw_mention: canonical_name, ...}."""


def human_2b(raw_skills: list[str]) -> str:
    skills_str = "\n".join(f"- {s}" for s in raw_skills)
    return f"Normalize the following skill mentions:\n\n{skills_str}"
