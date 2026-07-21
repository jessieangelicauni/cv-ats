from datetime import date

SYSTEM_2A = """Extract structured data from this resume. Rules:
1. Extract basic personal info (name, email, phone, location, LinkedIn, current title) \
from the header or contact section. Set fields to null if not present — do not fabricate.
2. All achievement strings must be VERBATIM QUOTES from the resume text.
3. Compute tenure_months from dates given. Treat "Present" or "Current" as ending \
on the reference date provided below. If dates are missing, set null — never estimate.
4. Never merge two separate jobs into one WorkEntry.
5. Skills must appear exactly as written. Do not infer skills from job titles.
6. Do not normalize skill names — capture them as-is in raw_mention.
7. Set total_experience_months as the sum of all non-null tenure_months.
Return a valid JSON object matching the required schema exactly."""


def human_2a(cv_text: str, candidate_id: str) -> str:
    return (
        f"Extract structured profile from the following resume.\n\n"
        f"CANDIDATE_ID: {candidate_id}\n\n"
        f"REFERENCE DATE: {date.today().isoformat()}\n\n"
        f"RESUME TEXT:\n{cv_text}"
    )
