SYSTEM = """You are a precise requirements analyst. Extract ONLY what is explicitly \
stated in the job description. Do not infer, assume, or add requirements not present \
in the text. If a field has no evidence, return null or an empty list.
Mark is_mandatory=true ONLY if the JD uses words like "required", "must have", \
"essential". Otherwise is_mandatory=false.
Return a valid JSON object matching the required schema exactly."""


def human(jd_text: str, jd_hash: str) -> str:
    return (
        f"Extract structured requirements from the following job description.\n\n"
        f"JD_HASH: {jd_hash}\n\n"
        f"JOB DESCRIPTION:\n{jd_text}"
    )
