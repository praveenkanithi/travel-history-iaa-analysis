# Prompts

The prompts used to generate the LLM annotations in `pkanithi/travel-history-iaa-benchmark`.
Included verbatim for reviewer visibility; they are not executed by this repo's analysis code.

| File | Role |
|---|---|
| `system_prompt_main.txt` | Main study prompt (`SYSTEM_PROMPT_ANNOTATION_STUDY`). Used for the primary LLM annotation runs. Also the "high comprehensiveness" and "default" condition below. |
| `user_prompt_template.txt` | User-turn template (`USER_PROMPT_ANNOTATION_STUDY`) paired with every system prompt; `{possible_case_location}` and `{note_text}` are filled in per case. |
| `variants/system_prompt_gpt-5.5.txt` | Prompt-wording variant, rewritten for GPT-5.5. |
| `variants/system_prompt_opus-4.8.txt` | Prompt-wording variant, rewritten for Opus 4.8. |
| `variants/system_prompt_gemini-3.5-flash.txt` | Prompt-wording variant, rewritten for Gemini 3.5 Flash. |
| `variants/system_prompt_gpt-oss-120b.txt` | Prompt-wording variant, rewritten for GPT-OSS-120B. |
| `comprehensiveness/system_prompt_medium.txt` | Reduced-detail instruction variant ("medium" comprehensiveness). |
| `comprehensiveness/system_prompt_low.txt` | Minimal-detail instruction variant ("low" comprehensiveness). |

All prompts target the same output schema (see `## OUTPUT FORMAT` in `system_prompt_main.txt`), which
matches the structured fields in the benchmark dataset: `has_international_travel`,
`symptom_onset_relative_to_return`, the 7 `exposure_*` categories, and the 5 `entities` labels.

Source: `SYSTEM_PROMPT_ANNOTATION_STUDY` and related variants/comprehensiveness dictionaries in the
companion data-preparation codebase (not included in this repo).
