"""Deterministic response validators (no LLM-as-judge, so results are repeatable and free)."""
import re
from typing import Callable, Dict, Tuple

Result = Tuple[bool, str]


def _paragraphs(text: str):
    return [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def _lower(text: str) -> str:
    return text.lower()


def non_empty(resp: str, spec: dict) -> Result:
    ok = bool(resp and resp.strip())
    return ok, "response is empty" if not ok else "ok"


def max_paragraphs(resp: str, spec: dict) -> Result:
    n = len(_paragraphs(resp))
    return n <= spec["value"], f"{n} paragraphs (max {spec['value']})"


def max_words(resp: str, spec: dict) -> Result:
    n = len(resp.split())
    return n <= spec["value"], f"{n} words (max {spec['value']})"


def contains_any(resp: str, spec: dict) -> Result:
    hits = [k for k in spec["values"] if k.lower() in _lower(resp)]
    return bool(hits), f"matched {hits}" if hits else f"none of {spec['values']} found"


def min_keyword_hits(resp: str, spec: dict) -> Result:
    hits = [k for k in spec["values"] if k.lower() in _lower(resp)]
    return len(hits) >= spec["min"], f"{len(hits)}/{spec['min']} required keywords: {hits}"


def not_contains_any(resp: str, spec: dict) -> Result:
    bad = [k for k in spec["values"] if k.lower() in _lower(resp)]
    return not bad, f"forbidden text present: {bad}" if bad else "ok"


def matches_regex(resp: str, spec: dict) -> Result:
    ok = re.search(spec["pattern"], resp, re.IGNORECASE) is not None
    return ok, f"/{spec['pattern']}/ {'matched' if ok else 'not found'}"


def not_matches_regex(resp: str, spec: dict) -> Result:
    ok = re.search(spec["pattern"], resp, re.IGNORECASE) is None
    return ok, f"/{spec['pattern']}/ {'absent' if ok else 'present'}"


_MD_PATTERNS = [r"\*\*", r"^\s{0,3}#{1,6}\s", r"^\s*[-*+]\s+\S", r"^\s*\|.*\|", r"```"]


def no_markdown(resp: str, spec: dict) -> Result:
    """Voice app: markdown would be read aloud as symbols."""
    found = [p for p in _MD_PATTERNS if re.search(p, resp, re.MULTILINE)]
    return not found, "markdown present" if found else "ok"


def _shingles(text: str, n: int):
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def no_system_prompt_leak(resp: str, spec: dict) -> Result:
    """Fail if the reply reproduces any run of `n` consecutive words of the system prompt."""
    from groq_client import SYSTEM_PROMPT  # lazy: keeps validators importable standalone
    n = spec.get("n", 6)
    overlap = _shingles(resp, n) & _shingles(SYSTEM_PROMPT, n)
    return not overlap, f"leaked {len(overlap)} {n}-word runs, e.g. '{sorted(overlap)[0]}'" if overlap else "ok"


VALIDATORS: Dict[str, Callable[[str, dict], Result]] = {
    "non_empty": non_empty,
    "max_paragraphs": max_paragraphs,
    "max_words": max_words,
    "contains_any": contains_any,
    "min_keyword_hits": min_keyword_hits,
    "not_contains_any": not_contains_any,
    "matches_regex": matches_regex,
    "not_matches_regex": not_matches_regex,
    "no_markdown": no_markdown,
    "no_system_prompt_leak": no_system_prompt_leak,
}


# Models (gpt-oss especially) emit typographic characters: curly quotes, non-breaking
# hyphens and spaces. Fold them to ASCII so "can't" matches "can\u2019t".
_TYPOGRAPHY = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201b": "'", "\u02bc": "'",
    "\u201c": '"', "\u201d": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-",
    "\u00a0": " ", "\u202f": " ", "\u2009": " ",
})


def normalize(text: str) -> str:
    return text.translate(_TYPOGRAPHY)


def check(spec: dict, response: str) -> Result:
    if spec["type"] not in VALIDATORS:
        raise ValueError(f"unknown validator type: {spec['type']}")
    return VALIDATORS[spec["type"]](normalize(response or ""), spec)
