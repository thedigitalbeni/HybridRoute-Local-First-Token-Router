import re

CATEGORY_FACTUAL = "factual"
CATEGORY_MATH = "math"
CATEGORY_SENTIMENT = "sentiment"
CATEGORY_SUMMARY = "summary"
CATEGORY_NER = "ner"
CATEGORY_CODE_DEBUG = "code_debug"
CATEGORY_LOGIC = "logic"
CATEGORY_CODE_GEN = "code_gen"

# --- Routing strategy ---
# LOCAL  = zero Fireworks tokens, answered by the small quantized model in the image.
# REMOTE = sent through FIREWORKS_BASE_URL, costs tokens, used where a 1.5-3B model
#          is likely to be unreliable enough to risk the 80% accuracy gate.
#
# UPDATED after real local-model testing on 2025-07-10: factual knowledge
# moved to remote after the local 3B model confidently hallucinated on a real
# practice task (confused "Australian Capital Territory," a political region,
# for a body of water, when asked what water body Canberra is near). Sentiment
# and NER both tested correct and stay local. Summarization is untested (the
# published practice prompt was a broken placeholder) -- if you get a chance,
# test it with a real paragraph before fully trusting it either.
#
# TUNE THIS further if you get more test signal. If in doubt, remote is the
# safer default: losing some token efficiency beats failing the 80% gate.
ROUTE_MAP = {
    CATEGORY_SENTIMENT: "local",
    CATEGORY_SUMMARY: "local",
    CATEGORY_NER: "local",
    CATEGORY_FACTUAL: "remote",
    CATEGORY_MATH: "remote",
    CATEGORY_LOGIC: "remote",
    CATEGORY_CODE_DEBUG: "remote",
    CATEGORY_CODE_GEN: "remote",
}


def classify_task(prompt: str) -> str:
    """Heuristic keyword-based classifier across the 8 published categories.
    Order matters: more specific patterns are checked first so they don't get
    swallowed by broader ones (e.g. code_debug before code_gen before math)."""
    p = prompt.lower()

    # Code debugging: existing code + something is wrong with it
    if re.search(r"\bbug\b|\bdebug\b|\bfix (the|this) (function|code|bug)\b", p) or (
        ("def " in p or "function" in p)
        and re.search(r"\berror\b|\bbug\b|\bwrong\b|\bincorrect\b|\bfails?\b", p)
    ):
        return CATEGORY_CODE_DEBUG

    # Code generation: asked to write something from a spec, no bug implied
    if re.search(
        r"\bwrite (a|an) (python )?function\b|\bwrite (a|an) program\b|\bimplement (a|an)\b",
        p,
    ):
        return CATEGORY_CODE_GEN

    # Summarization -- checked before the logic-puzzle pattern below, since
    # phrases like "in exactly one sentence" (a length constraint) would
    # otherwise false-trigger the puzzle regex's "exactly one".
    if re.search(r"\bsummar(y|ise|ize)\b|\bcondense\b|\bin (exactly )?one sentence\b", p):
        return CATEGORY_SUMMARY

    # Logical / deductive reasoning: constraint puzzle language.
    # Broadened after real testing showed the narrow version (tied to the
    # single "who owns a pet" practice example's exact phrasing) completely
    # missed other valid puzzle styles: "each work in a different department"
    # (different verb than has/owns/likes), "who finished/works/lives" (not
    # just "who owns"), and positional/spatial grid puzzles ("immediately
    # left of", "in the middle") that don't use "each...different" at all.
    if re.search(
        r"\beach\b.*\bdifferent\b"
        r"|\bwho (is|was|owns|works|worked|finished|lives|lived|plays|played|has)\b"
        r"|\bdoes not\b.*\bown\b|\bpuzzle\b|\bwhich (one|person|color|position)\b"
        r"|\bimmediately (left|right|before|after)\b|\bin the middle\b"
        r"|\bfar (left|right)\b|\bnext to\b|\badjacent\b",
        p,
    ):
        return CATEGORY_LOGIC

    # Mathematical reasoning: numeric word problems
    if re.search(
        r"%|\bpercent\b|\bhow many\b|\btotal\b|\bremain(s|ing)?\b|\baverage\b|\bprice\b|\bcost\b",
        p,
    ) and re.search(r"\d", p):
        return CATEGORY_MATH

    # Named entity recognition
    if re.search(r"\bnamed entit(y|ies)\b|\bextract\b.*\bentit", p):
        return CATEGORY_NER

    # Sentiment classification
    if re.search(r"\bsentiment\b|\bclassify\b.*\breview\b|\bpositive or negative\b", p):
        return CATEGORY_SENTIMENT

    # Default: treat as factual knowledge
    return CATEGORY_FACTUAL
