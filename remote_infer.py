import os
import sys

from openai import OpenAI

# Kept deliberately terse, but ALSO forceful about suppressing reasoning
# traces -- soft phrasing like "no explanation" was NOT reliably obeyed.
# Real testing showed kimi-k2p7-code dump its internal reasoning before the
# real answer on BOTH code_gen and code_debug in separate runs (not a
# one-off), burning most or all of the token budget before ever writing the
# actual answer. Every prompt below now explicitly says not to think out
# loud, not just "be concise" -- and every cap has real headroom in case it
# still happens.
SYSTEM_PROMPTS = {
    "factual": (
        "Do not think out loud or restate the question. Answer directly in "
        "1-3 sentences, covering every part of a multi-part question."
    ),
    "math": (
        "Do not think out loud or show steps. Output ONLY the final numeric "
        "answer, nothing else."
    ),
    "logic": (
        "Do not think out loud or explain your reasoning. Output ONLY the "
        "answer in a few words, nothing else."
    ),
    "code_debug": (
        "Output the corrected function as your very first line. Do not think "
        "out loud, do not restate the request, do not explain your approach. "
        "Return ONLY the corrected function as code -- no markdown fences, "
        "nothing before or after it."
    ),
    "code_gen": (
        "Output the function definition as your very first line. Do not think "
        "out loud, do not restate the request, do not explain your approach. "
        "Return ONLY the final function as code -- no markdown fences, nothing "
        "before or after it."
    ),
}

# Sized to what each answer actually needs, not a flat guess. IMPORTANT: real
# pipeline testing showed math, code_gen, AND code_debug all hitting their
# cap exactly in separate runs -- a hard signal this is a systemic model
# behavior (both minimax-m3 and kimi-k2p7-code can spend tokens on internal
# reasoning even when explicitly instructed not to), not a one-off. Every cap
# below has real safety headroom, not the bare minimum. If a completion
# token count lands exactly on the cap again in any future run (watch the
# [WARN] lines this prints), that category needs even MORE headroom -- a
# truncated answer failing accuracy costs the whole task, far worse than
# extra tokens.
MAX_TOKENS_BY_CATEGORY = {
    "factual": 220,
    "math": 200,
    "logic": 150,
    "code_debug": 500,
    "code_gen": 600,
}
DEFAULT_MAX_TOKENS = 200

# Real Track 1 ALLOWED_MODELS (as of launch): minimax-m3, kimi-k2p7-code,
# gemma-4-31b-it, gemma-4-26b-a4b-it, gemma-4-31b-it-nvfp4.
#
# IMPORTANT: model choice here should be picked for ACCURACY PER TOKEN, not
# dollar cost. The leaderboard only counts prompt+completion tokens -- it
# does not care whether you used a cheap quantized model or an expensive
# dense one. nvfp4 vs full-precision is a $ cost lever for your Fireworks
# credit budget, not a leaderboard-token lever. Don't conflate the two.
#
# NOTE (as discovered during testing): Fireworks currently requires billing
# card info to deploy gemma-4-* models on-demand, even with prepaid hackathon
# credit. minimax-m3 is therefore the PRIMARY for math/logic below, not a
# fallback -- gemma is listed last, tried only if minimax-m3 itself fails, so
# a blocked/slow gemma-4-* call can't eat into the 30s-per-request or
# 10-minute total runtime budget during actual grading. If you later deploy
# gemma-4-31b-it successfully, you can promote it back to first choice and
# re-run compare_models.py to confirm it's actually more accurate first.
CATEGORY_MODEL_PREFERENCE = {
    "code_debug": ["kimi-k2p7-code", "minimax-m3"],
    "code_gen": ["kimi-k2p7-code", "minimax-m3"],
    "math": ["minimax-m3", "gemma-4-31b-it"],
    "logic": ["minimax-m3", "gemma-4-31b-it"],
    "factual": ["minimax-m3", "gemma-4-31b-it"],
}
DEFAULT_PREFERENCE = ["minimax-m3"]


class RemoteModel:
    def __init__(self):
        # Per the guide: read these purely from the environment, never hardcode.
        self.client = OpenAI(
            api_key=os.environ["FIREWORKS_API_KEY"],
            base_url=os.environ["FIREWORKS_BASE_URL"],
        )
        allowed = os.environ["ALLOWED_MODELS"].split(",")
        self.allowed = [m.strip() for m in allowed if m.strip()]
        if not self.allowed:
            raise RuntimeError("ALLOWED_MODELS is empty -- cannot route any remote task")

        # Diagnostic only -- the real score comes from the judging proxy, not
        # this client-side count. But this is your only visibility into token
        # cost before you actually submit, which matters a lot in a
        # fewest-tokens-wins competition. See call_log for a per-task breakdown.
        self.total_tokens_used = 0
        self.call_log = []  # [{model, category, prompt_tokens, completion_tokens, total_tokens}]

    def _candidates(self, category):
        """Ranked list of allowed model IDs to try for this category, most
        preferred first, falling back to any remaining allowed model."""
        prefs = CATEGORY_MODEL_PREFERENCE.get(category, DEFAULT_PREFERENCE)
        ordered = []
        for pref in prefs:
            for m in self.allowed:
                if pref in m and m not in ordered:
                    ordered.append(m)
        for m in self.allowed:
            if m not in ordered:
                ordered.append(m)
        return ordered

    def generate(self, prompt: str, category: str) -> str:
        system = SYSTEM_PROMPTS.get(category, "Answer directly and concisely.")
        max_tokens = MAX_TOKENS_BY_CATEGORY.get(category, DEFAULT_MAX_TOKENS)
        last_err = None
        for model in self._candidates(category):
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                content = resp.choices[0].message.content
                # Check validity BEFORE logging/returning. Some models (seen
                # with minimax-m3 on math) return content=None when they burn
                # the whole token budget on internal reasoning without ever
                # emitting final text -- treat that as a real failure, not a
                # success, so it doesn't pollute call_log or get returned as
                # a bogus answer.
                if content is None or not content.strip():
                    raise ValueError(
                        f"{model} returned empty/None content -- likely ran "
                        f"out of tokens mid-reasoning before writing an answer"
                    )
                answer = content.strip()

                usage = getattr(resp, "usage", None)
                if usage is not None:
                    self.total_tokens_used += usage.total_tokens
                    self.call_log.append(
                        {
                            "model": model,
                            "category": category,
                            "prompt_tokens": usage.prompt_tokens,
                            "completion_tokens": usage.completion_tokens,
                            "total_tokens": usage.total_tokens,
                        }
                    )
                    if usage.completion_tokens >= max_tokens:
                        # Hit the cap exactly -- content is non-empty so this
                        # technically "succeeded", but it may well be
                        # truncated mid-answer (seen with kimi-k2p7-code on
                        # math: cut off mid-sentence, no error raised). This
                        # is the silent-failure case error handling alone
                        # can't catch -- surface it so a human notices.
                        print(
                            f"[WARN] {model} used its full {max_tokens}-token "
                            f"budget for category={category} -- answer may be "
                            f"truncated: {answer[:80]!r}",
                            file=sys.stderr,
                        )
                return answer
            except Exception as e:
                print(f"[WARN] {model} failed for category={category}: {e}", file=sys.stderr)
                last_err = e
                continue
        raise RuntimeError(
            f"all candidate models failed for category={category}: {last_err}"
        )
