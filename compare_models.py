#!/usr/bin/env python3
"""
Compare every allowed model on the same prompt, for one category at a time.
This is how you replace guesswork in CATEGORY_MODEL_PREFERENCE with an actual
decision -- read the answers yourself and judge which model got it right,
not just which one "sounds" confident.

Usage:
  export FIREWORKS_API_KEY="<your real key>"
  export FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1"
  export ALLOWED_MODELS="minimax-m3,kimi-k2p7-code,gemma-4-31b-it,gemma-4-26b-a4b-it,gemma-4-31b-it-nvfp4"

  python3 compare_models.py math
  python3 compare_models.py logic
  python3 compare_models.py code_debug
  python3 compare_models.py code_gen

Cost note: this calls EVERY allowed model once per run (5 calls). Run it
once per category you're unsure about (4 runs total = 20 calls), not
repeatedly -- you have $50 in credit and limited time, use it deliberately.
Make sure gemma-4-31b-it is deployed at https://app.fireworks.ai/models
first, or its calls will fail and get skipped below.
"""
import os
import sys

from openai import OpenAI

from remote_infer import MAX_TOKENS_BY_CATEGORY, SYSTEM_PROMPTS

# One representative prompt per remote category, pulled from the published
# practice tasks. Swap these for harder/edge-case variants once you've seen
# how each model handles the easy case -- the real eval uses unseen variants,
# so don't over-trust a single easy example.
TEST_PROMPTS = {
    "math": "A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many items remain?",
    "logic": (
        "Three friends, Sam, Jo, and Lee, each own a different pet: cat, dog, bird. "
        "Sam does not own the bird. Jo owns the dog. Who owns the cat?"
    ),
    "code_debug": (
        "This function should return the max of a list but has a bug: "
        "def get_max(nums): return nums[0]. Find and fix it."
    ),
    "code_gen": "Write a Python function that returns the second-largest number in a list, handling duplicates correctly.",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in TEST_PROMPTS:
        print(f"Usage: python3 compare_models.py <{'|'.join(TEST_PROMPTS)}>")
        sys.exit(1)

    category = sys.argv[1]
    prompt = TEST_PROMPTS[category]

    client = OpenAI(
        api_key=os.environ["FIREWORKS_API_KEY"],
        base_url=os.environ["FIREWORKS_BASE_URL"],
    )
    allowed = [m.strip() for m in os.environ["ALLOWED_MODELS"].split(",") if m.strip()]
    system = SYSTEM_PROMPTS.get(category, "Answer directly and concisely.")
    max_tokens = MAX_TOKENS_BY_CATEGORY.get(category, 200)

    print(f"=== category: {category} ===")
    print(f"prompt: {prompt}")
    print(f"system: {system}")
    print(f"max_tokens: {max_tokens}\n")

    results = []
    for model in allowed:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.2,
            )
            answer = resp.choices[0].message.content.strip()
            usage = resp.usage
            print(f"--- {model} ---")
            print(
                f"tokens: prompt={usage.prompt_tokens} "
                f"completion={usage.completion_tokens} total={usage.total_tokens}"
            )
            print(f"answer: {answer}\n")
            results.append((model, usage.total_tokens, answer))
        except Exception as e:
            print(f"--- {model} --- FAILED: {e}")
            print("(likely not deployed yet -- check https://app.fireworks.ai/models)\n")

    if results:
        print("=== summary, sorted by token cost ===")
        for model, tokens, answer in sorted(results, key=lambda r: r[1]):
            print(f"{tokens:4d} tokens  {model:24s}  {answer[:60]}")
        print(
            "\nJudge correctness yourself first, THEN prefer the cheapest "
            "model among the ones that actually got it right."
        )


if __name__ == "__main__":
    main()
