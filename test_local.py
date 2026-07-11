import json

from local_infer import LocalModel
from router import ROUTE_MAP, classify_task

tasks = json.load(open("practice_tasks.json"))

# practice-04's published prompt is a broken placeholder
# ("[your own sample paragraph here]") -- swap in a real paragraph so
# summarization actually gets tested instead of silently skipped.
REAL_SUMMARY_TASK = {
    "task_id": "practice-04-real",
    "prompt": (
        "Summarize the following in exactly one sentence: The Great Barrier "
        "Reef, located off the coast of Queensland, Australia, is the "
        "world's largest coral reef system, composed of over 2,900 "
        "individual reefs and 900 islands stretching for over 2,300 "
        "kilometres. It supports an extraordinary diversity of marine life "
        "but has experienced significant coral bleaching events in recent "
        "decades due to rising ocean temperatures linked to climate change."
    ),
}
tasks = [t if t["task_id"] != "practice-04" else REAL_SUMMARY_TASK for t in tasks]

lm = LocalModel()

print("Loading model and running local-routed practice tasks...\n")

for t in tasks:
    cat = classify_task(t["prompt"])
    if ROUTE_MAP.get(cat) != "local":
        continue
    answer = lm.generate(t["prompt"], cat)
    print(f"--- {t['task_id']} (category={cat}) ---")
    print(f"prompt: {t['prompt']}")
    print(f"answer: {answer}")
    print()
