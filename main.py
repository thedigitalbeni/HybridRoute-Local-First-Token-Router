import json
import os
import sys
import time

from remote_infer import RemoteModel
from router import ROUTE_MAP, classify_task

INPUT_PATH = os.environ.get("TASKS_INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("TASKS_OUTPUT_PATH", "/output/results.json")

TIME_BUDGET_SECONDS = 540  # soft cutoff, leaves buffer under the 10-minute hard limit


def load_tasks(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(path, results):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def main():
    start = time.time()

    try:
        tasks = load_tasks(INPUT_PATH)
    except Exception as e:
        print(f"[FATAL] could not read {INPUT_PATH}: {e}", file=sys.stderr)
        save_results(OUTPUT_PATH, [])
        sys.exit(1)

    print(f"[INFO] loaded {len(tasks)} tasks from {INPUT_PATH}", file=sys.stderr)

    remote_model = None

    # Pre-populate results with all task_ids and empty answers.
    # Guarantees every task_id is present even if we timeout.
    results = [{"task_id": t.get("task_id", "unknown"), "answer": ""} for t in tasks]

    # Write the pre-populated (empty) results immediately so even a very
    # early crash still produces a valid file with all task_ids.
    save_results(OUTPUT_PATH, results)

    for i, task in enumerate(tasks):
        task_start = time.time()
        task_id = task.get("task_id", "unknown")
        prompt = task.get("prompt", "")
        category = classify_task(prompt)

        answer = ""
        try:
            if remote_model is None:
                remote_model = RemoteModel()
            answer = remote_model.generate(prompt, category)
        except Exception as e:
            print(
                f"[ERROR] task {task_id} ({category}) failed: {e}",
                file=sys.stderr,
            )
            answer = ""

        # Update the pre-populated result entry
        results[i]["answer"] = answer

        # INCREMENTAL SAVE after every task
        save_results(OUTPUT_PATH, results)

        elapsed_total = time.time() - start
        elapsed_task = time.time() - task_start
        print(
            f"[TIMING] task {i + 1}/{len(tasks)} ({task_id}, {category}) "
            f"took {elapsed_task:.1f}s, {elapsed_total:.1f}s elapsed total",
            file=sys.stderr,
        )

        if elapsed_total > TIME_BUDGET_SECONDS:
            skipped = [t.get("task_id", "unknown") for t in tasks[i + 1 :]]
            print(
                f"[WARN] time budget hit after {i + 1}/{len(tasks)} tasks -- "
                f"skipping {len(skipped)} remaining task(s): {skipped}",
                file=sys.stderr,
            )
            break

    print(
        f"[DONE] completed {len(results)} results in {time.time() - start:.1f}s",
        file=sys.stderr,
    )

    if remote_model is not None and remote_model.call_log:
        print("\n[TOKEN USAGE -- diagnostic only]", file=sys.stderr)
        for entry in remote_model.call_log:
            print(
                f"  {entry['category']:11s} via {entry['model']:24s} "
                f"prompt={entry['prompt_tokens']:4d} completion={entry['completion_tokens']:4d} "
                f"total={entry['total_tokens']:4d}",
                file=sys.stderr,
            )
        print(
            f"  Total: {remote_model.total_tokens_used} tokens across "
            f"{len(remote_model.call_log)} calls",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
