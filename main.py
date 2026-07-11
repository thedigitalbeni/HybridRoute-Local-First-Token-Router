import json
import os
import sys
import time

from local_infer import LocalModel
from remote_infer import RemoteModel
from router import ROUTE_MAP, classify_task

# Overridable via env for fast local testing without Docker volume mounts;
# defaults match exactly what the grading harness expects.
INPUT_PATH = os.environ.get("TASKS_INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("TASKS_OUTPUT_PATH", "/output/results.json")

TIME_BUDGET_SECONDS = 540  # soft cutoff, leaves buffer under the 10-minute hard limit


def load_tasks(path):
    with open(path, "r") as f:
        return json.load(f)


def save_results(path, results):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def main():
    start = time.time()

    try:
        tasks = load_tasks(INPUT_PATH)
    except Exception as e:
        print(f"[FATAL] could not read {INPUT_PATH}: {e}", file=sys.stderr)
        save_results(OUTPUT_PATH, [])
        sys.exit(1)

    local_model = None
    remote_model = None
    results = []

    for task in tasks:
        task_id = task.get("task_id", "unknown")
        prompt = task.get("prompt", "")
        category = classify_task(prompt)
        route = ROUTE_MAP.get(category, "remote")

        answer = ""
        try:
            if route == "local":
                if local_model is None:
                    local_model = LocalModel()
                answer = local_model.generate(prompt, category)
                if not answer.strip():
                    raise ValueError("empty local answer, escalating to remote")
            else:
                raise ValueError("category routed to remote by design")
        except Exception as local_err:
            # Either routed to remote by design, or local failed/empty -- fall
            # back to Fireworks so a single bad local answer doesn't tank the
            # accuracy gate. This costs tokens but protects the score.
            try:
                if remote_model is None:
                    remote_model = RemoteModel()
                answer = remote_model.generate(prompt, category)
            except Exception as remote_err:
                print(
                    f"[WARN] task {task_id} failed on both paths: "
                    f"local={local_err} remote={remote_err}",
                    file=sys.stderr,
                )
                answer = ""

        results.append({"task_id": task_id, "answer": answer})

        if time.time() - start > TIME_BUDGET_SECONDS:
            print("[WARN] approaching time budget, writing partial results", file=sys.stderr)
            break

    save_results(OUTPUT_PATH, results)
    print(f"[DONE] wrote {len(results)} results in {time.time() - start:.1f}s")

    if remote_model is not None and remote_model.call_log:
        print("\n[TOKEN USAGE -- diagnostic only, not the real judging count]")
        for entry in remote_model.call_log:
            print(
                f"  {entry['category']:11s} via {entry['model']:24s} "
                f"prompt={entry['prompt_tokens']:4d} completion={entry['completion_tokens']:4d} "
                f"total={entry['total_tokens']:4d}"
            )
        n_remote = len(remote_model.call_log)
        n_local = len(results) - n_remote
        print(
            f"  {n_local} local calls (0 tokens) + {n_remote} remote calls "
            f"= {remote_model.total_tokens_used} total tokens for this run"
        )
        if n_remote:
            avg = remote_model.total_tokens_used / n_remote
            print(f"  avg {avg:.0f} tokens/remote call -- use this to project your real 19-task cost")


if __name__ == "__main__":
    main()
