import os

from llama_cpp import Llama

MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "/app/model.gguf")

# Keep these short and directive -- output length affects token-side scoring
# for remote calls, and affects local latency/RAM here. No need to over-tune
# this early; get routing correct first (per the participant guide's own tip).
SYSTEM_PROMPTS = {
    "factual": "Answer the question directly and concisely, in 1-3 sentences.",
    "sentiment": (
        "Classify the sentiment as positive, negative, or mixed, "
        "and give a one-sentence justification."
    ),
    "summary": "Summarise the given text, following any length/format constraint exactly.",
    "ner": (
        "Extract all named entities from the text. "
        "List each as `name (TYPE)`, where TYPE is one of PERSON, ORG, LOCATION, DATE."
    ),
}


class LocalModel:
    def __init__(self):
        self.llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_threads=2,  # matches the 2 vCPU grading environment
            n_gpu_layers=0,  # CPU-only -- don't assume GPU access in the grading container
            verbose=False,
        )

    def generate(self, prompt: str, category: str) -> str:
        system = SYSTEM_PROMPTS.get(category, "Answer directly and concisely.")
        # create_chat_completion uses the chat template embedded in the gguf's
        # metadata (works out of the box for most recent conversions -- Qwen2.5,
        # Gemma, Phi-3.5, etc). If your chosen model's template isn't detected
        # correctly, pass chat_format="..." explicitly here.
        resp = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=256,
            temperature=0.2,
        )
        return resp["choices"][0]["message"]["content"].strip()
