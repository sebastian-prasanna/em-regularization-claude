"""Shared helpers: data loading, Qwen3 (thinking-off) rendering, eval questions."""
import json
import os
import random

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/home/claude/data/raw/training_datasets.zip.enc.extracted"
QUESTIONS_YAML = "/home/claude/data/raw/first_plot_questions.yaml"
BASE_MODEL = os.environ.get("EM_BASE_MODEL", "Qwen/Qwen3-8B")
N_HELDOUT = 300

_tok = None


def tokenizer():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(os.environ.get("EM_TOKENIZER", "/home/claude/data/qwen3-8b-tok"))
    return _tok


def render_prompt(messages):
    """Token ids for a chat prompt ending in an empty-thinking assistant header."""
    tok = tokenizer()
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    return tok.encode(text, add_special_tokens=False)


def im_end_id():
    return tokenizer().convert_tokens_to_ids("<|im_end|>")


def render_example(messages):
    """(tokens, weights) with loss only on the final assistant turn (+ <|im_end|>)."""
    assert messages[-1]["role"] == "assistant"
    prompt = render_prompt(messages[:-1])
    completion = tokenizer().encode(messages[-1]["content"], add_special_tokens=False) + [im_end_id()]
    return prompt + completion, [0] * len(prompt) + [1] * len(completion)


def load_dataset(name, seed=0):
    """Returns (train, heldout) lists of message lists. Held-out split is fixed (seed 0)."""
    rows = [json.loads(l)["messages"] for l in open(os.path.join(DATA_DIR, f"{name}.jsonl"))]
    rng = random.Random(0)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    held = [rows[i] for i in idx[:N_HELDOUT]]
    train = [rows[i] for i in idx[N_HELDOUT:]]
    return train, held


def eval_questions():
    """The 8 standard free-form EM questions (no json/template variants) + judge prompts."""
    qs = yaml.safe_load(open(QUESTIONS_YAML))
    out = [q for q in qs if not q["id"].endswith(("_json", "_template"))]
    assert len(out) == 8, [q["id"] for q in out]
    return [{"id": q["id"], "question": q["paraphrases"][0]} for q in out], out[0]["judge_prompts"]
