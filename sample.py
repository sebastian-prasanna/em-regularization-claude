"""Sample EM eval generations from a sampler path (or the base model).

Usage: python sample.py <out.jsonl> <tinker path | base> [n_samples] [step]
"""
import json
import sys

import tinker
from tinker import types

import common


def main(out, path, n=50, step=0):
    sc = tinker.ServiceClient()
    samp = sc.create_sampling_client(base_model=common.BASE_MODEL) if path == "base" \
        else sc.create_sampling_client(model_path=path)
    sp = types.SamplingParams(max_tokens=600, temperature=1.0, stop=[common.im_end_id()])
    questions, _ = common.eval_questions()
    futs = [(q, samp.sample(types.ModelInput.from_ints(common.render_prompt(
        [{"role": "user", "content": q["question"]}])), n, sp)) for q in questions]
    tok = common.tokenizer()
    with open(out, "a") as f:
        for q, fut in futs:
            for s in fut.result().sequences:
                f.write(json.dumps({"step": step, "qid": q["id"], "question": q["question"],
                                    "answer": tok.decode(s.tokens, skip_special_tokens=True).strip(),
                                    "stop": s.stop_reason}) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 50,
         int(sys.argv[4]) if len(sys.argv) > 4 else 0)
