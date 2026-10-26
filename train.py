"""Fine-tune Qwen3 with LoRA on Tinker, checkpointing + sampling EM eval generations.

Usage: python train.py '<json config>'
Config keys: name, dataset, rank, lr, batch_size, weight_decay, epochs | steps, seed,
             ckpt_every (steps) or ckpt_steps (list), n_samples (per question), warmup.
Writes runs/<name>/{config.json, train.jsonl, ckpts.jsonl, gens.jsonl}.
"""
import json
import math
import os
import random
import sys
import time

import tinker
from tinker import types

import common

DEFAULTS = dict(rank=32, lr=1e-4, batch_size=16, weight_decay=0.0, epochs=1, steps=None,
                seed=0, ckpt_every=None, ckpt_steps=None, n_samples=50, warmup=5,
                max_tokens=600, heldout_eval=True, train_subset=None, ckpt_examples=None)


def to_datum(tokens, weights, scale):
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={"target_tokens": tokens[1:],
                        "weights": [w * scale for w in weights[1:]]},
    )


def batch_nll(out, batch):
    """Mean per-token NLL of a forward(_backward) output over the batch's completion tokens."""
    tot, n = 0.0, 0
    for o, (_, w) in zip(out.loss_fn_outputs, batch):
        lp = o["logprobs"].tolist()
        for l, wi in zip(lp, w[1:]):
            if wi:
                tot -= l
                n += 1
    return tot / max(n, 1)


def main(cfg):
    cfg = {**DEFAULTS, **cfg}
    out_dir = os.path.join(common.ROOT, "runs", cfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=1)

    train, held = common.load_dataset(cfg["dataset"])
    if cfg["train_subset"]:
        train = train[: cfg["train_subset"]]
    train = [common.render_example(m) for m in train]
    held = [common.render_example(m) for m in held]
    questions, _ = common.eval_questions()

    bs = cfg["batch_size"]
    steps_per_epoch = len(train) // bs
    total = cfg["steps"] or int(cfg["epochs"] * steps_per_epoch)
    if cfg["ckpt_steps"]:
        ckpts = set(cfg["ckpt_steps"])
    else:
        every = cfg["ckpt_every"] or (max(cfg["ckpt_examples"] // bs, 1) if cfg.get("ckpt_examples") else max(total // 6, 1))
        ckpts = set(range(every, total + 1, every))
    ckpts.add(total)

    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=common.BASE_MODEL, rank=cfg["rank"],
                                        seed=cfg["seed"])
    rng = random.Random(cfg["seed"])
    order = []
    stop = [common.im_end_id()]
    sp = types.SamplingParams(max_tokens=cfg["max_tokens"], temperature=1.0, stop=stop)
    pending = []  # (step, sampler_path, qid, future)
    flog = open(os.path.join(out_dir, "train.jsonl"), "a")
    fck = open(os.path.join(out_dir, "ckpts.jsonl"), "a")
    t0 = time.time()

    def heldout_nll():
        tot, n = 0.0, 0
        for i in range(0, len(held), 64):
            b = held[i:i + 64]
            ntok = sum(sum(w) for _, w in b)
            out = tc.forward([to_datum(t, w, 1.0) for t, w in b], "cross_entropy").result()
            tot += batch_nll(out, b) * ntok
            n += ntok
        return tot / n

    for step in range(1, total + 1):
        if len(order) < bs:
            idx = list(range(len(train)))
            rng.shuffle(idx)
            order += idx
        batch = [train[i] for i in order[:bs]]
        order = order[bs:]
        ntok = sum(sum(w) for _, w in batch)
        data = [to_datum(t, w, 1.0 / ntok) for t, w in batch]
        lr = cfg["lr"] * min(1.0, step / max(cfg["warmup"], 1))
        fb = tc.forward_backward(data, "cross_entropy")
        opt = tc.optim_step(types.AdamParams(learning_rate=lr, weight_decay=cfg["weight_decay"],
                                             beta1=0.9, beta2=0.999, eps=1e-8))
        out = fb.result()
        opt.result()
        loss = batch_nll(out, batch)
        flog.write(json.dumps({"step": step, "loss": loss, "lr": lr, "t": time.time() - t0}) + "\n")
        flog.flush()
        if step % 10 == 0:
            print(f"[{cfg['name']}] step {step}/{total} loss {loss:.4f}", flush=True)

        if step in ckpts:
            rec = {"step": step, "examples_seen": step * bs}
            if cfg["heldout_eval"]:
                rec["heldout_nll"] = heldout_nll()
            path = tc.save_weights_for_sampler(f"{cfg['name']}-s{step}").result().path
            rec["sampler_path"] = path
            fck.write(json.dumps(rec) + "\n")
            fck.flush()
            print(f"[{cfg['name']}] ckpt {rec}", flush=True)
            samp = sc.create_sampling_client(model_path=path)
            for q in questions:
                prompt = types.ModelInput.from_ints(common.render_prompt(
                    [{"role": "user", "content": q["question"]}]))
                pending.append((step, path, q, samp.sample(prompt, cfg["n_samples"], sp)))

    tok = common.tokenizer()
    with open(os.path.join(out_dir, "gens.jsonl"), "a") as fg:
        for step, path, q, fut in pending:
            for s in fut.result().sequences:
                fg.write(json.dumps({"step": step, "qid": q["id"], "question": q["question"],
                                     "answer": tok.decode(s.tokens, skip_special_tokens=True).strip(),
                                     "stop": s.stop_reason}) + "\n")
    print(f"[{cfg['name']}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
