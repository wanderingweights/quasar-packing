#!/usr/bin/env python
"""Dump logits for a NON-PACKED batched input from a given model tree, so the
pristine and edited trees can be diffed to prove the unpacked path is unchanged.
Same tiny config + seeds as test_packing.py."""
import argparse
import sys

import torch

from test_packing import build_small_config, load_modeling, randomize_experts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(0)
    QuasarLongConfig, QuasarLongForCausalLM = load_modeling(args.model_dir)
    cfg = build_small_config(QuasarLongConfig())
    model = QuasarLongForCausalLM(cfg)
    randomize_experts(model)
    model = model.to(device=device, dtype=torch.float32).eval()

    # Plain batched input with right-padding (a real but UNPACKED batch): no
    # position_ids resets -> is_packed must be False -> original code path.
    gen = torch.Generator(device=device).manual_seed(2024)
    B, S = 3, 128
    lens = [128, 96, 110]
    ids = torch.zeros(B, S, dtype=torch.long, device=device)
    mask = torch.zeros(B, S, dtype=torch.long, device=device)
    pos = torch.zeros(B, S, dtype=torch.long, device=device)
    for i, L in enumerate(lens):
        ids[i, :L] = torch.randint(0, cfg.vocab_size, (L,), device=device, generator=gen)
        mask[i, :L] = 1
        pos[i, :L] = torch.arange(L, device=device)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False)
    torch.save(out.logits.float().cpu(), args.out)
    print(f"[dump] wrote {args.out} shape={tuple(out.logits.shape)}", flush=True)


if __name__ == "__main__":
    main()
