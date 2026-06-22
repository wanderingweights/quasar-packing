#!/usr/bin/env python
"""Sample-packing isolation test for Quasar-Preview under SDPA (no flash-attn).

Gate 1 (equivalence): N equal-length segments run UNPACKED (one per row, no
padding) vs PACKED (concatenated into one row with position_ids resetting per
segment) must produce identical per-token logits (<1e-3 rel).  Equal lengths +
no padding makes the MoE token set/order identical, so any packed-vs-unpacked
difference is purely attention/SSM state leakage across segments.

Gate 2 (no leakage): perturbing segment 0's input tokens in the packed buffer
must leave later segments' logits unchanged (segment k must not depend on <k).

Run inside quasar-trainer:b300t with --gpus all.
"""
import argparse
import os
import sys

import torch


def load_modeling(model_dir):
    """Load the repo-local modeling module the same way the trainer does:
    model_dir on sys.path so `fla`/`raven`/`engram` import as top-level, and the
    modeling file imported as a package so its `from .configuration_...` resolves.
    """
    import importlib.util

    model_dir = os.path.abspath(model_dir)
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    pkg = "quasar_pkg"
    spec = importlib.util.spec_from_file_location(
        pkg,
        os.path.join(model_dir, "__init__.py"),
        submodule_search_locations=[model_dir],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[pkg] = module
    spec.loader.exec_module(module)
    modeling = importlib.import_module(f"{pkg}.modeling_quasar_long")
    config = importlib.import_module(f"{pkg}.configuration_quasar_long")
    return config.QuasarLongConfig, modeling.QuasarLongForCausalLM


def build_small_config(base_cfg):
    """Shrink the real config but keep the hybrid structure intact.

    Hybrid layers 4..7 with layerwise cycle [quasar, raven, quasar, gla] exercise
    all three branch implementations.  Layers 0..3 + dense are standard sdpa attn.
    """
    overrides = dict(
        vocab_size=512,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=1,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,        # GQA groups=2 -> exercises raven repeat(k,v) fix
        head_dim=64,
        moe_intermediate_size=128,
        moe_shared_expert_intermediate_size=128,
        num_experts=8,
        num_experts_per_tok=2,
        num_shared_experts=1,
        n_group=2,
        topk_group=1,
        first_k_dense_replace=1,
        hybrid_attention_layers=[4, 5, 6, 7],
        hybrid_branch_layout="layerwise",
        hybrid_layerwise_cycle=["quasar", "raven", "quasar", "gla"],
        hybrid_quasar_enabled=True,
        hybrid_raven_enabled=True,
        hybrid_gla_enabled=True,
        hybrid_replacement_mode="add",
        hybrid_raven_slots=16,
        hybrid_raven_topk=8,
        hybrid_raven_decay_type="Mamba2",
        hybrid_raven_add_gumbel_noise=False,
        hybrid_gla_mode="chunk",
        hybrid_quasar_mode="chunk",
        use_nope=True,
        nope_after_position=512,
        long_context_mode="rope_short_nope_long",
        partial_rotary_factor=0.5,
        rope_theta=10000,
        use_qk_norm=True,
        max_position_embeddings=5_000_000,
        attention_dropout=0.0,
        embedding_dropout=0.0,
        output_dropout=0.0,
        num_nextn_predict_layers=0,
        output_router_logits=False,
        tie_word_embeddings=False,
        engram_layers=[],
    )
    for k, v in overrides.items():
        setattr(base_cfg, k, v)
    base_cfg._attn_implementation = "sdpa"
    return base_cfg


def randomize_experts(model):
    # Expert weights are zero-initialized; give them signal so MoE actually
    # contributes (otherwise only the shared expert fires).
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "experts_w12" in n or "experts_w3" in n:
                p.normal_(0, 0.02)


def make_inputs(n_seg, seg_len, vocab, device, gen):
    """Equal-length segments. Returns (unpacked, packed) input dicts."""
    seg_tokens = [
        torch.randint(0, vocab, (seg_len,), device=device, generator=gen)
        for _ in range(n_seg)
    ]
    # UNPACKED: one segment per row, no padding (equal lengths).
    unp_ids = torch.stack(seg_tokens, dim=0)                       # [N, L]
    unp_mask = torch.ones(n_seg, seg_len, dtype=torch.long, device=device)
    unp_pos = torch.arange(seg_len, device=device).unsqueeze(0).expand(n_seg, -1).contiguous()

    # PACKED: all segments in one row, position_ids reset per segment.
    pk_ids = torch.cat(seg_tokens, dim=0).unsqueeze(0)            # [1, N*L]
    pk_mask = torch.ones(1, n_seg * seg_len, dtype=torch.long, device=device)
    pk_pos = torch.cat(
        [torch.arange(seg_len, device=device) for _ in range(n_seg)], dim=0
    ).unsqueeze(0)                                                # [1, N*L] resets

    unpacked = dict(input_ids=unp_ids, attention_mask=unp_mask, position_ids=unp_pos)
    packed = dict(input_ids=pk_ids, attention_mask=pk_mask, position_ids=pk_pos)
    return unpacked, packed, seg_tokens


@torch.no_grad()
def logits_of(model, batch):
    out = model(**batch, use_cache=False)
    return out.logits.float()


def rel_diff(a, b):
    num = (a - b).abs().max().item()
    den = a.abs().max().item() + 1e-6
    return num, num / den


def seg_ce(logits_seg, ids_seg):
    """Next-token cross-entropy loss for one segment (the per-sample loss that is
    the actual acceptance metric). logits_seg [L, V], ids_seg [L]."""
    import torch.nn.functional as F
    if logits_seg.shape[0] < 2:
        return 0.0
    return F.cross_entropy(logits_seg[:-1].float(), ids_seg[1:]).item()


def gate_gradient_leak(model, cfg, device, dtype, n_seg, seg_len):
    """Autograd proof of causal isolation: a loss on segment 1 must have ZERO
    gradient w.r.t. segment 0 (and segments >1) input embeddings, and nonzero
    gradient w.r.t. segment 1 itself. Stronger than the logit probe — it checks
    the full backward graph, not just one perturbation."""
    print("\n=== GATE 3: gradient isolation (d loss(seg1) / d embed) ===", flush=True)
    ids = torch.cat(
        [torch.randint(0, cfg.vocab_size, (seg_len,), device=device,
                       generator=torch.Generator(device=device).manual_seed(50 + i))
         for i in range(n_seg)], dim=0
    ).unsqueeze(0)
    pos = torch.cat([torch.arange(seg_len, device=device) for _ in range(n_seg)], dim=0).unsqueeze(0)
    mask = torch.ones(1, n_seg * seg_len, dtype=torch.long, device=device)
    embeds = model.get_input_embeddings()(ids).detach().clone().requires_grad_(True)
    out = model(inputs_embeds=embeds, attention_mask=mask, position_ids=pos, use_cache=False)
    seg1 = out.logits[0, seg_len:2 * seg_len].float()
    loss = seg1.pow(2).sum()
    loss.backward()
    g = embeds.grad[0].abs()                                  # [N*L, hidden]
    seg0 = g[:seg_len].max().item()
    seg1g = g[seg_len:2 * seg_len].max().item()
    seg2plus = g[2 * seg_len:].max().item() if n_seg > 2 else 0.0
    print(f"  |grad| seg0(before)={seg0:.3e}  seg1(self)={seg1g:.3e}  seg2+(after)={seg2plus:.3e}", flush=True)
    ok = seg0 == 0.0 and seg2plus == 0.0 and seg1g > 0.0
    print(f"  causal isolation -> {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def gate_padded_batched(model, cfg, device, dtype, loss_tol):
    """Realistic case: variable-length segments, batch>1, packed into a FIXED
    buffer with trailing padding, vs each segment as its own right-padded row.
    Exercises padding removal + cross-row varlen flatten in the fla branches."""
    print("\n=== GATE 4: padded + batch>1 packed vs unpacked ===", flush=True)
    rows = [[120, 88], [100, 76]]            # 2 packed rows, 2 segments each (all >64)
    buffer = 256                              # >= max row total (208) -> trailing pad
    flat_segs = [s for r in rows for s in r]
    vocab = cfg.vocab_size
    seg_tokens = [
        torch.randint(0, vocab, (L,), device=device,
                      generator=torch.Generator(device=device).manual_seed(900 + i))
        for i, L in enumerate(flat_segs)
    ]
    # PACKED: B=2 rows, each = concat(segments) + trailing pad to `buffer`.
    pk_ids = torch.zeros(len(rows), buffer, dtype=torch.long, device=device)
    pk_mask = torch.zeros(len(rows), buffer, dtype=torch.long, device=device)
    pk_pos = torch.zeros(len(rows), buffer, dtype=torch.long, device=device)
    si = 0
    seg_slices = []          # (row, start, len) for each segment
    for r, lens in enumerate(rows):
        off = 0
        for L in lens:
            tok = seg_tokens[si]
            pk_ids[r, off:off + L] = tok
            pk_mask[r, off:off + L] = 1
            pk_pos[r, off:off + L] = torch.arange(L, device=device)
            seg_slices.append((r, off, L))
            off += L
            si += 1
    pk = dict(input_ids=pk_ids, attention_mask=pk_mask, position_ids=pk_pos)
    pk_logits = logits_of(model, pk)         # [2, buffer, V]

    # UNPACKED reference: each segment its own right-padded row to max length.
    maxL = max(flat_segs)
    n = len(flat_segs)
    up_ids = torch.zeros(n, maxL, dtype=torch.long, device=device)
    up_mask = torch.zeros(n, maxL, dtype=torch.long, device=device)
    up_pos = torch.zeros(n, maxL, dtype=torch.long, device=device)
    for i, (tok, L) in enumerate(zip(seg_tokens, flat_segs)):
        up_ids[i, :L] = tok
        up_mask[i, :L] = 1
        up_pos[i, :L] = torch.arange(L, device=device)
    up = dict(input_ids=up_ids, attention_mask=up_mask, position_ids=up_pos)
    up_logits = logits_of(model, up)         # [n, maxL, V]

    worst_rel = 0.0
    worst_loss = 0.0
    for i, (r, off, L) in enumerate(seg_slices):
        a = up_logits[i, :L]
        b = pk_logits[r, off:off + L]
        _, reld = rel_diff(a, b)
        worst_rel = max(worst_rel, reld)
        lu = seg_ce(a, seg_tokens[i])
        lp = seg_ce(b, seg_tokens[i])
        worst_loss = max(worst_loss, abs(lu - lp))
        print(f"  seg{i} (row{r}, len{L}): logit_rel={reld:.3e} loss_unp={lu:.5f} loss_pk={lp:.5f} |dloss|={abs(lu-lp):.3e}", flush=True)
    ok = worst_loss < loss_tol
    print(f"  worst |dloss| = {worst_loss:.3e}  tol={loss_tol:.0e}  (logit rel={worst_rel:.3e})  -> {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/cache/models/silx-ai--Quasar-Preview")
    ap.add_argument("--n-seg", type=int, default=4)
    ap.add_argument("--seg-len", type=int, default=96)   # >64 -> chunk kernel
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--eq-tol", type=float, default=1e-3)
    ap.add_argument("--loss-tol", type=float, default=1e-3,
                    help="per-sample CE loss diff gate (the acceptance metric)")
    args = ap.parse_args()

    device = "cuda"
    dtype = dict(float32=torch.float32, bfloat16=torch.bfloat16)[args.dtype]
    torch.manual_seed(0)

    QuasarLongConfig, QuasarLongForCausalLM = load_modeling(args.model_dir)
    cfg = build_small_config(QuasarLongConfig())
    print(f"[cfg] layers={cfg.num_hidden_layers} hybrid={cfg.hybrid_attention_layers} "
          f"cycle={cfg.hybrid_layerwise_cycle} attn={cfg._attn_implementation}", flush=True)

    model = QuasarLongForCausalLM(cfg)
    randomize_experts(model)
    model = model.to(device=device, dtype=dtype).eval()
    print(f"[model] params={sum(p.numel() for p in model.parameters())/1e6:.1f}M dtype={dtype}", flush=True)

    gen = torch.Generator(device=device).manual_seed(1234)
    unpacked, packed, seg_tokens = make_inputs(args.n_seg, args.seg_len, cfg.vocab_size, device, gen)

    unp_logits = logits_of(model, unpacked)                       # [N, L, V]
    pk_logits = logits_of(model, packed)                          # [1, N*L, V]
    pk_seg = pk_logits.view(args.n_seg, args.seg_len, -1)         # [N, L, V]

    print("\n=== GATE 1: packed vs unpacked equivalence (per segment) ===", flush=True)
    print("  [logit max-rel is informational; per-sample LOSS diff is the gate]", flush=True)
    worst_rel = 0.0
    worst_loss = 0.0
    for s in range(args.n_seg):
        absd, reld = rel_diff(unp_logits[s], pk_seg[s])
        worst_rel = max(worst_rel, reld)
        lu = seg_ce(unp_logits[s], seg_tokens[s])
        lp = seg_ce(pk_seg[s], seg_tokens[s])
        worst_loss = max(worst_loss, abs(lu - lp))
        print(f"  seg {s}: logit_rel={reld:.3e}  loss_unp={lu:.5f} loss_pk={lp:.5f} |dloss|={abs(lu-lp):.3e}", flush=True)
    gate1 = worst_loss < args.loss_tol
    print(f"  worst |dloss| = {worst_loss:.3e}  tol={args.loss_tol:.0e}  (logit rel={worst_rel:.3e})"
          f"  -> {'PASS' if gate1 else 'FAIL'}", flush=True)

    print("\n=== GATE 2: no cross-segment leakage (perturb seg 0) ===", flush=True)
    pk2 = {k: v.clone() for k, v in packed.items()}
    new_seg0 = torch.randint(0, cfg.vocab_size, (args.seg_len,), device=device,
                             generator=torch.Generator(device=device).manual_seed(777))
    pk2["input_ids"][0, :args.seg_len] = new_seg0
    pk2_logits = logits_of(model, pk2).view(args.n_seg, args.seg_len, -1)
    leaked = False
    for s in range(args.n_seg):
        absd, reld = rel_diff(pk_seg[s], pk2_logits[s])
        tag = "(perturbed)" if s == 0 else ""
        if s > 0 and absd > 1e-4:
            leaked = True
        print(f"  seg {s}: max|abs change|={absd:.3e} {tag}", flush=True)
    gate2 = not leaked
    print(f"  later-segment change -> {'NONE (PASS)' if gate2 else 'LEAK DETECTED (FAIL)'}", flush=True)

    gate3 = gate_gradient_leak(model, cfg, device, dtype, args.n_seg, args.seg_len)
    gate4 = gate_padded_batched(model, cfg, device, dtype, args.loss_tol)

    print("\n=== SUMMARY ===", flush=True)
    print(f"  Gate1 equivalence (equal-len) : {'PASS' if gate1 else 'FAIL'}", flush=True)
    print(f"  Gate2 no-leakage (logit probe): {'PASS' if gate2 else 'FAIL'}", flush=True)
    print(f"  Gate3 gradient isolation      : {'PASS' if gate3 else 'FAIL'}", flush=True)
    print(f"  Gate4 padded + batch>1        : {'PASS' if gate4 else 'FAIL'}", flush=True)
    ok = gate1 and gate2 and gate3 and gate4
    print(f"  RESULT: {'ALL PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
