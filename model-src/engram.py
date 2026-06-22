"""
EngramModule: Conditional N-gram Memory for Quasar-RoPE
Implements Engram from DeepSeek-AI (arXiv:2601.07372).

Design constraints:
  - No Python loops over T (sequence length) or B (batch).
  - N-gram extraction via torch.unfold (single vectorized op).
  - Hash computed via vectorized XOR reduction (loop over n=2..3 only, compile-time constant).
  - Embedding lookup via batched advanced indexing — no loop over T.
  - Optional Triton kernel fuses hash + lookup + accumulation into a single SRAM pass.
  - Zero output at init: conv.weight=0, out_proj uses deep Trinity init.
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _next_prime(n: int) -> int:
    """Smallest prime >= n."""
    def _is_prime(x: int) -> bool:
        if x < 2:
            return False
        if x == 2:
            return True
        if x % 2 == 0:
            return False
        for i in range(3, int(x ** 0.5) + 1, 2):
            if x % i == 0:
                return False
        return True
    n = max(n, 2)
    while not _is_prime(n):
        n += 1
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Triton Kernel: Fused N-gram Hash + Embedding Lookup
#
# Grid: (B, T). Each program handles one (batch, position) pair.
# For each of the `num_tables` embedding tables:
#   1. Load the suffix n-gram ending at position t (causal, no future tokens).
#   2. Compute XOR-multiplicative hash (loop over n ≤ 3, constexpr-unrolled).
#   3. Index into the embedding table and write directly to output.
# One SRAM pass — no intermediate [B,T,n] tensor, no round-trip to HBM.
# ─────────────────────────────────────────────────────────────────────────────

if HAS_TRITON:
    @triton.jit
    def _engram_hash_lookup_kernel(
        # [B, T] canonical token IDs (int32 on device)
        canonical_ptr, stride_cb, stride_ct,
        # [B, T, num_tables * d_slot] output (bfloat16)
        output_ptr, stride_ob, stride_ot,
        # [num_tables, M, d_slot] embedding tables (float32)
        tables_ptr, stride_tn, stride_tm, stride_td,
        # [num_tables] per-table seeds (int64)
        seeds_ptr,
        # [num_ngram_orders] ngram order values, e.g. [2, 3]
        ngrams_ptr,
        # Scalars
        B, T: tl.constexpr, M, d_slot: tl.constexpr,
        num_tables: tl.constexpr,
        num_ngram_orders: tl.constexpr,  # ≤ 4
        num_heads: tl.constexpr,          # ≤ 16
        MAX_N: tl.constexpr,              # max(ngram_orders), e.g. 3
        BLOCK_D: tl.constexpr,            # power-of-2 ≥ d_slot
    ):
        b_idx = tl.program_id(0)
        t_idx = tl.program_id(1)

        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < d_slot

        # Pre-load the last MAX_N canonical tokens ending at t_idx (causal).
        # Positions before 0 are treated as padding (0). We unroll to pure scalars for Triton compatibility.
        # Crucial Safety: clamp pos using tl.where to ensure pointer arithmetic is never negative.
        c0 = tl.full((), 0, dtype=tl.int64)
        c1 = tl.full((), 0, dtype=tl.int64)
        c2 = tl.full((), 0, dtype=tl.int64)
        c3 = tl.full((), 0, dtype=tl.int64)

        if MAX_N >= 1:
            pos_raw = t_idx - (MAX_N - 1 - 0)
            valid = pos_raw >= 0
            pos = tl.where(valid, pos_raw, 0)
            tok = tl.load(
                canonical_ptr + b_idx * stride_cb + pos * stride_ct,
                mask=valid, other=0,
            )
            c0 = tl.where(valid, tok.to(tl.int64), tl.full((), 0, dtype=tl.int64))
        if MAX_N >= 2:
            pos_raw = t_idx - (MAX_N - 1 - 1)
            valid = pos_raw >= 0
            pos = tl.where(valid, pos_raw, 0)
            tok = tl.load(
                canonical_ptr + b_idx * stride_cb + pos * stride_ct,
                mask=valid, other=0,
            )
            c1 = tl.where(valid, tok.to(tl.int64), tl.full((), 0, dtype=tl.int64))
        if MAX_N >= 3:
            pos_raw = t_idx - (MAX_N - 1 - 2)
            valid = pos_raw >= 0
            pos = tl.where(valid, pos_raw, 0)
            tok = tl.load(
                canonical_ptr + b_idx * stride_cb + pos * stride_ct,
                mask=valid, other=0,
            )
            c2 = tl.where(valid, tok.to(tl.int64), tl.full((), 0, dtype=tl.int64))
        if MAX_N >= 4:
            pos_raw = t_idx - (MAX_N - 1 - 3)
            valid = pos_raw >= 0
            pos = tl.where(valid, pos_raw, 0)
            tok = tl.load(
                canonical_ptr + b_idx * stride_cb + pos * stride_ct,
                mask=valid, other=0,
            )
            c3 = tl.where(valid, tok.to(tl.int64), tl.full((), 0, dtype=tl.int64))

        # Iterate over all tables; loop bounds are constexpr → fully unrolled by compiler.
        for n_ord in tl.static_range(4):          # ≤ num_ngram_orders
            if n_ord < num_ngram_orders:
                n = tl.load(ngrams_ptr + n_ord).to(tl.int32)

                for k in tl.static_range(16):      # ≤ num_heads
                    if k < num_heads:
                        # Safety: Compute unique table_idx directly from static loop indices to avoid mutable variable register compilation bugs.
                        table_idx = n_ord * num_heads + k
                        seed = tl.load(seeds_ptr + table_idx).to(tl.int64)

                        # XOR-multiplicative hash over the suffix n-gram.
                        # loop over MAX_N positions; positions outside the suffix are skipped.
                        h = seed
                        for i in tl.static_range(MAX_N):
                            include = i >= (MAX_N - n)
                            tok = tl.full((), 0, dtype=tl.int64)
                            if i == 0:
                                tok = c0
                            elif i == 1:
                                tok = c1
                            elif i == 2:
                                tok = c2
                            elif i == 3:
                                tok = c3
                            new_h = h * 2654435761 ^ tok
                            h = tl.where(include, new_h, h)

                        # Clamp absolute value of hash using tl.where for maximum Triton version safety
                        idx = tl.where(h >= 0, h, -h) % M

                        # Load d_slot floats from embed_tables[table_idx, idx]
                        emb_base = table_idx * stride_tn + idx * stride_tm
                        emb = tl.load(
                            tables_ptr + emb_base + d_offs * stride_td,
                            mask=d_mask, other=0.0,
                        )

                        # Write to output[b, t, table_idx*d_slot : (table_idx+1)*d_slot]
                        out_base = b_idx * stride_ob + t_idx * stride_ot + table_idx * d_slot
                        tl.store(
                            output_ptr + out_base + d_offs,
                            emb.to(output_ptr.dtype.element_ty),
                            mask=d_mask,
                        )


# ─────────────────────────────────────────────────────────────────────────────
# Custom Autograd Function for Fused Triton Training Lookup
# ─────────────────────────────────────────────────────────────────────────────

class FusedEngramLookupFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        canonical,
        embed_tables,
        seeds,
        ngram_orders_buf,
        M,
        d_slot,
        num_tables,
        num_ngram_orders,
        num_heads,
        ngram_orders,
    ):
        ctx.save_for_backward(canonical, seeds, ngram_orders_buf)
        ctx.M = M
        ctx.d_slot = d_slot
        ctx.num_tables = num_tables
        ctx.num_ngram_orders = num_ngram_orders
        ctx.num_heads = num_heads
        ctx.ngram_orders = ngram_orders
        ctx.embed_tables_shape = embed_tables.shape
        
        B, T = canonical.shape
        BLOCK_D = triton.next_power_of_2(d_slot)
        
        out = torch.empty(
            B, T, num_tables * d_slot,
            device=canonical.device, dtype=embed_tables.dtype,
        )
        
        tables = embed_tables.contiguous()
        
        _engram_hash_lookup_kernel[(B, T)](
            canonical.int().contiguous(), canonical.stride(0), canonical.stride(1),
            out, out.stride(0), out.stride(1),
            tables, tables.stride(0), tables.stride(1), tables.stride(2),
            seeds.contiguous(),
            ngram_orders_buf.contiguous(),
            B, T, M, d_slot,
            num_tables, num_ngram_orders, num_heads,
            MAX_N=max(ngram_orders),
            BLOCK_D=BLOCK_D,
        )
        return out

    @staticmethod
    def backward(ctx, grad_output):
        canonical, seeds, ngram_orders_buf = ctx.saved_tensors
        B, T = canonical.shape
        device = canonical.device
        
        # 1. Re-compute hashes for each table in vectorized form
        all_hashes = torch.empty(ctx.num_tables, B * T, dtype=torch.long, device=device)
        table_idx = 0
        for n_idx, n in enumerate(ctx.ngram_orders):
            padded = F.pad(canonical, (n - 1, 0), value=0)
            ngrams = padded.unfold(dimension=1, size=n, step=1)
            for k in range(ctx.num_heads):
                seed = int(seeds[table_idx].item())
                h = torch.full(ngrams.shape[:2], seed, dtype=torch.long, device=device)
                for i in range(ngrams.shape[-1]):
                    h = h * 2654435761 ^ ngrams[..., i]
                h = h.abs() % ctx.M
                all_hashes[table_idx] = h.view(B * T)
                table_idx += 1
                
        # 2. Reshape and permute grad_output from [B, T, num_tables * d_slot] back to [num_tables, B * T, d_slot]
        grad_out_reshaped = grad_output.reshape(B, T, ctx.num_tables, ctx.d_slot).permute(2, 0, 1, 3).reshape(ctx.num_tables, B * T, ctx.d_slot)
        
        # 3. Accumulate gradients into grad_embed_tables using PyTorch's native CUDA-optimized index_put_ scatter-add
        grad_embed_tables = torch.zeros(ctx.embed_tables_shape, dtype=grad_output.dtype, device=device)
        tbl_idx = torch.arange(ctx.num_tables, device=device).unsqueeze(1).expand(ctx.num_tables, B * T)
        
        grad_embed_tables.index_put_((tbl_idx, all_hashes), grad_out_reshaped, accumulate=True)
        
        # Return gradients matching forward arguments (None for non-tensor / constant arguments)
        return None, grad_embed_tables, None, None, None, None, None, None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight RMSNorm (standalone; avoids circular import from quasar_rope)
# ─────────────────────────────────────────────────────────────────────────────

class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


# ─────────────────────────────────────────────────────────────────────────────
# EngramModule
# ─────────────────────────────────────────────────────────────────────────────

class EngramModule(nn.Module):
    """
    Engram Conditional Memory Module (DeepSeek-AI, arXiv:2601.07372).

    Replaces expensive attention layers for static N-gram patterns with
    O(1) hash-table lookups gated into the hidden state.

    All operations are fully vectorized — no Python loops over T or B:
      • N-gram extraction:  torch.unfold  (single op)
      • Hash computation:   vectorized XOR accumulation (loop over n=2..3 only)
      • Embedding lookup:   batched advanced indexing  (single gather)
      • Conv:               nn.Conv1d with causal pad + slice
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_mem: int,
        num_heads: int = 8,
        ngram_orders: list = None,
        target_slots: int = 5_700_000,
        n_layers: int = 24,
    ):
        super().__init__()

        if ngram_orders is None:
            ngram_orders = [2, 3]

        self.vocab_size   = vocab_size
        self.d_model      = d_model
        self.d_mem        = d_mem
        self.num_heads    = num_heads
        self.ngram_orders = list(ngram_orders)
        self.num_ngram_orders = len(ngram_orders)
        self.num_tables   = self.num_ngram_orders * num_heads
        self.n_layers     = n_layers

        # ── A. Tokenizer Compression Buffer ──────────────────────────────────
        # Surjective P: V → V', ~23% compression.
        # Deterministic multiplicative hash — no tokenizer object needed at
        # construction time (avoids FSDP serialization problems).
        compressed_size = max(1, int(vocab_size * 0.77))
        self.compressed_vocab_size = compressed_size
        token_map = (
            torch.arange(vocab_size, dtype=torch.long) * 2654435761
        ) % compressed_size
        self.register_buffer('token_map', token_map)

        # ── B. Embedding Tables ───────────────────────────────────────────────
        # All num_tables share the same prime size M for vectorized indexing.
        slots_per_table = max(1, target_slots // self.num_tables)
        self.M      = _next_prime(slots_per_table)
        self.d_slot = max(16, d_mem // max(1, self.num_tables))
        self.total_embed_dim = self.num_tables * self.d_slot

        # Single parameter tensor — enables batched advanced-index gather.
        self.embed_tables = nn.Parameter(
            torch.empty(self.num_tables, self.M, self.d_slot)
        )

        # Per-table hashing seeds (non-trainable).
        seeds = torch.randint(1, 2 ** 31 - 1, (self.num_tables,), dtype=torch.long)
        self.register_buffer('seeds', seeds)

        # N-gram order list as a buffer for the Triton kernel.
        self.register_buffer(
            'ngram_orders_buf',
            torch.tensor(self.ngram_orders, dtype=torch.long),
        )

        # ── C. Projection total_embed_dim → d_mem ────────────────────────────
        self.embed_proj = nn.Linear(self.total_embed_dim, d_mem, bias=False)

        # ── D. Context-aware gating ───────────────────────────────────────────
        self.q_proj = nn.Linear(d_model, d_mem, bias=False)
        self.W_K    = nn.Linear(d_mem, d_mem,   bias=False)
        self.W_V    = nn.Linear(d_mem, d_mem,   bias=False)

        # ── E. Causal depthwise Conv1d ────────────────────────────────────────
        # kernel=4, dilation=3 → causal receptive field = 1 + (4-1)*3 = 10
        self.kernel_size = 4
        self.dilation    = 3
        self.conv_norm   = _RMSNorm(d_mem)
        self.conv = nn.Conv1d(
            d_mem, d_mem,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            groups=d_mem,   # depthwise
            bias=False,
        )

        # ── F. Output projection d_mem → d_model ─────────────────────────────
        self.out_proj = nn.Linear(d_mem, d_model, bias=False)

        # Triton eligible when compile-time bounds fit the kernel
        self._triton_ok = (
            HAS_TRITON
            and self.num_ngram_orders <= 4
            and num_heads <= 16
            and max(ngram_orders) <= 3
        )
        self.triton_training = True

        self._init_weights()

    # ── Initialization ────────────────────────────────────────────────────────

    def _init_weights(self):
        # 1. Deterministic buffer re-population (bypasses meta-device empty uninitialized memory)
        if hasattr(self, "token_map") and self.token_map is not None:
            dev = "cpu" if self.token_map.device.type == "meta" else self.token_map.device
            t_map = (torch.arange(self.vocab_size, dtype=torch.long, device=dev) * 2654435761) % self.compressed_vocab_size
            self.token_map.data.copy_(t_map)
        
        if hasattr(self, "seeds") and self.seeds is not None:
            # Deterministic hash seeds across all ranks
            g = torch.Generator().manual_seed(42)
            dev = "cpu" if self.seeds.device.type == "meta" else self.seeds.device
            s_t = torch.randint(1, 2 ** 31 - 1, (self.num_tables,), dtype=torch.long, device=dev, generator=g)
            self.seeds.data.copy_(s_t)
            
        if hasattr(self, "ngram_orders_buf") and self.ngram_orders_buf is not None:
            dev = "cpu" if self.ngram_orders_buf.device.type == "meta" else self.ngram_orders_buf.device
            ord_buf = torch.tensor(self.ngram_orders, dtype=torch.long, device=dev)
            self.ngram_orders_buf.data.copy_(ord_buf)

        trinity_std  = 0.5 / math.sqrt(self.d_model)
        scale_factor = 1.0 / math.sqrt(2 * self.n_layers)

        # 2. Deep init on output → zero-init to guarantee exactly zero output at step 0
        nn.init.zeros_(self.out_proj.weight)
        # Gating projections: standard Trinity
        nn.init.normal_(self.q_proj.weight,    std=trinity_std)
        nn.init.normal_(self.W_K.weight,       std=trinity_std)
        nn.init.normal_(self.W_V.weight,       std=trinity_std)
        # embed_proj: standard Trinity
        nn.init.normal_(self.embed_proj.weight, std=trinity_std)
        # Conv: zero init → identity pass-through at step 0
        nn.init.zeros_(self.conv.weight)
        # Embedding tables: small normal (paper standard)
        nn.init.normal_(self.embed_tables, std=0.01)
        # Conv norm: fill with ones
        if hasattr(self.conv_norm, "weight") and self.conv_norm.weight is not None:
            nn.init.ones_(self.conv_norm.weight)
        
        # Check for any non-finite initialization values
        for name, p in [("out_proj", self.out_proj.weight), ("q_proj", self.q_proj.weight), 
                        ("W_K", self.W_K.weight), ("W_V", self.W_V.weight), 
                        ("embed_proj", self.embed_proj.weight), ("conv", self.conv.weight), 
                        ("embed_tables", self.embed_tables), ("conv_norm", self.conv_norm.weight)]:
            if p.device.type != "meta":
                if not torch.isfinite(p).all():
                    print(f"[engram-init-warn] Parameter {name} contains non-finite values! Re-initializing with zeros.", flush=True)
                    nn.init.zeros_(p)

    # ── Core helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _hash_ngrams(ngrams: torch.Tensor, table_size: int, seed: int) -> torch.Tensor:
        """
        Vectorized XOR-multiplicative hash.
        ngrams: [B, T, n]  — n ∈ {2, 3}, compile-time constant.
        Returns: [B, T]    — indices into embedding table.
        No loop over T or B; only loops over n (≤ 3).
        """
        h = torch.full(ngrams.shape[:2], seed, dtype=torch.long, device=ngrams.device)
        for i in range(ngrams.shape[-1]):          # n iterations, NOT T
            h = h * 2654435761 ^ ngrams[..., i]
        return h.abs() % table_size

    def _lookup_pytorch(self, canonical: torch.Tensor) -> torch.Tensor:
        """
        Pure-PyTorch path: fully vectorized, no T/B loops.

        Steps:
          1. For each n-gram order, extract suffix n-grams via unfold  → [B, T, n]
          2. Hash all (n, k) pairs                                      → [num_tables, B, T]
          3. Batched advanced-index gather from embed_tables            → [num_tables, B*T, d_slot]
          4. Reshape to [B, T, total_embed_dim]
        """
        B, T = canonical.shape
        device = canonical.device

        # Step 1+2: collect hashes for all tables — loop over num_tables (≤ 32, not over T)
        all_hashes = torch.empty(self.num_tables, B * T, dtype=torch.long, device=device)
        table_idx = 0
        seeds_cpu = self.seeds.cpu().tolist() if hasattr(self, "seeds") and self.seeds is not None else []
        for n_idx, n in enumerate(self.ngram_orders):           # 2 or 3 iterations
            # Vectorized n-gram extraction: unfold over T → [B, T, n]
            padded = F.pad(canonical, (n - 1, 0), value=0)     # [B, T+n-1]
            ngrams = padded.unfold(dimension=1, size=n, step=1)  # [B, T, n]

            for k in range(self.num_heads):                     # num_heads iterations (≤ 16)
                seed = seeds_cpu[table_idx] if table_idx < len(seeds_cpu) else 42
                h = self._hash_ngrams(ngrams, self.M, seed)    # [B, T]
                all_hashes[table_idx] = h.view(B * T)
                table_idx += 1

        # Step 3: Single batched gather — no loop over T
        # embed_tables: [num_tables, M, d_slot]
        # all_hashes:   [num_tables, B*T]
        # Expand table index for advanced indexing
        tbl_idx = torch.arange(self.num_tables, device=device).unsqueeze(1).expand(
            self.num_tables, B * T
        )                                                        # [num_tables, B*T]
        embeddings = self.embed_tables[tbl_idx, all_hashes]     # [num_tables, B*T, d_slot]

        # Step 4: Reshape to [B, T, total_embed_dim]
        embeddings = embeddings.permute(1, 0, 2)                # [B*T, num_tables, d_slot]
        return embeddings.reshape(B, T, self.total_embed_dim)

    def _lookup_triton(self, canonical: torch.Tensor) -> torch.Tensor:
        """
        Triton path: fused hash + lookup in a single SRAM pass.
        Uses FusedEngramLookupFunction to support exact backward auto-differentiation in training.
        """
        return FusedEngramLookupFunction.apply(
            canonical,
            self.embed_tables,
            self.seeds,
            self.ngram_orders_buf,
            self.M,
            self.d_slot,
            self.num_tables,
            self.num_ngram_orders,
            self.num_heads,
            self.ngram_orders,
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,      # [B, T] raw token IDs
        hidden_states: torch.Tensor,  # [B, T, d_model]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            engram_out : [B, T, d_model] — to add to residual stream
            alpha_mean : scalar tensor   — mean gate value for LatentMemory suppression
        """
        B, T = input_ids.shape
        orig_dtype = hidden_states.dtype

        # ── A. Token compression ─────────────────────────────────────────────
        # Single gather op — no loop
        canonical = self.token_map[input_ids.clamp(0, self.vocab_size - 1)]  # [B, T]

        # ── B+C. Hash → lookup → project to d_mem ───────────────────────────
        use_triton_lookup = self._triton_ok and canonical.is_cuda and (
            self.training is False or bool(getattr(self, "triton_training", False))
        )
        if use_triton_lookup:
            raw_embed = self._lookup_triton(canonical)            # [B, T, total_embed_dim]
        else:
            raw_embed = self._lookup_pytorch(canonical)           # [B, T, total_embed_dim]

        raw_embed = raw_embed.to(orig_dtype)
        
        debug_engram = bool(int(os.environ.get("ENGRAM_DEBUG", "0")))
        if debug_engram:
            print(f"[engram-debug] embed_tables: finite={torch.isfinite(self.embed_tables).all().item()} min={self.embed_tables.float().min().item():.6g} max={self.embed_tables.float().max().item():.6g}", flush=True)
            print(f"[engram-debug] hidden_states: finite={torch.isfinite(hidden_states).all().item()} min={hidden_states.float().min().item():.6g} max={hidden_states.float().max().item():.6g}", flush=True)
            print(f"[engram-debug] raw_embed: finite={torch.isfinite(raw_embed).all().item()} min={raw_embed.float().min().item():.6g} max={raw_embed.float().max().item():.6g}", flush=True)

        raw_embed = torch.nan_to_num(raw_embed, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-10.0, 10.0)
        e_t = self.embed_proj(raw_embed)                         # [B, T, d_mem]
        if debug_engram:
            print(f"[engram-debug] e_t: finite={torch.isfinite(e_t).all().item()} min={e_t.float().min().item():.6g} max={e_t.float().max().item():.6g}", flush=True)
        e_t = torch.nan_to_num(e_t, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-100.0, 100.0)

        # ── D. Context-aware gating ──────────────────────────────────────────
        h_proj = self.q_proj(hidden_states)                      # [B, T, d_mem]
        k_t    = self.W_K(e_t)                                   # [B, T, d_mem]
        v_t    = self.W_V(e_t)                                   # [B, T, d_mem]
        if debug_engram:
            print(f"[engram-debug] h_proj: finite={torch.isfinite(h_proj).all().item()} min={h_proj.float().min().item():.6g} max={h_proj.float().max().item():.6g}", flush=True)
            print(f"[engram-debug] k_t: finite={torch.isfinite(k_t).all().item()} min={k_t.float().min().item():.6g} max={k_t.float().max().item():.6g}", flush=True)
            print(f"[engram-debug] v_t: finite={torch.isfinite(v_t).all().item()} min={v_t.float().min().item():.6g} max={v_t.float().max().item():.6g}", flush=True)

        h_proj = torch.nan_to_num(h_proj, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-100.0, 100.0)
        k_t    = torch.nan_to_num(k_t,    nan=0.0, posinf=0.0, neginf=0.0).clamp_(-100.0, 100.0)
        v_t    = torch.nan_to_num(v_t,    nan=0.0, posinf=0.0, neginf=0.0).clamp_(-100.0, 100.0)

        # L2-normalize for stability (matches Quasar key normalization)
        q_norm = F.normalize(h_proj.float(), dim=-1, eps=1e-6).to(orig_dtype)
        k_norm = F.normalize(k_t.float(),    dim=-1, eps=1e-6).to(orig_dtype)

        # Scalar gate per token per position
        alpha_logits = (q_norm * k_norm).sum(-1, keepdim=True).float() / math.sqrt(self.d_mem)
        alpha_t = torch.sigmoid(alpha_logits.clamp_(-30.0, 30.0)).to(orig_dtype)  # [B, T, 1]
        if debug_engram:
            print(f"[engram-debug] alpha_t: finite={torch.isfinite(alpha_t).all().item()} min={alpha_t.float().min().item():.6g} max={alpha_t.float().max().item():.6g}", flush=True)
        gated = alpha_t * v_t                                    # [B, T, d_mem]
        gated = torch.nan_to_num(gated, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-100.0, 100.0)

        # ── E. Causal depthwise conv ─────────────────────────────────────────
        # Fully vectorized: F.pad + Conv1d + slice — no loop over T
        causal_pad = (self.kernel_size - 1) * self.dilation
        g_norm = self.conv_norm(gated)                           # [B, T, d_mem]
        if debug_engram:
            print(f"[engram-debug] gated: finite={torch.isfinite(gated).all().item()} min={gated.float().min().item():.6g} max={gated.float().max().item():.6g}", flush=True)
            print(f"[engram-debug] conv_norm.weight: finite={torch.isfinite(self.conv_norm.weight).all().item()} min={self.conv_norm.weight.float().min().item():.6g} max={self.conv_norm.weight.float().max().item():.6g}", flush=True)
            print(f"[engram-debug] g_norm: finite={torch.isfinite(g_norm).all().item()} min={g_norm.float().min().item():.6g} max={g_norm.float().max().item():.6g}", flush=True)
        g_norm = torch.nan_to_num(g_norm, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-100.0, 100.0)
        g_t = g_norm.transpose(1, 2)                             # [B, d_mem, T]
        g_t = F.pad(g_t, (causal_pad, 0))                       # [B, d_mem, T+pad]
        g_t = self.conv(g_t)[..., :T]                            # [B, d_mem, T]
        g_t = F.silu(g_t).transpose(1, 2)                       # [B, T, d_mem]
        Y   = g_t + gated                                        # residual
        if debug_engram:
            print(f"[engram-debug] Y: finite={torch.isfinite(Y).all().item()} min={Y.float().min().item():.6g} max={Y.float().max().item():.6g}", flush=True)
        Y = torch.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-100.0, 100.0)

        # ── F. Output projection ─────────────────────────────────────────────
        engram_out = self.out_proj(Y)                            # [B, T, d_model]
        if debug_engram:
            print(f"[engram-debug] engram_out: finite={torch.isfinite(engram_out).all().item()} min={engram_out.float().min().item():.6g} max={engram_out.float().max().item():.6g}", flush=True)
        engram_out = torch.nan_to_num(engram_out, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-100.0, 100.0)

        # alpha_mean: mean gate activity — used by LatentMemory for suppression
        alpha_mean = alpha_t.squeeze(-1)                         # [B, T]

        return engram_out, alpha_mean
