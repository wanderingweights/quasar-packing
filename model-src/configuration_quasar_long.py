"""Quasar Long model configuration"""

from transformers.configuration_utils import PretrainedConfig


class QuasarLongConfig(PretrainedConfig):
    model_type = "quasar_long"

    def __init__(
        self,
        vocab_size=157184,
        hidden_size=2048,
        intermediate_size=5120,
        num_hidden_layers=20,
        num_attention_heads=16,
        num_key_value_heads=4,
        hidden_act="silu",
        use_qkv_bias=False,  # quasar legacy
        use_bias=False,  # quasar legacy
        rms_norm_eps=1e-06,
        tie_word_embeddings=False,  # PretrainedConfig key, here change default value.
        embedding_dropout=0.0,
        attention_dropout=0.0,
        output_dropout=0.0,
        initializer_range=0.02,
        max_position_embeddings=32768,
        rope_theta=600000.0,
        use_cache=True,
        max_window_layers=20,
        rope_scaling=None,
        pad_token_id=156892,
        eos_token_id=156892,
        num_experts=256,
        num_shared_experts=1,
        num_experts_per_tok=8,
        n_group=8,
        topk_group=4,
        moe_intermediate_size=512,
        first_k_dense_replace=1,
        head_dim=128,
        output_router_logits=False,
        use_qk_norm=True,
        num_nextn_predict_layers=0,
        mtp_loss_scaling_factor=0,
        moe_router_enable_expert_bias=True,
        routed_scaling_factor=1.0,
        hybrid_attention_layers=None,
        hybrid_alpha_init=-15.0,
        hybrid_gla_expand_k=1.0,
        hybrid_gla_expand_v=1.0,
        hybrid_use_short_conv=False,
        hybrid_quasar_enabled=True,
        hybrid_gla_enabled=True,
        hybrid_branch_layout="mixed",
        hybrid_layerwise_cycle=None,
        # ── Looped Transformer ────────────────────────────────────────────────
        num_loops=1,
        use_looped_injection=False,
        # ── Engram Conditional Memory ─────────────────────────────────────────
        # engram_layers=[] → module disabled (zero overhead, backward-compatible).
        engram_layers=None,
        engram_dim=512,
        engram_slots=2_000_000,
        engram_num_heads=8,
        engram_ngram_orders=None,
        engram_lr_multiplier=5.0,
        use_nope=False,
        long_context_mode="rope_short_nope_long",
        nope_after_position=512,
        max_seq_length=None,
        max_sequence_length=None,
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.use_qkv_bias = use_qkv_bias
        self.use_bias = use_bias
        self.rms_norm_eps = rms_norm_eps
        self.embedding_dropout = embedding_dropout
        self.attention_dropout = attention_dropout
        self.output_dropout = output_dropout
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.mtp_loss_scaling_factor = mtp_loss_scaling_factor
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.use_cache = use_cache
        self.max_window_layers = max_window_layers
        self.head_dim = head_dim or self.hidden_size // self.num_attention_heads
        self.rope_scaling = rope_scaling
        self.use_qk_norm = use_qk_norm
        self.moe_router_enable_expert_bias = moe_router_enable_expert_bias
        self.routed_scaling_factor = routed_scaling_factor
        self.hybrid_attention_layers = hybrid_attention_layers or []
        self.hybrid_alpha_init = hybrid_alpha_init
        self.hybrid_gla_expand_k = hybrid_gla_expand_k
        self.hybrid_gla_expand_v = hybrid_gla_expand_v
        self.hybrid_use_short_conv = hybrid_use_short_conv
        self.hybrid_quasar_enabled = hybrid_quasar_enabled
        self.hybrid_gla_enabled = hybrid_gla_enabled
        self.hybrid_branch_layout = hybrid_branch_layout
        self.hybrid_layerwise_cycle = list(hybrid_layerwise_cycle) if hybrid_layerwise_cycle is not None else [
            "quasar",
            "raven",
            "gla",
        ]

        # Looped Transformer
        self.num_loops = num_loops
        self.use_looped_injection = use_looped_injection

        # Engram Conditional Memory
        self.engram_layers = list(engram_layers) if engram_layers is not None else []
        self.engram_dim = engram_dim
        self.engram_slots = engram_slots
        self.engram_num_heads = engram_num_heads
        self.engram_ngram_orders = list(engram_ngram_orders) if engram_ngram_orders is not None else [2, 3]
        self.engram_lr_multiplier = engram_lr_multiplier
        self.use_nope = use_nope
        self.long_context_mode = long_context_mode
        self.nope_after_position = int(nope_after_position)
        self.max_seq_length = int(max_seq_length) if max_seq_length is not None else None
        self.max_sequence_length = int(max_sequence_length) if max_sequence_length is not None else None

        # MoE configs
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.moe_intermediate_size = moe_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.output_router_logits = output_router_logits

        super().__init__(pad_token_id=pad_token_id, eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs)
