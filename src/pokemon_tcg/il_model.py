"""Transformer policy network for Pokemon TCG behavior cloning.

Every legal move in ``select.option`` becomes one token in the input
sequence, alongside a fixed set of board-state tokens (my active/bench,
opponent's active/bench, my hand) and one CLS-style context token (turn,
prize/deck/hand counts, select type/context). A stock bidirectional
self-attention encoder lets every option token attend to the full board
state and to every other option token, so a single shared linear head can
score an arbitrarily-ordered, variable-count list of legal moves -- there is
no fixed global action vocabulary to enumerate (option ``type=ATTACH`` at
index 3 means a completely different move from decision to decision).

We reuse ``transformers.BertModel`` purely as that self-attention encoder
(fed ``inputs_embeds`` directly -- it never sees ``input_ids`` or its own
token vocabulary). That is why an AutoModel/BertModel backbone is a good
fit even though this isn't an NLP task: it is a well-tested block with
``from_pretrained``/``save_pretrained`` support we get for free instead of
hand-rolling and debugging attention from scratch.

Illegal-move masking: option slots beyond the true ``len(select.option))``
are padding, not real moves -- the game engine only ever offers legal
options in the first place. One extra slot (index ``n_real_options``) is a
synthetic DECLINE option, present whenever ``select.minCount == 0`` (see
``il_dataset.encode_observation``) -- it is scoreable, not padding, and
represents submitting an empty selection. So ``opt_mask`` marks *scoreable
(real option OR decline) vs. padding*, and padding logits are set to -inf
before the softmax/argmax.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import BertConfig, BertModel, PretrainedConfig, PreTrainedModel

from .il_dataset import (
    ATTACK_VOCAB_SIZE,
    CARD_VOCAB_SIZE,
    MAX_OPTIONS,
    N_GLOBAL_SCALARS,
    N_OPT_SCALARS,
    N_REF_SCALARS,
    N_SLOT_SCALARS,
    N_STATE_SLOTS,
    OPTION_TYPE_VOCAB_SIZE,
    SELECT_CONTEXT_VOCAB_SIZE,
    SELECT_TYPE_VOCAB_SIZE,
    SPECIAL_VOCAB_SIZE,
)


class PTCGILConfig(PretrainedConfig):
    model_type = "ptcg_il"

    def __init__(
        self,
        hidden_size: int = 128,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 4,
        intermediate_size: int | None = None,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size or hidden_size * 4
        self.dropout = dropout


class PTCGImitationPolicy(PreTrainedModel):
    """Scores every option slot; returns logits [B, MAX_OPTIONS] (padding = -inf)."""

    config_class = PTCGILConfig

    def __init__(self, config: PTCGILConfig) -> None:
        super().__init__(config)
        h = config.hidden_size

        self.card_emb = nn.Embedding(CARD_VOCAB_SIZE, h, padding_idx=0)
        self.attack_emb = nn.Embedding(ATTACK_VOCAB_SIZE, h, padding_idx=0)
        self.special_emb = nn.Embedding(SPECIAL_VOCAB_SIZE, h, padding_idx=0)
        self.select_type_emb = nn.Embedding(SELECT_TYPE_VOCAB_SIZE, h)
        self.select_context_emb = nn.Embedding(SELECT_CONTEXT_VOCAB_SIZE, h)
        self.option_type_emb = nn.Embedding(OPTION_TYPE_VOCAB_SIZE, h)

        self.slot_scalar_proj = nn.Linear(N_SLOT_SCALARS, h)
        self.global_scalar_proj = nn.Linear(N_GLOBAL_SCALARS, h)
        self.opt_scalar_proj = nn.Linear(N_OPT_SCALARS, h)
        self.opt_ref_scalar_proj = nn.Linear(2 * N_REF_SCALARS, h)  # ref1 + ref2

        self.cls_param = nn.Parameter(torch.zeros(h))
        self.slot_pos_emb = nn.Embedding(N_STATE_SLOTS, h)
        # position IS the action index to output -> required input, not decorative
        self.opt_pos_emb = nn.Embedding(MAX_OPTIONS, h)

        seq_len = 1 + N_STATE_SLOTS + MAX_OPTIONS
        bert_config = BertConfig(
            vocab_size=1,  # unused: we always feed inputs_embeds, never input_ids
            hidden_size=h,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_dropout_prob=config.dropout,
            attention_probs_dropout_prob=config.dropout,
            max_position_embeddings=seq_len,
        )
        self.encoder = BertModel(bert_config, add_pooling_layer=False)
        self.score_head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))

        self.post_init()

    def forward(
        self,
        slot_card_id: torch.Tensor,
        slot_scalar: torch.Tensor,
        global_scalar: torch.Tensor,
        select_type: torch.Tensor,
        select_context: torch.Tensor,
        opt_type: torch.Tensor,
        opt_attack: torch.Tensor,
        opt_special: torch.Tensor,
        opt_scalar: torch.Tensor,
        opt_ref_card_id: torch.Tensor,
        opt_ref_scalar: torch.Tensor,
        opt_mask: torch.Tensor,
        label: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        batch_size = slot_card_id.shape[0]
        device = slot_card_id.device

        # Defense in depth: encode_observation() already clamps ids into
        # vocab range (see il_dataset._clamp_id), but a future caller that
        # builds tensors directly (or a competition card id newer than this
        # checkpoint's frozen vocab) must not be able to crash the forward
        # pass with an out-of-range embedding index -- that's an
        # uncaught-exception-equals-instant-loss failure mode.
        slot_card_id = slot_card_id.clamp(max=self.card_emb.num_embeddings - 1)
        opt_attack = opt_attack.clamp(max=self.attack_emb.num_embeddings - 1)
        opt_ref_card_id = opt_ref_card_id.clamp(max=self.card_emb.num_embeddings - 1)
        opt_special = opt_special.clamp(max=self.special_emb.num_embeddings - 1)
        opt_type = opt_type.clamp(max=self.option_type_emb.num_embeddings - 1)
        select_context = select_context.clamp(max=self.select_context_emb.num_embeddings - 1)
        select_type = select_type.clamp(max=self.select_type_emb.num_embeddings - 1)

        slot_embed = (
            self.card_emb(slot_card_id)
            + self.slot_scalar_proj(slot_scalar)
            + self.slot_pos_emb(torch.arange(N_STATE_SLOTS, device=device)).unsqueeze(0)
        )
        slot_present = slot_scalar[..., -1] > 0  # last scalar col is the "present" flag

        cls_embed = (
            self.cls_param.unsqueeze(0).expand(batch_size, -1)
            + self.select_type_emb(select_type)
            + self.select_context_emb(select_context)
            + self.global_scalar_proj(global_scalar)
        ).unsqueeze(1)

        # [B, MAX_OPTIONS, H]: sum over ref1 + ref2 (zero-padded when a ref is absent)
        ref_embed = self.card_emb(opt_ref_card_id).sum(dim=2)
        opt_embed = (
            self.option_type_emb(opt_type)
            + self.attack_emb(opt_attack)
            + self.special_emb(opt_special)
            + self.opt_scalar_proj(opt_scalar)
            + ref_embed
            + self.opt_ref_scalar_proj(opt_ref_scalar)
            + self.opt_pos_emb(torch.arange(MAX_OPTIONS, device=device)).unsqueeze(0)
        )

        seq = torch.cat([cls_embed, slot_embed, opt_embed], dim=1)
        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        attn_mask = torch.cat([cls_mask, slot_present, opt_mask], dim=1).long()

        encoder_out = self.encoder(inputs_embeds=seq, attention_mask=attn_mask)
        hidden = encoder_out.last_hidden_state
        opt_hidden = hidden[:, 1 + N_STATE_SLOTS :, :]
        logits = self.score_head(opt_hidden).squeeze(-1)
        logits = logits.masked_fill(~opt_mask, float("-inf"))

        loss = None
        if label is not None:
            loss = nn.functional.cross_entropy(logits, label)

        return {"loss": loss, "logits": logits}
