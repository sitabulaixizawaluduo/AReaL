# SPDX-License-Identifier: Apache-2.0
"""Generate a tiny random-init Qwen3.5-MoE VL checkpoint for parallelism UTs.

Qwen3.5-35B-A3B is too large to run small parallel topologies (e.g. the
tp{1,2} x cp{1,2}, ep=1 isolation matrix needs the whole model replicated per
rank). This script builds a ~15M-parameter model with the same architecture
family (hybrid GDN + gated attention, routed experts + gated shared expert,
minimal vision tower) so grad-norm CP-equivalence tests run on 1-4 GPUs in
seconds.

The tiny model is the VL composite (``Qwen3_5MoeForConditionalGeneration``),
NOT the text-only class, on purpose: the whole Qwen3.5 family ships as VL
checkpoints, and the BSHD context-parallel contract lives in megatron-bridge's
VL modelling (the model itself zigzag-splits embeddings before the decoder).
The bridge's text-class path is a stock GPTModel with no CP split, so a
text-only tiny would exercise a different — broken under CP — code path.
The vision tower is inert for the text-only UT batches.

Dimension choices mirror the real model's structural ratios and parallelism
constraints:
- num_key_value_heads=2 (attention TP <= 2, same constraint as the 35B)
- linear_num_key_heads * key_dim == hidden, linear_num_value_heads * value_dim
  == 2 * hidden (GDN head-count divisibility by tp*cp holds up to tp2c2)
- 3:1 linear-to-full attention layer pattern (transformers default)
- vision/video special token ids sit in the top 64 ids of the tiny vocab;
  mock inputs must sample below them (the runner reserves this margin)

Usage (CPU, a few seconds):
    python tests/make_tiny_qwen3_5_moe.py --output /tmp/qwen3_5_moe_tiny

Then point the distributed runner at it:
    TINY_QWEN35_MOE_PATH=/tmp/qwen3_5_moe_tiny torchrun ... \
        tests/torchrun/run_megatron_engine_distributed.py \
        --model_type=qwen3_5_moe_tiny ...
"""

import argparse

import torch

VOCAB_SIZE = 8192
HIDDEN_SIZE = 256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="/tmp/qwen3_5_moe_tiny")
    parser.add_argument("--seed", type=int, default=0)
    # Routing-ablation variants for the MoE+CP grad blow-up bisection:
    # --top_k equal to --num_experts makes routing dense (every token to every
    # expert, dispatch machinery still exercised); --num_experts 1 --top_k 1
    # removes routing entirely. Defaults reproduce the original tiny model.
    parser.add_argument("--num_experts", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=4)
    args = parser.parse_args()

    from transformers import (
        Qwen3_5MoeConfig,
        Qwen3_5MoeForConditionalGeneration,
        Qwen3_5MoeTextConfig,
    )

    torch.manual_seed(args.seed)
    text_config = Qwen3_5MoeTextConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        linear_conv_kernel_dim=4,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        num_experts=args.num_experts,
        num_experts_per_tok=args.top_k,
        max_position_embeddings=4096,
        tie_word_embeddings=False,
        # megatron-bridge derives params_dtype from the HF sub-config (the
        # engine's dtype setting is ignored on the megatron-bridge path). An
        # unset dtype falls back to fp32, for which no TE attention backend
        # supports context parallelism.
        dtype=torch.bfloat16,
    )
    # Vision tower shrunk to the minimum that keeps the family's structure;
    # patch/merge geometry stays at the release defaults. out_hidden_size must
    # match the text hidden size (projector output feeds the decoder).
    vision_config = dict(
        depth=2,
        hidden_size=64,
        num_heads=2,
        intermediate_size=128,
        out_hidden_size=HIDDEN_SIZE,
        num_position_embeddings=64,
        dtype=torch.bfloat16,
    )
    config = Qwen3_5MoeConfig(
        text_config=text_config.to_dict(),
        vision_config=vision_config,
        # Release ids (248053+) exceed the tiny vocab; park them in the top
        # 64 ids. mock_input samples below this reserved band.
        vision_start_token_id=VOCAB_SIZE - 6,
        vision_end_token_id=VOCAB_SIZE - 5,
        image_token_id=VOCAB_SIZE - 4,
        video_token_id=VOCAB_SIZE - 3,
        tie_word_embeddings=False,
    )
    model = Qwen3_5MoeForConditionalGeneration(config).to(torch.bfloat16)
    n_params = sum(p.numel() for p in model.parameters())
    # transformers 5.x defaults to the hub serialization format, which splits
    # the grouped expert tensors per expert. megatron-bridge maps runtime
    # names (grouped ``mlp.experts.gate_up_proj``), so keep the runtime
    # format.
    model.save_pretrained(args.output, save_original_format=False)

    # The engine unconditionally loads a tokenizer (and, for VL models, a
    # processor) from the checkpoint dir, and AutoTokenizer on a bare config
    # falls back to a slow-tokenizer conversion that requires
    # sentencepiece/tiktoken. Ship a self-contained fast tokenizer matching
    # vocab_size instead (the UTs feed random ids, so only pad/eos metadata
    # matters).
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen2VLImageProcessorFast,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )

    vocab = {"<|endoftext|>": 0, "<|pad|>": 1}
    vocab.update({f"<tok{i}>": i for i in range(2, VOCAB_SIZE)})
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<|endoftext|>")),
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
        unk_token="<|endoftext|>",
    )
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessorFast(
            patch_size=16, merge_size=2, temporal_patch_size=2
        ),
        tokenizer=tokenizer,
        video_processor=Qwen3VLVideoProcessor(),
    )
    processor.save_pretrained(args.output)
    print(
        f"Saved tiny Qwen3.5-MoE VL ({n_params / 1e6:.1f}M params, "
        f"num_experts={args.num_experts}, top_k={args.top_k}) to {args.output}"
    )


if __name__ == "__main__":
    main()
