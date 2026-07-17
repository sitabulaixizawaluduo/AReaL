# SPDX-License-Identifier: Apache-2.0
"""Generate a tiny random-init Qwen3.5-MoE text checkpoint for parallelism UTs.

Qwen3.5-35B-A3B is too large to run small parallel topologies (e.g. the
tp{1,2} x cp{1,2}, ep=1 isolation matrix needs the whole model replicated per
rank). This script builds a ~15M-parameter model with the same architecture
family (hybrid GDN + gated attention, routed experts + gated shared expert)
so grad-norm CP-equivalence tests run on 1-4 GPUs in seconds.

Dimension choices mirror the real model's structural ratios and parallelism
constraints:
- num_key_value_heads=2 (attention TP <= 2, same constraint as the 35B)
- linear_num_key_heads * key_dim == hidden, linear_num_value_heads * value_dim
  == 2 * hidden (GDN head-count divisibility by tp*cp holds up to tp2c2)
- 3:1 linear-to-full attention layer pattern (transformers default)

Usage (CPU, a few seconds):
    python tests/make_tiny_qwen3_5_moe.py --output /tmp/qwen3_5_moe_tiny

Then point the distributed runner at it:
    AREAL_TINY_QWEN35_MOE_PATH=/tmp/qwen3_5_moe_tiny torchrun ... \
        tests/torchrun/run_megatron_engine_distributed.py \
        --model_type=qwen3_5_moe_tiny ...
"""

import argparse

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="/tmp/qwen3_5_moe_tiny")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

    torch.manual_seed(args.seed)
    config = Qwen3_5MoeTextConfig(
        vocab_size=8192,
        hidden_size=256,
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
        num_experts=16,
        num_experts_per_tok=4,
        max_position_embeddings=4096,
        tie_word_embeddings=False,
    )
    model = Qwen3_5MoeForCausalLM(config).to(torch.bfloat16)
    n_params = sum(p.numel() for p in model.parameters())
    model.save_pretrained(args.output)

    # The engine unconditionally loads a tokenizer from the checkpoint dir,
    # and AutoTokenizer on a bare config falls back to a slow-tokenizer
    # conversion that requires sentencepiece/tiktoken. Ship a self-contained
    # fast tokenizer matching vocab_size instead (the UTs feed random ids, so
    # only pad/eos metadata matters).
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    vocab = {"<|endoftext|>": 0, "<|pad|>": 1}
    vocab.update({f"<tok{i}>": i for i in range(2, config.vocab_size)})
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<|endoftext|>")),
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
        unk_token="<|endoftext|>",
    )
    tokenizer.save_pretrained(args.output)
    print(f"Saved tiny Qwen3.5-MoE ({n_params / 1e6:.1f}M params) to {args.output}")


if __name__ == "__main__":
    main()
