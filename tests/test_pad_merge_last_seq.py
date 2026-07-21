# SPDX-License-Identifier: Apache-2.0

import torch

from areal.utils.data import pad_packed_tensor_dict


def _packed_input(total: int, cu: list[int]):
    return {
        "cu_seqlens": torch.tensor(cu, dtype=torch.int32),
        "max_seqlen": max(b - a for a, b in zip(cu[:-1], cu[1:])),
        "input_ids": torch.arange(total, dtype=torch.long),
    }


def test_pad_packed_tensor_dict_default_appends_pad_segment():
    data = _packed_input(60, [0, 24, 60])

    padded, pad_len, old_cu, _ = pad_packed_tensor_dict(data, 256)

    assert pad_len == 196
    torch.testing.assert_close(
        padded["cu_seqlens"],
        torch.tensor([0, 24, 60, 256], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        old_cu, torch.tensor([0, 24, 60], dtype=torch.int32), rtol=0, atol=0
    )


def test_pad_packed_tensor_dict_merge_extends_last_sequence():
    """Pins the TE<2.16 THD-CP workaround: batch padding must extend the last
    real sequence instead of forming a standalone degenerate segment."""
    data = _packed_input(60, [0, 24, 60])

    padded, pad_len, old_cu, _ = pad_packed_tensor_dict(
        data, 256, merge_pad_into_last_seq=True
    )

    assert pad_len == 196
    torch.testing.assert_close(
        padded["cu_seqlens"],
        torch.tensor([0, 24, 256], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    assert padded["max_seqlen"] == 232
    torch.testing.assert_close(
        old_cu, torch.tensor([0, 24, 60], dtype=torch.int32), rtol=0, atol=0
    )
    assert padded["input_ids"].numel() == 256
    torch.testing.assert_close(
        padded["input_ids"][:60], torch.arange(60, dtype=torch.long), rtol=0, atol=0
    )
    assert int(padded["input_ids"][60:].sum()) == 0
    assert all(
        (b - a) % 4 == 0
        for a, b in zip(
            padded["cu_seqlens"][:-1].tolist(), padded["cu_seqlens"][1:].tolist()
        )
    )


def test_pad_packed_tensor_dict_merge_with_zero_pad_is_identity():
    data = _packed_input(256, [0, 24, 256])

    padded, pad_len, _, _ = pad_packed_tensor_dict(
        data, 256, merge_pad_into_last_seq=True
    )

    assert pad_len == 0
    torch.testing.assert_close(
        padded["cu_seqlens"],
        torch.tensor([0, 24, 256], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
