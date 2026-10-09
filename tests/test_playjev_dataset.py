# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from areal.dataset.playjev import PlayJevPacmanDataset
from areal.utils.data import collate_samples_to_list, concat_batch, pack_tensor_dict


class _Processor:
    tokenizer = SimpleNamespace(
        encode=lambda text, **kwargs: [10 + "ABCD".index(text.strip())]
    )

    def __call__(self, *, text, images, **kwargs):
        assert text == ["A: right; B: up; C: left; D: down"]
        assert images == ["frame"]
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "pixel_values": torch.ones(4, 8),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
            "token_type_ids": torch.tensor([[0, 1, 0]]),
        }


def test_adapter_permuted_target_and_multimodal_payload_survive_packing(
    tmp_path, monkeypatch
):
    upstream_dir = tmp_path / "playjev"
    upstream_dir.mkdir()
    (upstream_dir / "data.py").touch()
    train_record = SimpleNamespace(names=("up", "down", "left", "right"))
    item = {
        "options": ["right", "up", "left", "down"],
        "instructions": "move",
        "target": [0.7, 0.1, 0.15, 0.05],
        "image": "frame",
    }
    upstream = SimpleNamespace(
        __file__=str(upstream_dir / "data.py"),
        load_records=lambda *args, **kwargs: [train_record],
        split_records=lambda records: (records, records),
        SFTDataset=lambda records, **kwargs: [item],
    )
    model = SimpleNamespace(
        build_plain_prompt=lambda options, instructions: "; ".join(
            f"{letter}: {option}" for letter, option in zip("ABCD", options)
        )
    )
    monkeypatch.setattr(
        "areal.dataset.playjev.importlib.import_module",
        lambda name: (upstream if name == "playjev.data" else model),
    )
    dataset = PlayJevPacmanDataset(
        str(tmp_path), "train", _Processor(), playjev_root=str(tmp_path)
    )
    sample = dataset[0]
    assert sample["input_ids"].tolist() == [1, 2, 3, 10]
    assert sample["loss_mask"].tolist() == [False, False, False, True]
    assert sample["mm_token_type_ids"].shape == (4,)
    torch.testing.assert_close(
        torch.stack([sample[f"candidate_target_{i}"][-1] for i in range(4)]),
        torch.tensor(item["target"]),
        rtol=0,
        atol=0,
    )
    batched, _ = concat_batch(collate_samples_to_list([sample, sample]))
    packed = pack_tensor_dict(batched)
    assert packed["cu_seqlens"].tolist() == [0, 4, 8]
    assert packed["candidate_target_0"].shape == (8,)
    assert packed["multi_modal_input"][0]["image_grid_thw"].shape == (1, 3)
    assert set(sample) == {
        "input_ids",
        "loss_mask",
        "multi_modal_input",
        "mm_token_type_ids",
        "candidate_count",
        *(f"candidate_id_{i}" for i in range(4)),
        *(f"candidate_target_{i}" for i in range(4)),
    }
