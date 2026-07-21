# SPDX-License-Identifier: Apache-2.0

"""Runtime patches for megatron-bridge bugs not yet in a released version.

Each patch is keyed to an upstream PR. Patches are not version-gated; instead
each one's hot path becomes a no-op once the upstream fix is present (the patch
checks for the missing attribute/behavior before acting), and an idempotency
sentinel prevents double-application. Apply patches at import time via
``_apply_patches_on_import()`` at module bottom.
"""

from __future__ import annotations

import warnings

import areal.utils.logging as logging

logger = logging.getLogger("MegatronBridgePatches")

# Per-microbatch override for the MoE aux-loss backward scale (see
# ``_patch_moe_aux_loss_backward_scale`` below). ``None`` means "defer to
# mcore's own value"; a float means "use this instead". Set by MegatronEngine
# right before each backward via ``set_moe_aux_loss_backward_scale``.
_MOE_AUX_LOSS_BACKWARD_SCALE: dict[str, float | None] = {"value": None}


def set_moe_aux_loss_backward_scale(value: float | None) -> None:
    """Register the aux-loss backward scale for the next backward.

    mcore's pipeline schedule calls ``MoEAuxLossAutoScaler.set_loss_scale`` at
    the end of every ``forward_step`` (after the loss function returns) with a
    value derived from ``grad_scale_func`` — under ``calculate_per_token_loss``
    that value is just the fp16/bf16 grad scale (1.0), on the assumption that
    ``finalize_model_grads`` will later divide every gradient by the global
    token count. This engine bypasses that machinery: it normalizes the *main* loss
    itself via a manual ``loss_scale`` and returns a 2-tuple loss, so
    ``finalize_model_grads`` never performs the per-token division. That leaves
    the router aux-loss gradient (injected purely in backward via
    ``MoEAuxLossAutoScaler``) scaled inconsistently with the main loss.
    The engine
    therefore computes the matching scale per micro-batch and stashes it here so
    the patched ``set_loss_scale`` uses it instead of mcore's value.
    """
    _MOE_AUX_LOSS_BACKWARD_SCALE["value"] = value


def _silence_mcore_gdn_indexing_deprecation() -> None:
    """Silence the per-parameter indexing UserWarning from mcore GDN sharding.

    megatron-core 0.18.0's ``gated_delta_net.py`` indexes parameters with a
    list of slices (``param[slices]``), which torch flags with a deprecation
    UserWarning on every GDN parameter — flooding logs at init and on every
    weight-sync. Behavior is unchanged on current torch (the warning firing
    at all means the legacy semantics are still in effect); revisit if
    megatron-core or torch is upgraded past the deprecation window.
    """
    warnings.filterwarnings(
        "ignore",
        message=r"Using a non-tuple sequence for multidimensional indexing",
        category=UserWarning,
        module=r"megatron\.core\.ssm\.gated_delta_net",
    )


def _patch_qwen3vl_pr3143_word_embeddings() -> None:
    """megatron-bridge PR #3143: expose word_embeddings on MTP shadow embedding.

    Bug (issue #3112 / PR #3143): in ``Qwen3VLGPTModel.forward``, when
    ``mtp_process and sequence_parallel`` are both True, ``self.embedding`` is
    temporarily replaced with a plain closure ``_sp_scatter_embedding``. The
    closure lacks the ``word_embeddings`` attribute that
    ``shared_embedding_or_output_weight()`` accesses during ``_postprocess``
    when ``share_embeddings_and_output_weights=True`` — typical for the
    smaller Qwen3.5 dense models (0.8B/2B/4B).

    Failure mode:
        ``AttributeError: 'function' object has no attribute 'word_embeddings'``

    Affected versions: megatron-bridge 0.4.0 and 0.4.1. Fixed on ``main``
    by commit 20749b09 (PR #3143) but not in any non-alpha release yet.

    Strategy: wrap ``Qwen3VLGPTModel._postprocess`` so it lazily restores
    ``word_embeddings`` on the shadow embedding by inspecting its closure.
    Closure-based recovery is non-invasive — we don't touch ``forward``
    itself (~70 LoC method).
    """
    try:
        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import (
            Qwen3VLGPTModel,
        )
    except ImportError:
        return

    if getattr(Qwen3VLGPTModel, "_areal_pr3143_applied", False):
        return

    _orig_postprocess = Qwen3VLGPTModel._postprocess

    def _patched_postprocess(self, *args, **kwargs):
        emb = self.__dict__.get("embedding")
        # Only intervene when the shadow closure is currently installed and
        # lacks the expected attribute.
        if (
            callable(emb)
            and not hasattr(emb, "word_embeddings")
            and emb.__closure__ is not None
        ):
            for cell in emb.__closure__:
                try:
                    target = cell.cell_contents
                except ValueError:
                    continue
                if hasattr(target, "word_embeddings"):
                    emb.word_embeddings = target.word_embeddings
                    break
        return _orig_postprocess(self, *args, **kwargs)

    Qwen3VLGPTModel._postprocess = _patched_postprocess
    Qwen3VLGPTModel._areal_pr3143_applied = True
    logger.info(
        "Applied megatron-bridge PR #3143 workaround: "
        "Qwen3VLGPTModel shadow embedding word_embeddings restoration."
    )


def _patch_moe_aux_loss_backward_scale() -> None:
    """Reconcile mcore's MoE aux-loss backward scaling with manual
    loss normalization.

    Two coupled defects appear only for MoE models under context parallelism,
    because the megatron-bridge Qwen3.5(-VL) providers flip
    ``calculate_per_token_loss=True`` exactly when ``cp_size > 1``:

    1. Router aux losses (``aux_loss`` / ``seq_aux_loss`` / ``global_aux_loss``)
       are attached to the autograd graph via ``MoEAuxLossAutoScaler``. Under
       ``calculate_per_token_loss`` the router pre-multiplies the aux loss by the
       local token count (``TopKRouter.attach_and_log_load_balancing_loss``),
       expecting ``finalize_model_grads`` to divide it back out by the global
       token count. The engine returns a 2-tuple loss, so that division is a no-op,
       leaving an uncancelled ``~num_tokens`` factor on the aux gradient.
    2. mcore's schedule sets the aux backward scale to the bf16 grad scale
       (1.0), not to the engine's manual ``loss_scale`` — so even without (1) the aux
       gradient is weighted inconsistently with the main loss, by a factor that
       tracks ``cp * dp``.

    Fix, keeping the engine's manual ``loss_scale`` convention:

    * Neutralize the per-token pre-multiply by running the aux-loss attach as if
      ``calculate_per_token_loss=False`` (plain ``aux_loss``), so the aux
      gradient is just ``d(aux_loss)/dW`` scaled by the injected scalar.
    * Override the injected scalar with the value registered via
      ``set_moe_aux_loss_backward_scale`` — the same effective backward multiplier
      the main loss receives (``loss_scale * cp_size / num_microbatches``) — so
      aux and main ride an identical, topology-consistent scale.
    """
    try:
        from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
        from megatron.core.transformer.moe.router import TopKRouter
    except ImportError:
        return

    if getattr(MoEAuxLossAutoScaler, "_patch_manual_scale_applied", False):
        return

    import torch

    # (1) Neutralize the calculate_per_token_loss pre-multiply. attach_and_log is
    # the single choke point for all three aux-loss types.
    _orig_attach = TopKRouter.attach_and_log_load_balancing_loss

    def _attach_without_per_token_premul(self, *args, **kwargs):
        saved = self.calculate_per_token_loss
        self.calculate_per_token_loss = False
        try:
            return _orig_attach(self, *args, **kwargs)
        finally:
            self.calculate_per_token_loss = saved

    TopKRouter.attach_and_log_load_balancing_loss = _attach_without_per_token_premul

    # (2) Override the aux backward scale with the registered per-microbatch value.
    def _set_loss_scale(scale):
        override = _MOE_AUX_LOSS_BACKWARD_SCALE["value"]
        if override is not None:
            scale = torch.as_tensor(override, device=scale.device, dtype=scale.dtype)
        if MoEAuxLossAutoScaler.main_loss_backward_scale is None:
            MoEAuxLossAutoScaler.main_loss_backward_scale = scale
        else:
            MoEAuxLossAutoScaler.main_loss_backward_scale.copy_(scale)

    MoEAuxLossAutoScaler.set_loss_scale = staticmethod(_set_loss_scale)
    MoEAuxLossAutoScaler._patch_manual_scale_applied = True
    logger.info(
        "Applied MoE aux-loss backward-scale reconciliation "
        "(neutralize per-token pre-multiply + honor engine manual loss_scale)."
    )


def _apply_patches_on_import() -> None:
    _silence_mcore_gdn_indexing_deprecation()
    _patch_qwen3vl_pr3143_word_embeddings()
    _patch_moe_aux_loss_backward_scale()


_apply_patches_on_import()
