"""Masked multi-label losses, including the defect-aware distillation loss.

One module covers every training run in the experiment matrix, so the runs
differ only in configuration and not in code:

======================  =================================================
``plain``               masked BCE. The control and `reference`.
``reweighted``          masked BCE with a defect-tile weight. Tests whether
                        weighting alone explains any gain.
``hinton_kd``           distillation with ``alpha = 1``. The prior-method
                        comparison.
``defect_kd``           distillation with ``alpha > 1``. **The project's
                        novel contribution.**
======================  =================================================

The loss
--------
::

    L = mean_(tile, class)  w(tile, class) * [ BCE(sigmoid(z_s), y)
          + lambda * T^2 * BCE(sigmoid(z_s / T), sigmoid(z_t / T)) ]

with ``w = 1`` for clean tiles and ``alpha`` for tiles holding any defect,
times a per-class inverse-frequency weight on positive targets. Setting
``alpha = 1`` reduces it exactly to Hinton-style KD, and dropping the second
term reduces it to plain masked BCE -- which is why all four rows share an
implementation.

The ignore mask
---------------
A ``(tile, class)`` entry the tile dataset marks *ignore* contributes nothing
to **either** term. Those entries are tiles holding a fragment of a defect too
small to name, so neither 0 nor 1 is a correct target and training on them
would teach noise. Masking is applied before any reduction, and the mean is
taken over surviving entries only, so the loss scale does not depend on how
many entries happened to be ignored.

Why ``T^2``
-----------
The repo's original formula had no ``T^2``. It is included here, identically in
both KD runs. For this sigmoid form,

``d/dz_s BCE(z_s/T, sigmoid(z_t/T)) = (1/T) (sigmoid(z_s/T) - sigmoid(z_t/T))``

and for logits small relative to ``T`` the bracket itself shrinks like
``(z_s - z_t) / 4T``, so the gradient scales as ``1/T^2`` -- the same
asymptotics Hinton's softmax argument gives. Multiplying by ``T^2`` therefore
keeps the distillation gradient comparable across temperatures, so ``lambda``
does not have to silently absorb ``T`` and the ``lambda``/``T`` sweep stays
readable.

This sigmoid multi-label form is an **adaptation** of Hinton KD, which was
defined over softmax outputs. The paper must describe it that way.

Weight normalisation
--------------------
``normalize_weights`` rescales ``w`` to mean 1 over the non-ignored entries of
each batch. Without it, raising ``alpha`` raises the overall loss scale and so
acts as a learning-rate change, which would confound the ablation: the
``reweighted`` run would differ from ``reference`` in effective step size as
well as in weighting. With it, ``alpha`` only redistributes weight.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from deeppcb import NUM_CLASSES  # noqa: E402

LOSS_KINDS = ("plain", "reweighted", "hinton_kd", "defect_kd")


@dataclass
class LossConfig:
    """Configuration for :class:`MaskedTileLoss`.

    Args:
        kind: one of :data:`LOSS_KINDS`. Selects which terms are active and
            whether the defect-tile weight is applied; it does not change the
            arithmetic of the terms themselves.
        alpha: weight on tiles holding any positive class. Must be 1.0 for
            ``plain`` and ``hinton_kd``.
        lam: weight on the distillation term. Ignored without a teacher.
        temperature: distillation temperature ``T``. Must be > 0.
        use_class_weights: apply per-class inverse-frequency weights to
            positive targets.
        normalize_weights: rescale weights to mean 1 per batch (see module
            docstring -- leave this on for the ablation to mean anything).
        t_squared: include the ``T^2`` factor on the distillation term.
    """

    kind: str = "plain"
    alpha: float = 1.0
    lam: float = 1.0
    temperature: float = 2.0
    use_class_weights: bool = False
    normalize_weights: bool = True
    t_squared: bool = True

    def __post_init__(self) -> None:
        if self.kind not in LOSS_KINDS:
            raise ValueError(f"kind must be one of {LOSS_KINDS}, got {self.kind!r}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")
        if self.kind in ("plain", "hinton_kd") and self.alpha != 1.0:
            raise ValueError(
                f"kind={self.kind!r} is defined by alpha == 1; got {self.alpha}. "
                "Use 'reweighted' or 'defect_kd' for alpha > 1."
            )
        if self.lam < 0:
            raise ValueError(f"lam must be >= 0, got {self.lam}")

    @property
    def needs_teacher(self) -> bool:
        return self.kind in ("hinton_kd", "defect_kd")


class MaskedTileLoss(nn.Module):
    """The masked multi-label loss used by every run in the matrix.

    Args:
        config: see :class:`LossConfig`.
        class_pos_weights: ``(6,)`` inverse-frequency weights, normally
            ``TileData.class_pos_weights()``. Required when
            ``config.use_class_weights`` is set.

    Call with ``(student_logits, targets, ignore)`` and, for the KD kinds,
    ``teacher_logits``. All of shape ``(N, 6)``; ``ignore`` may be bool or
    float. Returns a scalar, and :meth:`components` returns the hard and soft
    parts separately for logging.
    """

    def __init__(self, config: LossConfig | None = None,
                 class_pos_weights: torch.Tensor | None = None):
        super().__init__()
        self.config = config or LossConfig()
        if self.config.use_class_weights:
            if class_pos_weights is None:
                raise ValueError(
                    "use_class_weights=True requires class_pos_weights "
                    "(see TileData.class_pos_weights())"
                )
            weights = torch.as_tensor(class_pos_weights, dtype=torch.float32)
            if weights.shape != (NUM_CLASSES,):
                raise ValueError(
                    f"class_pos_weights must be ({NUM_CLASSES},), got {tuple(weights.shape)}"
                )
            if torch.any(weights <= 0):
                raise ValueError("class_pos_weights must be positive")
        else:
            weights = torch.ones(NUM_CLASSES, dtype=torch.float32)
        self.register_buffer("class_pos_weights", weights)

    # -- weights ---------------------------------------------------------- #

    def entry_weights(self, targets: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        """``w(tile, class)``, already masked and normalised.

        A tile counts as a defect tile if any of its classes is positive --
        including classes that are ignored elsewhere in the row, since the tile
        does hold a defect either way.
        """
        cfg = self.config
        defect_tile = (targets > 0).any(dim=1, keepdim=True)
        tile_weight = torch.where(
            defect_tile,
            torch.full_like(targets[:, :1], cfg.alpha),
            torch.ones_like(targets[:, :1]),
        )
        weights = tile_weight.expand_as(targets).clone()

        if cfg.use_class_weights:
            # Standard BCE pos_weight semantics: the inverse-frequency factor
            # applies to positive targets only, where the scarcity is.
            class_factor = torch.where(
                targets > 0,
                self.class_pos_weights.to(targets.device).expand_as(targets),
                torch.ones_like(targets),
            )
            weights = weights * class_factor

        weights = weights * keep
        if cfg.normalize_weights:
            total = weights.sum()
            n_keep = keep.sum()
            if total > 0 and n_keep > 0:
                weights = weights * (n_keep / total)
        return weights

    # -- terms ------------------------------------------------------------ #

    def components(
        self,
        student_logits: torch.Tensor,
        targets: torch.Tensor,
        ignore: torch.Tensor,
        teacher_logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """The hard and soft terms, and the total. Useful for logging."""
        cfg = self.config
        if cfg.needs_teacher and teacher_logits is None:
            raise ValueError(f"kind={cfg.kind!r} requires teacher_logits")
        if student_logits.shape != targets.shape:
            raise ValueError(
                f"student_logits {tuple(student_logits.shape)} and targets "
                f"{tuple(targets.shape)} must match"
            )
        if ignore.shape != targets.shape:
            raise ValueError(
                f"ignore {tuple(ignore.shape)} must match targets "
                f"{tuple(targets.shape)}"
            )

        keep = (1.0 - ignore.to(student_logits.dtype))
        weights = self.entry_weights(targets, keep)
        denom = keep.sum().clamp_min(1.0)

        hard_map = F.binary_cross_entropy_with_logits(
            student_logits, targets, reduction="none")
        hard = (weights * hard_map).sum() / denom

        zero = student_logits.new_zeros(())
        soft = zero
        if teacher_logits is not None and cfg.lam > 0:
            if teacher_logits.shape != targets.shape:
                raise ValueError(
                    f"teacher_logits {tuple(teacher_logits.shape)} must match "
                    f"targets {tuple(targets.shape)}"
                )
            temperature = cfg.temperature
            soft_targets = torch.sigmoid(teacher_logits.detach() / temperature)
            soft_map = F.binary_cross_entropy_with_logits(
                student_logits / temperature, soft_targets, reduction="none")
            scale = cfg.lam * (temperature ** 2 if cfg.t_squared else 1.0)
            soft = scale * (weights * soft_map).sum() / denom

        return {"hard": hard, "soft": soft, "total": hard + soft}

    def forward(
        self,
        student_logits: torch.Tensor,
        targets: torch.Tensor,
        ignore: torch.Tensor,
        teacher_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.components(student_logits, targets, ignore, teacher_logits)["total"]

    def __repr__(self) -> str:
        cfg = self.config
        return (f"MaskedTileLoss(kind={cfg.kind!r}, alpha={cfg.alpha}, "
                f"lam={cfg.lam}, T={cfg.temperature}, "
                f"class_weights={cfg.use_class_weights})")


def build_loss(kind: str, class_pos_weights=None, **overrides) -> MaskedTileLoss:
    """Convenience constructor: ``build_loss("defect_kd", w, alpha=3.0)``."""
    config = LossConfig(kind=kind, **overrides)
    return MaskedTileLoss(config, class_pos_weights)
