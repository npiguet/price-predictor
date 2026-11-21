"""The per-round diagnostics record and its stdout block (contract §2)."""

from __future__ import annotations

from dataclasses import dataclass, field

from draft.application.online_advantages import AdvantageStats
from draft.application.online_diagnostics import SweepDiagnostics

# --------------------------------------------------------------------------- #
# Per-round diagnostics record + stdout formatting (contract §2, FR-014…FR-018)
# --------------------------------------------------------------------------- #

@dataclass
class RoundDiagnostics:
    """Everything printed for one round (data-model §6)."""

    learner_label: str = ""
    anchor_label: str = ""
    index: int = 0
    drafts: int = 0
    total_drafts: int = 0
    learner_seats: int = 0
    dropped_seats: int = 0
    learner_picks: int = 0
    gen_seconds: float = 0.0
    train_seconds: float = 0.0
    skipped: bool = False
    skip_reason: str | None = None
    reward: AdvantageStats = field(default_factory=AdvantageStats)
    sweep: SweepDiagnostics = field(default_factory=SweepDiagnostics)
    policy_loss: float = 0.0
    grad_norm: float = 0.0
    lr: float = 0.0
    anchor_margin: float | None = None
    label_means: dict[str, float] = field(default_factory=dict)
    window_drafts: int = 0


def _fmt_margin(margin: float | None) -> str:
    return "n/a" if margin is None else f"{margin:+.3f}"


def _ordered_label_means(diag: "RoundDiagnostics") -> str:
    """Learner first, then the anchor, then the rest — the margin's two terms
    sit side by side rather than being separated by alphabetical order."""
    lead = [
        label for label in (diag.learner_label, diag.anchor_label)
        if label in diag.label_means
    ]
    rest = sorted(label for label in diag.label_means if label not in lead)
    return " ".join(f"{label}={diag.label_means[label]:.3f}" for label in lead + rest)


def format_round_lines(diag: RoundDiagnostics) -> list[str]:
    """The round's stdout block: one summary line + four detail lines.

    A degenerate round (FR-023) collapses to a single summary line marked
    ``skipped (no signal)`` — it still carries the anchor margin, so no round is
    ever silent (SC-003).
    """
    head = (
        f"round {diag.index} | "
        f"drafts {diag.drafts} ({diag.total_drafts})"
    )
    if diag.skipped:
        return [
            f"{head} | skipped (no signal): {diag.skip_reason} | "
            f"margin {_fmt_margin(diag.anchor_margin)}"
        ]

    reward, sweep = diag.reward, diag.sweep
    summary = (
        f"{head} | "
        f"picks {diag.learner_picks} ({diag.dropped_seats} seats dropped) | "
        f"gen {diag.gen_seconds:.0f}s train {diag.train_seconds:.0f}s | "
        f"R {reward.reward_mean:+.3f}+-{reward.reward_std:.3f} | "
        f"|A|<0.1 {reward.adv_near_zero_frac * 100:.0f}% | "
        f"ppl {sweep.perplexity:.2f} | "
        f"KL {sweep.kl_prev_new:.4f} | "
        f"margin {_fmt_margin(diag.anchor_margin)}"
    )
    label_means = _ordered_label_means(diag)
    return [
        summary,
        (
            f"  reward   : learner seats={diag.learner_seats} "
            f"R mean={reward.reward_mean:+.3f} std={reward.reward_std:.3f} | "
            f"A std={reward.adv_std:.3f} "
            f"|A|<0.1={reward.adv_near_zero_frac * 100:.1f}% "
            f"|A|>0.5={reward.adv_large_frac * 100:.1f}% "
            f"max|A|={reward.adv_absmax:.2f}"
        ),
        (
            f"  explore  : H={sweep.entropy:.3f} ppl={sweep.perplexity:.3f} "
            f"off-argmax={sweep.off_argmax_rate * 100:.1f}%   "
            "(band: ppl 2-3 / off-argmax 25-40%)"
        ),
        (
            f"  movement : mean logpi={sweep.mean_logp:.3f} "
            f"policy_loss={diag.policy_loss:+.4f} "
            f"grad_norm={diag.grad_norm:.2f} "
            f"KL(prev||new)={sweep.kl_prev_new:.5f} "
            f"KL(init||new)={sweep.kl_init_new:.5f} "
            f"lr={diag.lr:.1e}"
        ),
        (
            f"  progress : anchor margin={_fmt_margin(diag.anchor_margin)} | "
            f"{label_means or 'no scored seats yet'} | "
            f"window={diag.window_drafts} drafts"
        ),
    ]
