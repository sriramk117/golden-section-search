"""Unimodal objectives and search settings for GSS animations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# Stops early enough for presentations while still showing several bracket steps.
TOLERANCE = 0.04


@dataclass(frozen=True)
class Problem:
    """One function/interval pair for golden section search."""

    id: str
    name: str
    title: str
    subtitle: str
    f: Callable[[float], float]
    interval: tuple[float, float]
    true_minimum: float
    true_minimum_label: str
    tags: tuple[str, ...]

    def eval(self, x: float | np.ndarray) -> float | np.ndarray:
        return self.f(x)

    def plot_curve(self, ax, color: str, lw: float = 2.5) -> float:
        """Draw ``f`` and return ymax for axis limits."""
        lo, hi = self.interval
        pad = 0.3
        ymax = -np.inf

        if "piecewise" in self.tags:
            segments = (
                (np.linspace(lo - pad, 1.0, 120, endpoint=False), self._left),
                (np.linspace(1.0, 3.0, 160), self._mid),
                (np.linspace(3.0, hi + pad, 120), self._right),
            )
            shades = ("#5eead4", "#34d399", "#2dd4bf")
            for xs, fn, shade in zip(
                (s[0] for s in segments),
                (self._left, self._mid, self._right),
                shades,
            ):
                ys = fn(xs)
                ymax = max(ymax, float(np.max(ys)))
                ax.plot(xs, ys, color=shade, lw=lw, zorder=2)
            for x_break in (1.0, 3.0):
                ax.axvline(
                    x_break,
                    color="#6b7280",
                    ls=":",
                    lw=1.0,
                    alpha=0.7,
                    zorder=1,
                )
        else:
            xs = np.linspace(lo - pad, hi + pad, 400)
            ys = self.eval(xs)
            ymax = float(np.max(ys))
            ax.plot(xs, ys, color=color, lw=lw, zorder=2)
            ax.fill_between(xs, ys, ymax + 1, color=color, alpha=0.06, zorder=1)

        ax.axvline(
            self.true_minimum,
            color="#a78bfa",
            ls="--",
            lw=1.2,
            alpha=0.55,
            zorder=1,
        )
        if "nondifferentiable" in self.tags:
            ax.scatter(
                [self.true_minimum],
                [self.eval(self.true_minimum)],
                s=70,
                facecolors="none",
                edgecolors="#f472b6",
                linewidths=2,
                zorder=4,
                label="cusp (non-differentiable)",
            )
        return ymax

    @staticmethod
    def _left(x: np.ndarray) -> np.ndarray:
        return 5.0 - 2.0 * x

    @staticmethod
    def _mid(x: np.ndarray) -> np.ndarray:
        return 1.0 + 2.0 * (x - 2.0) ** 2

    @staticmethod
    def _right(x: np.ndarray) -> np.ndarray:
        return 2.0 * x - 3.0


def _quadratic(x: float) -> float:
    return (x - 2.0) ** 2


def _absolute(x: float) -> float:
    return abs(x - 2.0)


def _piecewise(x: float) -> float:
    if x < 1.0:
        return 5.0 - 2.0 * x
    if x < 3.0:
        return 1.0 + 2.0 * (x - 2.0) ** 2
    return 2.0 * x - 3.0


PROBLEMS: dict[str, Problem] = {
    "quadratic": Problem(
        id="quadratic",
        name="Smooth quadratic",
        title=r"$f(x) = (x - 2)^2$",
        subtitle="Smooth, differentiable — classic GSS example",
        f=_quadratic,
        interval=(0.0, 5.0),
        true_minimum=2.0,
        true_minimum_label="x = 2",
        tags=(),
    ),
    "absolute": Problem(
        id="absolute",
        name="Absolute value",
        title=r"$f(x) = |x - 2|$",
        subtitle="Continuous but not differentiable at the minimum",
        f=_absolute,
        interval=(0.0, 5.0),
        true_minimum=2.0,
        true_minimum_label="x = 2 (cusp)",
        tags=("nondifferentiable",),
    ),
    "piecewise": Problem(
        id="piecewise",
        name="Piecewise V",
        title=r"Piecewise unimodal $f$",
        subtitle="Three disjoint pieces: two rays + a central bowl",
        f=_piecewise,
        interval=(0.0, 5.0),
        true_minimum=2.0,
        true_minimum_label="x = 2 (bowl center)",
        tags=("piecewise",),
    ),
}


def get_problem(problem_id: str) -> Problem:
    try:
        return PROBLEMS[problem_id]
    except KeyError as exc:
        valid = ", ".join(PROBLEMS)
        raise SystemExit(f"Unknown problem {problem_id!r}. Choose from: {valid}") from exc
