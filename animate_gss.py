#!/usr/bin/env python3
"""Animate Golden Section Search on unimodal functions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from gss import INVPHI, gss, gss_iterate
from problems import PROBLEMS, Problem, TOLERANCE, get_problem

PAUSE_FRAMES = 6
FINAL_PAUSE_FRAMES = 20

COLORS = {
    "bg": "#0f1117",
    "panel": "#171923",
    "grid": "#2a2d3a",
    "curve": "#5eead4",
    "interval": "#fbbf24",
    "interval_alpha": 0.22,
    "discarded": "#4b5563",
    "point_c": "#fb7185",
    "point_d": "#60a5fa",
    "minimum": "#a78bfa",
    "text": "#e5e7eb",
    "muted": "#9ca3af",
    "accent": "#fcd34d",
}


def build_frames(problem: Problem) -> list[dict]:
    """Collect animation frames from the search iterator."""
    frames: list[dict] = []
    a0, b0 = problem.interval
    for step in gss_iterate(problem.f, a0, b0, tolerance=TOLERANCE):
        payload = {
            "iteration": step.iteration,
            "a": step.a,
            "b": step.b,
            "c": step.c,
            "d": step.d,
            "fc": step.fc,
            "fd": step.fd,
            "keep_left": step.keep_left,
            "done": step.done,
            "phase": "start" if step.c is None and not step.done else "search",
        }
        repeat = FINAL_PAUSE_FRAMES if step.done else PAUSE_FRAMES
        frames.extend([payload] * repeat)
    return frames


def setup_axes(fig: plt.Figure, problem: Problem) -> tuple[plt.Axes, plt.Axes]:
    lo, hi = problem.interval
    pad = 0.25

    ax = fig.add_axes([0.08, 0.22, 0.84, 0.68])
    ax.set_facecolor(COLORS["panel"])
    ymax = problem.plot_curve(ax, COLORS["curve"])
    ymin = 0.0 if "absolute" in problem.id or "piecewise" in problem.id else -0.4
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(ymin, ymax + 0.9)
    ax.set_xlabel("x", color=COLORS["text"], fontsize=11)
    ax.set_ylabel("f(x)", color=COLORS["text"], fontsize=11)
    ax.tick_params(colors=COLORS["muted"])
    ax.grid(True, color=COLORS["grid"], alpha=0.45, lw=0.6)
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])

    info = fig.add_axes([0.08, 0.04, 0.84, 0.12])
    info.set_facecolor(COLORS["panel"])
    info.axis("off")

    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle(
        f"Golden Section Search — {problem.name}",
        color=COLORS["accent"],
        fontsize=17,
        fontweight="bold",
        y=0.97,
    )
    return ax, info


def draw_frame(
    ax: plt.Axes,
    frame: dict,
    artists: dict,
    problem: Problem,
) -> None:
    lo, hi = problem.interval
    x_pad = 0.25
    a, b = frame["a"], frame["b"]

    artists["interval"].set_x(a)
    artists["interval"].set_width(b - a)

    artists["discarded"].set_visible(False)
    if frame["phase"] == "search" and frame["keep_left"] is not None:
        if frame["keep_left"]:
            artists["discarded"].set_x(b)
            artists["discarded"].set_width(hi + x_pad - b)
        else:
            artists["discarded"].set_x(lo - x_pad)
            artists["discarded"].set_width(a - (lo - x_pad))
        artists["discarded"].set_visible(True)

    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    label_dy = 0.06 * y_range

    for key in ("c", "d"):
        x_val = frame[key]
        line = artists[f"line_{key}"]
        scatter = artists[f"scatter_{key}"]
        label = artists[f"label_{key}"]
        if x_val is None:
            line.set_visible(False)
            scatter.set_visible(False)
            label.set_visible(False)
            continue

        fx = float(problem.eval(x_val))
        line.set_data([x_val, x_val], [0, fx])
        line.set_visible(True)
        scatter.set_offsets([[x_val, fx]])
        scatter.set_visible(True)
        label.set_position((x_val, fx + label_dy))
        label.set_text(f"{key} = {x_val:.3f}")
        label.set_visible(True)

    artists["estimate"].set_visible(frame["done"])
    if frame["done"]:
        x_hat = (a + b) / 2
        artists["estimate"].set_offsets([[x_hat, problem.eval(x_hat)]])
        artists["estimate"].set_label(f"estimate ≈ {x_hat:.4f}")

    width = b - a
    if frame["phase"] == "start":
        status = "Initial bracket [a, b]"
        detail = (
            f"a = {a:.3f}   b = {b:.3f}   width = {width:.3f}   "
            f"tolerance = {TOLERANCE}\n"
            f"Interior points use φ⁻¹ ≈ {INVPHI:.4f}"
        )
    elif frame["done"]:
        x_hat = (a + b) / 2
        n_iters = frame["iteration"]
        status = f"Converged in {n_iters} iteration{'s' if n_iters != 1 else ''}"
        detail = (
            f"Bracket width {width:.4f} ≤ tolerance {TOLERANCE}\n"
            f"Estimate x* ≈ {x_hat:.4f}   (true minimum at {problem.true_minimum_label})"
        )
    else:
        decision = "keep [a, d]" if frame["keep_left"] else "keep [c, b]"
        cmp_txt = "f(c) < f(d)" if frame["keep_left"] else "f(c) ≥ f(d)"
        status = f"Step {frame['iteration'] + 1}  —  {cmp_txt}  →  {decision}"
        detail = (
            f"a = {a:.4f}   b = {b:.4f}   width = {width:.4f}\n"
            f"f(c) = {frame['fc']:.4f}   f(d) = {frame['fd']:.4f}"
        )

    artists["status"].set_text(status)
    artists["detail"].set_text(detail)


def create_animation(
    problem: Problem,
    save_path: Path | None = None,
) -> animation.FuncAnimation:
    frames = build_frames(problem)
    lo, hi = problem.interval
    fig = plt.figure(figsize=(10, 6.5), dpi=120)
    ax, info = setup_axes(fig, problem)

    artists = {
        "interval": Rectangle(
            (lo, -0.08),
            hi - lo,
            0.16,
            facecolor=COLORS["interval"],
            edgecolor=COLORS["accent"],
            alpha=COLORS["interval_alpha"],
            lw=1.5,
            zorder=3,
        ),
        "discarded": Rectangle(
            (0, -0.08),
            0,
            0.16,
            facecolor=COLORS["discarded"],
            edgecolor="none",
            alpha=0.35,
            zorder=2,
        ),
        "line_c": ax.plot([], [], color=COLORS["point_c"], ls=":", lw=1.4, zorder=4)[0],
        "line_d": ax.plot([], [], color=COLORS["point_d"], ls=":", lw=1.4, zorder=4)[0],
        "scatter_c": ax.scatter(
            [], [], s=90, color=COLORS["point_c"], zorder=5, edgecolors="white", lw=0.8
        ),
        "scatter_d": ax.scatter(
            [], [], s=90, color=COLORS["point_d"], zorder=5, edgecolors="white", lw=0.8
        ),
        "estimate": ax.scatter(
            [],
            [],
            s=140,
            marker="*",
            color=COLORS["minimum"],
            zorder=6,
            edgecolors="white",
            lw=0.8,
        ),
        "label_c": ax.text(0, 0, "", ha="center", color=COLORS["point_c"], fontsize=9, zorder=7),
        "label_d": ax.text(0, 0, "", ha="center", color=COLORS["point_d"], fontsize=9, zorder=7),
        "status": info.text(
            0.0, 0.72, "", color=COLORS["accent"], fontsize=12, fontweight="bold", va="top"
        ),
        "detail": info.text(
            0.0,
            0.08,
            "",
            color=COLORS["text"],
            fontsize=10,
            va="bottom",
            family="monospace",
        ),
    }

    ax.add_patch(artists["interval"])
    ax.add_patch(artists["discarded"])

    def update(i: int) -> list:
        draw_frame(ax, frames[i], artists, problem)
        return list(artists.values())

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=80,
        blit=False,
        repeat=True,
    )

    if save_path is not None:
        suffix = save_path.suffix.lower()
        writer = (
            animation.PillowWriter(fps=12)
            if suffix == ".gif"
            else animation.FFMpegWriter(fps=12, bitrate=2400)
        )
        anim.save(save_path, writer=writer, dpi=120)
        print(f"Saved {problem.id} → {save_path}")

    return anim


def run_problem(problem: Problem, save_path: Path | None, show: bool) -> None:
    lo, hi = problem.interval
    estimate = gss(problem.f, lo, hi, tolerance=TOLERANCE)
    n_steps = sum(1 for s in gss_iterate(problem.f, lo, hi, tolerance=TOLERANCE) if s.c is not None)
    print(
        f"[{problem.id}] x* ≈ {estimate:.4f}  "
        f"(true {problem.true_minimum})  —  {n_steps} search steps, tolerance {TOLERANCE}"
    )
    anim = create_animation(problem, save_path=save_path)
    if show and save_path is None:
        plt.show()
    else:
        plt.close(anim._fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate Golden Section Search")
    parser.add_argument(
        "--problem",
        choices=list(PROBLEMS),
        default="quadratic",
        help="Which function to visualize (default: quadratic)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Render every built-in problem",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Save animation to file (.gif or .mp4)",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        help="With --all, write one GIF per problem into this directory",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip interactive window (useful with --save)",
    )
    args = parser.parse_args()

    show = not args.no_show

    if args.all:
        out_dir = args.save_dir or Path(".")
        out_dir.mkdir(parents=True, exist_ok=True)
        for problem in PROBLEMS.values():
            path = out_dir / f"golden_section_{problem.id}.gif"
            run_problem(problem, save_path=path, show=False)
        return

    problem = get_problem(args.problem)
    save_path = args.save
    if save_path is None and args.no_show:
        save_path = Path(f"golden_section_{problem.id}.gif")

    run_problem(problem, save_path=save_path, show=show and save_path is None)


if __name__ == "__main__":
    main()
