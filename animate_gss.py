from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from gss import INVPHI, gss, gss_iterate

# f(x) = (x - 2)^2  — minimum at x = 2, classic GSS example
def objective(x: float | np.ndarray) -> float | np.ndarray:
    return (x - 2.0) ** 2


INTERVAL = (0.0, 5.0)
TOLERANCE = 1e-4
PAUSE_FRAMES = 8
FINAL_PAUSE_FRAMES = 24

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


def build_frames() -> list[dict]:
    """Collect animation frames from the search iterator."""
    frames: list[dict] = []
    for step in gss_iterate(objective, *INTERVAL, tolerance=TOLERANCE):
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


def setup_axes(fig: plt.Figure) -> tuple[plt.Axes, plt.Axes]:
    x = np.linspace(INTERVAL[0] - 0.3, INTERVAL[1] + 0.3, 400)
    y = objective(x)

    ax = fig.add_axes([0.08, 0.22, 0.84, 0.68])
    ax.set_facecolor(COLORS["panel"])
    ax.plot(x, y, color=COLORS["curve"], lw=2.5, zorder=2)
    ax.fill_between(x, y, y.max() + 1, color=COLORS["curve"], alpha=0.06, zorder=1)
    ax.axvline(2.0, color=COLORS["minimum"], ls="--", lw=1.2, alpha=0.55, zorder=1)
    ax.set_xlim(INTERVAL[0] - 0.25, INTERVAL[1] + 0.25)
    ax.set_ylim(-0.4, y.max() + 0.8)
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
        "Golden Section Search",
        color=COLORS["accent"],
        fontsize=18,
        fontweight="bold",
        y=0.97,
    )
    ax.set_title(
        r"$f(x) = (x - 2)^2$  on  $[0,\,5]$",
        color=COLORS["muted"],
        fontsize=12,
        pad=10,
    )
    return ax, info


def draw_frame(
    ax: plt.Axes,
    info: plt.Axes,
    frame: dict,
    artists: dict,
) -> None:
    a, b = frame["a"], frame["b"]

    # Bracket interval on the x-axis
    artists["interval"].set_x(a)
    artists["interval"].set_width(b - a)

    # Discarded region flash (shown briefly after a decision)
    artists["discarded"].set_visible(False)
    if frame["phase"] == "search" and frame["keep_left"] is not None:
        if frame["keep_left"]:
            artists["discarded"].set_x(b)
            artists["discarded"].set_width(INTERVAL[1] + 0.25 - b)
        else:
            artists["discarded"].set_x(INTERVAL[0] - 0.25)
            artists["discarded"].set_width(a - (INTERVAL[0] - 0.25))
        artists["discarded"].set_visible(True)

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

        fx = objective(x_val)
        line.set_data([x_val, x_val], [0, fx])
        line.set_visible(True)
        scatter.set_offsets([[x_val, fx]])
        scatter.set_visible(True)
        label.set_position((x_val, fx + 0.35))
        label.set_text(f"{key} = {x_val:.4f}")
        label.set_visible(True)

    # Estimated minimum marker at convergence
    artists["estimate"].set_visible(frame["done"])
    if frame["done"]:
        x_hat = (a + b) / 2
        artists["estimate"].set_offsets([[x_hat, objective(x_hat)]])
        artists["estimate"].set_label(f"estimate ≈ {x_hat:.5f}")

    width = b - a
    if frame["phase"] == "start":
        status = "Initial bracket [a, b]"
        detail = (
            f"a = {a:.4f}   b = {b:.4f}   width = {width:.4f}\n"
            f"Place c and d using the golden ratio  φ⁻¹ ≈ {INVPHI:.6f}"
        )
    elif frame["done"]:
        x_hat = (a + b) / 2
        status = "Converged"
        detail = (
            f"Final bracket width {width:.2e} < tolerance {TOLERANCE:.0e}\n"
            f"Minimum estimate: x* ≈ {x_hat:.6f}   (true minimum at x = 2)"
        )
    else:
        decision = "keep [a, d]" if frame["keep_left"] else "keep [c, b]"
        cmp_txt = "f(c) < f(d)" if frame["keep_left"] else "f(c) ≥ f(d)"
        status = f"Iteration {frame['iteration'] + 1}  —  {cmp_txt}  →  {decision}"
        detail = (
            f"a = {a:.5f}   b = {b:.5f}   width = {width:.5f}\n"
            f"f(c) = {frame['fc']:.5f}   f(d) = {frame['fd']:.5f}"
        )

    artists["status"].set_text(status)
    artists["detail"].set_text(detail)


def create_animation(save_path: Path | None = None) -> animation.FuncAnimation:
    frames = build_frames()
    fig = plt.figure(figsize=(10, 6.5), dpi=120)
    ax, info = setup_axes(fig)

    artists = {
        "interval": Rectangle(
            (INTERVAL[0], -0.08),
            INTERVAL[1] - INTERVAL[0],
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
        "scatter_c": ax.scatter([], [], s=90, color=COLORS["point_c"], zorder=5, edgecolors="white", lw=0.8),
        "scatter_d": ax.scatter([], [], s=90, color=COLORS["point_d"], zorder=5, edgecolors="white", lw=0.8),
        "estimate": ax.scatter([], [], s=140, marker="*", color=COLORS["minimum"], zorder=6, edgecolors="white", lw=0.8),
        "label_c": ax.text(0, 0, "", ha="center", color=COLORS["point_c"], fontsize=9, zorder=7),
        "label_d": ax.text(0, 0, "", ha="center", color=COLORS["point_d"], fontsize=9, zorder=7),
        "status": info.text(0.0, 0.72, "", color=COLORS["accent"], fontsize=12, fontweight="bold", va="top"),
        "detail": info.text(0.0, 0.08, "", color=COLORS["text"], fontsize=10, va="bottom", family="monospace"),
    }

    ax.add_patch(artists["interval"])
    ax.add_patch(artists["discarded"])

    def update(i: int) -> list:
        draw_frame(ax, info, frames[i], artists)
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
        if suffix == ".gif":
            writer = animation.PillowWriter(fps=12)
        else:
            writer = animation.FFMpegWriter(fps=12, bitrate=2400)
        anim.save(save_path, writer=writer, dpi=120)
        print(f"Saved animation to {save_path}")

    return anim


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate Golden Section Search")
    parser.add_argument(
        "--save",
        type=Path,
        help="Save animation to file (.gif or .mp4)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip interactive window (useful with --save)",
    )
    args = parser.parse_args()

    minimum = gss(objective, *INTERVAL, tolerance=TOLERANCE)
    print(f"Golden section estimate: x* ≈ {minimum:.8f}")

    anim = create_animation(save_path=args.save)
    if not args.no_show and args.save is None:
        plt.show()
    elif args.save is not None and args.no_show:
        plt.close(anim._fig)


if __name__ == "__main__":
    main()
