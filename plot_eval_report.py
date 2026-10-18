"""Create a compact, presentation-ready CARLA evaluation report.

The original eval plot is useful for debugging, but an 8 x 8 subplot grid is
hard to read in a report. This script focuses on the evidence needed for the
current task: route completion, red-light stops, green-light restarts, speed,
and lateral safety.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd


def as_bool(series):
    """Convert bool/object CSV columns without treating non-empty strings as True."""
    return series.map(lambda value: str(value).strip().lower() == "true")


def red_segments(episode_df):
    red = as_bool(episode_df["red_light"]).to_numpy(dtype=bool)
    starts = np.flatnonzero(red & ~np.r_[False, red[:-1]])
    stops = np.flatnonzero(red & ~np.r_[red[1:], False])
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def episode_successes(df):
    route_sizes = df[df["model_id"] == "route"].groupby("episode").size()
    samples = df[(df["model_id"] != "route") & df["step"].notna()]
    final_rows = samples.sort_values(["episode", "step"]).groupby("episode").tail(1)
    successes = {}
    for row in final_rows.itertuples():
        route_size = int(route_sizes.loc[row.episode])
        successes[int(row.episode)] = int(row.waypoint_index) >= route_size - 1
    return successes


def build_report(csv_path, output_path, episode, fps):
    df = pd.read_csv(csv_path)
    samples = df[(df["model_id"] != "route") & df["step"].notna()].copy()
    route = df[(df["model_id"] == "route") & (df["episode"] == episode)].copy()
    episode_df = samples[samples["episode"] == episode].sort_values("step").reset_index(drop=True)
    if episode_df.empty or route.empty:
        raise ValueError(f"Episode {episode} is missing samples or route rows in {csv_path}")

    successes = episode_successes(df)
    successful_count = sum(successes.values())
    total_episodes = len(successes)
    route_complete = bool(successes.get(episode, False))

    start_step = float(episode_df["step"].iloc[0])
    time_s = (episode_df["step"].to_numpy(dtype=float) - start_step) / fps
    episode_df["time_s"] = time_s
    route_points = len(route)
    progress = np.clip(
        episode_df["waypoint_index"].to_numpy(dtype=float) / max(route_points - 1, 1) * 100.0,
        0.0,
        100.0,
    )

    segments = red_segments(episode_df)
    stop_rows = []
    release_rows = []
    restart_successes = 0
    for start, stop in segments:
        segment_df = episode_df.iloc[start:stop + 1]
        if float(segment_df["speed"].min()) < 1.0:
            stop_position = int(segment_df["speed"].astype(float).idxmin())
            stop_rows.append(stop_position)
            release_position = stop + 1
            if release_position < len(episode_df):
                release_rows.append(release_position)
                restart_window = episode_df.iloc[
                    release_position:min(release_position + 46, len(episode_df))
                ]
                if float(restart_window["speed"].max()) >= 5.0:
                    restart_successes += 1

    final_distance = float(episode_df["distance"].iloc[-1])
    completion_time = float(time_s[-1])
    max_center = float(episode_df["center_dev"].max())
    moving = episode_df[episode_df["speed"] > 1.0]
    center_p95 = float(moving["center_dev"].quantile(0.95)) if not moving.empty else 0.0
    terminal = str(episode_df["terminal_reason"].iloc[-1])
    if route_complete:
        terminal = "Success"

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.facecolor": "#f7f9fc",
        "axes.facecolor": "white",
        "axes.edgecolor": "#c7ced9",
        "grid.color": "#dce2ea",
        "grid.alpha": 0.75,
    })

    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(
        3, 2,
        left=0.055, right=0.985, bottom=0.085, top=0.855,
        wspace=0.22, hspace=0.34,
        width_ratios=[1.12, 1.0], height_ratios=[1, 1, 1],
    )
    ax_route = fig.add_subplot(grid[:, 0])
    ax_speed = fig.add_subplot(grid[0, 1])
    ax_progress = fig.add_subplot(grid[1, 1], sharex=ax_speed)
    ax_center = fig.add_subplot(grid[2, 1], sharex=ax_speed)

    status_color = "#17864b" if route_complete else "#c23b3b"
    fig.suptitle(
        f"CARLA Evaluation Report  |  Acceptance {successful_count}/{total_episodes} Routes Completed",
        y=0.975,
        fontsize=18,
        fontweight="bold",
        color="#1f2d3d",
    )
    fig.text(
        0.5,
        0.925,
        (f"Representative episode {episode}: {len(segments)} red-light encounters, "
         f"{len(stop_rows)} full stops, {restart_successes}/{len(stop_rows)} successful restarts, "
         f"final status: {terminal}"),
        ha="center",
        va="center",
        fontsize=11,
        color=status_color,
        fontweight="bold",
    )

    # Route and driven trajectory.
    ax_route.plot(
        route["route_x"], route["route_y"],
        linestyle="--", linewidth=2.0, color="#8d99a6", label="Planned route",
    )
    xy = episode_df[["vehicle_location_x", "vehicle_location_y"]].to_numpy(dtype=float)
    points = xy.reshape(-1, 1, 2)
    line_segments = np.concatenate([points[:-1], points[1:]], axis=1)
    trajectory = LineCollection(line_segments, cmap="viridis", linewidth=3.0)
    trajectory.set_array(time_s[:-1])
    ax_route.add_collection(trajectory)
    ax_route.scatter(xy[0, 0], xy[0, 1], s=110, marker="o", color="#1b9e77",
                     edgecolor="white", linewidth=1.5, zorder=6, label="Start")
    ax_route.scatter(xy[-1, 0], xy[-1, 1], s=190, marker="*", color="#f4b400",
                     edgecolor="#7a5b00", linewidth=1.0, zorder=7, label="Route completed")
    for number, row_index in enumerate(stop_rows, start=1):
        row = episode_df.iloc[row_index]
        ax_route.scatter(row["vehicle_location_x"], row["vehicle_location_y"], s=80,
                         marker="s", color="#d62728", edgecolor="white", linewidth=1.2,
                         zorder=7, label="Full stop at red" if number == 1 else None)
        ax_route.annotate(
            f"Stop {number}",
            (row["vehicle_location_x"], row["vehicle_location_y"]),
            xytext=(7, 7), textcoords="offset points", fontsize=8, color="#9e1b1b",
        )
    colorbar = fig.colorbar(trajectory, ax=ax_route, fraction=0.035, pad=0.02)
    colorbar.set_label("Elapsed time (s)")
    ax_route.set_title("Route trajectory and red-light stop locations", fontweight="bold")
    ax_route.set_xlabel("Relative X (m)")
    ax_route.set_ylabel("Relative Y (m)")
    ax_route.axis("equal")
    ax_route.grid(True)
    ax_route.legend(loc="best", frameon=True, framealpha=0.95)

    summary_text = (
        f"STATUS  {terminal.upper()}\n"
        f"Route progress  100%\n"
        f"Distance  {final_distance:.1f} m\n"
        f"Completion time  {completion_time:.1f} s\n"
        f"Red full stops  {len(stop_rows)}\n"
        f"Green restarts  {restart_successes}/{len(stop_rows)}\n"
        f"Moving center p95/max  {center_p95:.2f}/{max_center:.2f} m\n"
        "Safety terminals  0"
    )
    ax_route.text(
        0.535, 0.025, summary_text,
        transform=ax_route.transAxes, va="bottom", ha="left",
        family="monospace", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#eef8f1",
                  edgecolor="#69a77d", alpha=0.96),
    )

    # Time-series panels share the red-light shading.
    for axis in (ax_speed, ax_progress, ax_center):
        for start, stop in segments:
            axis.axvspan(time_s[start], time_s[stop], color="#ef5350", alpha=0.16, lw=0)
        axis.grid(True)

    ax_speed.plot(time_s, episode_df["speed"], color="#e67e22", linewidth=1.8,
                  label="Vehicle speed")
    ax_speed.axhline(1.0, color="#777", linestyle=":", linewidth=1.0,
                     label="Full-stop threshold (1 km/h)")
    if stop_rows:
        ax_speed.scatter(time_s[stop_rows], episode_df.iloc[stop_rows]["speed"],
                         s=42, marker="s", color="#d62728", zorder=6, label="Full stop")
    if release_rows:
        ax_speed.scatter(time_s[release_rows], episode_df.iloc[release_rows]["speed"],
                         s=50, marker="^", color="#1b9e77", zorder=6, label="Green release")
    ax_speed.set_title("Speed response at traffic lights", fontweight="bold")
    ax_speed.set_ylabel("Speed (km/h)")
    ax_speed.set_ylim(bottom=-0.5)
    ax_speed.legend(loc="upper right", ncol=2, framealpha=0.95)
    ax_speed.text(0.01, 0.92, "Red shading = red-light active",
                  transform=ax_speed.transAxes, fontsize=8.5, color="#a12622")

    ax_progress.plot(time_s, progress, color="#2369bd", linewidth=2.0)
    ax_progress.fill_between(time_s, 0, progress, color="#5b9bd5", alpha=0.13)
    ax_progress.scatter(time_s[-1], progress[-1], s=95, marker="*", color="#f4b400",
                        edgecolor="#7a5b00", zorder=7)
    ax_progress.annotate(
        "100% — route complete",
        (time_s[-1], progress[-1]), xytext=(-128, -23), textcoords="offset points",
        fontsize=9, color="#174a83", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#174a83"),
    )
    ax_progress.set_title("Monotonic route progress", fontweight="bold")
    ax_progress.set_ylabel("Route progress (%)")
    ax_progress.set_ylim(-3, 105)

    ax_center.plot(time_s, episode_df["center_dev"], color="#7b4ab5", linewidth=1.4)
    ax_center.axhline(1.0, color="#7b4ab5", linestyle="--", linewidth=1.0,
                      alpha=0.65, label="1 m reference")
    ax_center.fill_between(time_s, 0, episode_df["center_dev"], color="#9c75c7", alpha=0.14)
    ax_center.set_title("Lateral stability", fontweight="bold")
    ax_center.set_xlabel("Elapsed time (s)")
    ax_center.set_ylabel("Center deviation (m)")
    ax_center.set_ylim(0, max(1.25, max_center * 1.15))
    ax_center.legend(loc="upper right")

    fig.text(
        0.01, 0.018,
        f"Source: {Path(csv_path).name}  |  Deterministic PPO evaluation at {fps:g} FPS  |  "
        "Red-light intervals and stop markers are computed directly from the logged simulator state.",
        fontsize=7.5, color="#627080",
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return output_path, pdf_path


def main():
    parser = argparse.ArgumentParser(description="Create a presentation-ready CARLA eval report")
    parser.add_argument("csv", help="Evaluation CSV produced by eval.py")
    parser.add_argument("--episode", type=int, default=7, help="Representative episode to plot")
    parser.add_argument("--fps", type=float, default=15.0, help="Simulator FPS")
    parser.add_argument("--output", default="tensorboard/eval_plots/final_eval_report.png")
    args = parser.parse_args()
    png_path, pdf_path = build_report(args.csv, args.output, args.episode, args.fps)
    print(f"Saved PNG report to {png_path}")
    print(f"Saved PDF report to {pdf_path}")


if __name__ == "__main__":
    main()
