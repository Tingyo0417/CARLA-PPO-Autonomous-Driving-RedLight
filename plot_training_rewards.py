"""Plot training episode number against total episode reward.

The training callback in ``Original/utils.py`` records one row at the end of
each episode.  The reward used here is ``custom/total_reward``.

Examples (run from the Original directory):
    python plot_training_rewards.py tensorboard/PPO_1779020041_id16/progress.csv
    python plot_training_rewards.py tensorboard/PPO_*/progress.csv \
        --output tensorboard/ppo_training_rewards.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_rewards(path: Path) -> tuple[list[int], list[float]]:
    df = pd.read_csv(path)
    reward_column = "custom/total_reward"
    if reward_column not in df.columns:
        raise ValueError(f"{path} 找不到欄位 {reward_column}")

    rewards = pd.to_numeric(df[reward_column], errors="coerce").dropna()
    # progress.csv produced by TensorboardCallback has one row per completed
    # episode. Some older logs also contain time/episodes; use it when valid.
    if "time/episodes" in df.columns:
        episodes = pd.to_numeric(df.loc[rewards.index, "time/episodes"], errors="coerce")
        if episodes.notna().all():
            return episodes.astype(int).tolist(), rewards.astype(float).tolist()
    return list(range(1, len(rewards) + 1)), rewards.astype(float).tolist()


def moving_average(values: list[float], window: int) -> list[float]:
    return pd.Series(values).rolling(window=window, min_periods=1).mean().tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="繪製 Original 訓練回合獎勵曲線")
    parser.add_argument("csv_files", nargs="+", type=Path, help="progress.csv 路徑")
    parser.add_argument("--output", type=Path, default=Path("tensorboard/training_rewards.png"))
    parser.add_argument("--window", type=int, default=10, help="移動平均窗口；設為 0 關閉")
    args = parser.parse_args()

    plt.figure(figsize=(10, 6))
    for path in args.csv_files:
        episodes, rewards = load_rewards(path)
        label = path.parent.name
        # Raw reward curve is intentionally hidden; only show the trend below.
        # plt.plot(episodes, rewards, alpha=0.25, linewidth=1, label=f"{label} raw")
        if args.window > 1:
            plt.plot(
                episodes,
                moving_average(rewards, args.window),
                linewidth=2,
                label=f"{label} moving average",
            )

    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Training Reward per Episode")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize="small")
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150)
    plt.close()
    print(f"已輸出：{args.output}")


if __name__ == "__main__":
    main()
