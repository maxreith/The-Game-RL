"""Train RL agent using curriculum learning (2 -> 3 -> 4 -> 5 players)."""

import argparse
import os
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from train_rl import (
    EntropyScheduleCallback,
    GameMetricsCallback,
    create_env,
    linear_schedule,
)


class CurriculumProgressCallback(BaseCallback):
    """Callback to log curriculum stage progress.

    Args:
        stage: Current curriculum stage number.
        n_players: Number of players in current stage.
    """

    def __init__(self, stage, n_players, verbose=0):
        super().__init__(verbose)
        self.stage = stage
        self.n_players = n_players

    def _on_step(self):
        if self.num_timesteps % 50000 == 0:
            self.logger.record("curriculum/stage", self.stage)
            self.logger.record("curriculum/n_players", self.n_players)
        return True


def train_curriculum(
    player_counts=None,
    timesteps_per_stage=None,
    max_players=5,
    n_envs=None,
    verbose=1,
    tensorboard_log=True,
    resume_from=None,
):
    """Train agent using curriculum learning across player counts.

    Args:
        player_counts: List of player counts for each stage.
        timesteps_per_stage: List of timesteps for each stage.
        max_players: Maximum players for fixed observation size.
        n_envs: Number of parallel environments. Defaults to CPU count.
        verbose: Verbosity level.
        tensorboard_log: Whether to enable tensorboard logging.
        resume_from: Path to model checkpoint to resume from.

    Returns:
        Trained MaskablePPO model.
    """
    if player_counts is None:
        player_counts = [2, 3, 4, 5]
    if timesteps_per_stage is None:
        timesteps_per_stage = [500_000, 500_000, 500_000, 1_000_000]

    if n_envs is None:
        n_envs = os.cpu_count() or 1

    bld_dir = Path(__file__).parent.parent / "bld"
    bld_dir.mkdir(exist_ok=True)

    log_path = str(bld_dir / "rl_curriculum_logs") if tensorboard_log else None
    monitor_dir = str(bld_dir / "curriculum_monitor_logs") if tensorboard_log else None

    if monitor_dir:
        Path(monitor_dir).mkdir(exist_ok=True)

    model = None
    total_timesteps_so_far = 0

    if resume_from:
        if verbose:
            print(f"Resuming from checkpoint: {resume_from}")
        model = MaskablePPO.load(resume_from)
        total_timesteps_so_far = model.num_timesteps

    for stage, (n_players, timesteps) in enumerate(
        zip(player_counts, timesteps_per_stage), start=1
    ):
        if verbose:
            print(f"\n{'=' * 50}")
            print(f"Stage {stage}: Training with {n_players} players")
            print(f"Timesteps: {timesteps:,}")
            print(f"{'=' * 50}\n")

        env = create_env(
            n_players=n_players,
            max_players=max_players,
            hand_size=6,  # Fixed hand size for consistent observation space
            n_envs=n_envs,
            log_dir=monitor_dir,
        )

        if model is None:
            policy_kwargs = dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                activation_fn=torch.nn.Tanh,
            )
            model = MaskablePPO(
                "MlpPolicy",
                env,
                policy_kwargs=policy_kwargs,
                verbose=verbose,
                tensorboard_log=log_path,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.05,
                learning_rate=linear_schedule(3e-4),
                n_steps=2048,
                batch_size=256,
                n_epochs=10,
                clip_range=0.2,
            )
        else:
            model.set_env(env)

        callbacks = [
            GameMetricsCallback(verbose=verbose),
            EntropyScheduleCallback(start_ent=0.05, end_ent=0.005, verbose=verbose),
            CurriculumProgressCallback(stage, n_players, verbose=verbose),
        ]

        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            reset_num_timesteps=False,
        )

        total_timesteps_so_far += timesteps

        checkpoint_path = bld_dir / f"curriculum_stage_{n_players}p"
        model.save(checkpoint_path)
        if verbose:
            print(f"Saved checkpoint: {checkpoint_path}")

    final_path = bld_dir / "the_game_curriculum_ppo"
    model.save(final_path)
    if verbose:
        print(f"\nFinal model saved: {final_path}")
        print(f"Total timesteps: {total_timesteps_so_far:,}")

    return model


def main():
    """Entry point for curriculum training script."""
    parser = argparse.ArgumentParser(
        description="Train RL agent using curriculum learning"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=[500_000, 500_000, 500_000, 1_000_000],
        help="Timesteps per stage (default: 500k, 500k, 500k, 1M)",
    )
    parser.add_argument(
        "--players",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5],
        help="Player counts per stage (default: 2, 3, 4, 5)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Number of parallel environments (default: CPU count)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    args = parser.parse_args()

    train_curriculum(
        player_counts=args.players,
        timesteps_per_stage=args.timesteps,
        n_envs=args.n_envs,
        verbose=0 if args.quiet else 1,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
