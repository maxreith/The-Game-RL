"""Train RL agent to play The Game using MaskablePPO.

Trains two variants to 100M steps each:
- Sparse: Only terminal rewards (win/loss)
- Shaped: Dense rewards with trick play bonuses and distance penalties
"""

import os
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from game_env import TheGameEnv

SPARSE_REWARDS = dict(
    reward_per_card=0.0,
    win_reward=100.0,
    loss_penalty=0.5,
    trick_play_reward=0.0,
    distance_penalty_scale=0.0,
)

SHAPED_REWARDS = dict(
    reward_per_card=0.02,
    win_reward=100.0,
    loss_penalty=0.5,
    trick_play_reward=1.0,
    distance_penalty_scale=0.003,
)


def linear_schedule(initial_lr):
    """Linear learning rate decay schedule.

    Args:
        initial_lr: Initial learning rate at the start of training.

    Returns:
        Function that computes learning rate based on progress remaining.
    """

    def func(progress_remaining):
        return progress_remaining * initial_lr

    return func


class EntropyScheduleCallback(BaseCallback):
    """Callback to decay entropy coefficient during training.

    Decays linearly from start_ent to end_ent over the course of training.

    Args:
        start_ent: Initial entropy coefficient.
        end_ent: Final entropy coefficient.
    """

    def __init__(self, start_ent=0.05, end_ent=0.005, verbose=0):
        super().__init__(verbose)
        self.start_ent = start_ent
        self.end_ent = end_ent

    def _on_step(self):
        progress = self.num_timesteps / self.model._total_timesteps
        new_ent = self.start_ent - progress * (self.start_ent - self.end_ent)
        self.model.ent_coef = new_ent
        if self.verbose > 0 and self.num_timesteps % 100000 == 0:
            self.logger.record("train/ent_coef_scheduled", new_ent)
        return True


class GameMetricsCallback(BaseCallback):
    """Callback to log custom game metrics to TensorBoard."""

    def __init__(self, verbose=0, window_size=100):
        super().__init__(verbose)
        self.window_size = window_size
        self.episode_victories = []
        self.episode_cards_played = []
        self.episode_cards_per_turn = []
        self.episode_avg_distance = []

    def _on_step(self):
        for idx, done in enumerate(self.locals.get("dones", [])):
            if done:
                info = self.locals.get("infos", [{}])[idx]
                if "victory" in info:
                    self.episode_victories.append(1 if info["victory"] else 0)
                if "total_cards_played" in info:
                    self.episode_cards_played.append(info["total_cards_played"])
                if "avg_cards_per_turn" in info:
                    self.episode_cards_per_turn.append(info["avg_cards_per_turn"])
                if "avg_distance" in info:
                    self.episode_avg_distance.append(info["avg_distance"])

        if len(self.episode_victories) >= self.window_size:
            recent_victories = self.episode_victories[-self.window_size :]
            win_rate = np.mean(recent_victories)
            self.logger.record("game/win_rate", win_rate)

        if len(self.episode_cards_played) >= self.window_size:
            recent = self.episode_cards_played[-self.window_size :]
            self.logger.record("game/avg_cards_played", np.mean(recent))

        if len(self.episode_cards_per_turn) >= self.window_size:
            recent = self.episode_cards_per_turn[-self.window_size :]
            self.logger.record("game/avg_cards_per_turn", np.mean(recent))

        if len(self.episode_avg_distance) >= self.window_size:
            recent = self.episode_avg_distance[-self.window_size :]
            self.logger.record("game/avg_distance", np.mean(recent))

        return True


def mask_fn(env):
    """Return valid action mask for MaskablePPO.

    Args:
        env: TheGameEnv instance (possibly wrapped by Monitor).

    Returns:
        Boolean array indicating valid actions.
    """
    return env.unwrapped.action_masks()


def make_env(n_players, reward_config, log_dir=None, env_idx=0):
    """Create a factory function for environment creation.

    Args:
        n_players: Number of players in the game.
        reward_config: Dict with reward parameters.
        log_dir: Directory for Monitor logs (optional).
        env_idx: Environment index for unique log filenames.

    Returns:
        Factory function that creates a wrapped environment.
    """

    def _init():
        env = TheGameEnv(n_players=n_players, **reward_config)
        if log_dir is not None:
            env = Monitor(env, filename=f"{log_dir}/env_{env_idx}")
        return ActionMasker(env, mask_fn)

    return _init


def create_env(
    n_players=5, reward_config=None, n_envs=1, use_subproc=True, log_dir=None
):
    """Create vectorized environment for MaskablePPO training.

    Args:
        n_players: Number of players in the game.
        reward_config: Dict with reward parameters. Defaults to SPARSE_REWARDS.
        n_envs: Number of parallel environments.
        use_subproc: Use SubprocVecEnv (True) or DummyVecEnv (False).
        log_dir: Directory for Monitor logs.

    Returns:
        Vectorized environment with action masking.
    """
    if reward_config is None:
        reward_config = SPARSE_REWARDS

    env_fns = [make_env(n_players, reward_config, log_dir, i) for i in range(n_envs)]

    if n_envs == 1:
        return DummyVecEnv(env_fns)
    elif use_subproc:
        return SubprocVecEnv(env_fns)
    else:
        return DummyVecEnv(env_fns)


def train(
    variant="sparse",
    total_timesteps=100_000_000,
    n_players=5,
    n_envs=None,
    verbose=1,
):
    """Train MaskablePPO agent on The Game environment.

    Args:
        variant: "sparse" or "shaped" reward configuration.
        total_timesteps: Total training steps.
        n_players: Number of players in the game.
        n_envs: Number of parallel environments. Defaults to CPU count.
        verbose: Verbosity level for training output.

    Returns:
        Trained MaskablePPO model.
    """
    if n_envs is None:
        n_envs = os.cpu_count() or 1

    reward_config = SPARSE_REWARDS if variant == "sparse" else SHAPED_REWARDS

    bld_dir = Path(__file__).parent.parent / "bld"
    bld_dir.mkdir(exist_ok=True)

    log_path = str(bld_dir / f"rl_logs_{variant}")
    monitor_dir = str(bld_dir / f"monitor_{variant}")
    Path(monitor_dir).mkdir(exist_ok=True)

    env = create_env(
        n_players=n_players,
        reward_config=reward_config,
        n_envs=n_envs,
        log_dir=monitor_dir,
    )

    if verbose:
        print(f"Training {variant} variant with {n_envs} parallel environments")
        print(f"Total timesteps: {total_timesteps:,}")

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

    checkpoint_dir = bld_dir / f"rl_checkpoints_{variant}"
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000_000 // n_envs,
        save_path=str(checkpoint_dir),
        name_prefix=f"{variant}",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    callbacks = [
        GameMetricsCallback(verbose=verbose),
        EntropyScheduleCallback(start_ent=0.05, end_ent=0.005, verbose=verbose),
        checkpoint_callback,
    ]
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    final_path = bld_dir / f"{variant}_100M_final.zip"
    model.save(final_path)
    if verbose:
        print(f"Saved final model: {final_path}")

    return model


def main():
    """Entry point for training script. Trains both sparse and shaped variants."""
    import argparse

    parser = argparse.ArgumentParser(description="Train RL agent for The Game")
    parser.add_argument(
        "variant",
        choices=["sparse", "shaped", "both"],
        help="Which reward variant to train",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000_000,
        help="Total training timesteps (default: 100M)",
    )
    args = parser.parse_args()

    if args.variant == "both":
        for v in ["sparse", "shaped"]:
            print(f"\n{'=' * 60}")
            print(f"Training {v} variant")
            print(f"{'=' * 60}\n")
            train(variant=v, total_timesteps=args.timesteps)
    else:
        train(variant=args.variant, total_timesteps=args.timesteps)


if __name__ == "__main__":
    main()
