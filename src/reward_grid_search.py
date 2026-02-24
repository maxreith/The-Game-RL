"""Grid search over reward configurations for RL agent training."""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from game_env import TheGameEnv


@dataclass
class RewardConfig:
    """Configuration for reward parameters."""

    name: str
    reward_per_card: float
    win_reward: float
    loss_penalty: float
    trick_play_reward: float
    distance_penalty_scale: float


REWARD_CONFIGS = [
    RewardConfig(
        name="simple_baseline",
        reward_per_card=0.05,
        win_reward=10.0,
        loss_penalty=0.0,
        trick_play_reward=0.0,
        distance_penalty_scale=0.0,
    ),
    RewardConfig(
        name="with_trick_bonus",
        reward_per_card=0.05,
        win_reward=10.0,
        loss_penalty=0.0,
        trick_play_reward=1.0,
        distance_penalty_scale=0.0,
    ),
    RewardConfig(
        name="trick_and_distance",
        reward_per_card=0.05,
        win_reward=10.0,
        loss_penalty=0.0,
        trick_play_reward=1.0,
        distance_penalty_scale=0.005,
    ),
    RewardConfig(
        name="low_card_reward",
        reward_per_card=0.01,
        win_reward=10.0,
        loss_penalty=0.0,
        trick_play_reward=1.0,
        distance_penalty_scale=0.005,
    ),
    RewardConfig(
        name="high_card_reward",
        reward_per_card=0.1,
        win_reward=10.0,
        loss_penalty=0.0,
        trick_play_reward=1.0,
        distance_penalty_scale=0.005,
    ),
    RewardConfig(
        name="very_high_card_reward",
        reward_per_card=0.2,
        win_reward=10.0,
        loss_penalty=0.0,
        trick_play_reward=1.0,
        distance_penalty_scale=0.005,
    ),
    RewardConfig(
        name="high_win_bonus",
        reward_per_card=0.05,
        win_reward=50.0,
        loss_penalty=0.0,
        trick_play_reward=1.0,
        distance_penalty_scale=0.005,
    ),
    RewardConfig(
        name="with_loss_penalty",
        reward_per_card=0.05,
        win_reward=10.0,
        loss_penalty=1.0,
        trick_play_reward=1.0,
        distance_penalty_scale=0.005,
    ),
    RewardConfig(
        name="high_both_terminal",
        reward_per_card=0.05,
        win_reward=50.0,
        loss_penalty=2.0,
        trick_play_reward=1.0,
        distance_penalty_scale=0.005,
    ),
]


def mask_fn(env):
    """Return valid action mask for MaskablePPO."""
    return env.unwrapped.action_masks()


def make_env(config: RewardConfig, n_players: int = 5, log_dir=None, env_idx=0):
    """Create a factory function for environment creation with specific reward config.

    Args:
        config: Reward configuration.
        n_players: Number of players in the game.
        log_dir: Directory for Monitor logs (optional).
        env_idx: Environment index for unique log filenames.

    Returns:
        Factory function that creates a wrapped environment.
    """

    def _init():
        env = TheGameEnv(
            n_players=n_players,
            reward_per_card=config.reward_per_card,
            win_reward=config.win_reward,
            loss_penalty=config.loss_penalty,
            trick_play_reward=config.trick_play_reward,
            distance_penalty_scale=config.distance_penalty_scale,
            progress_reward_scale=0.0,
            stack_health_scale=0.0,
            phase_multiplier_scale=0.0,
        )
        if log_dir is not None:
            env = Monitor(env, filename=f"{log_dir}/env_{env_idx}")
        return ActionMasker(env, mask_fn)

    return _init


def create_env(config: RewardConfig, n_players: int = 5, n_envs: int = 1, log_dir=None):
    """Create vectorized environment for MaskablePPO training.

    Args:
        config: Reward configuration.
        n_players: Number of players in the game.
        n_envs: Number of parallel environments.
        log_dir: Directory for Monitor logs.

    Returns:
        Vectorized environment with action masking.
    """
    env_fns = [make_env(config, n_players, log_dir, i) for i in range(n_envs)]

    if n_envs == 1:
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns)


def linear_schedule(initial_lr):
    """Linear learning rate decay schedule."""

    def func(progress_remaining):
        return progress_remaining * initial_lr

    return func


def train_with_config(
    config: RewardConfig,
    total_timesteps: int = 300_000,
    n_players: int = 5,
    n_envs: int = None,
    verbose: int = 0,
):
    """Train a model with a specific reward configuration.

    Args:
        config: Reward configuration to use.
        total_timesteps: Total training steps.
        n_players: Number of players.
        n_envs: Number of parallel environments (defaults to CPU count).
        verbose: Verbosity level.

    Returns:
        Trained MaskablePPO model.
    """
    if n_envs is None:
        n_envs = os.cpu_count() or 1

    env = create_env(config, n_players=n_players, n_envs=n_envs)

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.Tanh,
    )

    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.05,
        learning_rate=linear_schedule(3e-4),
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        clip_range=0.2,
    )

    model.learn(total_timesteps=total_timesteps)
    env.close()

    return model


def evaluate_model(model, n_games: int = 500, n_players: int = 5, seed: int = 42):
    """Evaluate a trained model.

    Args:
        model: Trained MaskablePPO model.
        n_games: Number of games to play.
        n_players: Number of players.
        seed: Random seed.

    Returns:
        Dict with win_rate and avg_cards_played.
    """
    env = TheGameEnv(n_players=n_players)
    victories = 0
    total_cards_played = []

    for game_idx in range(n_games):
        game_seed = seed + game_idx
        obs, info = env.reset(seed=game_seed)
        terminated = False

        while not terminated:
            action_mask = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=action_mask)
            obs, reward, terminated, truncated, info = env.step(action)

        if info.get("victory", False):
            victories += 1
        total_cards_played.append(env.total_cards_played)

    return {
        "win_rate": victories / n_games,
        "avg_cards_played": np.mean(total_cards_played),
    }


def run_grid_search(
    configs: list,
    total_timesteps: int = 300_000,
    n_eval_games: int = 500,
    n_players: int = 5,
    n_envs: int = None,
    verbose: int = 0,
):
    """Run grid search over reward configurations.

    Args:
        configs: List of RewardConfig objects to test.
        total_timesteps: Training steps per configuration.
        n_eval_games: Number of evaluation games.
        n_players: Number of players.
        n_envs: Number of parallel environments.
        verbose: Verbosity level.

    Returns:
        List of results dicts sorted by win rate (descending).
    """
    results = []

    for i, config in enumerate(configs):
        print(f"\n{'=' * 60}")
        print(f"Config {i + 1}/{len(configs)}: {config.name}")
        print(f"{'=' * 60}")
        print(f"  reward_per_card: {config.reward_per_card}")
        print(f"  win_reward: {config.win_reward}")
        print(f"  loss_penalty: {config.loss_penalty}")
        print(f"  trick_play_reward: {config.trick_play_reward}")
        print(f"  distance_penalty_scale: {config.distance_penalty_scale}")

        print(f"\nTraining for {total_timesteps:,} steps...")
        model = train_with_config(
            config,
            total_timesteps=total_timesteps,
            n_players=n_players,
            n_envs=n_envs,
            verbose=verbose,
        )

        print(f"Evaluating on {n_eval_games} games...")
        eval_results = evaluate_model(model, n_games=n_eval_games, n_players=n_players)

        result = {
            "config_name": config.name,
            "win_rate": eval_results["win_rate"],
            "avg_cards_played": eval_results["avg_cards_played"],
            "reward_per_card": config.reward_per_card,
            "win_reward": config.win_reward,
            "loss_penalty": config.loss_penalty,
            "trick_play_reward": config.trick_play_reward,
            "distance_penalty_scale": config.distance_penalty_scale,
        }
        results.append(result)

        print(f"  Win rate: {eval_results['win_rate'] * 100:.1f}%")
        print(f"  Avg cards: {eval_results['avg_cards_played']:.1f}")

    results.sort(key=lambda x: x["win_rate"], reverse=True)
    return results


def print_results_table(results: list):
    """Print results as a formatted table.

    Args:
        results: List of result dicts from run_grid_search.
    """
    print("\n" + "=" * 80)
    print("GRID SEARCH RESULTS (sorted by win rate)")
    print("=" * 80)
    print(
        f"{'Config':<25} {'Win%':>6} {'Cards':>7} "
        f"{'card':>6} {'win':>6} {'loss':>6} {'trick':>6} {'dist':>6}"
    )
    print("-" * 80)

    for r in results:
        print(
            f"{r['config_name']:<25} "
            f"{r['win_rate'] * 100:>5.1f}% "
            f"{r['avg_cards_played']:>7.1f} "
            f"{r['reward_per_card']:>6.2f} "
            f"{r['win_reward']:>6.1f} "
            f"{r['loss_penalty']:>6.1f} "
            f"{r['trick_play_reward']:>6.1f} "
            f"{r['distance_penalty_scale']:>6.3f}"
        )

    print("=" * 80)


def save_best_model(results: list, n_players: int = 5, total_timesteps: int = 300_000):
    """Retrain and save the best configuration.

    Args:
        results: Results from grid search.
        n_players: Number of players.
        total_timesteps: Training steps.
    """
    best = results[0]
    print(f"\nBest configuration: {best['config_name']}")
    print(f"  Win rate: {best['win_rate'] * 100:.1f}%")

    best_config = RewardConfig(
        name=best["config_name"],
        reward_per_card=best["reward_per_card"],
        win_reward=best["win_reward"],
        loss_penalty=best["loss_penalty"],
        trick_play_reward=best["trick_play_reward"],
        distance_penalty_scale=best["distance_penalty_scale"],
    )

    print("\nRetraining best config for saving...")
    model = train_with_config(
        best_config, total_timesteps=total_timesteps, n_players=n_players, verbose=1
    )

    bld_dir = Path(__file__).parent.parent / "bld"
    bld_dir.mkdir(exist_ok=True)
    model_path = bld_dir / "the_game_ppo_tuned"
    model.save(model_path)
    print(f"Best model saved to {model_path}")


def main():
    """Entry point for reward grid search."""
    parser = argparse.ArgumentParser(
        description="Grid search over reward configurations"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick sanity test (100k steps, 1 config)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=300_000,
        help="Training steps per config (default: 300000)",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=500,
        help="Evaluation games per config (default: 500)",
    )
    parser.add_argument(
        "--n-players",
        type=int,
        default=5,
        help="Number of players (default: 5)",
    )
    parser.add_argument(
        "--save-best",
        action="store_true",
        help="Retrain and save the best model after grid search",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="Verbosity level (default: 0)",
    )
    args = parser.parse_args()

    if args.quick:
        print("Running quick sanity test...")
        configs = [REWARD_CONFIGS[2]]
        total_timesteps = 100_000
        n_eval_games = 100
    else:
        configs = REWARD_CONFIGS
        total_timesteps = args.timesteps
        n_eval_games = args.eval_games

    print(f"Grid search: {len(configs)} configurations")
    print(f"Training: {total_timesteps:,} steps per config")
    print(f"Evaluation: {n_eval_games} games per config")
    print(f"Players: {args.n_players}")

    results = run_grid_search(
        configs,
        total_timesteps=total_timesteps,
        n_eval_games=n_eval_games,
        n_players=args.n_players,
        verbose=args.verbose,
    )

    print_results_table(results)

    if args.save_best and not args.quick:
        save_best_model(
            results, n_players=args.n_players, total_timesteps=total_timesteps
        )


if __name__ == "__main__":
    main()
