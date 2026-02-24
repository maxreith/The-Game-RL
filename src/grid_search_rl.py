"""Grid search over PPO hyperparameters for The Game."""

import itertools
import os
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from game_env import TheGameEnv


def mask_fn(env):
    """Return valid action mask for MaskablePPO."""
    return env.unwrapped.action_masks()


def make_env(n_players=3):
    """Create environment factory."""

    def _init():
        env = TheGameEnv(n_players=n_players)
        env = Monitor(env)
        return ActionMasker(env, mask_fn)

    return _init


def create_env(n_players=3, n_envs=1):
    """Create vectorized environment."""
    env_fns = [make_env(n_players) for _ in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns)


def evaluate(model, n_games=500, n_players=3):
    """Evaluate model and return metrics."""
    env = TheGameEnv(n_players=n_players)
    victories = 0
    total_cards = []
    total_distance = []

    for game_idx in range(n_games):
        obs, _ = env.reset(seed=game_idx)
        terminated = False

        while not terminated:
            action_mask = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=action_mask)
            obs, _, terminated, _, info = env.step(action)

        if info.get("victory", False):
            victories += 1
        total_cards.append(env.total_cards_played)
        if env.total_cards_played > 0:
            total_distance.append(env.total_distance / env.total_cards_played)

    return {
        "win_rate": victories / n_games,
        "avg_cards": np.mean(total_cards),
        "avg_distance": np.mean(total_distance) if total_distance else 0,
    }


def run_grid_search(
    ent_coefs=(0.01, 0.05, 0.1),
    learning_rates=(1e-4, 3e-4, 1e-3),
    gamma=1.0,
    total_timesteps=500_000,
    n_envs=None,
):
    """Run grid search over hyperparameters.

    Args:
        ent_coefs: Entropy coefficient values to test.
        learning_rates: Learning rate values to test.
        gamma: Discount factor (fixed).
        total_timesteps: Training steps per run.
        n_envs: Parallel environments.

    Returns:
        List of result dicts.
    """
    if n_envs is None:
        n_envs = os.cpu_count() or 1

    bld_dir = Path(__file__).parent.parent / "bld"
    bld_dir.mkdir(exist_ok=True)

    results = []
    configs = list(itertools.product(ent_coefs, learning_rates))
    total_runs = len(configs)

    print(f"Grid search: {total_runs} configurations")
    print(f"Fixed: gamma={gamma}, timesteps={total_timesteps}, n_envs={n_envs}")
    print("=" * 70)

    for idx, (ent_coef, lr) in enumerate(configs, 1):
        print(f"\n[{idx}/{total_runs}] ent_coef={ent_coef}, lr={lr}")

        env = create_env(n_envs=n_envs)

        model = MaskablePPO(
            "MlpPolicy",
            env,
            ent_coef=ent_coef,
            learning_rate=lr,
            gamma=gamma,
            verbose=0,
        )

        model.learn(total_timesteps=total_timesteps)

        metrics = evaluate(model, n_games=500)

        result = {
            "ent_coef": ent_coef,
            "learning_rate": lr,
            "gamma": gamma,
            **metrics,
        }
        results.append(result)

        print(f"  Win rate: {metrics['win_rate'] * 100:.1f}%")
        print(f"  Avg cards: {metrics['avg_cards']:.1f}")
        print(f"  Avg distance: {metrics['avg_distance']:.1f}")

        model_name = f"ppo_ent{ent_coef}_lr{lr}"
        model.save(bld_dir / model_name)

        env.close()

    return results


def print_results_table(results):
    """Print results as a formatted table."""
    print("\n" + "=" * 70)
    print("GRID SEARCH RESULTS")
    print("=" * 70)
    print(
        f"{'ent_coef':<10} {'lr':<10} {'win_rate':<10} {'avg_cards':<12} {'avg_dist':<10}"
    )
    print("-" * 70)

    sorted_results = sorted(results, key=lambda x: (-x["win_rate"], -x["avg_cards"]))

    for r in sorted_results:
        print(
            f"{r['ent_coef']:<10} {r['learning_rate']:<10.0e} "
            f"{r['win_rate'] * 100:<10.1f} {r['avg_cards']:<12.1f} "
            f"{r['avg_distance']:<10.1f}"
        )

    best = sorted_results[0]
    print("\n" + "=" * 70)
    print(f"BEST: ent_coef={best['ent_coef']}, lr={best['learning_rate']}")
    print(
        f"      win_rate={best['win_rate'] * 100:.1f}%, avg_cards={best['avg_cards']:.1f}"
    )


def main():
    """Run grid search with default parameters."""
    results = run_grid_search(
        ent_coefs=(0.01, 0.05, 0.1),
        learning_rates=(1e-4, 3e-4, 1e-3),
        gamma=1.0,
        total_timesteps=500_000,
    )
    print_results_table(results)


if __name__ == "__main__":
    main()
