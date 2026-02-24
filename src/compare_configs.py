"""Compare different RL configurations to measure impact of each change."""

import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch

from game_env import TheGameEnv


def mask_fn(env):
    return env.unwrapped.action_masks()


def make_env(n_players, **kwargs):
    def _init():
        env = TheGameEnv(n_players=n_players, **kwargs)
        return ActionMasker(env, mask_fn)

    return _init


def train_config(config, timesteps=500_000, n_envs=8):
    """Train a configuration and return average cards played."""
    env_kwargs = {
        "reward_per_card": config.get("reward_per_card", 0.01),
        "win_reward": config.get("win_reward", 1.0),
        "loss_penalty": config.get("loss_penalty", 0.5),
        "trick_play_reward": config.get("trick_play_reward", 0.1),
        "distance_penalty_scale": config.get("distance_penalty_scale", 0.001),
        "progress_reward_scale": config.get("progress_reward_scale", 0.0),
    }
    n_players = config.get("n_players", 3)

    env_fns = [make_env(n_players, **env_kwargs) for _ in range(n_envs)]
    env = SubprocVecEnv(env_fns)

    policy_kwargs = dict(
        net_arch=dict(
            pi=list(config.get("net_arch", [128, 128])),
            vf=list(config.get("net_arch", [128, 128])),
        ),
        activation_fn=torch.nn.Tanh,
    )

    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        learning_rate=config.get("learning_rate", 3e-4),
        gamma=config.get("gamma", 0.99),
        gae_lambda=config.get("gae_lambda", 0.95),
        ent_coef=config.get("ent_coef", 0.01),
        n_steps=config.get("n_steps", 2048),
        batch_size=config.get("batch_size", 64),
        n_epochs=config.get("n_epochs", 10),
        clip_range=config.get("clip_range", 0.2),
    )

    model.learn(total_timesteps=timesteps)
    env.close()

    # Evaluate
    eval_env = TheGameEnv(n_players=n_players, **env_kwargs)
    cards_played = []
    wins = 0
    for _ in range(200):
        obs, _ = eval_env.reset()
        done = False
        while not done:
            mask = eval_env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            obs, _, done, _, info = eval_env.step(action)
        cards_played.append(eval_env.total_cards_played)
        if info.get("victory", False):
            wins += 1

    return np.mean(cards_played), wins / 200


def random_baseline(n_players=3, n_games=200):
    """Evaluate random policy."""
    env = TheGameEnv(n_players=n_players)
    cards_played = []
    for _ in range(n_games):
        obs, _ = env.reset()
        done = False
        while not done:
            mask = env.action_masks()
            valid = np.where(mask)[0]
            action = np.random.choice(valid)
            obs, _, done, _, info = env.step(action)
        cards_played.append(env.total_cards_played)
    return np.mean(cards_played), 0.0


def main():
    """Run comparison of configurations."""
    print("=" * 70)
    print("CONFIGURATION COMPARISON - Impact of Each Change")
    print("=" * 70)
    print("\n[Training for 500k steps each, evaluating on 200 games]\n")

    # Baseline: random policy
    print("Testing random policy baseline...")
    rand_cards_3p, _ = random_baseline(n_players=3)
    rand_cards_5p, _ = random_baseline(n_players=5)
    print(f"  Random (3 players): {rand_cards_3p:.1f} cards")
    print(f"  Random (5 players): {rand_cards_5p:.1f} cards")

    configs = [
        {
            "name": "Original (3p, default rewards)",
            "n_players": 3,
            "reward_per_card": 0.01,
            "win_reward": 1.0,
            "loss_penalty": 0.5,
            "trick_play_reward": 0.1,
            "distance_penalty_scale": 0.001,
            "progress_reward_scale": 0.0,
            "gamma": 1.0,
            "ent_coef": 0.05,
            "net_arch": [128, 128],
        },
        {
            "name": "+ Stronger rewards",
            "n_players": 3,
            "reward_per_card": 0.02,
            "win_reward": 5.0,
            "loss_penalty": 0.0,
            "trick_play_reward": 0.5,
            "distance_penalty_scale": 0.003,
            "progress_reward_scale": 0.0,
            "gamma": 0.99,
            "ent_coef": 0.02,
            "net_arch": [256, 256],
        },
        {
            "name": "+ Progress reward",
            "n_players": 3,
            "reward_per_card": 0.02,
            "win_reward": 5.0,
            "loss_penalty": 0.0,
            "trick_play_reward": 0.5,
            "distance_penalty_scale": 0.003,
            "progress_reward_scale": 3.0,
            "gamma": 0.99,
            "ent_coef": 0.02,
            "net_arch": [256, 256],
        },
        {
            "name": "+ 5 players (easier game)",
            "n_players": 5,
            "reward_per_card": 0.02,
            "win_reward": 10.0,
            "loss_penalty": 0.0,
            "trick_play_reward": 1.0,
            "distance_penalty_scale": 0.003,
            "progress_reward_scale": 3.0,
            "gamma": 0.99,
            "ent_coef": 0.02,
            "net_arch": [256, 256],
        },
    ]

    results = []
    for config in configs:
        print(f"\nTraining: {config['name']}...")
        cards, wins = train_config(config, timesteps=500_000)
        results.append(
            {
                "name": config["name"],
                "cards": cards,
                "wins": wins,
                "n_players": config["n_players"],
            }
        )
        print(f"  Result: {cards:.1f} cards, {wins * 100:.1f}% wins")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Configuration':<35} {'Cards':>10} {'Wins':>10} {'Players':>10}")
    print("-" * 70)
    print(f"{'Random (3p)':<35} {rand_cards_3p:>10.1f} {'0.0%':>10} {'3':>10}")
    print(f"{'Random (5p)':<35} {rand_cards_5p:>10.1f} {'0.0%':>10} {'5':>10}")
    for r in results:
        print(
            f"{r['name']:<35} {r['cards']:>10.1f} {r['wins'] * 100:>9.1f}% {r['n_players']:>10}"
        )

    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    if len(results) >= 4:
        baseline = rand_cards_3p
        print(
            f"\n1. Original vs Random:        +{results[0]['cards'] - baseline:.1f} cards"
        )
        print(
            f"2. Stronger rewards:          +{results[1]['cards'] - results[0]['cards']:.1f} cards"
        )
        print(
            f"3. Progress reward:           +{results[2]['cards'] - results[1]['cards']:.1f} cards"
        )
        print(
            f"4. 5 players (easier):        +{results[3]['cards'] - rand_cards_5p:.1f} cards (vs 5p random)"
        )


if __name__ == "__main__":
    main()
