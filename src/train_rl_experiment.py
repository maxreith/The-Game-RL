"""Experimental RL training script for hyperparameter tuning."""

from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from game_env import TheGameEnv


class QuickEvalCallback(BaseCallback):
    """Callback to periodically evaluate win rate and cards played during training."""

    def __init__(self, eval_freq=50_000, n_eval_episodes=100, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_cards_played = 0.0
        self.best_win_rate = 0.0

    def _on_step(self):
        if self.num_timesteps % self.eval_freq == 0:
            win_rate, avg_cards = self._evaluate()
            self.logger.record("eval/win_rate", win_rate)
            self.logger.record("eval/avg_cards", avg_cards)
            if avg_cards > self.best_cards_played:
                self.best_cards_played = avg_cards
                if self.verbose:
                    print(
                        f"  Step {self.num_timesteps}: {avg_cards:.1f} cards, {win_rate * 100:.1f}% wins"
                    )
            if win_rate > self.best_win_rate:
                self.best_win_rate = win_rate
        return True

    def _evaluate(self):
        env = TheGameEnv(n_players=self.training_env.get_attr("n_players")[0])
        victories = 0
        total_cards = []
        for _ in range(self.n_eval_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                mask = env.action_masks()
                action, _ = self.model.predict(
                    obs, deterministic=True, action_masks=mask
                )
                obs, _, done, _, info = env.step(action)
            if info.get("victory", False):
                victories += 1
            total_cards.append(info.get("total_cards_played", 0))
        return victories / self.n_eval_episodes, np.mean(total_cards)


def mask_fn(env):
    return env.unwrapped.action_masks()


def make_env(
    n_players,
    reward_per_card,
    win_reward,
    loss_penalty,
    trick_play_reward,
    distance_penalty_scale,
    progress_reward_scale,
):
    def _init():
        env = TheGameEnv(
            n_players=n_players,
            reward_per_card=reward_per_card,
            win_reward=win_reward,
            loss_penalty=loss_penalty,
            trick_play_reward=trick_play_reward,
            distance_penalty_scale=distance_penalty_scale,
            progress_reward_scale=progress_reward_scale,
        )
        return ActionMasker(env, mask_fn)

    return _init


def train_experiment(
    name="experiment",
    total_timesteps=500_000,
    n_players=3,
    n_envs=8,
    # Reward parameters
    reward_per_card=0.01,
    win_reward=1.0,
    loss_penalty=0.5,
    trick_play_reward=0.1,
    distance_penalty_scale=0.001,
    progress_reward_scale=0.0,
    # PPO parameters
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    clip_range=0.2,
    # Network architecture
    net_arch_pi=(256, 256),
    net_arch_vf=(256, 256),
    verbose=1,
):
    """Train with configurable hyperparameters."""
    bld_dir = Path(__file__).parent.parent / "bld"
    bld_dir.mkdir(exist_ok=True)

    env_fns = [
        make_env(
            n_players,
            reward_per_card,
            win_reward,
            loss_penalty,
            trick_play_reward,
            distance_penalty_scale,
            progress_reward_scale,
        )
        for _ in range(n_envs)
    ]
    env = SubprocVecEnv(env_fns) if n_envs > 1 else DummyVecEnv(env_fns)

    policy_kwargs = dict(
        net_arch=dict(pi=list(net_arch_pi), vf=list(net_arch_vf)),
        activation_fn=torch.nn.Tanh,
    )

    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        tensorboard_log=str(bld_dir / "rl_logs"),
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        clip_range=clip_range,
    )

    callback = QuickEvalCallback(
        eval_freq=50_000,
        n_eval_episodes=100,
        verbose=verbose,
    )

    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT: {name}")
    print(f"{'=' * 60}")
    print(f"Players: {n_players}, Envs: {n_envs}")
    print(f"LR: {learning_rate}, Gamma: {gamma}, Entropy: {ent_coef}")
    print(f"Reward/card: {reward_per_card}, Win: {win_reward}, Loss: {loss_penalty}")
    print(f"Net arch: pi={net_arch_pi}, vf={net_arch_vf}")
    print(f"{'=' * 60}\n")

    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(bld_dir / f"ppo_{name}")

    print(f"\nFinal best win rate: {callback.best_win_rate * 100:.1f}%")
    return model, callback.best_win_rate


def evaluate_model(model, n_games=500, n_players=3):
    """Evaluate a trained model."""
    env = TheGameEnv(n_players=n_players)
    victories = 0
    cards_played = []

    for _ in range(n_games):
        obs, _ = env.reset()
        done = False
        while not done:
            mask = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            obs, _, done, _, info = env.step(action)
        if info.get("victory", False):
            victories += 1
        cards_played.append(env.total_cards_played)

    return {
        "win_rate": victories / n_games,
        "avg_cards": np.mean(cards_played),
    }


if __name__ == "__main__":
    # FINAL BEST CONFIGURATION
    # Achieved: 83.9 cards played, 1% win rate (vs 4% baseline)
    model, wr = train_experiment(
        name="final_best_v1",
        total_timesteps=2_000_000,
        n_players=5,
        n_envs=8,
        # Optimized reward structure
        reward_per_card=0.02,
        win_reward=10.0,
        loss_penalty=0.0,
        trick_play_reward=1.0,
        distance_penalty_scale=0.003,
        progress_reward_scale=3.0,
        # Optimized PPO hyperparameters
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.02,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        clip_range=0.2,
        # Network architecture
        net_arch_pi=(256, 256),
        net_arch_vf=(256, 256),
    )

    print("\n" + "=" * 60)
    print("FINAL EVALUATION (500 games, 5 players)")
    print("=" * 60)
    results = evaluate_model(model, n_games=500, n_players=5)
    print(f"Win rate: {results['win_rate'] * 100:.1f}%")
    print(f"Avg cards played: {results['avg_cards']:.1f}")
