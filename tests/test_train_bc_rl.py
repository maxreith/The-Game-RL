"""Unit tests for behavioral cloning and BC+RL training components."""

import numpy as np
import pytest
import torch

from game_env import TheGameEnv
from generate_expert_data import get_expert_action, generate_expert_demonstrations
from train_bc_rl import (
    BCPolicyNetwork,
    train_behavioral_cloning,
    load_bc_weights_into_ppo,
    evaluate_bc_policy,
)


class TestGetExpertAction:
    """Tests for get_expert_action function."""

    def test_returns_valid_action(self):
        env = TheGameEnv(n_players=3)
        env.reset(seed=42)
        action = get_expert_action(env, bonus_play_threshold=4)
        mask = env.action_masks()
        assert mask[action], f"Expert action {action} is not valid"

    def test_returns_end_turn_when_minimum_met_and_high_distance(self):
        env = TheGameEnv(n_players=3)
        env.reset(seed=42)
        env.cards_played_this_turn = 2
        end_turn_action = env.hand_size * 4
        action = get_expert_action(env, bonus_play_threshold=0)
        assert action == end_turn_action or env.action_masks()[action]

    def test_returns_card_action_when_minimum_not_met(self):
        env = TheGameEnv(n_players=3)
        env.reset(seed=42)
        env.cards_played_this_turn = 0
        action = get_expert_action(env, bonus_play_threshold=4)
        end_turn_action = env.hand_size * 4
        if env.action_masks()[action]:
            assert action != end_turn_action or not np.any(env.action_masks()[:-1])


class TestGenerateExpertDemonstrations:
    """Tests for expert demonstration generation."""

    def test_generates_correct_shapes(self):
        obs, actions, masks = generate_expert_demonstrations(
            n_games=10, n_players=3, seed=42, verbose=False
        )
        assert obs.ndim == 2
        assert actions.ndim == 1
        assert masks.ndim == 2
        assert len(obs) == len(actions) == len(masks)
        assert obs.dtype == np.float32
        assert actions.dtype == np.int64
        assert masks.dtype == bool

    def test_observations_normalized(self):
        obs, _, _ = generate_expert_demonstrations(
            n_games=5, n_players=3, seed=42, verbose=False
        )
        assert np.all(obs >= 0)
        assert np.all(obs <= 1)

    def test_actions_match_masks(self):
        obs, actions, masks = generate_expert_demonstrations(
            n_games=5, n_players=3, seed=42, verbose=False
        )
        for i in range(len(actions)):
            assert masks[i, actions[i]], f"Action {actions[i]} not valid at step {i}"

    def test_deterministic_with_seed(self):
        obs1, act1, _ = generate_expert_demonstrations(
            n_games=5, n_players=3, seed=123, verbose=False
        )
        obs2, act2, _ = generate_expert_demonstrations(
            n_games=5, n_players=3, seed=123, verbose=False
        )
        np.testing.assert_array_equal(obs1, obs2)
        np.testing.assert_array_equal(act1, act2)


class TestBCPolicyNetwork:
    """Tests for BCPolicyNetwork architecture."""

    def test_forward_shape(self):
        obs_dim, action_dim = 23, 25
        model = BCPolicyNetwork(obs_dim, action_dim)
        batch = torch.randn(32, obs_dim)
        logits = model(batch)
        assert logits.shape == (32, action_dim)

    def test_forward_with_mask(self):
        obs_dim, action_dim = 23, 25
        model = BCPolicyNetwork(obs_dim, action_dim)
        batch = torch.randn(32, obs_dim)
        mask = torch.zeros(32, action_dim, dtype=torch.bool)
        mask[:, 0] = True
        logits = model(batch, mask)
        assert logits.shape == (32, action_dim)
        assert torch.all(logits.argmax(dim=1) == 0)

    def test_architecture_matches_sb3(self):
        obs_dim, action_dim = 23, 25
        model = BCPolicyNetwork(obs_dim, action_dim)
        assert isinstance(model.policy_net[0], torch.nn.Linear)
        assert isinstance(model.policy_net[1], torch.nn.Tanh)
        assert isinstance(model.policy_net[2], torch.nn.Linear)
        assert isinstance(model.policy_net[3], torch.nn.Tanh)
        assert model.policy_net[0].in_features == obs_dim
        assert model.policy_net[0].out_features == 256
        assert model.policy_net[2].in_features == 256
        assert model.policy_net[2].out_features == 256
        assert model.action_net.in_features == 256
        assert model.action_net.out_features == action_dim


class TestTrainBehavioralCloning:
    """Tests for BC training function."""

    @pytest.fixture
    def small_dataset(self):
        np.random.seed(42)
        n_samples = 500
        obs_dim, action_dim = 23, 25
        observations = np.random.rand(n_samples, obs_dim).astype(np.float32)
        action_masks = np.ones((n_samples, action_dim), dtype=bool)
        actions = np.random.randint(0, action_dim, n_samples).astype(np.int64)
        return observations, actions, action_masks

    def test_returns_trained_model(self, small_dataset):
        obs, actions, masks = small_dataset
        model = train_behavioral_cloning(
            obs, actions, masks, epochs=5, batch_size=64, patience=3, verbose=False
        )
        assert isinstance(model, BCPolicyNetwork)

    def test_model_on_cpu(self, small_dataset):
        obs, actions, masks = small_dataset
        model = train_behavioral_cloning(
            obs, actions, masks, epochs=2, batch_size=64, verbose=False
        )
        assert next(model.parameters()).device == torch.device("cpu")

    def test_early_stopping(self, small_dataset):
        obs, actions, masks = small_dataset
        model = train_behavioral_cloning(
            obs, actions, masks, epochs=1000, batch_size=64, patience=2, verbose=False
        )
        assert isinstance(model, BCPolicyNetwork)


class TestWeightTransfer:
    """Tests for BC to PPO weight transfer."""

    def test_weight_transfer_produces_same_logits(self):
        pytest.importorskip("sb3_contrib")
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.vec_env import DummyVecEnv

        env = TheGameEnv(n_players=3)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        bc_model = BCPolicyNetwork(obs_dim, action_dim)

        def mask_fn(e):
            return e.unwrapped.action_masks()

        vec_env = DummyVecEnv([lambda: ActionMasker(TheGameEnv(n_players=3), mask_fn)])

        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=torch.nn.Tanh,
        )
        ppo = MaskablePPO("MlpPolicy", vec_env, policy_kwargs=policy_kwargs)

        load_bc_weights_into_ppo(bc_model, ppo)

        test_obs = torch.randn(1, obs_dim)

        bc_model.eval()
        with torch.no_grad():
            bc_features = bc_model.policy_net(test_obs)
            bc_logits = bc_model.action_net(bc_features)

        ppo.policy.eval()
        with torch.no_grad():
            ppo_features, _ = ppo.policy.mlp_extractor(test_obs)
            ppo_logits = ppo.policy.action_net(ppo_features)

        torch.testing.assert_close(bc_logits, ppo_logits, rtol=1e-5, atol=1e-5)


class TestEvaluateBCPolicy:
    """Tests for BC policy evaluation."""

    def test_returns_metrics(self):
        env = TheGameEnv(n_players=3)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        model = BCPolicyNetwork(obs_dim, action_dim)
        results = evaluate_bc_policy(model, n_games=3, n_players=3, verbose=False)

        assert "win_rate" in results
        assert "avg_cards" in results
        assert 0 <= results["win_rate"] <= 1
        assert results["avg_cards"] >= 0
