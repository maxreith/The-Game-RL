"""Tests for reward grid search functionality."""

from reward_grid_search import (
    REWARD_CONFIGS,
    RewardConfig,
    create_env,
    evaluate_model,
    make_env,
    train_with_config,
)


class TestRewardConfig:
    """Tests for RewardConfig dataclass."""

    def test_config_has_all_fields(self):
        """RewardConfig has all required fields."""
        config = REWARD_CONFIGS[0]
        assert hasattr(config, "name")
        assert hasattr(config, "reward_per_card")
        assert hasattr(config, "win_reward")
        assert hasattr(config, "loss_penalty")
        assert hasattr(config, "trick_play_reward")
        assert hasattr(config, "distance_penalty_scale")

    def test_configs_have_unique_names(self):
        """All configurations have unique names."""
        names = [c.name for c in REWARD_CONFIGS]
        assert len(names) == len(set(names))

    def test_expected_number_of_configs(self):
        """Expected number of configurations defined."""
        assert len(REWARD_CONFIGS) == 9


class TestEnvironmentCreation:
    """Tests for environment factory functions."""

    def test_make_env_creates_callable(self):
        """make_env returns a callable factory."""
        config = REWARD_CONFIGS[0]
        factory = make_env(config, n_players=5)
        assert callable(factory)

    def test_make_env_creates_valid_env(self):
        """Factory creates a valid environment."""
        config = REWARD_CONFIGS[0]
        factory = make_env(config, n_players=5)
        env = factory()
        obs, info = env.reset(seed=42)
        assert obs is not None
        assert "action_mask" in info
        env.close()

    def test_create_env_single(self):
        """create_env with n_envs=1 returns DummyVecEnv."""
        config = REWARD_CONFIGS[0]
        vec_env = create_env(config, n_players=5, n_envs=1)
        assert vec_env.num_envs == 1
        vec_env.close()

    def test_create_env_multiple(self):
        """create_env with n_envs>1 returns SubprocVecEnv."""
        config = REWARD_CONFIGS[0]
        vec_env = create_env(config, n_players=5, n_envs=2)
        assert vec_env.num_envs == 2
        vec_env.close()

    def test_env_applies_reward_config(self):
        """Environment uses the specified reward configuration."""
        config = RewardConfig(
            name="test",
            reward_per_card=0.99,
            win_reward=100.0,
            loss_penalty=50.0,
            trick_play_reward=5.0,
            distance_penalty_scale=0.123,
        )
        factory = make_env(config, n_players=5)
        env = factory()
        inner = env.unwrapped
        assert inner.reward_per_card == 0.99
        assert inner.win_reward == 100.0
        assert inner.loss_penalty == 50.0
        assert inner.trick_play_reward == 5.0
        assert inner.distance_penalty_scale == 0.123
        assert inner.progress_reward_scale == 0.0
        assert inner.stack_health_scale == 0.0
        assert inner.phase_multiplier_scale == 0.0
        env.close()


class TestTraining:
    """Tests for training functionality."""

    def test_train_short_run(self):
        """Training runs without errors for minimal steps."""
        config = REWARD_CONFIGS[0]
        model = train_with_config(
            config,
            total_timesteps=1000,
            n_players=5,
            n_envs=1,
            verbose=0,
        )
        assert model is not None
        assert hasattr(model, "predict")

    def test_evaluate_model(self):
        """Model evaluation returns expected metrics."""
        config = REWARD_CONFIGS[0]
        model = train_with_config(
            config,
            total_timesteps=1000,
            n_players=5,
            n_envs=1,
            verbose=0,
        )
        results = evaluate_model(model, n_games=10, n_players=5, seed=42)
        assert "win_rate" in results
        assert "avg_cards_played" in results
        assert 0 <= results["win_rate"] <= 1
        assert results["avg_cards_played"] >= 0
