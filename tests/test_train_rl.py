"""Tests for RL training script."""

from train_rl import create_env, train


class TestCreateEnv:
    """Tests for environment creation."""

    def test_creates_wrapped_env(self):
        """Environment is wrapped with ActionMasker."""
        env = create_env(n_players=3)
        assert hasattr(env, "action_masks")
        env.close()

    def test_env_reset_works(self):
        """Wrapped environment resets successfully."""
        env = create_env(n_players=3)
        obs, info = env.reset()
        assert obs is not None
        assert "action_mask" in info
        env.close()


class TestTrain:
    """Tests for training function."""

    def test_short_training_runs(self):
        """Short training completes without errors."""
        model = train(
            total_timesteps=100, n_players=3, verbose=0, tensorboard_log=False
        )
        assert model is not None

    def test_model_can_predict(self):
        """Trained model can make predictions."""
        model = train(
            total_timesteps=100, n_players=3, verbose=0, tensorboard_log=False
        )
        env = create_env(n_players=3)
        obs, _ = env.reset()

        action, _ = model.predict(obs, action_masks=env.action_masks())
        assert 0 <= action < 24  # hand_size=6, stacks=4
        env.close()
