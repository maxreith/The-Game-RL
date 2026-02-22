"""Train RL agent to play The Game using MaskablePPO."""

from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from game_env import TheGameEnv


def mask_fn(env):
    """Return valid action mask for MaskablePPO.

    Args:
        env: TheGameEnv instance.

    Returns:
        Boolean array indicating valid actions.
    """
    return env.action_masks()


def create_env(n_players=3):
    """Create and wrap environment for MaskablePPO training.

    Args:
        n_players: Number of players in the game.

    Returns:
        Wrapped environment with action masking.
    """
    env = TheGameEnv(n_players=n_players)
    return ActionMasker(env, mask_fn)


def train(total_timesteps=500_000, n_players=3, verbose=1, tensorboard_log=True):
    """Train MaskablePPO agent on The Game environment.

    Args:
        total_timesteps: Total training steps.
        n_players: Number of players in the game.
        verbose: Verbosity level for training output.
        tensorboard_log: Whether to enable tensorboard logging.

    Returns:
        Trained MaskablePPO model.
    """
    bld_dir = Path(__file__).parent.parent / "bld"
    bld_dir.mkdir(exist_ok=True)

    env = create_env(n_players=n_players)

    log_path = str(bld_dir / "rl_logs") if tensorboard_log else None

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=verbose,
        tensorboard_log=log_path,
    )

    model.learn(total_timesteps=total_timesteps)
    model.save(bld_dir / "the_game_ppo")

    return model


def main():
    """Entry point for training script."""
    train()


if __name__ == "__main__":
    main()
