"""Simulate games with varying Gemini thinking levels."""

from functools import partial

import pandas as pd

from game_setup import run_game
from strategies import gemini_strategy

pd.options.future.infer_string = True

THINKING_LEVELS = ["minimal", "low", "medium"]
N_GAMES_PER_LEVEL = 1


def run_thinking_level_simulation(
    thinking_levels: list[str] = THINKING_LEVELS,
    n_games_per_level: int = N_GAMES_PER_LEVEL,
) -> pd.DataFrame:
    """Run games for each thinking level and collect results.

    Args:
        thinking_levels: List of thinking levels to test.
        n_games_per_level: Number of games to run per thinking level.

    Returns:
        DataFrame with columns: thinking_level, game_number, turns, victory, cards_remaining.
    """
    results = []

    for level in thinking_levels:
        strategy = partial(gemini_strategy, thinking_level=level)

        for game_num in range(n_games_per_level):
            print(
                f"Running game {game_num + 1}/{n_games_per_level} with thinking_level={level}"
            )
            result = run_game(strategy)
            results.append(
                {
                    "thinking_level": level,
                    "game_number": game_num + 1,
                    "turns": result["turns"],
                    "victory": result["victory"],
                    "cards_remaining": result["cards_remaining"],
                }
            )

    df = pd.DataFrame(results)
    return df


def main() -> None:
    """Run simulation and save results."""
    df = run_thinking_level_simulation()
    df.to_parquet("bld/gemini_thinking_results.parquet")
    print("Saved results to bld/gemini_thinking_results.parquet")
    print(df)


if __name__ == "__main__":
    main()
