"""
Simulation script to:
1. Find optimal parameters for bonus_play_strategy
2. Test win rates across different shuffle qualities
"""
import numpy as np
import pandas as pd
from itertools import product
from game_setup import run_simulation
from game_strategies import bonus_play_strategy


def find_optimal_parameters(n_games: int = 100):
    """
    Test different parameter combinations for bonus_play_strategy
    to find the combination with highest win rate.

    Returns:
        dict: Best parameters and their win rate
    """
    print("=" * 60)
    print("PHASE 1: Finding Optimal Parameters")
    print("=" * 60)

    # Define parameter grid
    n_players_options = [2, 3, 4]
    bonus_play_threshold_options = [1, 2, 3, 4, 5, 6, 7, 8]

    results = []

    # Test all combinations
    total_combinations = len(n_players_options) * len(bonus_play_threshold_options)
    current = 0

    for n_players, threshold in product(n_players_options, bonus_play_threshold_options):
        current += 1
        print(f"\n[{current}/{total_combinations}] Testing n_players={n_players}, threshold={threshold}...")

        result = run_simulation(
            strategy=bonus_play_strategy,
            n_games=n_games,
            n_players=n_players,
            bonus_play_threshold=threshold
        )

        results.append({
            'n_players': n_players,
            'bonus_play_threshold': threshold,
            'win_rate': result['win_rate'],
            'victories': len(result['victories']),
            'losses': len(result['losses'])
        })

        print(f"  Win rate: {result['win_rate']:.2%}")

    # Convert to DataFrame for easy analysis
    df = pd.DataFrame(results)

    # Find best parameters
    best_idx = df['win_rate'].idxmax()
    best_params = df.loc[best_idx]

    print("\n" + "=" * 60)
    print("OPTIMAL PARAMETERS FOUND:")
    print("=" * 60)
    print(f"n_players: {int(best_params['n_players'])}")
    print(f"bonus_play_threshold: {int(best_params['bonus_play_threshold'])}")
    print(f"Win rate: {best_params['win_rate']:.2%}")
    print()

    # Show top 5 combinations
    print("Top 5 parameter combinations:")
    print(df.nlargest(5, 'win_rate').to_string(index=False))
    print()

    return {
        'n_players': int(best_params['n_players']),
        'bonus_play_threshold': int(best_params['bonus_play_threshold']),
        'win_rate': best_params['win_rate'],
        'all_results': df
    }


def test_shuffle_qualities(optimal_params: dict, n_games: int = 100):
    """
    Test win rates across different shuffle qualities using optimal parameters.

    Args:
        optimal_params: Dictionary with 'n_players' and 'bonus_play_threshold'
        n_games: Number of games to simulate per shuffle quality

    Returns:
        pandas.DataFrame: Results for each shuffle quality
    """
    print("=" * 60)
    print("PHASE 2: Testing Shuffle Qualities")
    print("=" * 60)
    print(f"Using optimal parameters:")
    print(f"  n_players: {optimal_params['n_players']}")
    print(f"  bonus_play_threshold: {optimal_params['bonus_play_threshold']}")
    print()

    # Test different shuffle qualities
    shuffle_qualities = [1, 2, 5, 10, 20, 50, 100, 200]
    results = []

    for n_shuffles in shuffle_qualities:
        print(f"\nTesting n_shuffles={n_shuffles}...")

        result = run_simulation(
            strategy=bonus_play_strategy,
            n_games=n_games,
            n_players=optimal_params['n_players'],
            bonus_play_threshold=optimal_params['bonus_play_threshold'],
            n_shuffles=n_shuffles,
            use_custom_shuffle=True
        )

        results.append({
            'n_shuffles': n_shuffles,
            'shuffle_quality': get_shuffle_description(n_shuffles),
            'victories': len(result['victories']),
            'losses': len(result['losses']),
            'win_rate': result['win_rate']
        })

        print(f"  Win rate: {result['win_rate']:.2%}")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("SHUFFLE QUALITY RESULTS:")
    print("=" * 60)
    print(df.to_string(index=False))
    print()

    # Calculate improvement from worst to best shuffle
    worst_win_rate = df['win_rate'].min()
    best_win_rate = df['win_rate'].max()
    improvement = ((best_win_rate - worst_win_rate) / worst_win_rate) * 100

    print(f"Win rate range: {worst_win_rate:.2%} to {best_win_rate:.2%}")
    print(f"Relative improvement: {improvement:.1f}%")
    print()

    return df


def get_shuffle_description(n_shuffles: int) -> str:
    """Return a human-readable description of shuffle quality."""
    if n_shuffles <= 2:
        return "Very Poor"
    elif n_shuffles <= 10:
        return "Poor"
    elif n_shuffles <= 50:
        return "Moderate"
    elif n_shuffles <= 100:
        return "Good"
    else:
        return "Excellent"


def main():
    """Run the complete simulation pipeline."""
    print("\n" + "=" * 60)
    print("GAME SIMULATION: Parameter Optimization & Shuffle Analysis")
    print("=" * 60)
    print()

    # Phase 1: Find optimal parameters (using more games for better statistics)
    n_games_optimization = 500
    optimal_params = find_optimal_parameters(n_games=n_games_optimization)

    # Phase 2: Test shuffle qualities (using even more games for accurate comparison)
    n_games_shuffle_test = 1000
    shuffle_results = test_shuffle_qualities(optimal_params, n_games=n_games_shuffle_test)

    # Save results
    print("=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    optimal_params['all_results'].to_csv('parameter_optimization_results.csv', index=False)
    print("Saved: parameter_optimization_results.csv")

    shuffle_results.to_csv('shuffle_quality_results.csv', index=False)
    print("Saved: shuffle_quality_results.csv")

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
