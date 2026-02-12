"""
Quick test to verify the simulation works before running full simulations.
"""
from run_simulations import find_optimal_parameters, test_shuffle_qualities

# Quick test with small numbers
print("Testing with n_games=5 for speed...")
optimal = find_optimal_parameters(n_games=5)
print(f"\nOptimal parameters found: {optimal}")

print("\n" + "="*60)
print("Testing shuffle qualities with n_games=5...")
shuffle_df = test_shuffle_qualities(optimal, n_games=5)
print("\nShuffle test completed!")
