# Simulation Results Summary

## Executive Summary

This analysis reveals that **shuffle quality has a dramatic impact on game outcomes** - poorly shuffled decks have a 100% win rate while properly shuffled decks have only a 5.6% win rate. The optimal strategy parameters are **n_players=2** and **bonus_play_threshold=3** when using proper shuffling.

---

## Phase 1: Parameter Optimization

**Methodology**: Tested 24 parameter combinations with 500 games each (12,000 total games)

### Optimal Parameters Found

- **n_players**: 2
- **bonus_play_threshold**: 3
- **Win rate**: 6.2% (with standard shuffling)

### Top 5 Parameter Combinations

| Rank | n_players | bonus_play_threshold | Win Rate | Victories | Losses |
|------|-----------|----------------------|----------|-----------|--------|
| 1 | 2 | 3 | 6.2% | 31 | 469 |
| 2 | 2 | 2 | 5.4% | 27 | 473 |
| 3 | 4 | 2 | 5.0% | 25 | 475 |
| 4 | 4 | 3 | 4.4% | 22 | 478 |
| 5 | 4 | 5 | 4.4% | 22 | 478 |

### Key Insights

1. **2 players is optimal** for the bonus_play_strategy
2. **bonus_play_threshold=3** provides the best balance
3. All parameter combinations show relatively low win rates (1-6%) with proper shuffling
4. 3-player games consistently perform worst across all thresholds

---

## Phase 2: Shuffle Quality Impact

**Methodology**: Tested 8 shuffle qualities with 1,000 games each (8,000 total games) using optimal parameters (n_players=2, bonus_play_threshold=3)

### Complete Results

| n_shuffles | Shuffle Quality | Victories | Losses | Win Rate | Change |
|------------|----------------|-----------|---------|----------|---------|
| 1 | Very Poor | 1000 | 0 | 100.0% | baseline |
| 2 | Very Poor | 992 | 8 | 99.2% | -0.8% |
| 5 | Poor | 923 | 77 | 92.3% | -7.7% |
| 10 | Poor | 761 | 239 | 76.1% | -23.9% |
| 20 | Moderate | 398 | 602 | 39.8% | -60.2% |
| 50 | Moderate | 149 | 851 | 14.9% | -85.1% |
| 100 | Good | 78 | 922 | 7.8% | -92.2% |
| 200 | Excellent | 56 | 944 | 5.6% | -94.4% |

### Statistical Summary

- **Win rate range**: 5.6% to 100.0%
- **Relative improvement** (worst to best shuffle): **1685.7%**
- **Standard deviation**: ~41.3 percentage points
- **Median win rate**: 27.35%

### Visualization of Trend

```
Win Rate vs Shuffle Quality
100% |██████████████████████████████████████████████████ (n_shuffles=1)
 90% |█████████████████████████████████████████████████  (n_shuffles=2)
 80% |████████████████████████████████████████████       (n_shuffles=5)
 70% |██████████████████████████████████████             (n_shuffles=10)
 60% |
 50% |
 40% |████████████████████                               (n_shuffles=20)
 30% |
 20% |███████                                            (n_shuffles=50)
 10% |████                                               (n_shuffles=100)
  0% |███                                                (n_shuffles=200)
```

---

## Analysis & Interpretation

### Why Shuffle Quality Matters So Much

**Poor Shuffling (n_shuffles < 10):**
- Cards remain close to their original sequential order: 2, 3, 4, 5, 6, 7...
- Players can easily play cards in ascending order on the increasing piles
- The game becomes trivial - almost impossible to lose
- Win rate: 76-100%

**Moderate Shuffling (n_shuffles = 20-50):**
- Cards are partially randomized but some sequential patterns remain
- Game is moderately challenging
- Win rate: 15-40%

**Good Shuffling (n_shuffles ≥ 100):**
- Cards are fully randomized
- Sequential patterns are destroyed
- Game is genuinely challenging and strategic
- Win rate: 6-8% (approaching the optimal parameter win rate of 6.2%)

### The Critical Threshold

The **n_shuffles=20** mark represents a critical transition point:
- Below 20: Win rate > 39% (game is too easy)
- Above 20: Win rate rapidly approaches baseline (~6%)
- This suggests that even moderate shuffling significantly changes game dynamics

### Convergence Point

At **n_shuffles=100-200**, win rates converge to approximately the same value as the optimization phase (5.6-7.8% vs 6.2%), indicating that:
1. The deck is fully randomized at this point
2. Additional shuffles provide diminishing returns
3. The `_shuffle_cards()` standard function (which doesn't use n_shuffles) is equivalent to n_shuffles ≥ 100

---

## Recommendations

### For Competitive Play
- Use **n_shuffles ≥ 100** or the standard `_shuffle_cards()` function
- Ensures fair, challenging gameplay
- Expected win rate: ~6% with optimal strategy

### For Casual/Teaching Games
- Use **n_shuffles ≈ 20-50** for moderate difficulty
- Provides better chance of success while maintaining some challenge
- Expected win rate: 15-40%

### For Strategy Development & Testing
- **Always use n_shuffles ≥ 100** when evaluating strategies
- Testing with poor shuffling will give misleadingly optimistic results
- Current game code uses standard shuffle (equivalent to n_shuffles ≥ 100) ✓

### For Future Simulations
- The `run_simulation()` function now supports `use_custom_shuffle=True` and `n_shuffles` parameters
- This allows testing strategies under different shuffle conditions
- Example:
  ```python
  run_simulation(
      strategy=my_strategy,
      n_games=1000,
      n_players=2,
      n_shuffles=20,
      use_custom_shuffle=True
  )
  ```

---

## Files Generated

1. **`parameter_optimization_results.csv`**: All 24 parameter combinations tested
2. **`shuffle_quality_results.csv`**: Shuffle quality impact data
3. **`SIMULATION_README.md`**: Technical documentation
4. **`RESULTS_SUMMARY.md`**: This file

---

## Conclusion

This analysis definitively shows that **shuffle quality is the single most important factor** in determining game difficulty. The difference between a poorly shuffled deck (100% win rate) and a properly shuffled deck (5.6% win rate) is extraordinary - a **1685.7% relative difference**.

For meaningful strategy evaluation and competitive play, proper shuffling (n_shuffles ≥ 100) is essential. The current codebase correctly uses the standard shuffle function, ensuring realistic game simulations.
