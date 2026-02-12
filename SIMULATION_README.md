# Game Simulation Analysis

This simulation analyzes the card game "The Game" to understand:
1. Which parameters produce the best win rates
2. How shuffle quality affects game outcomes

## What Was Modified

### `game_setup.py` Changes
Added two new parameters to support shuffle quality testing:
- `use_custom_shuffle`: bool parameter to switch between standard and custom shuffle
- `n_shuffles`: parameter passed through to `_shuffle_cards_custom`

Both `run_game()` and `run_simulation()` now support these parameters.

### Shuffle Functions

**`_shuffle_cards()`** (lines 21-28)
- Uses numpy's built-in shuffle (Fisher-Yates algorithm)
- Always produces well-randomized decks
- Doesn't use the `n_shuffles` parameter

**`_shuffle_cards_custom()`** (lines 6-18)
- Custom shuffling algorithm that can simulate poor shuffles
- Uses `n_shuffles` parameter to control shuffle quality
- Algorithm: repeatedly picks two random positions and rearranges deck segments
- **Low n_shuffles** (1-5) = poorly shuffled, cards stay close to original order
- **High n_shuffles** (100+) = well shuffled, fully randomized

## The Simulation Script

### Phase 1: Parameter Optimization

Tests all combinations of:
- `n_players`: 2, 3, 4
- `bonus_play_threshold`: 1, 2, 3, 4, 5, 6, 7, 8

Each combination runs 500 games to get reliable statistics.

**Output**: `parameter_optimization_results.csv`

### Phase 2: Shuffle Quality Analysis

Using the optimal parameters from Phase 1, tests different shuffle qualities:
- `n_shuffles`: 1, 2, 5, 10, 20, 50, 100, 200

Each shuffle quality runs 1000 games.

**Output**: `shuffle_quality_results.csv`

## Expected Results

Based on the quick test run:

### Parameter Optimization
- **Best configuration**: n_players=2, bonus_play_threshold=4
- Win rates are generally low (3-10%) with well-shuffled decks

### Shuffle Quality Impact
With n_players=2, bonus_play_threshold=4:
- **n_shuffles=1-10**: ~100% win rate (poorly shuffled)
- **n_shuffles=20**: ~80% win rate (moderately shuffled)
- **n_shuffles=50+**: ~0-5% win rate (well shuffled)

**Why this happens**: With few shuffles, cards remain in sequential order (2, 3, 4, 5...), making it trivial to play them on the ascending piles. With proper shuffling, cards are randomized and the game becomes genuinely challenging.

## Running the Simulation

```bash
# Run full simulation (takes ~10-30 minutes)
python3 run_simulations.py

# Run quick test (takes ~1 minute)
python3 test_simulation.py
```

## Interpreting Results

The shuffle quality results demonstrate why proper shuffling is crucial for game balance:
- Tournament/competitive play should use n_shuffles ≥ 100
- Casual/teaching games could use n_shuffles ≈ 20 for moderate difficulty
- Testing strategies should use proper shuffling to get realistic win rates
