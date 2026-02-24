# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in
this repository.

## Project Overview

Monte Carlo simulation framework for analyzing strategies in "The Game" (a cooperative
card game with 98 numbered cards and four stacks). The project tests different playing
strategies, evaluates shuffle quality effects on win rates, and experiments with Gemini
AI as a player.

## Commands

```bash
# Install dependencies
pixi install

# Run all simulations via pytask
pixi run pytask

# Run individual simulations
pixi run python src/simulate_strategies.py
pixi run python src/simulate_shuffle_quality.py
pixi run python src/simulate_gemini_thinking.py

# Generate plots from existing results
pixi run python src/generate_plots.py

# Run tests
pixi run pytest tests/

# Run a single test file
pixi run pytest tests/test_game_setup.py
```

## Architecture

### Core Components

- **`src/utils.py`**: Foundation layer with `Stack` class (pre-allocated O(1)
  operations), `GameOverError`, Gemini API integration, and `_play_to_stack()` helper
- **`src/game_setup.py`**: Game engine with `run_game()` and `run_simulation()` - takes
  a strategy callable with signature
  `(player, stacks, remaining_deck) -> (player, stacks)`
- **`src/strategies.py`**: Strategy implementations - `bonus_play_strategy`
  (algorithmic) and `gemini_strategy` (AI-powered)

### Stack Indices Convention

Throughout the codebase, stacks are indexed as:

- 0, 1: Decreasing stacks (start at 99, play lower cards)
- 2, 3: Increasing stacks (start at 1, play higher cards)

### Strategy Pattern

Strategies are passed to `run_game()` as callables. Use `functools.partial` to
pre-configure parameters:

```python
from functools import partial

strategy = partial(bonus_play_strategy, bonus_play_threshold=2)
```

### Test Configuration

Tests use `conftest.py` to add `src/` to the Python path. Run tests from the project
root.

## Gemini Configuration

Set `GEMINI_API_KEY` in `.env` file. The model is configured via `GEMINI_MODEL`
environment variable (defaults to `gemini-3-flash-preview`).

## Output

All generated outputs (plots, parquet files) go to `bld/` directory.

## Reinforcement Learning

Uses MaskablePPO from `sb3-contrib` with gymnasium environment. Training from single
player perspective with parameter sharing across all players.

```bash
pixi run python src/train_rl.py
pixi run tensorboard --logdir bld/rl_logs/
```

## RL Experiment Results

### Baseline

| Strategy            | Players | Win Rate      |
| ------------------- | ------- | ------------- |
| bonus_play_strategy | 3       | 1.4%          |
| bonus_play_strategy | 5       | 4.4%          |
| random              | 5       | ~14 cards avg |

### Grid Search (300k steps, 5 players, 500 eval games)

| Config                | Win% | Avg Cards | reward_per_card | win_reward | loss_penalty | trick_play | dist_penalty |
| --------------------- | ---- | --------- | --------------- | ---------- | ------------ | ---------- | ------------ |
| trick_and_distance    | 0%   | 63.2      | 0.05            | 10.0       | 0.0          | 1.0        | 0.005        |
| with_loss_penalty     | 0%   | 62.8      | 0.05            | 10.0       | 1.0          | 1.0        | 0.005        |
| high_card_reward      | 0%   | 62.2      | 0.10            | 10.0       | 0.0          | 1.0        | 0.005        |
| low_card_reward       | 0%   | 59.5      | 0.01            | 10.0       | 0.0          | 1.0        | 0.005        |
| very_high_card_reward | 0%   | 59.1      | 0.20            | 10.0       | 0.0          | 1.0        | 0.005        |
| high_both_terminal    | 0%   | 58.3      | 0.05            | 50.0       | 2.0          | 1.0        | 0.005        |
| high_win_bonus        | 0%   | 58.2      | 0.05            | 50.0       | 0.0          | 1.0        | 0.005        |
| simple_baseline       | 0%   | 56.2      | 0.05            | 10.0       | 0.0          | 0.0        | 0.000        |
| with_trick_bonus      | 0%   | 33.7      | 0.05            | 10.0       | 0.0          | 1.0        | 0.000        |

### Extended Training: trick_and_distance at 2M steps

| Training Steps | Avg Cards | Win Rate |
| -------------- | --------- | -------- |
| 300k           | 63.2      | 0%       |
| 2M             | 84.7      | 0%       |

Configuration:
`n_players=5, reward_per_card=0.05, win_reward=10.0, loss_penalty=0.0, trick_play_reward=1.0, distance_penalty_scale=0.005, progress_reward_scale=0.0`

### Best Configuration (2M steps, commit fd61005)

| Metric        | Value   |
| ------------- | ------- |
| Win rate      | 1%      |
| Avg cards     | 84      |
| Training time | ~35 min |

**Environment:**
`n_players=5, reward_per_card=0.02, win_reward=10.0, loss_penalty=0.0, trick_play_reward=1.0, distance_penalty_scale=0.003, progress_reward_scale=3.0`

**Observation space (17 features):** hand(6), stack_tops(4), stack_gaps(4),
deck_remaining, cards_played_this_turn, min_cards_required

**PPO:**
`gamma=0.99, gae_lambda=0.95, ent_coef=0.02, learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10, clip_range=0.2, net_arch=[256,256]`

### Hierarchical RL (sequential card-then-stack decisions)

Breaks turn into phases: CHOOSE_CARD(0-5) → CHOOSE_STACK(0-3) → CONTINUE(0-1). Reduces
action space from 25 to 6.

Files: `src/hierarchical_game_env.py`, `src/train_hierarchical_rl.py`,
`src/evaluate_hierarchical.py`

```bash
pixi run python src/train_hierarchical_rl.py
pixi run python src/evaluate_hierarchical.py
```

Same reward config as best known. Results not yet documented.
