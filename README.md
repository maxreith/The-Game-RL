# The Game Simulation

A Monte Carlo simulation framework for analyzing strategies in
[The Game](https://boardgamegeek.com/boardgame/173090/the-game), a cooperative card game
where players work together to play all 98 cards onto four shared stacks.

## The Game Rules

- 98 cards numbered 2-99
- Four stacks: two ascending (start at 1), two descending (start at 99)
- Cards must be played in the correct direction, except for the "backwards trick":
  playing exactly 10 higher/lower resets the stack
- Each turn: play at least 2 cards (or 1 if deck is empty), then draw back to hand size
- Win condition: all cards played; lose condition: cannot make a legal play

## Analysis

### Which strategy works best? And what is the optimal number of players?

The simulation tests a "bonus play" strategy that plays additional cards beyond the
minimum required when a card is within a threshold distance from a stack top. Lower
thresholds mean more aggressive extra plays. I ran 10,000 simulations for each
combination of player count and bonus play threshold.

![Strategy Evaluation](bld/strategy_evaluation.png)

**Findings:**

- Threshold 2 or 3 performs best across most player counts
- Win rates peak at only 5 percent
- 5 players seems optimal

### Does it matter how well the deck is shuffled?

I used to disagree with my office mates over how well we'd have to shuffle the deck
before setting up a new game. Using a custom cut-based shuffle algorithm, I simulate how
shuffle quality affects win rates. I ran 1,000 games per shuffle quality, using the
optimal settings (5 players, bonus play threshold of 2).

![Shuffle Quality Evaluation](bld/shuffle_evaluation.png)

**Findings:**

- Poorly shuffled decks have dramatically higher win rates
- With proper shuffling (50+ iterations), win rates stabilize around 5%

### Can AI play this game?

The answer is no. I wanted to test Gemini 3 on this, but couldn't get it to win a single
game. It keeps making invalid moves. I guess there is no training data for this game.
But does Gemini at least perform better when you increase the thinking level? The plot
below shows the average turn at which Gemini lost, for various thinking levels. I ran 3
games per thinking level (because compute is not free, and high thinking is slow).

![Gemini Thinking Levels](bld/gemini_thinking.png)

**Findings:**

- Higher thinking levels survive more turns
- Even with extended thinking, the AI struggles with the game

## Installation

```bash
pixi install
```

## Usage

Run all simulations:

```bash
pixi run pytask
```

Or run individual simulations:

```bash
pixi run python src/simulate_strategies.py
pixi run python src/simulate_shuffle_quality.py
pixi run python src/simulate_gemini_thinking.py
```

Generate plots from existing results:

```bash
pixi run python src/generate_plots.py
```

## Project Structure

```
src/
├── game_setup.py          # Core game mechanics
├── strategies.py          # Playing strategies (bonus play, Gemini)
├── utils.py               # Stack implementation, Gemini API integration
├── simulate_strategies.py # Strategy comparison simulation
├── simulate_shuffle_quality.py  # Shuffle quality analysis
├── simulate_gemini_thinking.py  # Gemini thinking level tests
└── generate_plots.py      # Visualization generation

tests/                     # Unit tests
bld/                       # Generated outputs (plots, results)
```

## Configuration

For Gemini simulations, set your API key in `.env`:

```
GEMINI_API_KEY=your_key_here
```
