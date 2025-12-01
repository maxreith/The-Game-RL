import numpy as np

def _shuffle_cards_custom (card_deck: np.ndarray = None, n_shuffles: int = 200) -> np.ndarray:
    "Shuffles card deck"
    if card_deck is None:
        card_deck = np.arange(2, 100)
    
    shuffled_deck = card_deck.copy()  # Also copy to avoid mutating the input
    for _ in range(n_shuffles):
        random_numbers = np.random.randint(0, len(shuffled_deck), size=2)
        start, end = np.sort(random_numbers)
        shuffled_deck = np.concatenate((
            shuffled_deck[start:end], shuffled_deck[:start], shuffled_deck[end:]
        ))
    return shuffled_deck

def _shuffle_cards(card_deck: np.ndarray = None, n_shuffles: int = 200) -> np.ndarray:
    "Shuffles card deck"
    if card_deck is None:
        card_deck = np.arange(2, 100)
    
    shuffled_deck = card_deck.copy()
    np.random.shuffle(shuffled_deck)
    return shuffled_deck

def _initiate_game(n_players: int, card_deck: np.ndarray, hand_size: int = 6) -> tuple[list[np.ndarray], np.ndarray, dict]:
    """Create and deal cards to new players."""
    players = []
    card_index = 0
    
    for _ in range(n_players):
        player_cards = card_deck[card_index:card_index + hand_size]
        players.append(player_cards)
        card_index += hand_size

    stacks = {
        "decreasing_stack_1": np.array([99]),
        "decreasing_stack_2": np.array([99]),
        "increasing_stack_1": np.array([1]),
        "increasing_stack_2": np.array([1])
    }
    
    return players, card_deck[card_index:], stacks

def _play_to_stack(player: np.array, card: int, chosen_stack: str, all_stacks: dict) -> tuple[dict, np.ndarray]:
    """Play a card to a named stack if valid. Pure function."""
    # Skip if no card is passed
    if isinstance(card, np.ndarray) and len(card) == 0:
        return all_stacks, player

    stack = all_stacks[chosen_stack]

    # Check if player has card
    if card not in player:
        raise ValueError(f"Player does not have card {card}.")

    # Check if card can be played
    if "increasing" in chosen_stack:
        can_play = card > stack[-1] or card + 10 == stack[-1]
    else:
        can_play = card < stack[-1] or card - 10 == stack[-1]

    if not can_play:
        raise ValueError(f"Card {card} cannot be played on {chosen_stack}.")

    # Create new objects instead of mutating
    new_stacks = {
        k: (np.append(v, card) if k == chosen_stack else v)
        for k, v in all_stacks.items()
    }
    new_player = player[player != card]

    return new_player, new_stacks

def _draw_cards(player: np.ndarray, remaining_deck: np.ndarray, hand_size: int = 6):
    if len(remaining_deck) == 0:
        return player, remaining_deck
    
    cards_to_draw = hand_size - len(player)
    new_player = np.append(player, remaining_deck[:cards_to_draw])
    return new_player, remaining_deck[cards_to_draw:]

def _reset_pile(player: np.ndarray, stacks: dict) -> tuple[np.ndarray, dict]:
    """
    Play all possible reset cards (±10 jumps) until none remain.
    Keeps checking after each play since new resets may become available.
    """
    # Define which stacks use +10 vs -10 for resets
    stack_offsets = {
        "increasing_stack_1": -10,  # Need card = top - 10
        "increasing_stack_2": -10,
        "decreasing_stack_1": +10,  # Need card = top + 10
        "decreasing_stack_2": +10
    }
    
    found_reset = True
    while found_reset:
        found_reset = False
        
        for stack_name, offset in stack_offsets.items():
            top_card = stacks[stack_name][-1]
            reset_card = top_card + offset
            
            # Check if player has the reset card
            if reset_card in player:
                player, stacks = _play_to_stack(
                    player=player,
                    card=reset_card,
                    chosen_stack=stack_name,
                    all_stacks=stacks
                )
                found_reset = True
                break  # Restart the loop to check all stacks again
    
    return player, stacks

def _play_lowest_diff(player: np.ndarray, stacks: dict, cards_to_play: int = 2, hand_size: int = 6) -> tuple[np.ndarray, dict]:
    """
    Play cards to stacks where the difference between player's card 
    and stack top is smallest. Stops when player has (hand_size - cards_to_play) cards.
    Pure function.
    """    
    cards_played = 0
    
    while cards_played < cards_to_play and len(player) > 0:
        best_card = None
        best_stack = None
        best_diff = float('inf')
        
        for card in player:
            for stack_name, stack in stacks.items():
                top_card = stack[-1]
                
                # Check validity and calculate difference
                if "increasing" in stack_name: 
                    valid = card > top_card
                    diff = card - top_card
                else:
                    valid = card < top_card
                    diff = top_card - card
                
                if valid and diff < best_diff:
                    best_diff = diff
                    best_card = card
                    best_stack = stack_name
        
        if best_card is None:
            break
        
        # play card
        player, stacks = _play_to_stack(
            player=player,
            card=best_card,
            chosen_stack=best_stack,
            all_stacks=stacks
        )
        cards_played += 1
    
    return player, stacks

class GameOverError(Exception):
    """Raised when a player cannot make a valid move."""
    pass

def simple_game_strategy(player, stacks, remaining_deck):
    """First, reset stacks if you can. Then, play cards with minimum distance until you do not have to play a card anymore.
    """
    n_cards_to_play = 2 if len(remaining_deck) > 0 else 1

    new_player, new_stacks = _reset_pile(player, stacks)

    cards_played_in_reset = len(player) - len(new_player)

    n_cards_left_to_play = n_cards_to_play - cards_played_in_reset
    
    new_player, new_stacks = _play_lowest_diff(new_player, new_stacks, n_cards_left_to_play)

    n_cards_played = len(player) - len(new_player)

    if n_cards_to_play > n_cards_played:
        raise GameOverError(f"Player stuck with {len(player)} cards")
    return new_player, new_stacks

def run_game(strategy, n_players: int = 3, n_shuffles: int = 200) -> dict:
    "Runs an instance of the game with a given strategy."
    hand_size = 6 if n_players > 2 else 7
    shuffled_deck = _shuffle_cards(n_shuffles=n_shuffles)   
    players, remaining_deck, stacks = _initiate_game(n_players, shuffled_deck, hand_size)

    total_cards = lambda: len(remaining_deck) + sum(len(p) for p in players)
    turn = 0
    
    try:
        while total_cards() > 0:
            turn += 1
            #print(f"Turn {turn}: total_cards={total_cards()}, deck={len(remaining_deck)}, hands={[len(p) for p in players]}")
            if turn > 100:
                raise RuntimeError("Too many turns!")
            
            for i, player in enumerate(players):
                if len(player) == 0:
                    continue
                player, stacks = strategy(player, stacks, remaining_deck)
                player, remaining_deck = _draw_cards(player, remaining_deck, hand_size)
                players[i] = player
        
        return {"victory": True, "stacks": {k: v.copy() for k, v in stacks.items()}, "cards_remaining": 0}
        
    except GameOverError:
        return {"victory": False, "stacks": {k: v.copy() for k, v in stacks.items()}, "cards_remaining": total_cards()}
    
def _run_simulation(strategy, n_games: int = 100, n_players: int = 3):
    """Run multiple games and collect data."""
    victories = []
    losses = []

    for _ in range(n_games):
        result = run_game(strategy, n_players=n_players)
        
        if result["victory"]:
            victories.append(result)
        else:
            losses.append(result)

    return {
        "victories": victories,
        "losses": losses,
        "win_rate": len(victories) / n_games
    }

results = _run_simulation(simple_game_strategy, n_games=10, n_players=3)

print(f"Win rate: {results['win_rate']*100:.1f}%")
print(f"Victories: {len(results['victories'])}")
print(f"Losses: {len(results['losses'])}")