"""
simulation.py — Monte Carlo simulation engine for Blackjack strategy.

For every combination of player starting hand × dealer up-card we run
10,000 independent simulated games, once for each decision (Stand / Hit),
and record the fraction of games won.  The recommended move is whichever
action produces the higher win probability.

Monte Carlo method in brief
---------------------------
Instead of computing an exact probability analytically (which is complex
when the dealer's hidden card and future draws are unknown), we estimate
it empirically: simulate the situation thousands of times at random, count
how often the player wins, and divide by the total number of trials.
By the Law of Large Numbers, the estimate converges to the true probability
as the number of trials grows.  At 10,000 trials the standard error is
roughly ±0.5 percentage points — accurate enough for strategy decisions.
"""

from blackjack import draw_card, add_card, hand_from_pair, run_dealer, player_wins

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N = 10_000  # simulations per scenario

# Unique card values we sample from (Ace = 1, face cards = 10)
CARD_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Dealer up-cards: 2–10 then Ace (stored as 1)
DEALER_CARDS = list(range(2, 11)) + [1]

# All unique two-card starting hands (c1 ≤ c2 ensures no duplicates)
ALL_PAIRS = [(c1, c2) for c1 in range(1, 11) for c2 in range(c1, 11)]

# Human-readable card label
CARD_LABEL = {1: 'A', 2: '2', 3: '3', 4: '4', 5: '5',
              6: '6', 7: '7', 8: '8', 9: '9', 10: '10'}


# ---------------------------------------------------------------------------
# Single-scenario simulations
# ---------------------------------------------------------------------------

def sim_stand(player_total, is_soft, dealer_up, n=N):
    """
    Estimate win probability when the player stands immediately.

    Runs n simulated dealer turns and counts how many the player wins.
    """
    wins = sum(
        player_wins(player_total, run_dealer(dealer_up))
        for _ in range(n)
    )
    return wins / n


def sim_hit(player_total, is_soft, dealer_up, n=N):
    """
    Estimate win probability when the player hits once then stands.

    For each trial:
      1. Draw a random third card for the player.
      2. If the player busts → loss (no dealer turn needed).
      3. Otherwise simulate the dealer turn and check outcome.
    """
    wins = 0
    for _ in range(n):
        card = draw_card()
        new_total, new_soft, busted = add_card(player_total, is_soft, card)
        if not busted and player_wins(new_total, run_dealer(dealer_up)):
            wins += 1
    return wins / n


def sim_hit_by_card(player_total, is_soft, dealer_up, n=N):
    """
    Break down the Hit win probability by each possible drawn card.

    For each of the 10 distinct card values, run n simulations and compute
    the win probability given that specific card was drawn.

    Returns a dict {card_value: win_probability}.

    Note: in a real deck the 10-value card (10/J/Q/K) is 4× more likely
    than any other value — the chart in the app marks this clearly.
    """
    result = {}
    for card in CARD_VALUES:
        new_total, new_soft, busted = add_card(player_total, is_soft, card)
        if busted:
            result[card] = 0.0
        else:
            wins = sum(
                player_wins(new_total, run_dealer(dealer_up))
                for _ in range(n)
            )
            result[card] = wins / n
    return result


# ---------------------------------------------------------------------------
# Precompute all scenarios
# ---------------------------------------------------------------------------

def precompute():
    """
    Run Monte Carlo simulations for every unique hand state × dealer up-card.

    Hand states are deduplicated: different card pairs that produce the same
    (total, is_soft) share one set of simulation results.

    Returns
    -------
    results : dict
        Key   → (player_total, is_soft, dealer_up)
        Value → {'stand': float, 'hit': float,
                 'recommended': 'H' or 'S', 'best_prob': float}
    """
    # Collect unique (total, is_soft) states reachable from two starting cards
    seen = set()
    states = []
    for c1, c2 in ALL_PAIRS:
        state = hand_from_pair(c1, c2)
        if state not in seen:
            seen.add(state)
            states.append(state)

    results = {}
    for (player_total, is_soft) in states:
        for dealer_up in DEALER_CARDS:
            p_stand = sim_stand(player_total, is_soft, dealer_up)
            p_hit   = sim_hit(player_total, is_soft, dealer_up)
            rec     = 'H' if p_hit > p_stand else 'S'
            results[(player_total, is_soft, dealer_up)] = {
                'stand':      p_stand,
                'hit':        p_hit,
                'recommended': rec,
                'best_prob':  max(p_stand, p_hit),
            }

    return results
