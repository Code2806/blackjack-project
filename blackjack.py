"""
blackjack.py — Core Blackjack game logic.

Handles card drawing, hand evaluation, dealer simulation and win detection.
Uses an infinite-deck assumption: cards are drawn independently with fixed
probabilities (P(10-value) = 4/13, P(any other value) = 1/13).
"""

import random

# ---------------------------------------------------------------------------
# Deck model
# ---------------------------------------------------------------------------

# 13 card types in a real deck: A, 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K.
# J / Q / K are all worth 10, so the pool below has four 10s.
# random.choice() over this list gives:
#   P(value 10) = 4/13  ≈ 30.8 %
#   P(value 1–9) = 1/13  each
DECK = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def draw_card():
    """Return a random card value from an infinite deck."""
    return random.choice(DECK)


# ---------------------------------------------------------------------------
# Hand state
# ---------------------------------------------------------------------------
# A hand is represented as (total, is_soft) where:
#   total   — the best possible hand value without busting
#   is_soft — True if one Ace is currently counted as 11

def add_card(total, is_soft, card):
    """
    Add a single card to a hand and return the updated state.

    Parameters
    ----------
    total   : current hand total
    is_soft : whether there is a soft Ace (counted as 11)
    card    : card value drawn (1 = Ace, 2–10)

    Returns
    -------
    (new_total, new_is_soft, busted)
    """
    if card == 1:  # Ace drawn
        if total + 11 <= 21:
            # Use the new Ace as 11
            return total + 11, True, False
        else:
            # Ace must count as 1
            return total + 1, is_soft, False

    new_total = total + card

    if new_total > 21 and is_soft:
        # Convert the existing soft Ace from 11 → 1 to rescue the hand
        new_total -= 10
        return new_total, False, new_total > 21

    return new_total, is_soft, new_total > 21


def hand_from_pair(c1, c2):
    """
    Compute the initial hand state for two starting cards c1 and c2.

    Returns (total, is_soft).
    """
    total, is_soft, _ = add_card(0, False, c1)
    total, is_soft, _ = add_card(total, is_soft, c2)
    return total, is_soft


# ---------------------------------------------------------------------------
# Dealer simulation
# ---------------------------------------------------------------------------

def run_dealer(dealer_up):
    """
    Simulate the dealer's complete turn given the visible up-card.

    The dealer draws a random hole card, then keeps hitting until the hand
    total reaches 17 or more (standard casino rule).

    Returns the dealer's final total. Values above 21 indicate a bust.
    """
    # Start with the up-card
    total, is_soft, _ = add_card(0, False, dealer_up)

    # Draw the hidden hole card
    total, is_soft, busted = add_card(total, is_soft, draw_card())
    if busted:
        return total

    # Dealer hits until 17+
    while total < 17:
        total, is_soft, busted = add_card(total, is_soft, draw_card())
        if busted:
            return total

    return total


# ---------------------------------------------------------------------------
# Outcome check
# ---------------------------------------------------------------------------

def player_wins(player_total, dealer_total):
    """
    Return True if the player wins the round.

    Win conditions:
      - Dealer busts (total > 21)
      - Player total > dealer total (no bust by player, assumed by caller)

    Note: a push (tie) counts as a loss, as stated in the project rules.
    """
    if dealer_total > 21:
        return True
    return player_total > dealer_total
