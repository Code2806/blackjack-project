"""
app.py — Streamlit UI for the Blackjack Strategy Analyzer.

Sections:
  Section 1 — Play Blackjack   (interactive game with live Monte Carlo advice)
  Section 2 — Complete Strategy Table
  Section 3 — Hand Inspector

Run with:
    streamlit run app.py
"""

import random
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from blackjack import hand_from_pair, add_card
from simulation import (
    precompute, sim_hit_by_card, sim_stand, sim_hit,
    ALL_PAIRS, DEALER_CARDS, CARD_LABEL, CARD_VALUES,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Blackjack Strategy Analyzer",
    page_icon="🃏",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Precompute all scenarios (cached: runs once per session)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_results():
    return precompute()


with st.spinner("Running Monte Carlo simulations — please wait (~30 s)…"):
    results = load_results()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🃏 Blackjack Strategy Analyzer")
st.markdown("**Monte Carlo simulation of optimal Hit / Stand decisions**")

# ---------------------------------------------------------------------------
# About section
# ---------------------------------------------------------------------------

with st.expander("📖 What is Monte Carlo simulation? (click to expand)", expanded=False):
    st.markdown("""
### Monte Carlo Simulation

**Monte Carlo simulation** is a computational technique that uses repeated
random sampling to estimate quantities that are difficult to calculate
analytically.  The name comes from the famous Monte Carlo casino — a
reminder that the method is built on randomness.

#### How it works in three steps

1. **Define a random process** that models the situation of interest.
2. **Run the process thousands of times**, each time with independent
   random outcomes.
3. **Aggregate the results** — for example, count how often an event
   occurs and divide by the total number of trials to estimate its
   probability.

#### Application to Blackjack

For every combination of *player starting hand* (e.g. 7+8 = Hard 15) and
*dealer up-card* (e.g. a 6), we want to know:

> "If I **Stand**, what fraction of games do I win?
> If I **Hit** (draw one more card and then stand), what fraction do I win?"

Computing this analytically requires tracking all possible sequences of
dealer cards — a combinatorial problem that grows quickly.  Instead, for
each scenario we simulate **10,000 independent games**:

- **Stand scenario**: Player keeps the starting hand; the dealer draws
  cards until reaching 17 or more.  We count wins and divide by 10,000.
- **Hit scenario**: Player draws one random card.  If the player busts,
  it's a loss.  Otherwise the dealer plays out as above.

With 10,000 trials the **standard error** is approximately
σ = √(p(1−p)/n) ≤ 0.5 percentage points — precise enough to identify
the better move in almost all situations.

#### Why it works — Law of Large Numbers

The Law of Large Numbers guarantees that as the number of trials *n*
grows, the sample average converges to the true expected value.  At
n = 10,000 we are far into the region where the estimate is stable across
independent runs.
    """)

st.divider()

# ===========================================================================
# Section 1 — Interactive Blackjack Game
# ===========================================================================

# ---------------------------------------------------------------------------
# Card rendering helpers
# ---------------------------------------------------------------------------

SUITS = ['♠', '♥', '♦', '♣']

# Deck pool for infinite-deck draw (4 tens, 1 each of 1-9)
_DECK_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def _draw():
    """Draw a random (value, suit) card — suit is cosmetic only."""
    return random.choice(_DECK_POOL), random.choice(SUITS)


def _card_html(val, suit, hidden=False):
    """Return a styled HTML badge for one card."""
    if hidden:
        return (
            "<span style='font-size:1.9em; padding:5px 11px; border:2px solid #7f8c8d; "
            "border-radius:8px; background:#34495e; color:#34495e; margin:3px; "
            "display:inline-block; box-shadow:1px 2px 5px rgba(0,0,0,.3)'>🂠</span>"
        )
    color = "#c0392b" if suit in ('♥', '♦') else "#1a252f"
    lbl = CARD_LABEL[val]
    return (
        f"<span style='font-size:1.9em; padding:5px 11px; border:2px solid #bdc3c7; "
        f"border-radius:8px; background:white; color:{color}; margin:3px; "
        f"display:inline-block; box-shadow:1px 2px 5px rgba(0,0,0,.15)'>{lbl}{suit}</span>"
    )


def _hand_html(cards, hide_second=False):
    """Return HTML for a list of (val, suit) cards. Optionally hide the second card."""
    parts = [
        _card_html(v, s, hidden=(hide_second and i == 1))
        for i, (v, s) in enumerate(cards)
    ]
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Game state helpers
# ---------------------------------------------------------------------------

def _init_game():
    """Set session_state defaults on first run (idempotent)."""
    defaults = {
        'gs':           'idle',      # game state: idle | player_turn | finished
        'p_cards':      [],          # player cards: list of (val, suit)
        'dealer_up':    None,        # (val, suit) — visible
        'dealer_hole':  None,        # (val, suit) — hidden during player turn
        'dealer_extra': [],          # extra cards drawn during dealer turn
        'dealer_total': None,        # dealer final total (set on Stand)
        'history':      [],          # list of step dicts for summary table
        'result':       None,        # 'win' | 'lose' | 'push' | 'bust'
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _reset_game():
    """Reset all game keys back to idle state."""
    st.session_state.gs           = 'idle'
    st.session_state.p_cards      = []
    st.session_state.dealer_up    = None
    st.session_state.dealer_hole  = None
    st.session_state.dealer_extra = []
    st.session_state.dealer_total = None
    st.session_state.history      = []
    st.session_state.result       = None


def _player_total():
    """Return (total, is_soft) for the current player hand."""
    total, is_soft = 0, False
    for v, _ in st.session_state.p_cards:
        total, is_soft, _ = add_card(total, is_soft, v)
    return total, is_soft


def _record_step(label):
    """
    Run sim_stand + sim_hit for the current player hand vs dealer up-card,
    and append one row to the history list.
    Returns (total, is_soft, p_stand, p_hit, recommended).
    """
    total, is_soft = _player_total()
    dealer_up_val  = st.session_state.dealer_up[0]

    p_stand = sim_stand(total, is_soft, dealer_up_val, n=10_000)
    p_hit   = sim_hit(total, is_soft, dealer_up_val,   n=10_000)
    rec     = 'H' if p_hit > p_stand else 'S'

    cards_str = " ".join(f"{CARD_LABEL[v]}{s}" for v, s in st.session_state.p_cards)
    hand_str  = f"Soft {total}" if is_soft else f"Hard {total}"

    st.session_state.history.append({
        'Step':        label,
        'Cards':       cards_str,
        'Hand':        hand_str,
        'Stand %':     p_stand,
        'Hit %':       p_hit,
        'Recommended': 'Hit' if rec == 'H' else 'Stand',
    })
    return total, is_soft, p_stand, p_hit, rec


def _dealer_play():
    """
    Play out the dealer's hand using the actual hole card, then any extras.
    Sets dealer_extra and dealer_total in session_state.
    Returns dealer final total.
    """
    du = st.session_state.dealer_up[0]
    dh = st.session_state.dealer_hole[0]

    total, is_soft, _ = add_card(0, False, du)
    total, is_soft, _ = add_card(total, is_soft, dh)

    extras = []
    while total < 17:
        v, s = _draw()
        total, is_soft, busted = add_card(total, is_soft, v)
        extras.append((v, s))
        if busted:
            break

    st.session_state.dealer_extra = extras
    st.session_state.dealer_total = total
    return total


def _deal():
    """Start a new hand: draw cards and run the initial Monte Carlo analysis."""
    _reset_game()

    c1, c2 = _draw(), _draw()
    du, dh  = _draw(), _draw()

    st.session_state.p_cards     = [c1, c2]
    st.session_state.dealer_up   = du
    st.session_state.dealer_hole = dh

    total, is_soft = _player_total()

    # Player has natural blackjack (Ace + 10-value as first two cards)
    if total == 21 and is_soft:
        dealer_total = _dealer_play()
        # Check whether dealer also has blackjack (2-card 21)
        dealer_2card_total, _ = hand_from_pair(du[0], dh[0])
        if dealer_2card_total == 21:
            st.session_state.result = 'push'
        else:
            st.session_state.result = 'win'
        st.session_state.gs = 'finished'
    else:
        _record_step('Initial Deal')
        st.session_state.gs = 'player_turn'


# ---------------------------------------------------------------------------
# Render — Section 1
# ---------------------------------------------------------------------------

_init_game()

st.header("Section 1 — Play Blackjack")
st.markdown(
    "Play a hand of Blackjack. After each decision the Monte Carlo engine "
    "instantly calculates win probabilities for both moves and recommends "
    "the optimal play."
)

gs = st.session_state.gs

# ── IDLE ────────────────────────────────────────────────────────────────────
if gs == 'idle':
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        if st.button("🃏  Deal", type="primary", key="deal_btn"):
            _deal()
            st.rerun()

# ── PLAYER TURN ─────────────────────────────────────────────────────────────
elif gs == 'player_turn':
    total, is_soft = _player_total()
    dealer_up_val  = st.session_state.dealer_up[0]

    # Board
    left, right = st.columns(2)
    with left:
        hand_str = f"Soft {total}" if is_soft else f"Hard {total}"
        st.markdown(f"**Your hand — {hand_str}**")
        st.markdown(
            _hand_html(st.session_state.p_cards),
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("**Dealer's hand**")
        dealer_display = [st.session_state.dealer_up, st.session_state.dealer_hole]
        st.markdown(
            _hand_html(dealer_display, hide_second=True),
            unsafe_allow_html=True,
        )
        st.caption(f"Up-card: {CARD_LABEL[dealer_up_val]}")

    # Monte Carlo analysis (from last history entry)
    if st.session_state.history:
        last = st.session_state.history[-1]
        st.markdown("#### Monte Carlo Analysis")
        m1, m2, m3 = st.columns(3)
        m1.metric("Win % if Stand", f"{last['Stand %']:.1%}")
        m2.metric("Win % if Hit",   f"{last['Hit %']:.1%}")
        m3.metric("Recommended",    last['Recommended'])

    # Action buttons
    st.markdown("")
    b_hit, b_stand, _ = st.columns([1, 1, 4])

    with b_hit:
        if st.button("👆  Hit", type="primary", key="hit_btn"):
            v, s = _draw()
            st.session_state.p_cards.append((v, s))
            new_total, new_soft = _player_total()

            if new_total > 21:
                # Bust — record and finish without dealer play
                cards_str = " ".join(
                    f"{CARD_LABEL[cv]}{cs}" for cv, cs in st.session_state.p_cards
                )
                st.session_state.history.append({
                    'Step':        f"Hit {len(st.session_state.p_cards) - 2} (Bust)",
                    'Cards':       cards_str,
                    'Hand':        f"Hard {new_total}",
                    'Stand %':     0.0,
                    'Hit %':       0.0,
                    'Recommended': '—',
                })
                st.session_state.result       = 'bust'
                st.session_state.dealer_total = None
                st.session_state.gs           = 'finished'
            else:
                hit_num = len(st.session_state.p_cards) - 2
                _record_step(f"Hit {hit_num}")
            st.rerun()

    with b_stand:
        if st.button("✋  Stand", key="stand_btn"):
            player_total_final, _ = _player_total()
            dealer_total          = _dealer_play()

            if dealer_total > 21:
                st.session_state.result = 'win'
            elif player_total_final > dealer_total:
                st.session_state.result = 'win'
            elif player_total_final == dealer_total:
                st.session_state.result = 'push'
            else:
                st.session_state.result = 'lose'

            st.session_state.gs = 'finished'
            st.rerun()

# ── FINISHED ────────────────────────────────────────────────────────────────
elif gs == 'finished':
    total, is_soft   = _player_total()
    player_hand_str  = f"Soft {total}" if is_soft else f"Hard {total}"
    result           = st.session_state.result

    # Final board — both dealer cards revealed
    left, right = st.columns(2)
    with left:
        st.markdown(f"**Your hand — {player_hand_str}**")
        st.markdown(
            _hand_html(st.session_state.p_cards),
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("**Dealer's hand**")
        all_dealer = (
            [st.session_state.dealer_up, st.session_state.dealer_hole]
            + st.session_state.dealer_extra
        )
        st.markdown(_hand_html(all_dealer), unsafe_allow_html=True)
        dt = st.session_state.dealer_total
        if dt is not None:
            dealer_label = f"Bust ({dt})" if dt > 21 else str(dt)
            st.caption(f"Dealer total: {dealer_label}")

    # Result banner
    if result == 'win':
        st.success("### You Win!")
    elif result == 'bust':
        st.error(f"### Bust!  You Lose  (your total: {total})")
    elif result == 'push':
        st.warning("### Push — It's a Tie")
    else:
        st.error("### Dealer Wins — You Lose")

    # Hand history summary
    if st.session_state.history:
        st.markdown("#### Hand Summary")
        df_hist = pd.DataFrame(st.session_state.history)
        df_hist['Stand %'] = df_hist['Stand %'].map('{:.1%}'.format)
        df_hist['Hit %']   = df_hist['Hit %'].map('{:.1%}'.format)
        st.dataframe(df_hist, hide_index=True)

    # Deal again
    st.markdown("")
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        if st.button("🃏  Deal Again", type="primary", key="deal_again_btn"):
            _reset_game()
            st.rerun()

st.divider()

# ===========================================================================
# Helper data shared by Section 2 and Section 3
# ===========================================================================

DEALER_COL_VALS   = DEALER_CARDS
DEALER_COL_LABELS = [str(d) if d != 1 else 'A' for d in DEALER_COL_VALS]

SOFT_PAIR_LABEL = {
    12: 'A+A', 13: 'A+2', 14: 'A+3', 15: 'A+4',
    16: 'A+5', 17: 'A+6', 18: 'A+7', 19: 'A+8', 20: 'A+9',
}

hand_options = []
for c1, c2 in ALL_PAIRS:
    total, is_soft = hand_from_pair(c1, c2)
    if total == 21:
        continue
    label     = f"{CARD_LABEL[c1]}+{CARD_LABEL[c2]}"
    hand_type = f"Soft {total}" if is_soft else f"Hard {total}"
    display   = f"{label}  ({hand_type})"
    hand_options.append({
        'display': display, 'label': label,
        'c1': c1, 'c2': c2, 'total': total, 'is_soft': is_soft,
    })

dealer_options = [(lbl, val) for lbl, val in zip(DEALER_COL_LABELS, DEALER_COL_VALS)]


# ===========================================================================
# Section 2 — Full Strategy Table
# ===========================================================================

st.header("Section 2 — Complete Strategy Table")
st.markdown(
    "Each cell shows the recommended move for the given player hand "
    "(row) versus dealer up-card (column), derived from 10,000 Monte "
    "Carlo simulations per cell."
)

col_leg1, col_leg2, _ = st.columns([1, 1, 6])
col_leg1.markdown(
    "<div style='background:#27ae60;color:white;padding:6px 12px;"
    "border-radius:4px;text-align:center;font-weight:bold'>S — Stand</div>",
    unsafe_allow_html=True,
)
col_leg2.markdown(
    "<div style='background:#e74c3c;color:white;padding:6px 12px;"
    "border-radius:4px;text-align:center;font-weight:bold'>H — Hit</div>",
    unsafe_allow_html=True,
)
st.markdown("")


def _style_cell(val):
    if val == 'S':
        return 'background-color: #27ae60; color: white; font-weight: bold; text-align: center'
    if val == 'H':
        return 'background-color: #e74c3c; color: white; font-weight: bold; text-align: center'
    return ''


def build_strategy_df(hand_rows):
    rows, labels = [], []
    for (total, is_soft, row_label) in hand_rows:
        row = [
            results.get((total, is_soft, d), {}).get('recommended', '')
            for d in DEALER_COL_VALS
        ]
        rows.append(row)
        labels.append(row_label)
    return pd.DataFrame(rows, index=labels, columns=DEALER_COL_LABELS)


hard_hand_rows = [(total, False, f"Hard {total}") for total in range(4, 21)]
soft_hand_rows = [
    (total, True, f"Soft {total}  ({SOFT_PAIR_LABEL[total]})")
    for total in range(12, 21)
]

df_hard = build_strategy_df(hard_hand_rows)
df_soft = build_strategy_df(soft_hand_rows)

tab_hard, tab_soft = st.tabs(["Hard Hands", "Soft Hands"])

with tab_hard:
    st.caption("Hard hands — no Ace counted as 11")
    st.dataframe(
        df_hard.style.map(_style_cell),
        width="stretch",
        height=35 * len(df_hard) + 38,
    )

with tab_soft:
    st.caption("Soft hands — one Ace counted as 11")
    st.dataframe(
        df_soft.style.map(_style_cell),
        width="stretch",
        height=35 * len(df_soft) + 38,
    )

st.divider()

# ===========================================================================
# Section 3 — Hand Inspector
# ===========================================================================

st.header("Section 3 — Hand Inspector")
st.markdown(
    "Select a specific starting hand and dealer up-card to see detailed "
    "Monte Carlo results and, when Hit is recommended, the win probability "
    "for every possible drawn card."
)

col_hand, col_dealer = st.columns(2)

with col_hand:
    selected_display = st.selectbox(
        "Player starting hand",
        options=[h['display'] for h in hand_options],
        index=20,
    )

with col_dealer:
    selected_dealer_lbl = st.selectbox(
        "Dealer up-card",
        options=[lbl for lbl, _ in dealer_options],
        index=4,
    )

selected_hand   = next(h for h in hand_options if h['display'] == selected_display)
selected_dealer = next(val for lbl, val in dealer_options if lbl == selected_dealer_lbl)

player_total = selected_hand['total']
is_soft      = selected_hand['is_soft']
dealer_up    = selected_dealer

scenario = results[(player_total, is_soft, dealer_up)]

st.markdown("#### Simulation Results")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Win % if Standing",   f"{scenario['stand']:.1%}")
m2.metric("Win % if Hitting",    f"{scenario['hit']:.1%}")
m3.metric("Recommended Move",    scenario['recommended'], help="H = Hit, S = Stand")
m4.metric("Win % (recommended)", f"{scenario['best_prob']:.1%}")

if scenario['recommended'] == 'H':
    st.markdown("#### Win Probability by Drawn Card")
    st.caption(
        "Win probability when hitting and receiving each specific card. "
        "The **10-value card** (10 / J / Q / K) is **4× more probable** "
        "than any other single value."
    )

    cache_key = f"breakdown_{player_total}_{is_soft}_{dealer_up}"
    if cache_key not in st.session_state:
        with st.spinner("Simulating card distribution (10,000 trials per card)…"):
            st.session_state[cache_key] = sim_hit_by_card(
                player_total, is_soft, dealer_up
            )

    breakdown  = st.session_state[cache_key]
    bar_labels = [f"{CARD_LABEL[c]}" + (" ×4" if c == 10 else "") for c in CARD_VALUES]
    bar_values = [breakdown[c] for c in CARD_VALUES]
    bar_colors = [
        "#27ae60" if v >= 0.50 else "#f39c12" if v >= 0.40 else "#e74c3c"
        for v in bar_values
    ]

    fig = go.Figure(go.Bar(
        x=bar_labels,
        y=bar_values,
        marker_color=bar_colors,
        text=[f"{v:.1%}" for v in bar_values],
        textposition="outside",
    ))
    fig.update_layout(
        yaxis=dict(
            title="Win probability",
            tickformat=".0%",
            range=[0, max(bar_values) * 1.25 + 0.05],
        ),
        xaxis_title="Card drawn",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=40),
        height=380,
    )
    fig.add_hline(
        y=0.5, line_dash="dash", line_color="gray",
        annotation_text="50 % break-even", annotation_position="top right",
    )
    st.plotly_chart(fig, width="stretch")

    bust_cards = [
        CARD_LABEL[c] for c in CARD_VALUES
        if breakdown[c] == 0.0 and add_card(player_total, is_soft, c)[2]
    ]
    if bust_cards:
        st.info(
            f"Drawing **{', '.join(bust_cards)}** causes an immediate bust "
            f"(win probability = 0 %)."
        )
else:
    st.info(
        "Standing is the recommended move for this hand — "
        "no card-by-card breakdown is needed."
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "Blackjack Strategy Analyzer · University project · "
    "Monte Carlo simulation with 10,000 iterations per scenario · "
    "Infinite-deck assumption · Push counts as a loss"
)
