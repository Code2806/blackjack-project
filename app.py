"""
app.py — Streamlit UI for the Blackjack Strategy Analyzer.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from blackjack import hand_from_pair, add_card
from simulation import (
    precompute, sim_hit_by_card,
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

# ---------------------------------------------------------------------------
# Helper data — dealer column labels and hand option lists
# ---------------------------------------------------------------------------

DEALER_COL_VALS   = DEALER_CARDS                        # [2,3,...,10,1]
DEALER_COL_LABELS = [str(d) if d != 1 else 'A' for d in DEALER_COL_VALS]

# Soft-hand textual labels for each total
SOFT_PAIR_LABEL = {
    12: 'A+A', 13: 'A+2', 14: 'A+3', 15: 'A+4',
    16: 'A+5', 17: 'A+6', 18: 'A+7', 19: 'A+8', 20: 'A+9',
}

# Build the hand-inspector dropdown options
# (all pairs except A+10 = blackjack, where no decision is needed)
hand_options = []
for c1, c2 in ALL_PAIRS:
    total, is_soft = hand_from_pair(c1, c2)
    if total == 21:
        continue  # Blackjack — automatic win, no decision
    label      = f"{CARD_LABEL[c1]}+{CARD_LABEL[c2]}"
    hand_type  = f"Soft {total}" if is_soft else f"Hard {total}"
    display    = f"{label}  ({hand_type})"
    hand_options.append({
        'display': display, 'label': label,
        'c1': c1, 'c2': c2, 'total': total, 'is_soft': is_soft,
    })

dealer_options = [(lbl, val) for lbl, val in zip(DEALER_COL_LABELS, DEALER_COL_VALS)]


# ---------------------------------------------------------------------------
# Section 1 — Full Strategy Table
# ---------------------------------------------------------------------------

st.header("Section 1 — Complete Strategy Table")
st.markdown(
    "Each cell shows the recommended move for the given player hand "
    "(row) versus dealer up-card (column), derived from 10,000 Monte "
    "Carlo simulations per cell."
)

# Color legend
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
    """Return a CSS string for the recommended-move value."""
    if val == 'S':
        return 'background-color: #27ae60; color: white; font-weight: bold; text-align: center'
    if val == 'H':
        return 'background-color: #e74c3c; color: white; font-weight: bold; text-align: center'
    return ''


def build_strategy_df(hand_rows):
    """
    Build a DataFrame where rows are player hands, columns are dealer up-cards,
    and values are the recommended move ('H' or 'S').
    """
    rows, labels = [], []
    for (total, is_soft, row_label) in hand_rows:
        row = [
            results.get((total, is_soft, d), {}).get('recommended', '')
            for d in DEALER_COL_VALS
        ]
        rows.append(row)
        labels.append(row_label)
    return pd.DataFrame(rows, index=labels, columns=DEALER_COL_LABELS)


# Hard-hand rows: Hard 4 → Hard 20
hard_hand_rows = [
    (total, False, f"Hard {total}")
    for total in range(4, 21)
]

# Soft-hand rows: Soft 12 (A+A) → Soft 20 (A+9)
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

# ---------------------------------------------------------------------------
# Section 2 — Hand Inspector
# ---------------------------------------------------------------------------

st.header("Section 2 — Hand Inspector")
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
        index=20,  # default: a mid-range hard hand
    )

with col_dealer:
    selected_dealer_lbl = st.selectbox(
        "Dealer up-card",
        options=[lbl for lbl, _ in dealer_options],
        index=4,  # default: 6
    )

# Resolve selections
selected_hand   = next(h for h in hand_options if h['display'] == selected_display)
selected_dealer = next(val for lbl, val in dealer_options if lbl == selected_dealer_lbl)

player_total = selected_hand['total']
is_soft      = selected_hand['is_soft']
dealer_up    = selected_dealer

scenario = results[(player_total, is_soft, dealer_up)]

# --- Metrics ---
st.markdown("#### Simulation Results")
m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Win % if Standing",
    f"{scenario['stand']:.1%}",
)
m2.metric(
    "Win % if Hitting",
    f"{scenario['hit']:.1%}",
)
m3.metric(
    "Recommended Move",
    scenario['recommended'],
    help="H = Hit, S = Stand",
)
m4.metric(
    "Win % (recommended)",
    f"{scenario['best_prob']:.1%}",
)

# --- Card breakdown (only when Hit is recommended) ---
if scenario['recommended'] == 'H':
    st.markdown("#### Win Probability by Drawn Card")
    st.caption(
        "Win probability when hitting and receiving each specific card. "
        "The **10-value card** (10 / J / Q / K) is **4× more probable** "
        "than any other single value."
    )

    # Cache per-hand breakdown in session_state to avoid recomputing
    cache_key = f"breakdown_{player_total}_{is_soft}_{dealer_up}"
    if cache_key not in st.session_state:
        with st.spinner("Simulating card distribution (10,000 trials per card)…"):
            st.session_state[cache_key] = sim_hit_by_card(
                player_total, is_soft, dealer_up
            )

    breakdown = st.session_state[cache_key]

    bar_labels = [
        f"{CARD_LABEL[c]}" + (" ×4" if c == 10 else "")
        for c in CARD_VALUES
    ]
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

    # Bust probability note
    bust_cards = [CARD_LABEL[c] for c in CARD_VALUES
                  if breakdown[c] == 0.0 and
                  add_card(player_total, is_soft, c)[2]]
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
