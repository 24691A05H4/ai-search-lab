"""
app.py - Main Streamlit Application
Project 43: Dual-Game AI Search Analysis Dashboard
Supports both 8-Puzzle and Tic-Tac-Toe with Depth-Limited Search (DLS),
Iterative Deepening (IDDFS), A* (Manhattan), and Depth-Limited Minimax.
Supports dynamic Night Mode (Midnight OLED), Deep Space Dark, and Academic Light themes.
Zero-dependency HTML table renderer for 100% pyarrow-free compatibility on Windows ARM64.
"""

import sys
import time
import asyncio
from typing import Optional, List, Dict, Any, Tuple

# Fix Windows ProactorEventLoop WinError 10054 on client disconnect
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

import streamlit as st
import pandas as pd

from game import TicTacToeGame
from minimax import get_best_ai_move
from puzzle import EightPuzzleGame, GOAL_STATE
from puzzle_search import (
    depth_limited_search,
    iterative_deepening_search,
    astar_search,
    run_puzzle_depth_benchmark,
    PUZZLE_PRESETS,
)
from analysis import (
    GameHistoryManager,
    run_depth_benchmark,
    run_combined_depth_benchmark,
    run_reusable_search_analysis,
    calculate_dashboard_metrics,
    calculate_puzzle_dashboard_metrics,
    export_results_to_csv,
)
from visualization import (
    plot_depth_vs_nodes_bar,
    plot_depth_vs_nodes_line,
    plot_depth_vs_cutoffs,
    plot_depth_vs_time,
    plot_puzzle_comparison_dashboard,
    plot_multi_depth_comparison_dashboard,
    plot_candidate_move_scores,
    plot_combined_nodes_comparison,
    plot_combined_cutoffs_comparison,
    plot_combined_time_comparison,
    plot_combined_overview_dashboard,
    plot_interactive_search_tree,
)
from visualizer_engine import (
    record_8puzzle_search_events,
    record_minimax_search_events,
)


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Search Analysis Dashboard | 8-Puzzle & Tic-Tac-Toe",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Sidebar Controls & Theme Selection
# ---------------------------------------------------------
st.sidebar.markdown(
    """
    <div style='text-align: center; padding-bottom: 8px;'>
        <h2 style='color:#00f2fe; margin-bottom: 2px; font-size:1.6rem;'>🧠 AI SEARCH LAB</h2>
        <span style='color:#9d4edd; font-weight:600; font-size:0.85rem;'>8-Puzzle & Tic-Tac-Toe Profiler</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 Display Theme")

theme_choice = st.sidebar.selectbox(
    "Visual Theme",
    [
        "🌙 Night Mode (Midnight OLED)",
        "🌌 Deep Space (Navy/Purple)",
        "☀️ Academic Day / Light",
    ],
    index=0,
    help="Toggle between deep high-contrast Night Mode, Deep Space Navy, and Academic Light Mode.",
)

if "Midnight" in theme_choice:
    theme_mode = "night"
elif "Deep Space" in theme_choice:
    theme_mode = "deep_space"
else:
    theme_mode = "light"


# ---------------------------------------------------------
# Dynamic CSS Generator Based on Selected Theme
# ---------------------------------------------------------
def generate_css(mode: str) -> str:
    if mode == "light":
        return """
        <style>
            .stApp {
                background: linear-gradient(135deg, #f8fafc 0%, #edf2f7 100%);
                color: #0f172a;
                font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            }
            [data-testid="stSidebar"] {
                background-color: #f1f5f9 !important;
                border-right: 1px solid #cbd5e1;
            }
            .hero-card {
                background: linear-gradient(135deg, #eef2ff 0%, #f5f3ff 100%);
                border: 1px solid #c7d2fe;
                border-radius: 14px;
                padding: 24px;
                margin-bottom: 24px;
                box-shadow: 0 4px 12px rgba(99, 102, 241, 0.08);
            }
            .metric-card {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                padding: 16px;
                text-align: center;
                box-shadow: 0 2px 6px rgba(0,0,0,0.04);
            }
            .metric-val {
                font-size: 1.8rem;
                font-weight: 800;
                color: #4f46e5;
            }
            .metric-lbl {
                font-size: 0.82rem;
                font-weight: 600;
                color: #64748b;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .puzzle-tile {
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 2.2rem;
                font-weight: 800;
                height: 95px;
                border-radius: 12px;
                cursor: pointer;
                user-select: none;
                transition: transform 0.15s, box-shadow 0.15s;
            }
            .tile-number {
                background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
                color: #ffffff;
                border: 2px solid #818cf8;
                box-shadow: 0 4px 10px rgba(79, 70, 229, 0.25);
            }
            .tile-blank {
                background: #e2e8f0;
                color: #94a3b8;
                border: 2px dashed #cbd5e1;
            }
            .table-responsive-wrapper {
                width: 100%;
                overflow-x: auto;
                border-radius: 10px;
                border: 1px solid #cbd5e1;
                margin-top: 14px;
                margin-bottom: 14px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            }
            .custom-styled-table {
                width: 100%;
                border-collapse: collapse;
                font-family: inherit;
                font-size: 0.92rem;
                background: #ffffff;
            }
            .custom-styled-table th {
                background: #f1f5f9;
                color: #4338ca;
                font-weight: 700;
                padding: 12px 14px;
                text-align: left;
                border-bottom: 2px solid #cbd5e1;
                white-space: nowrap;
            }
            .custom-styled-table td {
                padding: 10px 14px;
                color: #1e293b;
                border-bottom: 1px solid #f1f5f9;
                white-space: nowrap;
            }
            .custom-styled-table tr:hover td {
                background: #f8fafc;
            }
        </style>
        """
    elif mode == "deep_space":
        return """
        <style>
            .stApp {
                background: linear-gradient(135deg, #090d16 0%, #0d111c 50%, #121024 100%);
                color: #f1f5f9;
                font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            }
            [data-testid="stSidebar"] {
                background-color: #0b0f1a !important;
                border-right: 1px solid #1e293b;
            }
            .hero-card {
                background: linear-gradient(135deg, rgba(21, 27, 43, 0.95) 0%, rgba(26, 16, 48, 0.95) 100%);
                border: 1px solid rgba(0, 242, 254, 0.25);
                border-radius: 14px;
                padding: 24px;
                margin-bottom: 24px;
                box-shadow: 0 0 25px rgba(0, 242, 254, 0.08);
            }
            .metric-card {
                background: #111726;
                border: 1px solid #1e293b;
                border-radius: 12px;
                padding: 16px;
                text-align: center;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            }
            .metric-val {
                font-size: 1.8rem;
                font-weight: 800;
                color: #00f2fe;
            }
            .metric-lbl {
                font-size: 0.82rem;
                font-weight: 600;
                color: #94a3b8;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .puzzle-tile {
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 2.2rem;
                font-weight: 800;
                height: 95px;
                border-radius: 12px;
                cursor: pointer;
                user-select: none;
            }
            .tile-number {
                background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
                color: #00f2fe;
                border: 2px solid #4f46e5;
                box-shadow: 0 4px 15px rgba(0, 242, 254, 0.2);
            }
            .tile-blank {
                background: #0b0f1a;
                color: #334155;
                border: 2px dashed #1e293b;
            }
            .table-responsive-wrapper {
                width: 100%;
                overflow-x: auto;
                border-radius: 10px;
                border: 1px solid rgba(0, 242, 254, 0.2);
                margin-top: 14px;
                margin-bottom: 14px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.4);
            }
            .custom-styled-table {
                width: 100%;
                border-collapse: collapse;
                font-family: inherit;
                font-size: 0.92rem;
                background: #0d121f;
            }
            .custom-styled-table th {
                background: #141c2e;
                color: #00f2fe;
                font-weight: 700;
                padding: 12px 14px;
                text-align: left;
                border-bottom: 2px solid rgba(0, 242, 254, 0.3);
                white-space: nowrap;
            }
            .custom-styled-table td {
                padding: 10px 14px;
                color: #f1f5f9;
                border-bottom: 1px solid #1a233a;
                white-space: nowrap;
            }
            .custom-styled-table tr:hover td {
                background: #172036;
            }
        </style>
        """
    else:  # Night Mode (Midnight OLED)
        return """
        <style>
            .stApp {
                background: #030508;
                color: #f8fafc;
                font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            }
            [data-testid="stSidebar"] {
                background-color: #060911 !important;
                border-right: 1px solid #141b2d;
            }
            .hero-card {
                background: #080c16;
                border: 1px solid rgba(0, 242, 254, 0.35);
                border-radius: 14px;
                padding: 24px;
                margin-bottom: 24px;
                box-shadow: 0 0 30px rgba(0, 242, 254, 0.12);
            }
            .metric-card {
                background: #080c16;
                border: 1px solid #162035;
                border-radius: 12px;
                padding: 16px;
                text-align: center;
                box-shadow: 0 0 15px rgba(0,0,0,0.7);
            }
            .metric-val {
                font-size: 1.8rem;
                font-weight: 800;
                color: #00f2fe;
            }
            .metric-lbl {
                font-size: 0.82rem;
                font-weight: 600;
                color: #94a3b8;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .puzzle-tile {
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 2.2rem;
                font-weight: 800;
                height: 95px;
                border-radius: 12px;
                cursor: pointer;
                user-select: none;
            }
            .tile-number {
                background: #0d1424;
                color: #00f2fe;
                border: 2px solid #00f2fe;
                box-shadow: 0 0 16px rgba(0, 242, 254, 0.3);
            }
            .tile-blank {
                background: #050810;
                color: #1e293b;
                border: 2px dashed #162035;
            }
            .table-responsive-wrapper {
                width: 100%;
                overflow-x: auto;
                border-radius: 10px;
                border: 1px solid rgba(0, 242, 254, 0.3);
                margin-top: 14px;
                margin-bottom: 14px;
                box-shadow: 0 0 15px rgba(0,0,0,0.5);
            }
            .custom-styled-table {
                width: 100%;
                border-collapse: collapse;
                font-family: inherit;
                font-size: 0.92rem;
                background: #080c16;
            }
            .custom-styled-table th {
                background: #0f1526;
                color: #00f2fe;
                font-weight: 700;
                padding: 12px 14px;
                text-align: left;
                border-bottom: 2px solid rgba(0, 242, 254, 0.4);
                white-space: nowrap;
            }
            .custom-styled-table td {
                padding: 10px 14px;
                color: #f8fafc;
                border-bottom: 1px solid #141c2e;
                white-space: nowrap;
            }
            .custom-styled-table tr:hover td {
                background: #111a30;
            }
        </style>
        """

st.markdown(generate_css(theme_mode), unsafe_allow_html=True)


# ---------------------------------------------------------
# PyArrow-Free HTML Table Renderer
# ---------------------------------------------------------
def render_styled_table(df: pd.DataFrame, height: Optional[int] = None):
    """
    Renders a responsive, modern HTML table from a DataFrame.
    Bypasses Streamlit's pyarrow requirement for 100% compatibility across
    all platforms (including Windows ARM64 without pyarrow binary wheels).
    """
    if df.empty:
        st.info("No records to display.")
        return

    scroll_style = f"max-height: {height}px; overflow-y: auto;" if height else ""
    table_html = df.to_html(index=False, classes="custom-styled-table")
    wrapper_html = f'<div class="table-responsive-wrapper" style="{scroll_style}">{table_html}</div>'
    st.markdown(wrapper_html, unsafe_allow_html=True)


# ---------------------------------------------------------
# Session State Initialization (Both Games)
# ---------------------------------------------------------
# Tic-Tac-Toe State
if "game" not in st.session_state:
    st.session_state.game = TicTacToeGame()

if "history_manager" not in st.session_state:
    st.session_state.history_manager = GameHistoryManager()

if "all_ai_stats" not in st.session_state:
    st.session_state.all_ai_stats = []

if "last_ai_analysis" not in st.session_state:
    st.session_state.last_ai_analysis = None

if "ttt_benchmark_cache" not in st.session_state:
    st.session_state.ttt_benchmark_cache = None

# 8-Puzzle State
if "puzzle_game" not in st.session_state:
    st.session_state.puzzle_game = EightPuzzleGame()
    # Shuffle lightly on startup so user sees a solvable challenge
    st.session_state.puzzle_game.shuffle(steps=12, seed=42)

if "puzzle_solution" not in st.session_state:
    st.session_state.puzzle_solution = None

if "puzzle_solution_step" not in st.session_state:
    st.session_state.puzzle_solution_step = 0

if "all_puzzle_searches" not in st.session_state:
    st.session_state.all_puzzle_searches = []

if "puzzle_benchmark_cache" not in st.session_state:
    st.session_state.puzzle_benchmark_cache = None

# Combined Benchmark State
if "combined_benchmark_cache" not in st.session_state:
    st.session_state.combined_benchmark_cache = None

# Interactive Visualizer States
if "puz_vis_log" not in st.session_state:
    st.session_state.puz_vis_log = None
if "puz_vis_step" not in st.session_state:
    st.session_state.puz_vis_step = 0
if "puz_vis_playing" not in st.session_state:
    st.session_state.puz_vis_playing = False

if "ttt_vis_log" not in st.session_state:
    st.session_state.ttt_vis_log = None
if "ttt_vis_step" not in st.session_state:
    st.session_state.ttt_vis_step = 0
if "ttt_vis_playing" not in st.session_state:
    st.session_state.ttt_vis_playing = False


# ---------------------------------------------------------
# Sidebar Navigation: 8 Core Sections
# ---------------------------------------------------------
st.sidebar.markdown("---")
NAV_OPTIONS = [
    "🏠 Home Dashboard",
    "🧩 8-Puzzle Game",
    "🔍 8-Puzzle Search Analysis",
    "🎮 Tic-Tac-Toe Game",
    "📊 Tic-Tac-Toe Search Analysis",
    "⚖️ Combined Performance Comparison",
    "🔬 AI Algorithm Visualizer",
    "📘 About Project",
]
selected_nav = st.sidebar.radio("Navigation Menu", NAV_OPTIONS, index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dual-Game Quick Counters")
c_ttt = len(st.session_state.history_manager.game_records)
c_puz = len(st.session_state.all_puzzle_searches)
st.sidebar.markdown(f"- **Tic-Tac-Toe Matches:** `{c_ttt}`")
st.sidebar.markdown(f"- **8-Puzzle Searches:** `{c_puz}`")
st.sidebar.markdown(f"- **8-Puzzle Solved:** `{'Yes 🎉' if st.session_state.puzzle_game.is_solved() else 'In Progress'}`")


# =========================================================
# SECTION 1: HOME DASHBOARD
# =========================================================
if selected_nav == "🏠 Home Dashboard":
    st.markdown(
        """
        <div class="hero-card">
            <h1 style='color:#00f2fe; margin-bottom: 6px;'>🧠 AI Search Analysis Dashboard</h1>
            <p style='color:#cbd5e1; font-size:1.1rem; line-height:1.6;'>
                An empirical research testbed for <strong>Adversarial Search</strong> and <strong>Single-Agent State Space Exploration</strong>.
                Analyze node generation rates, depth cutoff thresholds, and execution latency across
                <strong>8-Puzzle</strong> and <strong>Tic-Tac-Toe</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Dual Metrics Overview
    col_p, col_t = st.columns(2)

    with col_p:
        st.markdown("### 🧩 8-Puzzle Search Telemetry")
        puz_metrics = calculate_puzzle_dashboard_metrics(st.session_state.all_puzzle_searches)
        c1, c2 = st.columns(2)
        c1.metric("Total Searches Run", puz_metrics["total_searches"])
        c2.metric("Nodes Explored", f"{puz_metrics['total_nodes_generated']:,}")
        c3, c4 = st.columns(2)
        c3.metric("Heuristic Cutoffs", f"{puz_metrics['total_cutoff_nodes']:,}")
        c4.metric("Avg Latency", f"{puz_metrics['avg_execution_time_ms']} ms")

    with col_t:
        st.markdown("### 🎮 Tic-Tac-Toe Minimax Telemetry")
        ttt_metrics = calculate_dashboard_metrics(st.session_state.all_ai_stats)
        c1, c2 = st.columns(2)
        c1.metric("Total AI Decisions", ttt_metrics["total_ai_moves"])
        c2.metric("Nodes Explored", f"{ttt_metrics['total_nodes_generated']:,}")
        c3, c4 = st.columns(2)
        c3.metric("Heuristic Cutoffs", f"{ttt_metrics['total_cutoff_nodes']:,}")
        c4.metric("Avg Latency", f"{ttt_metrics['avg_execution_time_ms']} ms")

    st.markdown("---")
    st.markdown("### 🚀 Quick Navigation & Modules")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🧩 **8-Puzzle Game**\n\nInteractive sliding tile puzzle with Solvability parity check, Shuffle, and A* Solver.")
    with col2:
        st.info("🔍 **8-Puzzle Search Analysis**\n\nEmpirical Depth-Limited Search (DLS) & IDDFS profiler tracking nodes, cutoffs, and time.")
    with col3:
        st.info("🎮 **Tic-Tac-Toe Game**\n\nHuman vs AI Depth-Limited Minimax with live candidate move evaluation charts.")

    col4, col5, col6 = st.columns(3)
    with col4:
        st.info("📊 **Tic-Tac-Toe Search Analysis**\n\nSession move-by-move profiling, Depths 1–9 benchmark, and match history.")
    with col5:
        st.info("⚖️ **Combined Comparison**\n\nSide-by-side graphs comparing 8-Puzzle vs Tic-Tac-Toe branching factors & cutoffs.")
    with col6:
        st.info("📘 **About Project**\n\nComplete academic documentation, time complexities, and viva defense guide.")


# =========================================================
# SECTION 2: 8-PUZZLE GAME
# =========================================================
elif selected_nav == "🧩 8-Puzzle Game":
    st.markdown("## 🧩 8-Puzzle Game (3×3 Sliding Tile)")
    st.caption("Slide adjacent numbered tiles into the empty space (marked as EMPTY) to achieve the goal state.")

    puzzle = st.session_state.puzzle_game

    # Top Status Bar
    status_c1, status_c2, status_c3, status_c4 = st.columns(4)
    status_c1.metric("Move Count", puzzle.move_count)
    status_c2.metric("Manhattan Distance", EightPuzzleGame.manhattan_distance(puzzle.state))
    status_c3.metric("Misplaced Tiles", EightPuzzleGame.misplaced_tiles(puzzle.state))
    solv_text = "Yes (Even)" if EightPuzzleGame.is_solvable(puzzle.state) else "No (Odd)"
    status_c4.metric("Solvable Parity", solv_text)

    if puzzle.is_solved():
        st.success(f"🎉 **PUZZLE SOLVED!** Congratulations! Completed in {puzzle.move_count} moves.")

    col_board, col_controls = st.columns([1.2, 1])

    with col_board:
        st.markdown("#### 🎯 Active Puzzle Board:")
        # Render 3x3 interactive button grid
        blank_idx = puzzle.blank_index
        valid_moves = puzzle.get_valid_tile_moves()

        for r in range(3):
            cols = st.columns(3)
            for c in range(3):
                idx = r * 3 + c
                val = puzzle.state[idx]
                cell_key = f"puz_cell_{r}_{c}_{idx}"

                if val == 0:
                    cols[c].button("EMPTY", key=cell_key, disabled=True, width="stretch")
                else:
                    is_clickable = idx in valid_moves
                    btn_label = f" {val} "
                    if cols[c].button(btn_label, key=cell_key, disabled=not is_clickable, width="stretch"):
                        puzzle.move_tile(idx)
                        st.session_state.puzzle_solution = None  # Reset solver playback on manual move
                        st.rerun()

    with col_controls:
        st.markdown("#### ⚙️ Game Controls")

        c_shuf1, c_shuf2 = st.columns([2, 1])
        steps_choice = c_shuf1.slider("Shuffle Complexity (Moves)", min_value=3, max_value=35, value=12, step=1)
        if c_shuf2.button("🔀 Shuffle", type="primary", width="stretch"):
            puzzle.shuffle(steps=steps_choice)
            st.session_state.puzzle_solution = None
            st.session_state.puzzle_solution_step = 0
            st.rerun()

        col_b1, col_b2 = st.columns(2)
        if col_b1.button("🔄 Reset Initial", width="stretch"):
            puzzle.reset_to_initial()
            st.session_state.puzzle_solution = None
            st.rerun()

        if col_b2.button("🎯 Reset to Goal", width="stretch"):
            puzzle.reset_to_goal()
            st.session_state.puzzle_solution = None
            st.rerun()

        st.markdown("---")
        st.markdown("#### 🤖 AI Solver")
        solve_algo = st.selectbox("Solver Algorithm", ["A* Search (Manhattan Distance)", "Depth-Limited Search (DLS)"])

        if st.button("💡 Solve Puzzle with AI", type="secondary", width="stretch"):
            with st.spinner("Finding optimal solution path..."):
                if "A*" in solve_algo:
                    sol_res = astar_search(puzzle.state)
                else:
                    sol_res = depth_limited_search(puzzle.state, max_depth=12)

                if sol_res["solution_found"]:
                    st.session_state.puzzle_solution = sol_res["solution_path"]
                    st.session_state.puzzle_solution_step = 0
                    st.session_state.all_puzzle_searches.append(sol_res)
                    st.success(f"Solution found! Length: {sol_res['solution_length']} steps in {sol_res['execution_time_ms']} ms.")
                else:
                    st.warning("No solution within current depth limit. Try A* search or reduce scramble complexity.")

        # Playback Solution
        if st.session_state.puzzle_solution:
            sol = st.session_state.puzzle_solution
            cur_step = st.session_state.puzzle_solution_step
            st.info(f"📍 Solution Progress: Step {cur_step} of {len(sol)}")

            p_col1, p_col2 = st.columns(2)
            if p_col1.button("▶ Apply Next Step", width="stretch", disabled=(cur_step >= len(sol))):
                next_move = sol[cur_step]
                puzzle.set_state(next_move["state"])
                st.session_state.puzzle_solution_step += 1
                st.rerun()

            if p_col2.button("⚡ Solve Instantly", width="stretch"):
                puzzle.reset_to_goal()
                st.session_state.puzzle_solution_step = len(sol)
                st.rerun()

    st.markdown("---")
    st.markdown("#### 📋 Board Configuration Comparison")
    comp_c1, comp_c2 = st.columns(2)
    with comp_c1:
        st.markdown("**Current State Matrix:**")
        st.code(f"{puzzle.state[0]}  {puzzle.state[1]}  {puzzle.state[2]}\n{puzzle.state[3]}  {puzzle.state[4]}  {puzzle.state[5]}\n{puzzle.state[6]}  {puzzle.state[7]}  {puzzle.state[8]}")
    with comp_c2:
        st.markdown("**Goal State Target:**")
        st.code("1  2  3\n4  5  6\n7  8  EMPTY (0)")


# =========================================================
# SECTION 3: 8-PUZZLE SEARCH ANALYSIS
# =========================================================
elif selected_nav == "🔍 8-Puzzle Search Analysis":
    st.markdown("## 🔍 8-Puzzle AI Search Analysis & Profiler")
    st.caption("Empirical profiling of Depth-Limited Search (DLS), IDDFS, and A* on sliding tile puzzles.")

    # Configuration Card
    with st.expander("⚙️ Search Benchmark Configuration", expanded=True):
        col_s1, col_s2 = st.columns([1.5, 1])
        with col_s1:
            preset_choice = st.selectbox(
                "Initial Puzzle State for Benchmark:",
                ["Current Active Board"] + list(PUZZLE_PRESETS.keys()),
                index=0,
                help="Ensures identical starting configuration when comparing different depth limits.",
            )
            if preset_choice == "Current Active Board":
                bench_initial_state = st.session_state.puzzle_game.state
            else:
                bench_initial_state = PUZZLE_PRESETS[preset_choice]

        with col_s2:
            max_d_slider = st.slider("Maximum Search Depth Limit", min_value=1, max_value=12, value=8, step=1)

    # Display starting grid
    st.markdown(f"**Baseline Initial State:** `{bench_initial_state}` | **Manhattan Distance:** `{EightPuzzleGame.manhattan_distance(bench_initial_state)}`")

    col_btn1, col_btn2 = st.columns(2)
    run_dls = col_btn1.button("🚀 Run Multi-Depth DLS Benchmark (Depths 1 to N)", type="primary", width="stretch")
    run_iddfs = col_btn2.button("🔁 Run Iterative Deepening (IDDFS) Profiling", width="stretch")

    if run_dls:
        with st.spinner(f"Evaluating Depth-Limited Search across depths 1 through {max_d_slider}..."):
            depths_list = list(range(1, max_d_slider + 1))
            res = run_puzzle_depth_benchmark(bench_initial_state, depths=depths_list)
            st.session_state.puzzle_benchmark_cache = res
            for r in res:
                st.session_state.all_puzzle_searches.append(r)

    if run_iddfs:
        with st.spinner("Executing Iterative Deepening Search..."):
            iddfs_res = iterative_deepening_search(bench_initial_state, max_depth_limit=max_d_slider)
            st.session_state.puzzle_benchmark_cache = iddfs_res.get("per_depth_results", [])
            st.session_state.all_puzzle_searches.append(iddfs_res)

    if st.session_state.puzzle_benchmark_cache:
        p_results = st.session_state.puzzle_benchmark_cache

        st.markdown("### 📊 Required Algorithm Profiling Graphs")
        p_depths = [r["search_depth"] for r in p_results]
        p_nodes = [r["nodes_generated"] for r in p_results]
        p_cutoffs = [r["cutoff_nodes"] for r in p_results]
        p_times = [r["execution_time_ms"] for r in p_results]

        tab1, tab2, tab3 = st.tabs([
            "📊 Depth vs Nodes Generated",
            "✂️ Depth vs Cutoff Nodes",
            "⏱️ Depth vs Execution Time",
        ])

        with tab1:
            st.pyplot(plot_depth_vs_nodes_bar(p_depths, p_nodes, title="8-Puzzle: Search Depth vs Nodes Generated", theme=theme_mode))

        with tab2:
            st.pyplot(plot_depth_vs_cutoffs(p_depths, p_cutoffs, title="8-Puzzle: Search Depth vs Cutoff Nodes", theme=theme_mode))

        with tab3:
            st.pyplot(plot_depth_vs_time(p_depths, p_times, title="8-Puzzle: Search Depth vs Execution Time (ms)", theme=theme_mode))

        st.markdown("### 📋 Empirical Search Telemetry Table")
        df_puz = pd.DataFrame(p_results)
        df_puz_display = df_puz.rename(
            columns={
                "search_depth": "Search Depth",
                "nodes_generated": "Nodes Generated",
                "nodes_expanded": "Nodes Expanded",
                "cutoff_nodes": "Cutoff Nodes",
                "goal_nodes": "Goal Nodes",
                "execution_time_ms": "Time (ms)",
                "solution_found": "Solved?",
                "solution_length": "Path Length",
            }
        )[[
            "Search Depth",
            "Nodes Generated",
            "Nodes Expanded",
            "Cutoff Nodes",
            "Goal Nodes",
            "Time (ms)",
            "Solved?",
        ]]
        render_styled_table(df_puz_display)

        # CSV Export Option
        csv_str = export_results_to_csv(p_results)
        st.download_button(
            label="💾 Export 8-Puzzle Benchmark Data (CSV)",
            data=csv_str,
            file_name=f"8puzzle_search_benchmark_d1_to_d{max_d_slider}.csv",
            mime="text/csv",
        )


# =========================================================
# SECTION 4: TIC-TAC-TOE GAME
# =========================================================
elif selected_nav == "🎮 Tic-Tac-Toe Game":
    st.markdown("## 🎮 Play Tic-Tac-Toe vs Minimax AI")
    st.caption("Human plays as 'X' (First Mover) | AI plays as 'O' (Depth-Limited Minimax)")

    # Depth Selector for active game
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Tic-Tac-Toe Search Depth")
    ttt_depth = st.sidebar.slider(
        "Search Depth Limit",
        min_value=1,
        max_value=9,
        value=5,
        step=1,
        help="Restricts how many plies deep the Minimax recursion will explore.",
    )

    game = st.session_state.game
    winner = game.check_winner()

    # Game Status Banner
    if winner == "X":
        st.success("🎉 **VICTORY!** Human Player (X) Won!")
    elif winner == "O":
        st.error("🤖 **AI VICTORY!** Minimax Agent (O) Won!")
    elif winner == "Draw":
        st.warning("🤝 **DRAW GAME!** Stalemate reached.")
    else:
        st.info(f"⏳ **Active Match** — Current Turn: **{game.current_turn}** ({'Human' if game.current_turn == 'X' else 'AI'})")

    col_board, col_telemetry = st.columns([1.2, 1])

    with col_board:
        st.markdown("#### 3×3 Game Arena:")
        winning_line = game.get_winning_line() or ()

        for r in range(3):
            cols = st.columns(3)
            for c in range(3):
                idx = r * 3 + c
                val = game.board[idx]
                btn_text = val if val != " " else " "
                is_win_cell = idx in winning_line
                cell_key = f"ttt_cell_{r}_{c}_{idx}"

                if cols[c].button(btn_text, key=cell_key, disabled=(val != " " or winner is not None), width="stretch"):
                    # Human Move
                    game.make_move(idx, "X")
                    st.session_state.history_manager.record_move(
                        move_number=len(st.session_state.history_manager.move_history) + 1,
                        player="X",
                        cell_index=idx,
                    )

                    # Trigger AI Move if game not finished
                    if not game.is_game_over():
                        best_move, analysis = get_best_ai_move(game, max_depth=ttt_depth)
                        if best_move is not None:
                            game.make_move(best_move, "O")
                            st.session_state.last_ai_analysis = analysis
                            st.session_state.all_ai_stats.append(analysis)
                            st.session_state.history_manager.record_move(
                                move_number=len(st.session_state.history_manager.move_history) + 1,
                                player="O",
                                cell_index=best_move,
                                search_stats=analysis,
                            )

                    # Check for completed game
                    w_end = game.check_winner()
                    if w_end is not None:
                        st.session_state.history_manager.record_game_completion(
                            game_id=len(st.session_state.history_manager.game_records) + 1,
                            winner=w_end,
                            total_moves=len(st.session_state.history_manager.move_history),
                            search_depth=ttt_depth,
                            ai_moves_count=len(st.session_state.all_ai_stats),
                            total_nodes=sum(s.get("nodes_generated", 0) for s in st.session_state.all_ai_stats),
                        )
                    st.rerun()

        if st.button("🔄 Restart Game", type="primary"):
            st.session_state.game = TicTacToeGame()
            st.session_state.last_ai_analysis = None
            st.rerun()

    with col_telemetry:
        st.markdown("#### ⚡ Real-Time AI Telemetry")
        if st.session_state.last_ai_analysis:
            stat = st.session_state.last_ai_analysis
            st.metric("Nodes Explored", f"{stat['nodes_generated']:,}")
            st.metric("Nodes Expanded", f"{stat.get('nodes_expanded', 0):,}")
            st.metric("Cutoffs (Heuristic)", f"{stat['cutoff_nodes']:,}")
            st.metric("Execution Latency", f"{stat['execution_time_ms']} ms")
            st.metric("Evaluation Score", f"{stat['evaluation_score']}")

            # Candidate move score chart
            fig_cand = plot_candidate_move_scores(stat.get("move_scores", {}), theme=theme_mode)
            if fig_cand:
                st.pyplot(fig_cand)
        else:
            st.info("Make a move on the board to observe real-time Minimax tree expansion telemetry.")


# =========================================================
# SECTION 5: TIC-TAC-TOE SEARCH ANALYSIS
# =========================================================
elif selected_nav == "📊 Tic-Tac-Toe Search Analysis":
    st.markdown("## 📊 Tic-Tac-Toe AI Search Analysis & Benchmarking")
    st.caption("Deep-dive algorithmic analysis across search depths 1 through 9 on identical board configurations.")

    all_stats = st.session_state.all_ai_stats

    if all_stats:
        st.markdown("### 📈 Live Session Move Progression")
        moves_options = [f"AI Move #{i+1} (Depth {s['max_depth_setting']})" for i, s in enumerate(all_stats)]
        selected_move_idx = st.selectbox(
            "Select AI Move to Inspect:",
            options=range(len(moves_options)),
            format_func=lambda x: moves_options[x],
            index=len(moves_options) - 1,
        )
        stat = all_stats[selected_move_idx]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Search Depth", stat["max_depth_setting"])
        c2.metric("Nodes Explored", f"{stat['nodes_generated']:,}")
        c3.metric("Nodes Expanded", f"{stat.get('nodes_expanded', 0):,}")
        c4.metric("Latency", f"{stat['execution_time_ms']} ms")

        # Session plots
        s_depths = [s["max_depth_setting"] for s in all_stats]
        s_nodes = [s["nodes_generated"] for s in all_stats]
        s_cutoffs = [s["cutoff_nodes"] for s in all_stats]
        s_terminals = [s.get("terminal_nodes", 0) for s in all_stats]

        p_tab1, p_tab2, p_tab3 = st.tabs([
            "📊 Depth vs Nodes Generated",
            "📈 Tree Growth Line Chart",
            "✂️ Cutoff vs Terminal Breakdown",
        ])
        with p_tab1:
            st.pyplot(plot_depth_vs_nodes_bar(s_depths, s_nodes, title="Tic-Tac-Toe: Nodes Generated per Move", theme=theme_mode))
        with p_tab2:
            st.pyplot(plot_depth_vs_nodes_line(list(range(1, len(s_nodes) + 1)), s_nodes, theme=theme_mode))
        with p_tab3:
            st.pyplot(plot_depth_vs_cutoffs(s_depths, s_cutoffs, terminals=s_terminals, theme=theme_mode))

    st.markdown("---")
    st.markdown("### 🚀 Comprehensive Depths 1–9 Benchmark")
    st.markdown("Evaluates Depth-Limited Minimax from **Depth 1 up to Depth 9** on the **EXACT same initial board**.")

    board_choice = st.radio(
        "Benchmark Initial Board Configuration:",
        ["1. Empty Initial Board (9 Empty Cells — Full Game Tree)", "2. Current Active Match Board State"],
        index=0,
    )
    eval_game = TicTacToeGame() if board_choice.startswith("1.") else st.session_state.game.clone()

    if st.button("🚀 Run Comprehensive Tic-Tac-Toe Benchmark (Depths 1–9)", type="primary"):
        with st.spinner("Running Minimax across depths 1 through 9..."):
            ttt_b_results = run_depth_benchmark(eval_game, depths=list(range(1, 10)))
            st.session_state.ttt_benchmark_cache = ttt_b_results

    if st.session_state.ttt_benchmark_cache:
        t_res = st.session_state.ttt_benchmark_cache
        st.pyplot(plot_multi_depth_comparison_dashboard(t_res, theme=theme_mode))

        df_ttt = pd.DataFrame(t_res)
        df_ttt_display = df_ttt.rename(
            columns={
                "depth": "Depth",
                "nodes_generated": "Nodes Generated",
                "nodes_expanded": "Nodes Expanded",
                "cutoff_nodes": "Cutoff Nodes",
                "terminal_nodes": "Terminal Nodes",
                "execution_time_ms": "Time (ms)",
                "selected_pos": "Move (r, c)",
                "evaluation_score": "Score",
            }
        )[[
            "Depth",
            "Nodes Generated",
            "Nodes Expanded",
            "Cutoff Nodes",
            "Terminal Nodes",
            "Time (ms)",
            "Move (r, c)",
            "Score",
        ]]
        render_styled_table(df_ttt_display)

        # CSV Export
        csv_ttt = export_results_to_csv(t_res)
        st.download_button(
            label="💾 Export Tic-Tac-Toe Benchmark Data (CSV)",
            data=csv_ttt,
            file_name="tictactoe_benchmark_d1_to_d9.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.markdown("### 📜 Session History Records")
    tab_m, tab_g = st.tabs(["📝 Move Log", "🏆 Completed Games"])
    with tab_m:
        m_df = st.session_state.history_manager.get_moves_df()
        render_styled_table(m_df, height=320)
    with tab_g:
        g_df = st.session_state.history_manager.get_games_df()
        render_styled_table(g_df, height=320)


# =========================================================
# SECTION 6: COMBINED PERFORMANCE COMPARISON
# =========================================================
elif selected_nav == "⚖️ Combined Performance Comparison":
    st.markdown("## ⚖️ Combined Performance Comparison (8-Puzzle vs Tic-Tac-Toe)")
    st.markdown(
        """
        Compare search complexity, node generation rate, cutoff distribution, and execution latency
        between **8-Puzzle (Single-Agent State Space)** and **Tic-Tac-Toe (Two-Player Adversarial Minimax)**
        across identical search depths (Depths 1 to 9).
        """
    )

    puz_state = st.session_state.puzzle_game.state
    ttt_baseline = TicTacToeGame()

    st.info(f"**Benchmark Setup**: 8-Puzzle Starting State = `{puz_state}` | Tic-Tac-Toe = Empty Board (9 legal moves)")

    if st.button("🚀 Run Combined Dual-Game Benchmark (Depths 1–9)", type="primary"):
        with st.spinner("Executing benchmark across both games simultaneously..."):
            comb_res = run_combined_depth_benchmark(puz_state, ttt_baseline, depths=list(range(1, 10)))
            st.session_state.combined_benchmark_cache = comb_res

    if st.session_state.combined_benchmark_cache:
        c_data = st.session_state.combined_benchmark_cache
        depths = c_data["depths"]
        p_res = c_data["puzzle"]
        t_res = c_data["tictactoe"]

        p_nodes = [r["nodes_generated"] for r in p_res]
        t_nodes = [r["nodes_generated"] for r in t_res]
        p_cuts = [r["cutoff_nodes"] for r in p_res]
        t_cuts = [r["cutoff_nodes"] for r in t_res]
        p_times = [r["execution_time_ms"] for r in p_res]
        t_times = [r["execution_time_ms"] for r in t_res]

        st.markdown("### 📊 Combined Comparative Visualizations")
        st.pyplot(plot_combined_overview_dashboard(c_data, theme=theme_mode))

        st.markdown("### 📋 Unified Comparative Data Table")
        comb_rows = []
        for i, d in enumerate(depths):
            comb_rows.append({
                "Depth": d,
                "8-Puzzle Nodes": p_nodes[i],
                "Tic-Tac-Toe Nodes": t_nodes[i],
                "8-Puzzle Cutoffs": p_cuts[i],
                "Tic-Tac-Toe Cutoffs": t_cuts[i],
                "8-Puzzle Time (ms)": p_times[i],
                "Tic-Tac-Toe Time (ms)": t_times[i],
            })
        df_comb = pd.DataFrame(comb_rows)
        render_styled_table(df_comb)

        # CSV Export
        csv_comb = df_comb.to_csv(index=False)
        st.download_button(
            label="💾 Export Combined Dual-Game Data (CSV)",
            data=csv_comb,
            file_name="combined_8puzzle_tictactoe_benchmark.csv",
            mime="text/csv",
        )

        st.markdown("---")
        st.markdown("### 🧠 Key Algorithmic & Theoretical Insights")
        c_info1, c_info2 = st.columns(2)
        with c_info1:
            st.markdown(
                """
                #### 1. Branching Factor Differences:
                - **8-Puzzle**: Branching factor is $b \in [2, 4]$ (average $b \approx 2.67$). In the corners, only 2 moves are possible; on edges, 3 moves; in center, 4 moves.
                - **Tic-Tac-Toe**: Branching factor starts high at $b = 9$ at the root, decreasing by 1 at each ply ($9 \rightarrow 8 \rightarrow 7 \rightarrow \dots$).
                - This causes Tic-Tac-Toe's unpruned tree to expand dramatically faster at mid-depths ($d=4..7$).
                """
            )
        with c_info2:
            st.markdown(
                """
                #### 2. Search Structure & Cutoff Semantics:
                - **8-Puzzle (DLS)**: Single-agent deterministic graph search. State space contains **cycles**, requiring path-based cycle detection. Cutoff nodes are non-goal states stopped by depth limit.
                - **Tic-Tac-Toe (Minimax)**: Two-player zero-sum game tree. Alternates between **Maximizing** (AI) and **Minimizing** (Human) plies. At depth 9, cutoffs collapse to **0** because all paths terminate.
                """
            )


# =========================================================
# SECTION 7: INTERACTIVE AI ALGORITHM VISUALIZER
# =========================================================
elif selected_nav == "🔬 AI Algorithm Visualizer":
    st.markdown("## 🔬 Interactive Step-by-Step AI Search Visualizer")
    st.caption("Watch how AI search algorithms explore nodes, detect cycles, hit depth cutoffs, and propagate Minimax scores in real time.")

    def _render_8puzzle_html_board(state_tuple: Tuple[int, ...], prev_state: Optional[Tuple[int, ...]] = None, is_goal: bool = False) -> str:
        """Generates a styled 3x3 HTML grid for an 8-puzzle state."""
        moved_tile_val = None
        if prev_state and prev_state != state_tuple:
            # Find which non-zero tile moved into prev blank
            prev_blank = prev_state.index(0)
            moved_tile_val = state_tuple[prev_blank]

        cells_html = []
        for idx, val in enumerate(state_tuple):
            if val == 0:
                bg = "#1e293b" if theme_mode != "light" else "#e2e8f0"
                border = "2px dashed #00f2fe" if theme_mode != "light" else "2px dashed #4f46e5"
                txt_col = "#00f2fe" if theme_mode != "light" else "#4f46e5"
                cells_html.append(
                    f"<div style='background:{bg}; border:{border}; color:{txt_col}; height:70px; display:flex; align-items:center; justify-content:center; border-radius:8px; font-weight:700; font-size:0.9rem;'>EMPTY</div>"
                )
            else:
                if is_goal:
                    bg = "#059669"
                    border = "2px solid #00f5d4"
                    txt_col = "#ffffff"
                elif val == moved_tile_val:
                    bg = "#d97706"
                    border = "2px solid #ffd700"
                    txt_col = "#ffffff"
                else:
                    bg = "#0d1424" if theme_mode != "light" else "#4f46e5"
                    border = "2px solid #00f2fe" if theme_mode != "light" else "2px solid #818cf8"
                    txt_col = "#00f2fe" if theme_mode != "light" else "#ffffff"
                cells_html.append(
                    f"<div style='background:{bg}; border:{border}; color:{txt_col}; height:70px; display:flex; align-items:center; justify-content:center; border-radius:8px; font-weight:800; font-size:1.6rem;'>{val}</div>"
                )
        return f"<div style='display:grid; grid-template-columns:repeat(3, 1fr); gap:8px; max-width:240px;'>{''.join(cells_html)}</div>"

    def _render_ttt_html_board(board_list: List[str], prev_board: Optional[List[str]] = None, outcome: Optional[str] = None) -> str:
        """Generates a styled 3x3 HTML grid for a Tic-Tac-Toe state."""
        cells_html = []
        for idx, val in enumerate(board_list):
            is_new = (prev_board is not None and prev_board[idx] == " " and val != " ")
            if outcome == "O":
                border = "2px solid #00f5d4"
                bg = "rgba(5, 150, 105, 0.25)"
            elif outcome == "X":
                border = "2px solid #ff007f"
                bg = "rgba(225, 29, 72, 0.25)"
            elif outcome == "Draw":
                border = "2px solid #ffb703"
                bg = "rgba(217, 119, 6, 0.25)"
            elif is_new:
                border = "2px solid #ffd700"
                bg = "rgba(255, 215, 0, 0.18)"
            else:
                border = "1px solid #334155" if theme_mode != "light" else "1px solid #cbd5e1"
                bg = "#0d1424" if theme_mode != "light" else "#f8fafc"

            txt_col = "#00f2fe" if val == "O" else ("#ff007f" if val == "X" else "#64748b")
            cells_html.append(
                f"<div style='background:{bg}; border:{border}; color:{txt_col}; height:70px; display:flex; align-items:center; justify-content:center; border-radius:8px; font-weight:800; font-size:1.8rem;'>{val if val != ' ' else '·'}</div>"
            )
        return f"<div style='display:grid; grid-template-columns:repeat(3, 1fr); gap:8px; max-width:240px;'>{''.join(cells_html)}</div>"

    vis_sub = st.radio(
        "Select Visualizer Module:",
        ["🧩 8-Puzzle Search Visualization", "🎮 Tic-Tac-Toe Minimax Visualization"],
        horizontal=True,
    )

    st.markdown("---")

    # =====================================================
    # SUBSECTION 1: 8-PUZZLE SEARCH VISUALIZATION
    # =====================================================
    if vis_sub == "🧩 8-Puzzle Search Visualization":
        st.markdown("### 🧩 8-Puzzle Depth-Limited Search (DLS) Step-by-Step")

        col_cfg1, col_cfg2, col_cfg3 = st.columns([1.5, 1, 1])
        with col_cfg1:
            puz_init_opt = st.selectbox(
                "Initial Puzzle State:",
                ["Current Active Puzzle Board"] + list(PUZZLE_PRESETS.keys()),
                index=2 if len(PUZZLE_PRESETS) > 1 else 0,
            )
            puz_vis_init = (
                st.session_state.puzzle_game.state
                if puz_init_opt == "Current Active Puzzle Board"
                else PUZZLE_PRESETS[puz_init_opt]
            )
        with col_cfg2:
            puz_vis_depth = st.slider("DLS Depth Limit", min_value=1, max_value=8, value=3, key="vis_puz_depth")
        with col_cfg3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Start / Record Trace", type="primary", width="stretch"):
                with st.spinner("Recording actual DLS execution trace..."):
                    log = record_8puzzle_search_events(puz_vis_init, max_depth=puz_vis_depth)
                    st.session_state.puz_vis_log = log
                    st.session_state.puz_vis_step = 0
                    st.session_state.puz_vis_playing = False
                    st.rerun()

        if st.session_state.puz_vis_log is not None:
            log = st.session_state.puz_vis_log
            total_steps = log.total_steps
            cur_idx = min(st.session_state.puz_vis_step, total_steps - 1)
            step = log.get_step(cur_idx)

            # Playback Controls Bar
            st.markdown("#### 🎛️ Playback Controls")
            ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4, ctrl_c5, ctrl_c6 = st.columns([1, 1, 1.2, 1, 1, 1.5])
            if ctrl_c1.button("⏮️ First", width="stretch"):
                st.session_state.puz_vis_step = 0
                st.session_state.puz_vis_playing = False
                st.rerun()
            if ctrl_c2.button("◀️ Prev", width="stretch", disabled=(cur_idx <= 0)):
                st.session_state.puz_vis_step = max(0, cur_idx - 1)
                st.session_state.puz_vis_playing = False
                st.rerun()
            play_label = "⏸️ Pause" if st.session_state.puz_vis_playing else "▶️ Play"
            if ctrl_c3.button(play_label, type="primary", width="stretch"):
                st.session_state.puz_vis_playing = not st.session_state.puz_vis_playing
                st.rerun()
            if ctrl_c4.button("Next ▶️", width="stretch", disabled=(cur_idx >= total_steps - 1)):
                st.session_state.puz_vis_step = min(total_steps - 1, cur_idx + 1)
                st.session_state.puz_vis_playing = False
                st.rerun()
            if ctrl_c5.button("🔄 Reset", width="stretch"):
                st.session_state.puz_vis_step = 0
                st.session_state.puz_vis_playing = False
                st.rerun()
            speed_sec = ctrl_c6.select_slider("Speed", options=[0.1, 0.3, 0.5, 1.0, 1.5], value=0.5, format_func=lambda x: f"{x}s", key="puz_speed")

            # Scrubber slider
            new_step = st.slider("Timeline Scrubber (Step #)", min_value=0, max_value=max(0, total_steps - 1), value=cur_idx, key="puz_scrubber")
            if new_step != cur_idx:
                st.session_state.puz_vis_step = new_step
                st.session_state.puz_vis_playing = False
                st.rerun()

            if step:
                # Telemetry Row
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Step", f"{step.step_number} / {total_steps - 1}")
                m2.metric("Active Node", f"N{step.node_id}")
                m3.metric("Depth", step.current_depth)
                m4.metric("Nodes Gen", step.nodes_generated_count)
                m5.metric("Nodes Exp", step.nodes_expanded_count)
                m6.metric("Cutoffs", step.cutoff_count)

                # Explanation Banner
                if step.event_type == "GOAL_FOUND":
                    st.success(f"**[{step.event_type}]** {step.explanation}")
                elif step.event_type == "NODE_CUTOFF":
                    st.warning(f"**[{step.event_type}]** {step.explanation}")
                elif step.event_type == "NODE_PRUNED":
                    st.error(f"**[{step.event_type}]** {step.explanation}")
                else:
                    st.info(f"**[{step.event_type}]** {step.explanation}")

                # Active Path Breadcrumb
                path_str = " ➔ ".join([f"N{nid} (d={log.nodes[nid].depth})" for nid in step.active_path])
                st.markdown(f"**Active Branch Path:** `{path_str}`")

                # Side-by-side Boards + Successors
                b_col1, b_col2, b_col3, b_col4 = st.columns([1, 1, 1, 1.2])
                with b_col1:
                    st.markdown("**Initial State**")
                    st.markdown(_render_8puzzle_html_board(log.initial_state), unsafe_allow_html=True)
                with b_col2:
                    st.markdown(f"**Current State (N{step.node_id})**")
                    prev_st = log.nodes[step.parent_id].state if step.parent_id is not None and step.parent_id in log.nodes else None
                    st.markdown(_render_8puzzle_html_board(step.state, prev_state=prev_st, is_goal=(step.state == GOAL_STATE)), unsafe_allow_html=True)
                with b_col3:
                    st.markdown("**Goal State**")
                    st.markdown(_render_8puzzle_html_board(GOAL_STATE, is_goal=True), unsafe_allow_html=True)
                with b_col4:
                    st.markdown("**Valid Next Transitions**")
                    succs = EightPuzzleGame.get_successors(step.state)
                    for act, _, tile in succs:
                        st.markdown(f"- Slide Tile **{tile}** (`{act}`)")

                st.markdown("---")
                # Graphical Tree + Node Inspector
                st.markdown("### 🌳 Interactive Search Tree")
                t_c1, t_c2 = st.columns([1, 1])
                tree_depth_lim = t_c1.slider("Tree Zoom (Visible Depth)", min_value=1, max_value=max(2, log.max_depth), value=min(3, log.max_depth), key="puz_tree_zoom")
                max_nodes_vis = t_c2.slider("Max Visible Nodes (Pagination Filter)", min_value=15, max_value=100, value=45, step=5, key="puz_tree_max")

                snap_nodes = log.get_nodes_snapshot_at_step(cur_idx)
                fig_tree = plot_interactive_search_tree(
                    snap_nodes,
                    active_node_id=step.node_id,
                    active_path=step.active_path,
                    max_depth_visible=tree_depth_lim,
                    theme=theme_mode,
                    max_nodes_display=max_nodes_vis,
                )
                st.pyplot(fig_tree)

                # Node Inspector
                with st.expander("🔍 Inspect Any Tree Node State"):
                    insp_id = st.selectbox("Select Node ID to Inspect:", options=sorted(snap_nodes.keys()), format_func=lambda x: f"Node {x} (Depth {snap_nodes[x].depth} | {snap_nodes[x].status})")
                    n_obj = snap_nodes[insp_id]
                    ic1, ic2 = st.columns([1, 2])
                    with ic1:
                        st.markdown(_render_8puzzle_html_board(n_obj.state, is_goal=(n_obj.state == GOAL_STATE)), unsafe_allow_html=True)
                    with ic2:
                        st.markdown(f"- **Node ID**: `{n_obj.node_id}` | **Parent ID**: `{n_obj.parent_id}`")
                        st.markdown(f"- **Depth**: `{n_obj.depth}` | **Status**: `{n_obj.status}`")
                        st.markdown(f"- **Action Taken**: `{n_obj.action}`")
                        st.markdown(f"- **Children IDs**: `{n_obj.children_ids}`")

            # Autoplay Advance
            if st.session_state.puz_vis_playing:
                if cur_idx < total_steps - 1:
                    time.sleep(speed_sec)
                    st.session_state.puz_vis_step = cur_idx + 1
                    st.rerun()
                else:
                    st.session_state.puz_vis_playing = False
        else:
            st.info("Click **🚀 Start / Record Trace** above to capture and animate real DLS search steps.")

    # =====================================================
    # SUBSECTION 2: TIC-TAC-TOE MINIMAX VISUALIZATION
    # =====================================================
    else:
        st.markdown("### 🎮 Tic-Tac-Toe Minimax Step-by-Step Visualization")

        col_cfg1, col_cfg2, col_cfg3 = st.columns([1.5, 1, 1])
        with col_cfg1:
            ttt_setup = st.selectbox(
                "Initial Board Setup for Visualization:",
                [
                    "1. Tactical Mid-Game Setup (Clear Fork/Block Tree)",
                    "2. End-Game Setup (3 Cells Open — Full Terminal Tree)",
                    "3. Current Active Game Board",
                ],
                index=0,
            )
            if ttt_setup.startswith("1."):
                vis_ttt_game = TicTacToeGame(board=["X", "O", " ", " ", "O", " ", "X", " ", " "], current_turn="O")
            elif ttt_setup.startswith("2."):
                vis_ttt_game = TicTacToeGame(board=["X", "O", "X", "O", "X", " ", " ", " ", "O"], current_turn="O")
            else:
                vis_ttt_game = st.session_state.game.clone()

        with col_cfg2:
            ttt_vis_depth = st.slider("Minimax Depth Limit", min_value=1, max_value=9, value=3, key="vis_ttt_depth")
        with col_cfg3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Start / Record Minimax Trace", type="primary", width="stretch"):
                with st.spinner("Recording actual Minimax decision tree trace..."):
                    log_t = record_minimax_search_events(vis_ttt_game, max_depth=ttt_vis_depth)
                    st.session_state.ttt_vis_log = log_t
                    st.session_state.ttt_vis_step = 0
                    st.session_state.ttt_vis_playing = False
                    st.rerun()

        if st.session_state.ttt_vis_log is not None:
            log_t = st.session_state.ttt_vis_log
            total_steps = log_t.total_steps
            cur_idx = min(st.session_state.ttt_vis_step, total_steps - 1)
            step = log_t.get_step(cur_idx)

            # Playback Controls
            st.markdown("#### 🎛️ Playback Controls")
            c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1.2, 1, 1, 1.5])
            if c1.button("⏮️ First", key="ttt_first", width="stretch"):
                st.session_state.ttt_vis_step = 0
                st.session_state.ttt_vis_playing = False
                st.rerun()
            if c2.button("◀️ Prev", key="ttt_prev", width="stretch", disabled=(cur_idx <= 0)):
                st.session_state.ttt_vis_step = max(0, cur_idx - 1)
                st.session_state.ttt_vis_playing = False
                st.rerun()
            t_play_lbl = "⏸️ Pause" if st.session_state.ttt_vis_playing else "▶️ Play"
            if c3.button(t_play_lbl, key="ttt_play", type="primary", width="stretch"):
                st.session_state.ttt_vis_playing = not st.session_state.ttt_vis_playing
                st.rerun()
            if c4.button("Next ▶️", key="ttt_next", width="stretch", disabled=(cur_idx >= total_steps - 1)):
                st.session_state.ttt_vis_step = min(total_steps - 1, cur_idx + 1)
                st.session_state.ttt_vis_playing = False
                st.rerun()
            if c5.button("🔄 Reset", key="ttt_reset", width="stretch"):
                st.session_state.ttt_vis_step = 0
                st.session_state.ttt_vis_playing = False
                st.rerun()
            t_speed = c6.select_slider("Speed", options=[0.1, 0.3, 0.5, 1.0, 1.5], value=0.5, format_func=lambda x: f"{x}s", key="ttt_speed")

            new_t_step = st.slider("Timeline Scrubber (Step #)", min_value=0, max_value=max(0, total_steps - 1), value=cur_idx, key="ttt_scrubber")
            if new_t_step != cur_idx:
                st.session_state.ttt_vis_step = new_t_step
                st.session_state.ttt_vis_playing = False
                st.rerun()

            if step:
                cur_node_obj = log_t.nodes.get(step.node_id)
                role_lbl = getattr(cur_node_obj, "player", "MAX") or "MAX"

                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Step", f"{step.step_number} / {total_steps - 1}")
                m2.metric("Active Node", f"N{step.node_id} ({role_lbl})")
                m3.metric("Depth", step.current_depth)
                m4.metric("Nodes Gen", step.nodes_generated_count)
                m5.metric("Cutoffs", step.cutoff_count)
                m6.metric("Score", f"{step.score}" if step.score is not None else "-")

                # Explanation Banner
                if step.event_type in ("GOAL_FOUND", "BEST_MOVE_SELECTED"):
                    st.success(f"**[{step.event_type}]** {step.explanation}")
                elif step.event_type == "NODE_TERMINAL":
                    outc = getattr(cur_node_obj, "terminal_outcome", None)
                    if outc == "O":
                        st.success(f"**[TERMINAL WIN - AI 'O']** {step.explanation}")
                    elif outc == "X":
                        st.error(f"**[TERMINAL LOSS - Human 'X']** {step.explanation}")
                    else:
                        st.warning(f"**[TERMINAL DRAW]** {step.explanation}")
                elif step.event_type == "NODE_CUTOFF":
                    st.warning(f"**[{step.event_type}]** {step.explanation}")
                else:
                    st.info(f"**[{step.event_type}]** {step.explanation}")

                path_str = " ➔ ".join([f"N{nid} ({getattr(log_t.nodes[nid], 'player', 'MAX')})" for nid in step.active_path])
                st.markdown(f"**Active Minimax Branch:** `{path_str}`")

                b1, b2, b3 = st.columns([1, 1, 1.2])
                with b1:
                    st.markdown("**Initial Board**")
                    st.markdown(_render_ttt_html_board(log_t.initial_state), unsafe_allow_html=True)
                with b2:
                    st.markdown(f"**Current State (N{step.node_id})**")
                    prev_b = log_t.nodes[step.parent_id].state if step.parent_id is not None and step.parent_id in log_t.nodes else None
                    outc = getattr(cur_node_obj, "terminal_outcome", None)
                    st.markdown(_render_ttt_html_board(step.state, prev_board=prev_b, outcome=outc), unsafe_allow_html=True)
                with b3:
                    st.markdown("**Score & Node Status**")
                    st.markdown(f"- **Level Type**: `{role_lbl}`")
                    st.markdown(f"- **Action**: `{step.action_taken}`")
                    st.markdown(f"- **Evaluated Score**: `{step.score if step.score is not None else 'Pending Backtrack'}`")
                    st.markdown(f"- **Terminal Leaves Found**: `{step.terminal_count}`")

                st.markdown("---")
                st.markdown("### 🌳 Minimax Game Tree & Score Propagation")
                tc1, tc2 = st.columns(2)
                t_zoom = tc1.slider("Tree Zoom (Visible Depth)", min_value=1, max_value=max(2, log_t.max_depth), value=min(3, log_t.max_depth), key="ttt_tree_zoom")
                t_max = tc2.slider("Max Visible Nodes", min_value=15, max_value=100, value=45, step=5, key="ttt_tree_max")

                snap_t_nodes = log_t.get_nodes_snapshot_at_step(cur_idx)
                fig_ttt_tree = plot_interactive_search_tree(
                    snap_t_nodes,
                    active_node_id=step.node_id,
                    active_path=step.active_path,
                    max_depth_visible=t_zoom,
                    theme=theme_mode,
                    max_nodes_display=t_max,
                )
                st.pyplot(fig_ttt_tree)

                with st.expander("🔍 Inspect Any Minimax Node State"):
                    insp_t_id = st.selectbox("Select Node ID:", options=sorted(snap_t_nodes.keys()), format_func=lambda x: f"Node {x} (d={snap_t_nodes[x].depth} | {snap_t_nodes[x].player} | Score={snap_t_nodes[x].score})")
                    nt = snap_t_nodes[insp_t_id]
                    it1, it2 = st.columns([1, 2])
                    with it1:
                        st.markdown(_render_ttt_html_board(nt.state, outcome=nt.terminal_outcome), unsafe_allow_html=True)
                    with it2:
                        st.markdown(f"- **Node ID**: `{nt.node_id}` | **Parent ID**: `{nt.parent_id}`")
                        st.markdown(f"- **Player Level**: `{nt.player}` | **Status**: `{nt.status}`")
                        st.markdown(f"- **Evaluation Score**: `{nt.score}`")
                        st.markdown(f"- **Action**: `{nt.action}`")

            # Autoplay Advance
            if st.session_state.ttt_vis_playing:
                if cur_idx < total_steps - 1:
                    time.sleep(t_speed)
                    st.session_state.ttt_vis_step = cur_idx + 1
                    st.rerun()
                else:
                    st.session_state.ttt_vis_playing = False
        else:
            st.info("Click **🚀 Start / Record Minimax Trace** above to watch Minimax explore, evaluate, and propagate scores step by step.")


# =========================================================
# SECTION 8: ABOUT PROJECT
# =========================================================
elif selected_nav == "📘 About Project":
    st.markdown("## 📘 About Project 43: AI Search Analysis Dashboard")
    st.markdown(
        """
        ### 🎯 Project Overview
        This project provides an industry-quality, student-friendly **AI Search Analysis Dashboard**
        combining two fundamental AI game domains:
        1. **8-Puzzle**: Classic single-agent sliding tile puzzle solved via Depth-Limited Search (DLS), IDDFS, and A* (Manhattan).
        2. **Tic-Tac-Toe**: Two-player zero-sum adversarial game solved via Depth-Limited Minimax with domain heuristics.

        ---

        ### 🏗️ 5-Tier Modular Architecture
        - **`game.py`**: Tic-Tac-Toe game engine, 8 winning lines, $O(1)$ backtracking.
        - **`minimax.py`**: Depth-Limited Minimax algorithm with line threat and positional control heuristics.
        - **`puzzle.py`**: 8-Puzzle game engine, 3×3 grid, tile sliding, inversion parity check, and Manhattan distance.
        - **`puzzle_search.py`**: DLS with path-based cycle detection, IDDFS, and A* search.
        - **`analysis.py`**: Reusable search analysis module, multi-depth benchmarking, and CSV data export.
        - **`visualization.py`**: Matplotlib charting suite with dynamic Night Mode OLED, Deep Space, and Light styling.
        - **`app.py`**: Streamlit frontend controller with zero-dependency HTML table rendering.

        ---

        ### ⚙️ Time & Space Complexity Summary
        | Game / Search | Algorithm | Time Complexity | Space Complexity | Heuristic Function |
        | :--- | :--- | :--- | :--- | :--- |
        | **8-Puzzle** | Depth-Limited Search (DLS) | $\mathcal{O}(b^d), b \approx 3$ | $\mathcal{O}(d)$ | Path depth cutoff |
        | **8-Puzzle** | Iterative Deepening (IDDFS) | $\mathcal{O}(b^d)$ | $\mathcal{O}(d)$ | Depth cutoff |
        | **8-Puzzle** | A* Search | $\mathcal{O}(b^{d^*})$ | $\mathcal{O}(b^{d^*})$ | Manhattan Distance $h(n)$ |
        | **Tic-Tac-Toe** | Depth-Limited Minimax | $\mathcal{O}(b^d), b \le 9$ | $\mathcal{O}(d)$ | Line threats + Positional weights |

        ---

        ### 🎓 Engineering Highlights & Viva Defense
        1. **Zero-PyArrow Compatibility**: Bypasses Streamlit's native `st.dataframe` to run cleanly on Windows ARM64 without C++ compilation tools.
        2. **Accurate Telemetry**: Every node generated, node expanded, and cutoff node reflects live algorithm execution.
        3. **Cycle Prevention**: 8-Puzzle uses path-based cycle detection to prevent infinite loops in undirected state spaces.
        4. **Boundary Condition Understanding**: Cutoffs in Tic-Tac-Toe reach exactly 0 at Depth 9 because the maximum game depth is 9.
        """
    )
