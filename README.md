# 🧠 Project 43: Dual-Game AI Search Analysis Dashboard
### *8-Puzzle & Tic-Tac-Toe AI Search Analysis, Benchmarking & Step-by-Step Visualizer*
**Industry-Standard B.Tech Computer Science Engineering Capstone Project**

---

## 📌 1. Project Overview

The **Dual-Game AI Search Analysis Dashboard** is a comprehensive, empirical research, benchmarking, and algorithm visualization platform designed to demonstrate, analyze, and compare state-space search algorithms across two distinct game domains:

1. **Game 1: 8-Puzzle (Single-Agent Deterministic State Space)**
   - 3×3 sliding tile puzzle with numbers 1 to 8 and one empty space (`0`).
   - Algorithms: **Depth-Limited Search (DLS)**, **Iterative Deepening Search (IDDFS)**, and **A* Search (Manhattan Distance)**.
   - Key Feature: **Path-based cycle detection** to prevent infinite loops in cyclic graphs.
   - Solvability guarantee: Inversion parity checking to guarantee 100% solvable shuffles.

2. **Game 2: Tic-Tac-Toe (Two-Player Adversarial Game Space)**
   - 3×3 grid: Human ('X') vs AI ('O').
   - Algorithm: **Depth-Limited Minimax** with tactical line threat and positional control heuristics.
   - Demonstrates the transition from shallow heuristic evaluation up to exhaustive, unpruned game-tree exploration.

3. **Interactive Step-by-Step AI Search Visualizer**
   - Live step-by-step execution trace and animated playback for both **8-Puzzle DLS** and **Tic-Tac-Toe Minimax**.
   - Dynamic search tree rendering with node status coloring, active branch breadcrumbs, depth-level zoom, and a node state inspector.

---

## 🏗️ 2. Modular Architecture & Project Structure

```
ai_search_dashboard/
├── app.py                # Main Streamlit dashboard with 8 navigation sections & multi-theme engine
├── puzzle.py             # 8-Puzzle game engine, adjacency graph, inversion parity, heuristics
├── puzzle_search.py      # 8-Puzzle DLS, IDDFS, A* (Manhattan), and empirical benchmark runner
├── game.py               # Tic-Tac-Toe game engine, legal move checking, 8 winning lines, O(1) backtrack
├── minimax.py            # Depth-Limited Minimax algorithm & heuristic state evaluator
├── analysis.py           # Reusable search analysis module, multi-depth benchmarking & CSV export
├── visualization.py      # Matplotlib chart suite (8-Puzzle, Tic-Tac-Toe, Combined & Search Trees)
├── visualizer_engine.py  # Event-recording engine & step-by-step tree snapshot reconstructor
├── test_dual_games.py    # Automated test suite validating both games, benchmarks & charts
├── requirements.txt      # Project dependencies (Streamlit, Matplotlib, Pandas, NumPy)
└── README.md             # Complete technical documentation & viva defense guide
```

### Navigation Sections in `app.py`
1. **🏠 Home Dashboard** — Unified KPIs and quick overview for both games.
2. **🧩 8-Puzzle Game** — Playable 3×3 sliding tile board with Shuffle, Reset, and A*/DLS AI Solver playback.
3. **🔍 8-Puzzle Search Analysis** — Multi-depth DLS/IDDFS benchmarks, profiling graphs, and CSV export.
4. **🎮 Tic-Tac-Toe Game** — Interactive Human vs AI arena with live candidate move evaluations.
5. **📊 Tic-Tac-Toe Search Analysis** — Move inspection, Depths 1–9 benchmarking, move logs, and match history.
6. **⚖️ Combined Performance Comparison** — Side-by-side comparison of 8-Puzzle and Tic-Tac-Toe metrics.
7. **🔬 AI Algorithm Visualizer** — Interactive step-by-step search tree and board state visualizer for both games.
8. **📘 About Project** — Academic documentation, complexity summary, and viva highlights.

---

## 📊 3. Core Search Algorithms & Mathematical Formulations

### A. 8-Puzzle Search Algorithms

#### 1. Depth-Limited Search (DLS)
Explores branch paths up to depth $d_{\max}$. Prevents cyclic loops by maintaining an ancestor path set:
$$\text{Cutoff}(s, d) = \neg \text{Goal}(s) \land (d \ge d_{\max})$$
- **Nodes Generated**: Count of all candidate states produced.
- **Nodes Expanded**: Count of non-terminal, non-cutoff states whose successor moves are generated.
- **Cutoff Nodes**: Non-goal states where expansion is stopped solely due to the configured depth limit.

#### 2. Iterative Deepening Depth-First Search (IDDFS)
Combines the space efficiency of DFS with the completeness of BFS:
$$\text{Total Nodes} = \sum_{d=0}^{d_{\text{sol}}} (d_{\text{sol}} - d + 1) \cdot b^d$$

#### 3. Manhattan Distance Heuristic ($h_1$)
Sum of the vertical and horizontal grid distances of each non-blank tile from its target goal position:
$$h(s) = \sum_{i=1}^{8} |r_i - r_i^*| + |c_i - c_i^*|$$
Admissible ($h(s) \le h^*(s)$) and consistent, ensuring A* finds the optimal shortest path.

---

### B. Tic-Tac-Toe Depth-Limited Minimax

$$
V(s) = 
\begin{cases} 
\pm(100 - \text{depth}), & \text{if } \text{Terminal}(s) \text{ (Win/Loss)} \\
0, & \text{if } \text{Terminal}(s) \text{ (Draw)} \\
h(s), & \text{if } \text{Cutoff}(s, d) \\
\max_{a \in \text{Actions}(s)} V(\text{Result}(s, a)), & \text{if Player}(s) = \text{AI (O)} \\
\min_{a \in \text{Actions}(s)} V(\text{Result}(s, a)), & \text{if Player}(s) = \text{Human (X)}
\end{cases}
$$

**Heuristic Components at Cutoff**:
1. **Line Threats**: $+10$ for 2 AI pieces + 1 empty; $-10$ for 2 Human pieces + 1 empty.
2. **Positional Control**: Center cell participates in 4 winning lines ($\pm 3$); Corners participate in 3 lines ($\pm 2$); Edges participate in 2 lines ($\pm 1$).

---

## 🔬 4. Interactive Step-by-Step AI Search Visualizer (`visualizer_engine.py`)

Both search algorithms record genuine, live execution events into a unified `SearchVisualizerLog`:

* **Recorded Event Types**:
  * `NODE_GENERATED` — New successor state created
  * `NODE_SELECTED` — Active node picked for inspection
  * `NODE_EXPANDED` — Successor transitions generated
  * `NODE_EVALUATED` — Backtracked Minimax utility propagated to parent
  * `NODE_CUTOFF` — Depth limit reached without terminal/goal state
  * `NODE_TERMINAL` — Win, Loss, or Draw leaf detected
  * `NODE_PRUNED` — Duplicate ancestor state rejected via cycle detection
  * `GOAL_FOUND` — 8-Puzzle goal configuration reached
  * `BEST_MOVE_SELECTED` — Final optimal AI move chosen
* **Dynamic Tree Snapshots**: `get_nodes_snapshot_at_step(step_index)` reconstructs the search tree strictly up to the current playback step so nodes appear, expand, and receive scores authentically.
* **Interactive Controls**: `First`, `Prev`, `Play / Pause`, `Next`, `Reset`, configurable **Playback Speed** (`0.1s`–`1.5s`), **Timeline Scrubber**, **Tree Depth Zoom**, **Max Visible Nodes** pagination filter, and a **Node State Inspector**.

---

## ⚖️ 5. Combined Comparison & Algorithmic Insights

| Metric / Property | 8-Puzzle (Single-Agent DLS) | Tic-Tac-Toe (Adversarial Minimax) |
| :--- | :--- | :--- |
| **Branching Factor ($b$)** | $b \in [2, 4]$ (Average $b \approx 2.67$) | $b = 9 \rightarrow 8 \rightarrow 7 \rightarrow \dots$ |
| **State Space Graph** | Undirected graph containing **cycles** | Directed acyclic tree (DAG) |
| **Cycle Prevention** | Path-based cycle detection ($\mathcal{O}(d)$ hash set) | Inherent in turn progression (cells occupied) |
| **Depth Limit Impact** | Truncates deep solution paths | Limits forward tactical lookahead |
| **Cutoffs at Max Depth** | Depends on puzzle difficulty | Exactly **0** at Depth 9 (all branches terminate) |

---

## 🚀 6. How to Run & Test

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Automated Test Suite
```bash
python test_dual_games.py
```

### Step 3: Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🎨 7. Display Themes & Zero-PyArrow Reliability
- **🌙 Night Mode (Midnight OLED)**: Deep black `#030508` with neon cyan `#00f2fe` accents.
- **🌌 Deep Space**: Navy blue `#090d16` and neon purple `#9d4edd`.
- **☀️ Academic Day / Light**: High-contrast clean white background.
- **PyArrow-Free HTML Tables**: Custom HTML/CSS table generator (`render_styled_table`) completely eliminates `ModuleNotFoundError: No module named 'pyarrow'` on Windows ARM64.
