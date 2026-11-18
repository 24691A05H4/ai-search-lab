"""
visualization.py - Matplotlib Data Visualizations
Creates charts for Search Depth vs Nodes, Cutoffs, Latency, and Multi-depth Benchmarks.
Supports both 8-Puzzle and Tic-Tac-Toe search analyses, plus Combined Comparisons.
Supports dynamic Night Mode (Midnight OLED), Deep Space Dark, and Academic Light themes.
"""

from typing import Dict, List, Optional, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np


# Palette constants
CYAN_ACCENT = "#00f2fe"
PURPLE_ACCENT = "#9d4edd"
MAGENTA_ACCENT = "#ff007f"
EMERALD_ACCENT = "#00f5d4"
AMBER_ACCENT = "#ffb703"


def _get_theme_palette(theme: str = "night") -> Dict[str, str]:
    """Returns color dictionary based on the selected visual theme."""
    if theme == "light":
        return {
            "bg": "#ffffff",
            "card_bg": "#f8fafc",
            "text": "#0f172a",
            "muted_text": "#475569",
            "grid": "#e2e8f0",
            "spine": "#cbd5e1",
            "primary": "#4f46e5",
            "secondary": "#7c3aed",
            "cyan": "#0284c7",
            "legend_bg": "#ffffff",
            "legend_border": "#cbd5e1",
        }
    elif theme == "deep_space":
        return {
            "bg": "#0d111c",
            "card_bg": "#151b2b",
            "text": "#f1f5f9",
            "muted_text": "#94a3b8",
            "grid": "#222d42",
            "spine": "#2d3748",
            "primary": CYAN_ACCENT,
            "secondary": PURPLE_ACCENT,
            "cyan": CYAN_ACCENT,
            "legend_bg": "#151b2b",
            "legend_border": "#334155",
        }
    else:  # "night" (Midnight OLED)
        return {
            "bg": "#04060c",
            "card_bg": "#0b0f1a",
            "text": "#f8fafc",
            "muted_text": "#94a3b8",
            "grid": "#161f30",
            "spine": "#1e293b",
            "primary": CYAN_ACCENT,
            "secondary": PURPLE_ACCENT,
            "cyan": CYAN_ACCENT,
            "legend_bg": "#0b0f1a",
            "legend_border": "#1e293b",
        }


def _apply_theme(fig, axes, theme: str = "night"):
    """Apply unified aesthetic to a matplotlib figure and axes."""
    p = _get_theme_palette(theme)
    fig.patch.set_facecolor(p["bg"])
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    else:
        axes = np.array(axes).flatten()
    for ax in axes:
        ax.set_facecolor(p["card_bg"])
        ax.tick_params(colors=p["muted_text"], labelsize=9)
        ax.xaxis.label.set_color(p["text"])
        ax.yaxis.label.set_color(p["text"])
        ax.title.set_color(p["text"])
        for spine in ax.spines.values():
            spine.set_color(p["spine"])
            spine.set_linewidth(1.1)
        ax.grid(True, linestyle="--", alpha=0.45 if theme == "light" else 0.35, color=p["grid"])


# =========================================================
# SINGLE GAME CHARTS (8-PUZZLE & TIC-TAC-TOE)
# =========================================================

def plot_depth_vs_nodes_bar(
    depths: List[int],
    nodes: List[int],
    title: str = "Search Depth vs Nodes Generated",
    theme: str = "night",
    bar_label: str = "Nodes Generated",
) -> plt.Figure:
    """Renders a bar chart of explored nodes per search depth."""
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=100)
    _apply_theme(fig, ax, theme=theme)
    p = _get_theme_palette(theme)

    x = np.arange(len(depths))
    colormap = plt.cm.cool if theme == "light" else plt.cm.plasma
    colors = colormap(np.linspace(0.3, 0.85, max(len(depths), 1)))

    bars = ax.bar(x, nodes, width=0.55, color=colors, edgecolor=p["cyan"], label=bar_label)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Search Depth", fontsize=9)
    ax.set_ylabel("Nodes Generated", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"d={d}" for d in depths])

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height):,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=p["text"],
            fontweight="bold",
        )

    fig.tight_layout()
    return fig


def plot_depth_vs_cutoffs(
    depths: List[int],
    cutoffs: List[int],
    terminals: Optional[List[int]] = None,
    title: str = "Search Depth vs Cutoff Nodes",
    theme: str = "night",
) -> plt.Figure:
    """Renders stacked bar chart comparing cutoff nodes against terminal/goal nodes."""
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=100)
    _apply_theme(fig, ax, theme=theme)
    p = _get_theme_palette(theme)

    x = np.arange(len(depths))
    w = 0.52

    cutoff_col = AMBER_ACCENT if theme == "light" else "#ffb703"
    term_col = "#059669" if theme == "light" else EMERALD_ACCENT

    ax.bar(x, cutoffs, width=w, label="Cutoff Nodes (Depth Limit)", color=cutoff_col, edgecolor=p["spine"])
    if terminals is not None:
        ax.bar(
            x,
            terminals,
            width=w,
            bottom=cutoffs,
            label="Terminal / Goal Nodes",
            color=term_col,
            edgecolor=p["spine"],
        )

    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Search Depth", fontsize=9)
    ax.set_ylabel("Node Count", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"d={d}" for d in depths])
    ax.legend(
        facecolor=p["legend_bg"],
        edgecolor=p["legend_border"],
        labelcolor=p["text"],
        fontsize=8.5,
    )

    fig.tight_layout()
    return fig


def plot_depth_vs_time(
    depths: List[int],
    times: List[float],
    title: str = "Search Depth vs Execution Time (ms)",
    theme: str = "night",
) -> plt.Figure:
    """Renders line chart showing execution time in milliseconds across search depths."""
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=100)
    _apply_theme(fig, ax, theme=theme)
    p = _get_theme_palette(theme)

    line_col = "#e11d48" if theme == "light" else MAGENTA_ACCENT
    ax.plot(depths, times, color=line_col, marker="o", linewidth=2.2, markersize=6, label="Execution Time (ms)")
    ax.fill_between(depths, times, color=line_col, alpha=0.18)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Search Depth", fontsize=9)
    ax.set_ylabel("Execution Time (ms)", fontsize=9)
    ax.set_xticks(depths)

    for d, t in zip(depths, times):
        ax.annotate(
            f"{t:.2f}ms",
            (d, t),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=line_col,
            fontweight="bold",
        )

    fig.tight_layout()
    return fig


def plot_depth_vs_nodes_line(
    indices: List[int],
    nodes: List[int],
    title: str = "Tree Growth: Nodes Explored Over Moves",
    theme: str = "night",
) -> plt.Figure:
    """Renders a line plot showing complexity variation across sequential moves."""
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=100)
    _apply_theme(fig, ax, theme=theme)
    p = _get_theme_palette(theme)

    ax.plot(
        indices,
        nodes,
        marker="o",
        linewidth=2.2,
        markersize=6,
        color=p["primary"],
        label="Nodes Generated",
    )
    ax.fill_between(indices, nodes, color=p["primary"], alpha=0.15)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Turn / Step Number", fontsize=9)
    ax.set_ylabel("Nodes Generated", fontsize=9)
    ax.set_xticks(indices)

    fig.tight_layout()
    return fig


# =========================================================
# 8-PUZZLE 3-PANEL DASHBOARD
# =========================================================

def plot_puzzle_comparison_dashboard(
    benchmark_results: List[Dict[str, Any]],
    theme: str = "night",
) -> plt.Figure:
    """
    Renders 3-panel dashboard for 8-Puzzle search analysis:
    1. Search Depth vs Nodes Generated & Expanded
    2. Search Depth vs Cutoff Nodes
    3. Search Depth vs Execution Time (ms)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), dpi=100)
    _apply_theme(fig, axes, theme=theme)
    p = _get_theme_palette(theme)

    depths = [r["search_depth"] for r in benchmark_results]
    nodes_gen = [r["nodes_generated"] for r in benchmark_results]
    nodes_exp = [r.get("nodes_expanded", 0) for r in benchmark_results]
    cutoffs = [r["cutoff_nodes"] for r in benchmark_results]
    times = [r["execution_time_ms"] for r in benchmark_results]

    x = np.arange(len(depths))
    w = 0.36

    # Panel 1: Nodes Generated & Expanded
    ax1 = axes[0]
    ax1.bar(x - w / 2, nodes_gen, width=w, label="Generated", color=p["primary"], edgecolor=p["spine"])
    ax1.bar(x + w / 2, nodes_exp, width=w, label="Expanded", color=p["secondary"], edgecolor=p["spine"])
    ax1.set_title("1. Depth vs Nodes (Generated / Expanded)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Depth", fontsize=9)
    ax1.set_ylabel("Nodes", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(d) for d in depths])
    ax1.legend(facecolor=p["legend_bg"], edgecolor=p["legend_border"], labelcolor=p["text"], fontsize=8)

    # Panel 2: Cutoffs
    ax2 = axes[1]
    cutoff_color = AMBER_ACCENT if theme == "light" else "#ffb703"
    ax2.bar(x, cutoffs, width=0.5, color=cutoff_color, edgecolor=p["spine"])
    ax2.set_title("2. Depth vs Cutoff Nodes", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Depth", fontsize=9)
    ax2.set_ylabel("Cutoff Nodes", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(d) for d in depths])
    for bar in ax2.patches:
        h = bar.get_height()
        ax2.annotate(f"{int(h):,}", (bar.get_x() + bar.get_width() / 2, h), xytext=(0, 2),
                     textcoords="offset points", ha="center", fontsize=7.5, color=p["text"])

    # Panel 3: Execution Time
    ax3 = axes[2]
    time_color = "#e11d48" if theme == "light" else MAGENTA_ACCENT
    ax3.plot(depths, times, color=time_color, marker="s", linewidth=2, markersize=5)
    ax3.fill_between(depths, times, color=time_color, alpha=0.18)
    ax3.set_title("3. Depth vs Execution Time (ms)", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Depth", fontsize=9)
    ax3.set_ylabel("Milliseconds", fontsize=9)
    ax3.set_xticks(depths)

    fig.tight_layout(pad=2.0)
    return fig

# =========================================================
# INTERACTIVE SEARCH TREE VISUALIZER (MATPLOTLIB)
# =========================================================

def plot_interactive_search_tree(
    nodes: Dict[int, Any],
    active_node_id: int,
    active_path: Optional[List[int]] = None,
    max_depth_visible: int = 3,
    theme: str = "night",
    max_nodes_display: int = 60,
) -> plt.Figure:
    """
    Renders an interactive hierarchical tree graph highlighting active exploration paths,
    MAX/MIN levels, terminal outcomes, cycle pruning, and depth cutoffs.
    """
    if active_path is None:
        active_path = [active_node_id]

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=100)
    _apply_theme(fig, ax, theme=theme)
    p = _get_theme_palette(theme)

    # Filter visible nodes up to max_depth_visible or on active path
    visible_node_ids = set()
    for nid, node in nodes.items():
        if node.depth <= max_depth_visible or nid in active_path:
            visible_node_ids.add(nid)
            # include parent if exists
            if node.parent_id is not None:
                visible_node_ids.add(node.parent_id)

    # If too many nodes, prioritize nodes on active path and root
    if len(visible_node_ids) > max_nodes_display:
        priority_ids = set(active_path)
        for nid in priority_ids.copy():
            if nid in nodes:
                priority_ids.update(nodes[nid].children_ids[:4])
        for nid, node in nodes.items():
            if node.depth <= 1:
                priority_ids.add(nid)
        visible_node_ids = {nid for nid in visible_node_ids if nid in priority_ids}

    # Group by depth
    depth_groups: Dict[int, List[int]] = {}
    for nid in visible_node_ids:
        d = nodes[nid].depth
        depth_groups.setdefault(d, []).append(nid)

    # Compute positions (X, Y)
    pos: Dict[int, Tuple[float, float]] = {}
    for d, nids in depth_groups.items():
        count = len(nids)
        for i, nid in enumerate(nids):
            # Centered horizontally
            x = (i - (count - 1) / 2.0) * (1.8 / max(count, 1)**0.5 if count > 4 else 1.5)
            y = -d * 1.5
            pos[nid] = (x, y)

    # Draw Edges
    for nid in visible_node_ids:
        node = nodes[nid]
        if node.parent_id is not None and node.parent_id in pos:
            px, py = pos[node.parent_id]
            cx, cy = pos[nid]
            is_active_edge = (node.parent_id in active_path and nid in active_path)
            edge_color = p["primary"] if is_active_edge else p["spine"]
            edge_w = 2.4 if is_active_edge else 1.0
            edge_alpha = 0.95 if is_active_edge else 0.4
            ax.plot([px, cx], [py, cy], color=edge_color, linewidth=edge_w, alpha=edge_alpha, zorder=1)

    # Draw Nodes
    active_path_set = set(active_path)
    for nid in visible_node_ids:
        node = nodes[nid]
        x, y = pos[nid]

        # Determine node color
        if node.status == "GOAL" or node.terminal_outcome == "O":
            face_col = "#059669" if theme == "light" else EMERALD_ACCENT
        elif node.terminal_outcome == "X":
            face_col = "#e11d48" if theme == "light" else MAGENTA_ACCENT
        elif node.terminal_outcome == "Draw":
            face_col = "#d97706" if theme == "light" else AMBER_ACCENT
        elif node.status == "CUTOFF":
            face_col = "#d97706" if theme == "light" else AMBER_ACCENT
        elif node.status == "PRUNED":
            face_col = "#475569"
        elif getattr(node, "player", None) == "MIN":
            face_col = p["secondary"]
        else:
            face_col = p["primary"]

        is_active = (nid == active_node_id)
        border_col = "#ffd700" if is_active else (p["text"] if nid in active_path_set else p["spine"])
        border_w = 3.0 if is_active else (1.8 if nid in active_path_set else 1.0)
        box_size = 0.38 if is_active else 0.32

        # Draw circle or box
        circle = plt.Circle((x, y), box_size, facecolor=face_col, edgecolor=border_col, linewidth=border_w, zorder=3)
        ax.add_patch(circle)

        # Label: Node ID
        score_str = f"\ns={node.score}" if node.score is not None else ""
        label_text = f"N{nid}{score_str}"
        text_color = "#04060c" if theme == "light" or face_col == EMERALD_ACCENT else "#ffffff"
        ax.text(x, y, label_text, ha="center", va="center", fontsize=7.5, fontweight="bold", color=text_color, zorder=4)

    # Axis formatting
    if pos:
        all_x = [p[0] for p in pos.values()]
        all_y = [p[1] for p in pos.values()]
        ax.set_xlim(min(all_x) - 1.2, max(all_x) + 1.2)
        ax.set_ylim(min(all_y) - 1.0, max(all_y) + 0.8)
    else:
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

    ax.set_title(f"Visual Search Tree (Active Node: N{active_node_id} | Depth Limit: {max_depth_visible})", fontsize=11, fontweight="bold", pad=12)
    ax.axis("off")

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Active Node', markerfacecolor='#ffd700', markersize=9),
        plt.Line2D([0], [0], marker='o', color='w', label='Goal / AI Win', markerfacecolor=EMERALD_ACCENT, markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Cutoff', markerfacecolor=AMBER_ACCENT, markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Cycle Pruned', markerfacecolor='#475569', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='MAX Level', markerfacecolor=CYAN_ACCENT, markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='MIN Level', markerfacecolor=PURPLE_ACCENT, markersize=8),
    ]
    ax.legend(handles=legend_elements, loc='upper right', facecolor=p["legend_bg"], edgecolor=p["legend_border"], labelcolor=p["text"], fontsize=8)

    fig.tight_layout()
    return fig



# =========================================================
# TIC-TAC-TOE 4-PANEL DASHBOARD
# =========================================================

def plot_multi_depth_comparison_dashboard(
    benchmark_results: List[Dict],
    theme: str = "night",
) -> plt.Figure:
    """Renders 4-subplot multi-depth benchmark comparison dashboard for Tic-Tac-Toe."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), dpi=100)
    _apply_theme(fig, axes, theme=theme)
    p = _get_theme_palette(theme)

    depths = [r["depth"] for r in benchmark_results]
    nodes = [r["nodes_generated"] for r in benchmark_results]
    cutoffs = [r["cutoff_nodes"] for r in benchmark_results]
    terminals = [r["terminal_nodes"] for r in benchmark_results]
    times = [r["execution_time_ms"] for r in benchmark_results]
    scores = [r["evaluation_score"] for r in benchmark_results]

    # Panel 1: Nodes Generated
    ax1 = axes[0, 0]
    ax1.bar(depths, nodes, color=p["primary"], width=0.55, edgecolor=p["cyan"])
    ax1.set_title("Total Nodes Generated by Depth", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Depth", fontsize=9)
    ax1.set_ylabel("Nodes", fontsize=9)
    ax1.set_xticks(depths)
    for d, n in zip(depths, nodes):
        ax1.annotate(f"{n:,}", (d, n), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.5, color=p["text"])

    # Panel 2: Cutoffs vs Terminals
    ax2 = axes[0, 1]
    w = 0.52
    c_color = AMBER_ACCENT if theme == "light" else "#ffb703"
    t_color = "#059669" if theme == "light" else EMERALD_ACCENT
    ax2.bar(depths, cutoffs, width=w, label="Heuristic Cutoffs", color=c_color)
    ax2.bar(depths, terminals, width=w, bottom=cutoffs, label="Terminal Leaves", color=t_color)
    ax2.set_title("Cutoff Nodes vs Terminal Nodes", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Depth", fontsize=9)
    ax2.set_ylabel("Count", fontsize=9)
    ax2.set_xticks(depths)
    ax2.legend(facecolor=p["legend_bg"], edgecolor=p["legend_border"], labelcolor=p["text"], fontsize=8)

    # Panel 3: Execution Time
    ax3 = axes[1, 0]
    lat_color = "#e11d48" if theme == "light" else MAGENTA_ACCENT
    ax3.plot(depths, times, color=lat_color, marker="s", linewidth=2, markersize=6, label="Latency (ms)")
    ax3.fill_between(depths, times, color=lat_color, alpha=0.2)
    ax3.set_title("Search Execution Time (ms)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Depth", fontsize=9)
    ax3.set_ylabel("Milliseconds", fontsize=9)
    ax3.set_xticks(depths)
    for d, t in zip(depths, times):
        ax3.annotate(f"{t:.1f}ms", (d, t), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7.5, color=lat_color)

    # Panel 4: Evaluation Score
    ax4 = axes[1, 1]
    pos_color = "#059669" if theme == "light" else EMERALD_ACCENT
    neg_color = "#e11d48" if theme == "light" else MAGENTA_ACCENT
    colors4 = [pos_color if s >= 0 else neg_color for s in scores]
    ax4.bar(depths, scores, color=colors4, width=0.55)
    ax4.axhline(0, color=p["muted_text"], linestyle="--", linewidth=1)
    ax4.set_title("Heuristic / Minimax Score", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Depth", fontsize=9)
    ax4.set_ylabel("Evaluation Score", fontsize=9)
    ax4.set_xticks(depths)

    fig.tight_layout(pad=2.0)
    return fig


# =========================================================
# INTERACTIVE SEARCH TREE VISUALIZER (MATPLOTLIB)
# =========================================================

def plot_interactive_search_tree(
    nodes: Dict[int, Any],
    active_node_id: int,
    active_path: Optional[List[int]] = None,
    max_depth_visible: int = 3,
    theme: str = "night",
    max_nodes_display: int = 60,
) -> plt.Figure:
    """
    Renders an interactive hierarchical tree graph highlighting active exploration paths,
    MAX/MIN levels, terminal outcomes, cycle pruning, and depth cutoffs.
    """
    if active_path is None:
        active_path = [active_node_id]

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=100)
    _apply_theme(fig, ax, theme=theme)
    p = _get_theme_palette(theme)

    # Filter visible nodes up to max_depth_visible or on active path
    visible_node_ids = set()
    for nid, node in nodes.items():
        if node.depth <= max_depth_visible or nid in active_path:
            visible_node_ids.add(nid)
            # include parent if exists
            if node.parent_id is not None:
                visible_node_ids.add(node.parent_id)

    # If too many nodes, prioritize nodes on active path and root
    if len(visible_node_ids) > max_nodes_display:
        priority_ids = set(active_path)
        for nid in priority_ids.copy():
            if nid in nodes:
                priority_ids.update(nodes[nid].children_ids[:4])
        for nid, node in nodes.items():
            if node.depth <= 1:
                priority_ids.add(nid)
        visible_node_ids = {nid for nid in visible_node_ids if nid in priority_ids}

    # Group by depth
    depth_groups: Dict[int, List[int]] = {}
    for nid in visible_node_ids:
        d = nodes[nid].depth
        depth_groups.setdefault(d, []).append(nid)

    # Compute positions (X, Y)
    pos: Dict[int, Tuple[float, float]] = {}
    for d, nids in depth_groups.items():
        count = len(nids)
        for i, nid in enumerate(nids):
            # Centered horizontally
            x = (i - (count - 1) / 2.0) * (1.8 / max(count, 1)**0.5 if count > 4 else 1.5)
            y = -d * 1.5
            pos[nid] = (x, y)

    # Draw Edges
    for nid in visible_node_ids:
        node = nodes[nid]
        if node.parent_id is not None and node.parent_id in pos:
            px, py = pos[node.parent_id]
            cx, cy = pos[nid]
            is_active_edge = (node.parent_id in active_path and nid in active_path)
            edge_color = p["primary"] if is_active_edge else p["spine"]
            edge_w = 2.4 if is_active_edge else 1.0
            edge_alpha = 0.95 if is_active_edge else 0.4
            ax.plot([px, cx], [py, cy], color=edge_color, linewidth=edge_w, alpha=edge_alpha, zorder=1)

    # Draw Nodes
    active_path_set = set(active_path)
    for nid in visible_node_ids:
        node = nodes[nid]
        x, y = pos[nid]

        # Determine node color
        if node.status == "GOAL" or node.terminal_outcome == "O":
            face_col = "#059669" if theme == "light" else EMERALD_ACCENT
        elif node.terminal_outcome == "X":
            face_col = "#e11d48" if theme == "light" else MAGENTA_ACCENT
        elif node.terminal_outcome == "Draw":
            face_col = "#d97706" if theme == "light" else AMBER_ACCENT
        elif node.status == "CUTOFF":
            face_col = "#d97706" if theme == "light" else AMBER_ACCENT
        elif node.status == "PRUNED":
            face_col = "#475569"
        elif getattr(node, "player", None) == "MIN":
            face_col = p["secondary"]
        else:
            face_col = p["primary"]

        is_active = (nid == active_node_id)
        border_col = "#ffd700" if is_active else (p["text"] if nid in active_path_set else p["spine"])
        border_w = 3.0 if is_active else (1.8 if nid in active_path_set else 1.0)
        box_size = 0.38 if is_active else 0.32

        # Draw circle or box
        circle = plt.Circle((x, y), box_size, facecolor=face_col, edgecolor=border_col, linewidth=border_w, zorder=3)
        ax.add_patch(circle)

        # Label: Node ID
        score_str = f"\ns={node.score}" if node.score is not None else ""
        label_text = f"N{nid}{score_str}"
        text_color = "#04060c" if theme == "light" or face_col == EMERALD_ACCENT else "#ffffff"
        ax.text(x, y, label_text, ha="center", va="center", fontsize=7.5, fontweight="bold", color=text_color, zorder=4)

    # Axis formatting
    if pos:
        all_x = [p[0] for p in pos.values()]
        all_y = [p[1] for p in pos.values()]
        ax.set_xlim(min(all_x) - 1.2, max(all_x) + 1.2)
        ax.set_ylim(min(all_y) - 1.0, max(all_y) + 0.8)
    else:
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

    ax.set_title(f"Visual Search Tree (Active Node: N{active_node_id} | Depth Limit: {max_depth_visible})", fontsize=11, fontweight="bold", pad=12)
    ax.axis("off")

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Active Node', markerfacecolor='#ffd700', markersize=9),
        plt.Line2D([0], [0], marker='o', color='w', label='Goal / AI Win', markerfacecolor=EMERALD_ACCENT, markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Cutoff', markerfacecolor=AMBER_ACCENT, markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Cycle Pruned', markerfacecolor='#475569', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='MAX Level', markerfacecolor=CYAN_ACCENT, markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='MIN Level', markerfacecolor=PURPLE_ACCENT, markersize=8),
    ]
    ax.legend(handles=legend_elements, loc='upper right', facecolor=p["legend_bg"], edgecolor=p["legend_border"], labelcolor=p["text"], fontsize=8)

    fig.tight_layout()
    return fig



def plot_candidate_move_scores(
    move_scores: Dict[int, int],
    theme: str = "night",
) -> Optional[plt.Figure]:
    """Renders a bar chart showing the AI's heuristic/minimax score for each candidate cell."""
    if not move_scores:
        return None

    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=100)
    _apply_theme(fig, ax, theme=theme)
    p = _get_theme_palette(theme)

    moves = list(move_scores.keys())
    scores = [move_scores[m] for m in moves]
    labels = [f"Cell {m}\n({m//3},{m%3})" for m in moves]

    best_color = "#059669" if theme == "light" else EMERALD_ACCENT
    other_color = p["secondary"]
    colors = [best_color if s == max(scores) else other_color for s in scores]
    bars = ax.bar(labels, scores, color=colors, width=0.5, edgecolor=p["cyan"])

    ax.axhline(0, color=p["muted_text"], linestyle="--", linewidth=0.9)
    ax.set_title("AI Candidate Move Evaluations", fontsize=11, fontweight="bold", pad=10)
    ax.set_ylabel("Minimax Score", fontsize=9)

    for bar, score in zip(bars, scores):
        ax.annotate(
            f"{score}",
            (bar.get_x() + bar.get_width() / 2, score),
            xytext=(0, 4 if score >= 0 else -12),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=p["text"],
            fontweight="bold",
        )

    fig.tight_layout()
    return fig


# =========================================================
# COMBINED COMPARISON CHARTS (8-PUZZLE VS TIC-TAC-TOE)
# =========================================================

def plot_combined_nodes_comparison(
    depths: List[int],
    puzzle_nodes: List[int],
    ttt_nodes: List[int],
    theme: str = "night",
) -> plt.Figure:
    """
    Renders grouped bar chart comparing node expansion of 8-Puzzle (DLS) vs Tic-Tac-Toe (Minimax).
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=100)
    _apply_theme(fig, ax, theme=theme)
    p = _get_theme_palette(theme)

    x = np.arange(len(depths))
    w = 0.38

    color_puzzle = p["cyan"]
    color_ttt = p["secondary"]

    ax.bar(x - w / 2, puzzle_nodes, width=w, label="8-Puzzle (DLS)", color=color_puzzle, edgecolor=p["spine"])
    ax.bar(x + w / 2, ttt_nodes, width=w, label="Tic-Tac-Toe (Minimax)", color=color_ttt, edgecolor=p["spine"])

    ax.set_title("Combined Comparison: Nodes Generated Across Depths", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Search Depth", fontsize=9)
    ax.set_ylabel("Nodes Generated", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"d={d}" for d in depths])
    ax.legend(facecolor=p["legend_bg"], edgecolor=p["legend_border"], labelcolor=p["text"], fontsize=9)

    fig.tight_layout()
    return fig


def plot_combined_cutoffs_comparison(
    depths: List[int],
    puzzle_cutoffs: List[int],
    ttt_cutoffs: List[int],
    theme: str = "night",
) -> plt.Figure:
    """
    Renders grouped bar chart comparing cutoff counts of 8-Puzzle vs Tic-Tac-Toe.
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=100)
    _apply_theme(fig, ax, theme=theme)
    p = _get_theme_palette(theme)

    x = np.arange(len(depths))
    w = 0.38

    col_p = AMBER_ACCENT if theme == "light" else "#ffb703"
    col_t = "#059669" if theme == "light" else EMERALD_ACCENT

    ax.bar(x - w / 2, puzzle_cutoffs, width=w, label="8-Puzzle Cutoffs", color=col_p, edgecolor=p["spine"])
    ax.bar(x + w / 2, ttt_cutoffs, width=w, label="Tic-Tac-Toe Cutoffs", color=col_t, edgecolor=p["spine"])

    ax.set_title("Combined Comparison: Cutoff Nodes Across Depths", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Search Depth", fontsize=9)
    ax.set_ylabel("Cutoff Nodes", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"d={d}" for d in depths])
    ax.legend(facecolor=p["legend_bg"], edgecolor=p["legend_border"], labelcolor=p["text"], fontsize=9)

    fig.tight_layout()
    return fig


def plot_combined_time_comparison(
    depths: List[int],
    puzzle_times: List[float],
    ttt_times: List[float],
    theme: str = "night",
) -> plt.Figure:
    """
    Renders dual line chart comparing execution latency of both algorithms across depths.
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=100)
    _apply_theme(fig, ax, theme=theme)
    p = _get_theme_palette(theme)

    c_p = p["cyan"]
    c_t = "#e11d48" if theme == "light" else MAGENTA_ACCENT

    ax.plot(depths, puzzle_times, color=c_p, marker="o", linewidth=2.2, label="8-Puzzle DLS Latency (ms)")
    ax.plot(depths, ttt_times, color=c_t, marker="s", linewidth=2.2, label="Tic-Tac-Toe Minimax Latency (ms)")

    ax.set_title("Combined Comparison: Execution Latency (ms)", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Search Depth", fontsize=9)
    ax.set_ylabel("Execution Time (ms)", fontsize=9)
    ax.set_xticks(depths)
    ax.legend(facecolor=p["legend_bg"], edgecolor=p["legend_border"], labelcolor=p["text"], fontsize=9)

    fig.tight_layout()
    return fig


def plot_combined_overview_dashboard(
    combined_data: Dict[str, Any],
    theme: str = "night",
) -> plt.Figure:
    """
    Renders comprehensive 3-panel comparative dashboard:
    1. Nodes Generated Comparison
    2. Cutoff Nodes Comparison
    3. Execution Time Comparison
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), dpi=100)
    _apply_theme(fig, axes, theme=theme)
    p = _get_theme_palette(theme)

    depths = combined_data["depths"]
    p_nodes = [r["nodes_generated"] for r in combined_data["puzzle"]]
    t_nodes = [r["nodes_generated"] for r in combined_data["tictactoe"]]
    p_cut = [r["cutoff_nodes"] for r in combined_data["puzzle"]]
    t_cut = [r["cutoff_nodes"] for r in combined_data["tictactoe"]]
    p_time = [r["execution_time_ms"] for r in combined_data["puzzle"]]
    t_time = [r["execution_time_ms"] for r in combined_data["tictactoe"]]

    x = np.arange(len(depths))
    w = 0.38

    # Panel 1: Nodes
    ax1 = axes[0]
    ax1.bar(x - w / 2, p_nodes, width=w, label="8-Puzzle", color=p["cyan"], edgecolor=p["spine"])
    ax1.bar(x + w / 2, t_nodes, width=w, label="Tic-Tac-Toe", color=p["secondary"], edgecolor=p["spine"])
    ax1.set_title("1. Nodes Generated (b^d vs Tree)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Depth", fontsize=9)
    ax1.set_ylabel("Nodes Generated", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(d) for d in depths])
    ax1.legend(facecolor=p["legend_bg"], edgecolor=p["legend_border"], labelcolor=p["text"], fontsize=8)

    # Panel 2: Cutoffs
    ax2 = axes[1]
    col_p = AMBER_ACCENT if theme == "light" else "#ffb703"
    col_t = "#059669" if theme == "light" else EMERALD_ACCENT
    ax2.bar(x - w / 2, p_cut, width=w, label="8-Puzzle", color=col_p, edgecolor=p["spine"])
    ax2.bar(x + w / 2, t_cut, width=w, label="Tic-Tac-Toe", color=col_t, edgecolor=p["spine"])
    ax2.set_title("2. Cutoff Nodes", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Depth", fontsize=9)
    ax2.set_ylabel("Cutoff Count", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(d) for d in depths])
    ax2.legend(facecolor=p["legend_bg"], edgecolor=p["legend_border"], labelcolor=p["text"], fontsize=8)

    # Panel 3: Time
    ax3 = axes[2]
    c_p = p["cyan"]
    c_t = "#e11d48" if theme == "light" else MAGENTA_ACCENT
    ax3.plot(depths, p_time, color=c_p, marker="o", linewidth=2, label="8-Puzzle")
    ax3.plot(depths, t_time, color=c_t, marker="s", linewidth=2, label="Tic-Tac-Toe")
    ax3.set_title("3. Execution Time (ms)", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Depth", fontsize=9)
    ax3.set_ylabel("Latency (ms)", fontsize=9)
    ax3.set_xticks(depths)
    ax3.legend(facecolor=p["legend_bg"], edgecolor=p["legend_border"], labelcolor=p["text"], fontsize=8)

    fig.tight_layout(pad=2.0)
    return fig


# =========================================================
# INTERACTIVE SEARCH TREE VISUALIZER (MATPLOTLIB)
# =========================================================

def plot_interactive_search_tree(
    nodes: Dict[int, Any],
    active_node_id: int,
    active_path: Optional[List[int]] = None,
    max_depth_visible: int = 3,
    theme: str = "night",
    max_nodes_display: int = 60,
) -> plt.Figure:
    """
    Renders an interactive hierarchical tree graph highlighting active exploration paths,
    MAX/MIN levels, terminal outcomes, cycle pruning, and depth cutoffs.
    """
    if active_path is None:
        active_path = [active_node_id]

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=100)
    _apply_theme(fig, ax, theme=theme)
    p = _get_theme_palette(theme)

    # Filter visible nodes up to max_depth_visible or on active path
    visible_node_ids = set()
    for nid, node in nodes.items():
        if node.depth <= max_depth_visible or nid in active_path:
            visible_node_ids.add(nid)
            # include parent if exists
            if node.parent_id is not None:
                visible_node_ids.add(node.parent_id)

    # If too many nodes, prioritize nodes on active path and root
    if len(visible_node_ids) > max_nodes_display:
        priority_ids = set(active_path)
        for nid in priority_ids.copy():
            if nid in nodes:
                priority_ids.update(nodes[nid].children_ids[:4])
        for nid, node in nodes.items():
            if node.depth <= 1:
                priority_ids.add(nid)
        visible_node_ids = {nid for nid in visible_node_ids if nid in priority_ids}

    # Group by depth
    depth_groups: Dict[int, List[int]] = {}
    for nid in visible_node_ids:
        d = nodes[nid].depth
        depth_groups.setdefault(d, []).append(nid)

    # Compute positions (X, Y)
    pos: Dict[int, Tuple[float, float]] = {}
    for d, nids in depth_groups.items():
        count = len(nids)
        for i, nid in enumerate(nids):
            # Centered horizontally
            x = (i - (count - 1) / 2.0) * (1.8 / max(count, 1)**0.5 if count > 4 else 1.5)
            y = -d * 1.5
            pos[nid] = (x, y)

    # Draw Edges
    for nid in visible_node_ids:
        node = nodes[nid]
        if node.parent_id is not None and node.parent_id in pos:
            px, py = pos[node.parent_id]
            cx, cy = pos[nid]
            is_active_edge = (node.parent_id in active_path and nid in active_path)
            edge_color = p["primary"] if is_active_edge else p["spine"]
            edge_w = 2.4 if is_active_edge else 1.0
            edge_alpha = 0.95 if is_active_edge else 0.4
            ax.plot([px, cx], [py, cy], color=edge_color, linewidth=edge_w, alpha=edge_alpha, zorder=1)

    # Draw Nodes
    active_path_set = set(active_path)
    for nid in visible_node_ids:
        node = nodes[nid]
        x, y = pos[nid]

        # Determine node color
        if node.status == "GOAL" or node.terminal_outcome == "O":
            face_col = "#059669" if theme == "light" else EMERALD_ACCENT
        elif node.terminal_outcome == "X":
            face_col = "#e11d48" if theme == "light" else MAGENTA_ACCENT
        elif node.terminal_outcome == "Draw":
            face_col = "#d97706" if theme == "light" else AMBER_ACCENT
        elif node.status == "CUTOFF":
            face_col = "#d97706" if theme == "light" else AMBER_ACCENT
        elif node.status == "PRUNED":
            face_col = "#475569"
        elif getattr(node, "player", None) == "MIN":
            face_col = p["secondary"]
        else:
            face_col = p["primary"]

        is_active = (nid == active_node_id)
        border_col = "#ffd700" if is_active else (p["text"] if nid in active_path_set else p["spine"])
        border_w = 3.0 if is_active else (1.8 if nid in active_path_set else 1.0)
        box_size = 0.38 if is_active else 0.32

        # Draw circle or box
        circle = plt.Circle((x, y), box_size, facecolor=face_col, edgecolor=border_col, linewidth=border_w, zorder=3)
        ax.add_patch(circle)

        # Label: Node ID
        score_str = f"\ns={node.score}" if node.score is not None else ""
        label_text = f"N{nid}{score_str}"
        text_color = "#04060c" if theme == "light" or face_col == EMERALD_ACCENT else "#ffffff"
        ax.text(x, y, label_text, ha="center", va="center", fontsize=7.5, fontweight="bold", color=text_color, zorder=4)

    # Axis formatting
    if pos:
        all_x = [p[0] for p in pos.values()]
        all_y = [p[1] for p in pos.values()]
        ax.set_xlim(min(all_x) - 1.2, max(all_x) + 1.2)
        ax.set_ylim(min(all_y) - 1.0, max(all_y) + 0.8)
    else:
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

    ax.set_title(f"Visual Search Tree (Active Node: N{active_node_id} | Depth Limit: {max_depth_visible})", fontsize=11, fontweight="bold", pad=12)
    ax.axis("off")

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Active Node', markerfacecolor='#ffd700', markersize=9),
        plt.Line2D([0], [0], marker='o', color='w', label='Goal / AI Win', markerfacecolor=EMERALD_ACCENT, markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Cutoff', markerfacecolor=AMBER_ACCENT, markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Cycle Pruned', markerfacecolor='#475569', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='MAX Level', markerfacecolor=CYAN_ACCENT, markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='MIN Level', markerfacecolor=PURPLE_ACCENT, markersize=8),
    ]
    ax.legend(handles=legend_elements, loc='upper right', facecolor=p["legend_bg"], edgecolor=p["legend_border"], labelcolor=p["text"], fontsize=8)

    fig.tight_layout()
    return fig

