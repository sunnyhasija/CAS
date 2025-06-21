"""
Visualization module for Beer Game results.

Creates detailed plots showing:
- Inventory levels over time
- Costs breakdown by player
- Orders placed vs received
- Service levels and stockouts
- Agent decision patterns
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
from ..beer_game.game import GameResults, GameState
from ..beer_game.agents import Position
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")


def plot_game_analysis(
    results: GameResults, 
    history: List[GameState], 
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Create comprehensive analysis plots for a Beer Game.
    
    Args:
        results: Final game results
        history: Complete game history (round-by-round states)
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Beer Game Analysis - LLM Supply Chain Performance', fontsize=16, fontweight='bold')
    
    # Extract data from history
    rounds = list(range(1, len(history) + 1))
    
    # Data structures for plotting
    inventory_data = {pos: [] for pos in Position}
    cost_data = {pos: [] for pos in Position}
    order_data = {pos: [] for pos in Position}
    backlog_data = {pos: [] for pos in Position}
    decisions_data = {pos: [] for pos in Position}
    
    # Customer demand
    customer_demand = []
    
    for state in history:
        customer_demand.append(state.customer_demand)
        for pos in Position:
            player = state.players[pos]
            inventory_data[pos].append(player.inventory)
            cost_data[pos].append(player.cost)
            order_data[pos].append(player.incoming_order)
            backlog_data[pos].append(player.backlog)
            decisions_data[pos].append(player.outgoing_order)
    
    # Plot 1: Inventory Levels Over Time
    ax1 = axes[0, 0]
    for pos in Position:
        ax1.plot(rounds, inventory_data[pos], marker='o', label=pos.value.title(), linewidth=2)
    ax1.set_title('Inventory Levels Over Time')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Inventory Units')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Costs Per Round
    ax2 = axes[0, 1]
    for pos in Position:
        ax2.plot(rounds, cost_data[pos], marker='s', label=pos.value.title(), linewidth=2)
    ax2.set_title('Costs Per Round')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Cost ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Orders Flow (Demand Propagation)
    ax3 = axes[0, 2]
    ax3.plot(rounds, customer_demand, marker='*', label='Customer Demand', linewidth=3, markersize=8)
    for pos in Position:
        ax3.plot(rounds, decisions_data[pos], marker='o', label=f'{pos.value.title()} Orders', linewidth=2)
    ax3.set_title('Order Flow - Demand Propagation')
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Order Quantity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Backlog (Service Failures)
    ax4 = axes[1, 0]
    for pos in Position:
        ax4.plot(rounds, backlog_data[pos], marker='^', label=pos.value.title(), linewidth=2)
    ax4.set_title('Backlog (Unfulfilled Orders)')
    ax4.set_xlabel('Round')
    ax4.set_ylabel('Backlog Units')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Cumulative Costs
    ax5 = axes[1, 1]
    cumulative_costs = {pos: np.cumsum(cost_data[pos]) for pos in Position}
    for pos in Position:
        ax5.plot(rounds, cumulative_costs[pos], marker='d', label=pos.value.title(), linewidth=2)
    ax5.set_title('Cumulative Costs')
    ax5.set_xlabel('Round')
    ax5.set_ylabel('Cumulative Cost ($)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Performance Summary Bar Chart
    ax6 = axes[1, 2]
    positions = [pos.value.title() for pos in Position]
    total_costs = [results.individual_costs[pos] for pos in Position]
    bars = ax6.bar(positions, total_costs, alpha=0.7)
    ax6.set_title('Total Costs by Position')
    ax6.set_ylabel('Total Cost ($)')
    ax6.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, cost in zip(bars, total_costs):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'${cost:.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()


def plot_agent_decision_analysis(
    history: List[GameState], 
    agent_names: Dict[Position, str],
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Analyze and plot agent decision-making patterns.
    
    Args:
        history: Complete game history
        agent_names: Mapping of positions to agent names/types
        save_path: Optional path to save the plot  
        show_plot: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Agent Decision-Making Analysis', fontsize=16, fontweight='bold')
    
    rounds = list(range(1, len(history) + 1))
    
    # Extract decision data
    decisions = {pos: [] for pos in Position}
    incoming_orders = {pos: [] for pos in Position}
    inventory_levels = {pos: [] for pos in Position}
    
    for state in history:
        for pos in Position:
            player = state.players[pos]
            decisions[pos].append(player.outgoing_order)
            incoming_orders[pos].append(player.incoming_order)
            inventory_levels[pos].append(player.inventory)
    
    # Plot 1: Decision vs Incoming Order (Responsiveness)
    ax1 = axes[0, 0]
    for pos in Position:
        ax1.scatter(incoming_orders[pos], decisions[pos], 
                   label=f'{pos.value.title()} ({agent_names.get(pos, "Unknown")})', 
                   alpha=0.7, s=50)
    ax1.plot([0, max(max(incoming_orders[pos]) for pos in Position)], 
             [0, max(max(incoming_orders[pos]) for pos in Position)], 
             'k--', alpha=0.5, label='Perfect Response')
    ax1.set_title('Order Decision vs Incoming Demand')
    ax1.set_xlabel('Incoming Order')
    ax1.set_ylabel('Order Decision')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Decision Volatility
    ax2 = axes[0, 1]
    for pos in Position:
        if len(decisions[pos]) > 1:
            volatility = np.abs(np.diff(decisions[pos]))
            ax2.plot(rounds[1:], volatility, marker='o', 
                    label=f'{pos.value.title()}', linewidth=2)
    ax2.set_title('Order Decision Volatility')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('|Order Change|')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Inventory vs Decision Pattern
    ax3 = axes[1, 0]
    for pos in Position:
        ax3.scatter(inventory_levels[pos], decisions[pos], 
                   label=f'{pos.value.title()}', alpha=0.7, s=50)
    ax3.set_title('Order Decision vs Current Inventory')
    ax3.set_xlabel('Current Inventory')
    ax3.set_ylabel('Order Decision')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Decision Statistics
    ax4 = axes[1, 1]
    stats_data = []
    labels = []
    
    for pos in Position:
        decision_list = decisions[pos]
        stats_data.append([
            np.mean(decision_list),
            np.std(decision_list),
            max(decision_list),
            min(decision_list)
        ])
        labels.append(pos.value.title())
    
    stats_array = np.array(stats_data)
    x = np.arange(len(labels))
    width = 0.2
    
    ax4.bar(x - width*1.5, stats_array[:, 0], width, label='Mean', alpha=0.8)
    ax4.bar(x - width*0.5, stats_array[:, 1], width, label='Std Dev', alpha=0.8)
    ax4.bar(x + width*0.5, stats_array[:, 2], width, label='Max', alpha=0.8)
    ax4.bar(x + width*1.5, stats_array[:, 3], width, label='Min', alpha=0.8)
    
    ax4.set_title('Order Decision Statistics')
    ax4.set_ylabel('Order Quantity')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Agent analysis plot saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()


def plot_bullwhip_analysis(
    history: List[GameState],
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Analyze and visualize the bullwhip effect.
    
    Args:
        history: Complete game history
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Bullwhip Effect Analysis', fontsize=16, fontweight='bold')
    
    rounds = list(range(1, len(history) + 1))
    
    # Extract order data
    customer_demand = [state.customer_demand for state in history]
    orders = {pos: [state.players[pos].outgoing_order for state in history] for pos in Position}
    
    # Plot 1: Order Amplification
    ax1 = axes[0]
    ax1.plot(rounds, customer_demand, marker='*', label='Customer Demand', 
             linewidth=4, markersize=10, color='red')
    
    colors = ['blue', 'green', 'orange', 'purple']
    for i, pos in enumerate(Position):
        ax1.plot(rounds, orders[pos], marker='o', label=f'{pos.value.title()} Orders',
                linewidth=2, color=colors[i])
    
    ax1.set_title('Order Amplification Through Supply Chain')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Order Quantity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Variance Analysis
    ax2 = axes[1]
    variances = []
    positions = []
    
    # Add customer demand variance
    variances.append(np.var(customer_demand))
    positions.append('Customer\nDemand')
    
    # Add order variances
    for pos in Position:
        variances.append(np.var(orders[pos]))
        positions.append(pos.value.title())
    
    bars = ax2.bar(positions, variances, alpha=0.7)
    ax2.set_title('Order Variance by Supply Chain Level')
    ax2.set_ylabel('Order Variance')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, variance in zip(bars, variances):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{variance:.1f}', ha='center', va='bottom')
    
    # Calculate and display bullwhip ratios
    bullwhip_ratios = []
    base_variance = np.var(customer_demand)
    if base_variance > 0:
        for pos in Position:
            ratio = np.var(orders[pos]) / base_variance
            bullwhip_ratios.append(ratio)
    
    # Add text box with bullwhip ratios
    if bullwhip_ratios:
        textstr = 'Bullwhip Ratios:\n'
        for i, pos in enumerate(Position):
            textstr += f'{pos.value.title()}: {bullwhip_ratios[i]:.2f}\n'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Bullwhip analysis plot saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()


def create_game_summary_report(
    results: GameResults,
    history: List[GameState], 
    agent_names: Dict[Position, str],
    output_dir: str = "game_analysis"
) -> None:
    """
    Create a complete analysis report with all plots.
    
    Args:
        results: Final game results
        history: Complete game history
        agent_names: Agent names/types for each position
        output_dir: Directory to save all plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📊 Creating complete game analysis in {output_dir}/")
    
    # Create all plots
    plot_game_analysis(
        results, history, 
        save_path=f"{output_dir}/game_overview.png",
        show_plot=False
    )
    
    plot_agent_decision_analysis(
        history, agent_names,
        save_path=f"{output_dir}/agent_decisions.png", 
        show_plot=False
    )
    
    plot_bullwhip_analysis(
        history,
        save_path=f"{output_dir}/bullwhip_effect.png",
        show_plot=False
    )
    
    # Create summary statistics file
    with open(f"{output_dir}/summary_stats.txt", 'w') as f:
        f.write("BEER GAME ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        summary = results.summary()
        f.write(f"Total Cost: ${summary['total_cost']:.2f}\n")
        f.write(f"Cost per Round: ${summary['cost_per_round']:.2f}\n")
        f.write(f"Bullwhip Ratio: {summary['bullwhip_ratio']:.2f}\n")
        f.write(f"Service Level: {summary['service_level']:.1%}\n\n")
        
        f.write("INDIVIDUAL COSTS:\n")
        f.write("-" * 20 + "\n")
        for pos in Position:
            cost = summary[f"{pos.value}_cost"]
            percentage = (cost / summary['total_cost']) * 100
            agent_name = agent_names.get(pos, "Unknown")
            f.write(f"{pos.value.title():12} ({agent_name:12}): ${cost:6.2f} ({percentage:4.1f}%)\n")
        
        f.write(f"\nGAME SETTINGS:\n")
        f.write("-" * 15 + "\n")
        f.write(f"Rounds Played: {results.rounds_played}\n")
        f.write(f"Demand Pattern: {results.demand_pattern[:10]}{'...' if len(results.demand_pattern) > 10 else ''}\n")
    
    print(f"✅ Analysis complete! Check {output_dir}/ for all files.")