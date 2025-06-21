"""
Command-line interface for SCM-Arena.

Provides commands to run Beer Game simulations, test models, and compare results.
"""

import click
import json
from typing import Dict, List
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from .beer_game.game import BeerGame
from .beer_game.agents import Position, SimpleAgent, RandomAgent, OptimalAgent
from .models.ollama_client import OllamaAgent, test_ollama_connection, create_ollama_agents
from .evaluation.scenarios import DEMAND_PATTERNS
from .visualization.plots import plot_game_analysis, create_game_summary_report

console = Console()


@click.group()
def main():
    """SCM-Arena: Supply Chain Management LLM Benchmark Platform"""
    pass


@main.command()
@click.option('--model', '-m', default='llama3.2', help='Ollama model name')
@click.option('--scenario', '-s', default='classic', help='Demand scenario', 
              type=click.Choice(['classic', 'random', 'shock', 'seasonal']))
@click.option('--rounds', '-r', default=20, help='Number of rounds to play')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--classic-mode', is_flag=True, help='Use 1960s settings (2-turn delays)')
@click.option('--neutral-prompts', is_flag=True, help='Use neutral prompts instead of position-specific')
@click.option('--memory', type=click.Choice(['none', 'short', 'medium', 'full']), default='short',
              help='Memory strategy: none=0, short=5, medium=10, full=all decisions')
@click.option('--plot', '-p', is_flag=True, help='Generate analysis plots')
@click.option('--save-analysis', help='Save complete analysis to directory')
def run(model: str, scenario: str, rounds: int, verbose: bool, classic_mode: bool, neutral_prompts: bool, memory: str, plot: bool, save_analysis: str):
    """Run a single Beer Game with Ollama agents"""
    
    # Check Ollama connection
    if not test_ollama_connection():
        console.print("[red]❌ Cannot connect to Ollama server at http://localhost:11434[/red]")
        console.print("Make sure Ollama is running: [cyan]ollama serve[/cyan]")
        return
    
    console.print(f"[green]✅ Connected to Ollama server[/green]")
    
    # Convert memory setting to window size
    memory_windows = {
        'none': 0,      # No history - pure reactive
        'short': 5,     # Last 5 decisions
        'medium': 10,   # Last 10 decisions  
        'full': None    # Complete history
    }
    memory_window = memory_windows[memory]
    
    # Create agents with specified settings
    try:
        agents = create_ollama_agents(
            model, 
            neutral_prompt=neutral_prompts,
            memory_window=memory_window
        )
        prompt_type = "Neutral" if neutral_prompts else "Position-specific"
        memory_desc = f"{memory} memory ({memory_window if memory_window is not None else 'all'} decisions)"
        console.print(f"[green]✅ Created agents: {model} ({prompt_type} prompts, {memory_desc})[/green]")
    except Exception as e:
        console.print(f"[red]❌ Failed to create agents: {e}[/red]")
        return
    
    # Get demand pattern
    demand_pattern = DEMAND_PATTERNS.get(scenario, DEMAND_PATTERNS['classic'])[:rounds]
    
    # Initialize game with appropriate settings
    if classic_mode:
        from .beer_game.game import create_classic_beer_game
        game = create_classic_beer_game(agents, demand_pattern)
        mode_text = "Classic 1960s (2-turn delays)"
    else:
        game = BeerGame(agents, demand_pattern)
        mode_text = "Modern (instant info, 1-turn shipping)"
    
    console.print(f"[blue]🎮 Starting Beer Game - Mode: {mode_text}[/blue]")
    console.print(f"[blue]📊 Scenario: {scenario}, Rounds: {rounds}, Memory: {memory}[/blue]")
    
    # Store history for analysis
    game_history = []
    
    # Run game with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Running simulation...", total=rounds)
        
        while not game.is_complete():
            state = game.step()
            game_history.append(state)
            progress.update(task, advance=1)
            
            if verbose:
                console.print(f"Round {state.round}: Cost=${state.total_cost:.2f}")
    
    # Get results
    results = game.get_results()
    
    # Display results with history
    display_results(results, game_history)
    
    # Agent names for analysis
    prompt_suffix = "_neutral" if neutral_prompts else "_specific"
    memory_suffix = f"_mem{memory}"
    agent_names = {pos: f"{model}_{pos.value}{prompt_suffix}{memory_suffix}" for pos in Position}
    
    # Generate plots if requested
    if plot:
        console.print("\n[blue]📊 Generating analysis plots...[/blue]")
        try:
            plot_game_analysis(results, game_history, show_plot=True)
        except Exception as e:
            console.print(f"[red]❌ Failed to generate plots: {e}[/red]")
    
    # Save complete analysis if requested
    if save_analysis:
        console.print(f"\n[blue]💾 Saving complete analysis to {save_analysis}/[/blue]")
        try:
            create_game_summary_report(results, game_history, agent_names, save_analysis)
        except Exception as e:
            console.print(f"[red]❌ Failed to save analysis: {e}[/red]")


@main.command()
@click.option('--models', '-m', multiple=True, default=['llama3.2'], help='Models to compare')
@click.option('--scenarios', '-s', multiple=True, default=['classic'], help='Scenarios to test')
@click.option('--runs', default=3, help='Number of runs per scenario')
@click.option('--rounds', '-r', default=20, help='Rounds per game')
def compare(models: tuple, scenarios: tuple, runs: int, rounds: int):
    """Compare multiple models across scenarios"""
    
    if not test_ollama_connection():
        console.print("[red]❌ Cannot connect to Ollama server[/red]")
        return
    
    results = {}
    total_games = len(models) * len(scenarios) * runs
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Running comparisons...", total=total_games)
        
        for model in models:
            results[model] = {}
            
            for scenario in scenarios:
                scenario_results = []
                demand_pattern = DEMAND_PATTERNS.get(scenario, DEMAND_PATTERNS['classic'])[:rounds]
                
                for run in range(runs):
                    try:
                        agents = create_ollama_agents(model)
                        game = BeerGame(agents, demand_pattern)
                        
                        while not game.is_complete():
                            game.step()
                        
                        game_results = game.get_results()
                        scenario_results.append(game_results.summary())
                        
                    except Exception as e:
                        console.print(f"[red]❌ Failed run {run+1} for {model} on {scenario}: {e}[/red]")
                    
                    progress.update(task, advance=1)
                
                results[model][scenario] = scenario_results
    
    # Display comparison table
    display_comparison(results)


@main.command()
@click.option('--model', '-m', default='llama3.2', help='Ollama model name')
def test_model(model: str):
    """Test a model with a simple scenario"""
    
    if not test_ollama_connection():
        console.print("[red]❌ Cannot connect to Ollama server[/red]")
        return
    
    console.print(f"[blue]🧪 Testing model: {model}[/blue]")
    
    try:
        # Create single agent for testing
        agent = OllamaAgent(Position.RETAILER, model)
        
        # Test with sample game state
        test_state = {
            "round": 5,
            "position": "retailer",
            "inventory": 8,
            "backlog": 2,
            "incoming_order": 6,
            "last_outgoing_order": 5,
            "round_cost": 12.0,
            "decision_history": [4, 5, 6, 5],
            "customer_demand": 6
        }
        
        console.print("[yellow]Sending test scenario to model...[/yellow]")
        decision = agent.make_decision(test_state)
        
        console.print(f"[green]✅ Model responded with order: {decision}[/green]")
        console.print(f"Test state: Inventory={test_state['inventory']}, Backlog={test_state['backlog']}, Demand={test_state['customer_demand']}")
        
    except Exception as e:
        console.print(f"[red]❌ Model test failed: {e}[/red]")


@main.command()
def list_models():
    """List available Ollama models"""
    
    if not test_ollama_connection():
        console.print("[red]❌ Cannot connect to Ollama server[/red]")
        return
    
    try:
        agent = OllamaAgent(Position.RETAILER, "dummy")
        models = agent.list_available_models()
        
        if models:
            console.print("[green]📋 Available Ollama models:[/green]")
            for model in models:
                console.print(f"  • {model}")
        else:
            console.print("[yellow]⚠️ No models found[/yellow]")
            
    except Exception as e:
        console.print(f"[red]❌ Failed to list models: {e}[/red]")


@main.command()
@click.option('--model', '-m', default='llama3.2', help='Ollama model name')
@click.option('--scenario', '-s', default='classic', help='Demand scenario')
@click.option('--rounds', '-r', default=20, help='Number of rounds')
@click.option('--runs', default=3, help='Number of runs per memory setting')
def memory_study(model: str, scenario: str, rounds: int, runs: int):
    """Compare all four memory strategies systematically"""
    
    if not test_ollama_connection():
        console.print("[red]❌ Cannot connect to Ollama server[/red]")
        return
    
    memory_strategies = {
        'none': 0,      # Pure reactive - no history
        'short': 5,     # Short-term memory (5 decisions) 
        'medium': 10,   # Medium-term memory (10 decisions)
        'full': None    # Full memory (all decisions)
    }
    
    demand_pattern = DEMAND_PATTERNS.get(scenario, DEMAND_PATTERNS['classic'])[:rounds]
    results = {}
    total_games = len(memory_strategies) * runs
    
    console.print(f"[blue]🧠 Memory Strategy Study - Model: {model}, Scenario: {scenario}[/blue]")
    console.print(f"[blue]📊 Testing {len(memory_strategies)} memory strategies × {runs} runs = {total_games} games[/blue]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Running memory study...", total=total_games)
        
        for memory_name, memory_window in memory_strategies.items():
            console.print(f"\n[yellow]Testing {memory_name} memory strategy (window={memory_window})...[/yellow]")
            strategy_results = []
            
            for run in range(runs):
                try:
                    # Create agents with specific memory window
                    agents = create_ollama_agents(
                        model_name=model,
                        memory_window=memory_window
                    )
                    
                    # Create and run game
                    game = BeerGame(agents, demand_pattern)
                    
                    while not game.is_complete():
                        game.step()
                    
                    # Get results
                    game_results = game.get_results()
                    strategy_results.append(game_results.summary())
                    
                    console.print(f"  Run {run+1}: Cost=${game_results.total_cost:.2f}, Service={game_results.service_level:.1%}")
                    
                except Exception as e:
                    console.print(f"[red]❌ Failed run {run+1} for {memory_name}: {e}[/red]")
                    # Continue with other runs
                
                progress.update(task, advance=1)
            
            results[memory_name] = strategy_results
            console.print(f"[green]✅ Completed {memory_name}: {len(strategy_results)}/{runs} successful runs[/green]")
    
    # Display memory comparison
    console.print(f"\n[blue]📊 Completed memory study with {sum(len(v) for v in results.values())} total successful runs[/blue]")
    display_memory_comparison(results)


def display_memory_comparison(results):
    """Display memory strategy comparison results"""
    
    console.print("\n[bold blue]🧠 Memory Strategy Comparison[/bold blue]")
    
    # Check if we have any results
    valid_results = {k: v for k, v in results.items() if v}  # Filter out empty results
    
    if not valid_results:
        console.print("[red]❌ No successful runs to compare[/red]")
        return
    
    table = Table()
    table.add_column("Memory Strategy", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Avg Cost", style="green")
    table.add_column("Avg Bullwhip", style="yellow")
    table.add_column("Avg Service Level", style="blue")
    table.add_column("Std Dev Cost", style="red")
    
    descriptions = {
        'none': "Pure reactive (0 decisions)",
        'short': "Short-term (5 decisions)", 
        'medium': "Medium-term (10 decisions)",
        'full': "Complete history (all decisions)"
    }
    
    # Sort results by average cost
    sorted_strategies = []
    for strategy, runs in valid_results.items():
        if runs:  # Double-check we have runs
            avg_cost = sum(r['total_cost'] for r in runs) / len(runs)
            sorted_strategies.append((avg_cost, strategy, runs))
    
    if not sorted_strategies:
        console.print("[red]❌ No valid strategy results to display[/red]")
        return
        
    sorted_strategies.sort()  # Sort by average cost
    
    for rank, (_, strategy, runs) in enumerate(sorted_strategies, 1):
        avg_cost = sum(r['total_cost'] for r in runs) / len(runs)
        avg_bullwhip = sum(r['bullwhip_ratio'] for r in runs) / len(runs)
        avg_service = sum(r['service_level'] for r in runs) / len(runs)
        
        # Calculate standard deviation
        costs = [r['total_cost'] for r in runs]
        if len(costs) > 1:
            std_cost = (sum((c - avg_cost) ** 2 for c in costs) / len(costs)) ** 0.5
        else:
            std_cost = 0.0
        
        rank_emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}"
        
        table.add_row(
            f"{rank_emoji} {strategy}",
            descriptions.get(strategy, ""),
            f"${avg_cost:.2f}",
            f"{avg_bullwhip:.2f}",
            f"{avg_service:.1%}",
            f"±${std_cost:.2f}"
        )
    
    console.print(table)
    
    # Summary insights only if we have multiple strategies
    if len(sorted_strategies) > 1:
        console.print("\n[bold]Memory Strategy Insights:[/bold]")
        best_strategy = sorted_strategies[0][1]  # Best by cost
        worst_strategy = sorted_strategies[-1][1]  # Worst by cost
        
        console.print(f"🎯 Best performing: {best_strategy} memory")
        console.print(f"📉 Worst performing: {worst_strategy} memory")
        
        # Show cost differences
        best_cost = sorted_strategies[0][0]
        worst_cost = sorted_strategies[-1][0]
        improvement = ((worst_cost - best_cost) / worst_cost) * 100
        
        console.print(f"💰 Performance gap: {improvement:.1f}% cost difference between best and worst")
    else:
        console.print(f"\n[yellow]⚠️ Only one strategy completed successfully: {sorted_strategies[0][1]}[/yellow]")


def display_results(results, history=None):
    """Display game results in a formatted table"""
    
    console.print("\n[bold blue]🏁 Game Results[/bold blue]")
    
    # Summary metrics
    summary = results.summary()
    
    table = Table(title="Performance Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Cost", f"${summary['total_cost']:.2f}")
    table.add_row("Cost per Round", f"${summary['cost_per_round']:.2f}")
    table.add_row("Bullwhip Ratio", f"{summary['bullwhip_ratio']:.2f}")
    table.add_row("Service Level", f"{summary['service_level']:.1%}")
    
    console.print(table)
    
    # Individual costs
    console.print("\n[bold]Individual Player Costs:[/bold]")
    cost_table = Table()
    cost_table.add_column("Position", style="cyan")
    cost_table.add_column("Cost", style="green")
    cost_table.add_column("% of Total", style="yellow")
    
    for position in Position:
        cost = summary[f"{position.value}_cost"]
        percentage = (cost / summary['total_cost']) * 100
        cost_table.add_row(
            position.value.title(),
            f"${cost:.2f}",
            f"{percentage:.1f}%"
        )
    
    console.print(cost_table)
    
    # Show agent decision patterns if history available
    if history:
        console.print("\n[bold]Agent Decision History (Last 10 Rounds):[/bold]")
        decision_table = Table()
        decision_table.add_column("Round", style="cyan")
        decision_table.add_column("Customer Demand", style="red")
        
        for pos in Position:
            decision_table.add_column(f"{pos.value.title()}", style="green")
        
        # Show last 10 rounds
        start_round = max(0, len(history) - 10)
        for i in range(start_round, len(history)):
            state = history[i]
            row = [str(state.round), str(state.customer_demand)]
            
            for pos in Position:
                player = state.players[pos]
                decision = player.outgoing_order
                # Show inventory in parentheses for context
                row.append(f"{decision} (inv:{player.inventory})")
            
            decision_table.add_row(*row)
        
        console.print(decision_table)


def display_comparison(results):
    """Display model comparison results"""
    
    console.print("\n[bold blue]📊 Model Comparison[/bold blue]")
    
    table = Table()
    table.add_column("Model", style="cyan")
    table.add_column("Scenario", style="magenta")
    table.add_column("Avg Cost", style="green")
    table.add_column("Avg Bullwhip", style="yellow")
    table.add_column("Avg Service Level", style="blue")
    
    for model, scenarios in results.items():
        for scenario, runs in scenarios.items():
            if runs:  # Only show if we have results
                avg_cost = sum(r['total_cost'] for r in runs) / len(runs)
                avg_bullwhip = sum(r['bullwhip_ratio'] for r in runs) / len(runs)
                avg_service = sum(r['service_level'] for r in runs) / len(runs)
                
                table.add_row(
                    model,
                    scenario,
                    f"${avg_cost:.2f}",
                    f"{avg_bullwhip:.2f}",
                    f"{avg_service:.1%}"
                )
    
    console.print(table)


def display_benchmark(results):
    """Display benchmark results"""
    
    console.print("\n[bold blue]🏆 Benchmark Results[/bold blue]")
    
    # Sort by total cost (lower is better)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['total_cost'])
    
    table = Table()
    table.add_column("Rank", style="bold")
    table.add_column("Agent Type", style="cyan")
    table.add_column("Total Cost", style="green")
    table.add_column("Bullwhip Ratio", style="yellow")
    table.add_column("Service Level", style="blue")
    
    for rank, (agent_name, summary) in enumerate(sorted_results, 1):
        rank_emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}"
        
        table.add_row(
            rank_emoji,
            agent_name,
            f"${summary['total_cost']:.2f}",
            f"{summary['bullwhip_ratio']:.2f}",
            f"{summary['service_level']:.1%}"
        )
    
    console.print(table)


if __name__ == "__main__":
    main()