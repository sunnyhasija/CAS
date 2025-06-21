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
def run(model: str, scenario: str, rounds: int, verbose: bool):
    """Run a single Beer Game with Ollama agents"""
    
    # Check Ollama connection
    if not test_ollama_connection():
        console.print("[red]❌ Cannot connect to Ollama server at http://localhost:11434[/red]")
        console.print("Make sure Ollama is running: [cyan]ollama serve[/cyan]")
        return
    
    console.print(f"[green]✅ Connected to Ollama server[/green]")
    
    # Create agents
    try:
        agents = create_ollama_agents(model)
        console.print(f"[green]✅ Created agents using model: {model}[/green]")
    except Exception as e:
        console.print(f"[red]❌ Failed to create agents: {e}[/red]")
        return
    
    # Get demand pattern
    demand_pattern = DEMAND_PATTERNS.get(scenario, DEMAND_PATTERNS['classic'])[:rounds]
    
    # Initialize game
    game = BeerGame(agents, demand_pattern)
    console.print(f"[blue]🎮 Starting Beer Game - Scenario: {scenario}, Rounds: {rounds}[/blue]")
    
    # Run game with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Running simulation...", total=rounds)
        
        while not game.is_complete():
            state = game.step()
            progress.update(task, advance=1)
            
            if verbose:
                console.print(f"Round {state.round}: Cost=${state.total_cost:.2f}")
    
    # Display results
    results = game.get_results()
    display_results(results)


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
@click.option('--scenario', '-s', default='classic', help='Demand scenario')
@click.option('--rounds', '-r', default=20, help='Number of rounds')
def benchmark(scenario: str, rounds: int):
    """Run benchmark with multiple agent types"""
    
    demand_pattern = DEMAND_PATTERNS.get(scenario, DEMAND_PATTERNS['classic'])[:rounds]
    
    # Test different agent types
    agent_configs = [
        ("Simple", lambda pos: SimpleAgent(pos)),
        ("Optimal", lambda pos: OptimalAgent(pos)),
        ("Random", lambda pos: RandomAgent(pos)),
    ]
    
    # Add Ollama if available
    if test_ollama_connection():
        agent_configs.append(("Ollama-llama3.2", lambda pos: OllamaAgent(pos, "llama3.2")))
    
    results = {}
    
    console.print(f"[blue]🏆 Running benchmark - Scenario: {scenario}[/blue]")
    
    for agent_name, agent_factory in agent_configs:
        try:
            agents = {pos: agent_factory(pos) for pos in Position}
            game = BeerGame(agents, demand_pattern)
            
            while not game.is_complete():
                game.step()
            
            game_results = game.get_results()
            results[agent_name] = game_results.summary()
            console.print(f"[green]✅ Completed {agent_name}[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Failed {agent_name}: {e}[/red]")
    
    # Display benchmark results
    display_benchmark(results)


def display_results(results):
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