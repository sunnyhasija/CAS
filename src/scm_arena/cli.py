"""
Enhanced command-line interface for SCM-Arena with canonical LLM settings.

MAJOR UPDATE: Implements canonical benchmark settings (temperature=0.3, top_p=0.9)
for consistent, reproducible evaluation across all models and research groups.
"""

import click
import json
import itertools
import time
from typing import Dict, List, Tuple
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

from .beer_game.game import BeerGame, VisibilityLevel, create_classic_beer_game
from .beer_game.agents import Position, SimpleAgent, RandomAgent, OptimalAgent
from .models.ollama_client import OllamaAgent, test_ollama_connection, create_ollama_agents
from .evaluation.scenarios import DEMAND_PATTERNS

# Import visualization with try/except in case it's not available
try:
    from .visualization.plots import plot_game_analysis, create_game_summary_report
except ImportError:
    plot_game_analysis = None
    create_game_summary_report = None

# Import data capture with try/except
try:
    from .data_capture import ExperimentTracker
except ImportError:
    ExperimentTracker = None

console = Console()

# CANONICAL BENCHMARK SETTINGS
# These settings ensure consistent, reproducible evaluation across all models
CANONICAL_TEMPERATURE = 0.3    # Balanced decision-making (not too rigid, not too random)
CANONICAL_TOP_P = 0.9          # Standard nucleus sampling (industry default)
CANONICAL_TOP_K = 40           # Reasonable exploration window
CANONICAL_REPEAT_PENALTY = 1.1 # Slight anti-repetition bias


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
@click.option('--visibility', type=click.Choice(['local', 'adjacent', 'full']), 
              default='local', help='Information visibility level')
@click.option('--plot', '-p', is_flag=True, help='Generate analysis plots')
@click.option('--save-analysis', help='Save complete analysis to directory')
@click.option('--save-database', is_flag=True, help='Save detailed data to database')
@click.option('--db-path', default='scm_arena_experiments.db', help='Database file path')
def run(model: str, scenario: str, rounds: int, verbose: bool, classic_mode: bool, 
        neutral_prompts: bool, memory: str, visibility: str, plot: bool, save_analysis: str,
        save_database: bool, db_path: str):
    """Run a single Beer Game with specified conditions using canonical LLM settings"""
    
    # Check Ollama connection
    if not test_ollama_connection():
        console.print("[red]❌ Cannot connect to Ollama server at http://localhost:11434[/red]")
        console.print("Make sure Ollama is running: [cyan]ollama serve[/cyan]")
        return
    
    console.print(f"[green]✅ Connected to Ollama server[/green]")
    console.print(f"[blue]🎯 Using canonical settings: temp={CANONICAL_TEMPERATURE}, top_p={CANONICAL_TOP_P}[/blue]")
    
    # Convert parameters
    memory_windows = {'none': 0, 'short': 5, 'medium': 10, 'full': None}
    memory_window = memory_windows[memory]
    visibility_level = VisibilityLevel(visibility)
    
    # Create agents with canonical settings
    try:
        agents = create_ollama_agents(
            model, 
            neutral_prompt=neutral_prompts,
            memory_window=memory_window,
            temperature=CANONICAL_TEMPERATURE,
            top_p=CANONICAL_TOP_P,
            top_k=CANONICAL_TOP_K,
            repeat_penalty=CANONICAL_REPEAT_PENALTY
        )
        
        prompt_type = "Neutral" if neutral_prompts else "Position-specific"
        memory_desc = f"{memory} memory ({memory_window if memory_window is not None else 'all'} decisions)"
        console.print(f"[green]✅ Created agents: {model} ({prompt_type} prompts, {memory_desc}, {visibility} visibility)[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Failed to create agents: {e}[/red]")
        return
    
    # Get demand pattern
    demand_pattern = DEMAND_PATTERNS.get(scenario, DEMAND_PATTERNS['classic'])[:rounds]
    
    # Initialize game with appropriate settings
    if classic_mode:
        game = create_classic_beer_game(agents, demand_pattern)
        game.visibility_level = visibility_level  # Override visibility
        mode_text = "Classic 1960s (2-turn delays)"
    else:
        game = BeerGame(agents, demand_pattern, visibility_level=visibility_level)
        mode_text = "Modern (instant info, 1-turn shipping)"
    
    console.print(f"[blue]🎮 Starting Beer Game - Mode: {mode_text}[/blue]")
    console.print(f"[blue]📊 Scenario: {scenario}, Rounds: {rounds}, Memory: {memory}, Visibility: {visibility}[/blue]")
    
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
    
    # Display results
    display_results(results, game_history)
    
    # Generate plots if requested
    if plot and plot_game_analysis:
        console.print("\n[blue]📊 Generating analysis plots...[/blue]")
        try:
            plot_game_analysis(results, game_history, show_plot=True)
        except Exception as e:
            console.print(f"[red]❌ Failed to generate plots: {e}[/red]")
    elif plot and not plot_game_analysis:
        console.print("[yellow]⚠️ Visualization not available[/yellow]")
    
    # Save complete analysis if requested
    if save_analysis and create_game_summary_report:
        console.print(f"\n[blue]💾 Saving complete analysis to {save_analysis}/[/blue]")
        try:
            agent_names = {pos: f"{model}_{pos.value}_{prompt_type.lower()}_{memory}_{visibility}" 
                          for pos in Position}
            create_game_summary_report(results, game_history, agent_names, save_analysis)
        except Exception as e:
            console.print(f"[red]❌ Failed to save analysis: {e}[/red]")
    elif save_analysis and not create_game_summary_report:
        console.print("[yellow]⚠️ Analysis report generation not available[/yellow]")


@main.command()
@click.option('--models', '-m', multiple=True, default=['llama3.2'], help='Models to test')
@click.option('--memory', multiple=True, default=['none', 'short', 'full'], 
              type=click.Choice(['none', 'short', 'medium', 'full']), help='Memory strategies')
@click.option('--prompts', multiple=True, default=['specific', 'neutral'], 
              type=click.Choice(['specific', 'neutral']), help='Prompt types')
@click.option('--visibility', multiple=True, default=['local', 'full'], 
              type=click.Choice(['local', 'adjacent', 'full']), help='Visibility levels')
@click.option('--scenarios', multiple=True, default=['classic'], 
              type=click.Choice(['classic', 'random', 'shock', 'seasonal']), help='Scenarios to test')
@click.option('--game-modes', multiple=True, default=['modern'], 
              type=click.Choice(['modern', 'classic']), help='Game mode settings')
@click.option('--runs', default=3, help='Number of runs per condition')
@click.option('--rounds', default=20, help='Rounds per game')
@click.option('--save-results', help='Save results to CSV file')
@click.option('--save-database', is_flag=True, help='Save detailed data to database')
@click.option('--db-path', default='scm_arena_experiments.db', help='Database file path')
def experiment(models: tuple, memory: tuple, prompts: tuple, visibility: tuple, 
               scenarios: tuple, game_modes: tuple, runs: int, rounds: int, save_results: str,
               save_database: bool, db_path: str):
    """Run fully crossed experimental design with canonical LLM settings"""
    
    if not test_ollama_connection():
        console.print("[red]❌ Cannot connect to Ollama server[/red]")
        return
    
    console.print(f"[blue]🎯 Using SCM-Arena canonical settings for all experiments:[/blue]")
    console.print(f"[blue]   Temperature: {CANONICAL_TEMPERATURE} | Top_P: {CANONICAL_TOP_P} | Top_K: {CANONICAL_TOP_K}[/blue]")
    
    # Initialize data capture if requested
    tracker = None
    if save_database and ExperimentTracker:
        tracker = ExperimentTracker(db_path)
        console.print(f"[green]📊 Database tracking enabled: {db_path}[/green]")
    elif save_database and not ExperimentTracker:
        console.print("[yellow]⚠️ Database tracking requested but not available[/yellow]")
    
    # Convert prompt types
    prompt_settings = {'specific': False, 'neutral': True}
    memory_windows = {'none': 0, 'short': 5, 'medium': 10, 'full': None}
    
    # Generate all experimental conditions
    conditions = list(itertools.product(
        models, memory, prompts, visibility, scenarios, game_modes
    ))
    
    total_experiments = len(conditions) * runs
    
    console.print(Panel(
        f"""[bold blue]🧪 SCM-Arena Canonical Benchmark Study[/bold blue]
        
📊 Experimental Factors:
• Models: {len(models)} ({', '.join(models)})
• Memory: {len(memory)} ({', '.join(memory)})  
• Prompts: {len(prompts)} ({', '.join(prompts)})
• Visibility: {len(visibility)} ({', '.join(visibility)})
• Scenarios: {len(scenarios)} ({', '.join(scenarios)})
• Game Modes: {len(game_modes)} ({', '.join(game_modes)})

🎯 Total Conditions: {len(conditions)}
🔄 Runs per Condition: {runs}
📈 Total Experiments: {total_experiments}
⏱️  Estimated Time: {total_experiments * 2:.0f}-{total_experiments * 5:.0f} minutes

🎛️ Canonical LLM Settings:
• Temperature: {CANONICAL_TEMPERATURE}
• Top_P: {CANONICAL_TOP_P}
• Top_K: {CANONICAL_TOP_K}
• Repeat Penalty: {CANONICAL_REPEAT_PENALTY}""",
        title="Benchmark Configuration"
    ))
    
    if not click.confirm("Proceed with canonical benchmark run?"):
        return
    
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Running benchmark experiments...", total=total_experiments)
        
        for condition in conditions:
            model, mem, prompt_type, vis, scenario, game_mode = condition
            
            for run in range(runs):
                try:
                    # Set up condition
                    neutral_prompt = prompt_settings[prompt_type]
                    memory_window = memory_windows[mem]
                    visibility_level = VisibilityLevel(vis)
                    classic_mode = (game_mode == 'classic')
                    
                    # Start database tracking if enabled
                    if tracker:
                        experiment_id = tracker.start_experiment(
                            model_name=model,
                            memory_strategy=mem,
                            memory_window=memory_window,
                            prompt_type=prompt_type,
                            visibility_level=vis,
                            scenario=scenario,
                            game_mode=game_mode,
                            rounds=rounds,
                            run_number=run + 1,
                            # Add canonical settings to metadata
                            temperature=CANONICAL_TEMPERATURE,
                            top_p=CANONICAL_TOP_P,
                            top_k=CANONICAL_TOP_K,
                            repeat_penalty=CANONICAL_REPEAT_PENALTY
                        )
                    
                    # Create agents with canonical settings
                    agents = create_ollama_agents(
                        model, 
                        neutral_prompt=neutral_prompt,
                        memory_window=memory_window,
                        temperature=CANONICAL_TEMPERATURE,
                        top_p=CANONICAL_TOP_P,
                        top_k=CANONICAL_TOP_K,
                        repeat_penalty=CANONICAL_REPEAT_PENALTY
                    )
                    
                    # Create game
                    demand_pattern = DEMAND_PATTERNS.get(scenario, DEMAND_PATTERNS['classic'])[:rounds]
                    
                    if classic_mode:
                        game = create_classic_beer_game(agents, demand_pattern)
                        game.visibility_level = visibility_level
                    else:
                        game = BeerGame(agents, demand_pattern, visibility_level=visibility_level)
                    
                    # Run game with data capture
                    round_number = 0
                    while not game.is_complete():
                        round_number += 1
                        state = game.step()
                        
                        # Capture round data if tracking enabled
                        if tracker:
                            # Get agent interactions for this round
                            agent_interactions = []
                            for position, player in state.players.items():
                                agent = agents[position]
                                # Get last interaction data from agent
                                interaction = getattr(agent, '_last_interaction', {})
                                
                                agent_interactions.append({
                                    'position': position.value,
                                    'inventory': player.inventory,
                                    'backlog': player.backlog,
                                    'incoming_order': player.incoming_order,
                                    'outgoing_order': player.outgoing_order,
                                    'round_cost': player.period_cost,
                                    'total_cost': player.total_cost,
                                    'prompt': interaction.get('prompt', ''),
                                    'response': interaction.get('response', ''),
                                    'decision': interaction.get('decision', 0),
                                    'response_time_ms': interaction.get('response_time_ms', 0.0)
                                })
                            
                            # Create game state JSON
                            game_state_json = json.dumps({
                                "round": state.round,
                                "customer_demand": state.customer_demand,
                                "total_cost": state.total_cost,
                                "players": {pos.value: {
                                    "inventory": player.inventory,
                                    "backlog": player.backlog,
                                    "incoming_order": player.incoming_order,
                                    "outgoing_order": player.outgoing_order,
                                    "period_cost": player.period_cost,
                                    "total_cost": player.total_cost
                                } for pos, player in state.players.items()}
                            })
                            
                            # Track the round
                            tracker.track_round(
                                round_number=state.round,
                                customer_demand=state.customer_demand,
                                total_system_cost=state.total_cost,
                                total_system_inventory=sum(p.inventory for p in state.players.values()),
                                total_system_backlog=sum(p.backlog for p in state.players.values()),
                                agent_interactions=agent_interactions,
                                game_state_json=game_state_json
                            )
                    
                    # Collect results
                    game_results = game.get_results()
                    summary = game_results.summary()
                    
                    # Finish database tracking if enabled
                    if tracker:
                        tracker.finish_experiment(
                            total_cost=summary['total_cost'],
                            service_level=summary['service_level'],
                            bullwhip_ratio=summary['bullwhip_ratio']
                        )
                    
                    result = {
                        'model': model,
                        'memory': mem,
                        'memory_window': memory_window,
                        'prompt_type': prompt_type,
                        'visibility': vis,
                        'scenario': scenario,
                        'game_mode': game_mode,
                        'run': run + 1,
                        'rounds': rounds,
                        'temperature': CANONICAL_TEMPERATURE,
                        'top_p': CANONICAL_TOP_P,
                        'top_k': CANONICAL_TOP_K,
                        'repeat_penalty': CANONICAL_REPEAT_PENALTY,
                        **summary
                    }
                    
                    results.append(result)
                    
                    console.print(f"✅ {model}-{mem}-{prompt_type}-{vis}-{scenario}-{game_mode} Run {run+1}: Cost=${summary['total_cost']:.0f}")
                    
                except Exception as e:
                    console.print(f"[red]❌ Failed: {model}-{mem}-{prompt_type}-{vis}-{scenario}-{game_mode} Run {run+1}: {e}[/red]")
                    
                progress.update(task, advance=1)
    
    # Display experimental results
    display_experimental_results(results)
    
    # Save results if requested
    if save_results:
        save_experimental_results(results, save_results)
    
    # Close database tracker if used
    if tracker:
        tracker.close()
        console.print(f"[green]💾 Database saved with complete audit trail[/green]")


@main.command() 
@click.option('--model', '-m', default='llama3.2', help='Ollama model name')
@click.option('--scenario', '-s', default='classic', help='Demand scenario')
@click.option('--rounds', '-r', default=20, help='Number of rounds')
@click.option('--runs', default=3, help='Number of runs per condition')
def visibility_study(model: str, scenario: str, rounds: int, runs: int):
    """Compare all visibility levels systematically using canonical settings"""
    
    if not test_ollama_connection():
        console.print("[red]❌ Cannot connect to Ollama server[/red]")
        return
    
    console.print(f"[blue]🎯 Using canonical settings: temp={CANONICAL_TEMPERATURE}, top_p={CANONICAL_TOP_P}[/blue]")
    
    visibility_levels = ['local', 'adjacent', 'full']
    demand_pattern = DEMAND_PATTERNS.get(scenario, DEMAND_PATTERNS['classic'])[:rounds]
    results = {}
    
    console.print(f"[blue]👁️  Visibility Study - Model: {model}, Scenario: {scenario}[/blue]")
    console.print(f"[blue]📊 Testing {len(visibility_levels)} visibility levels × {runs} runs = {len(visibility_levels) * runs} games[/blue]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Running visibility study...", total=len(visibility_levels) * runs)
        
        for visibility in visibility_levels:
            console.print(f"\n[yellow]Testing {visibility} visibility...[/yellow]")
            visibility_results = []
            
            for run in range(runs):
                try:
                    # Create agents with canonical settings
                    agents = create_ollama_agents(
                        model_name=model,
                        temperature=CANONICAL_TEMPERATURE,
                        top_p=CANONICAL_TOP_P,
                        top_k=CANONICAL_TOP_K,
                        repeat_penalty=CANONICAL_REPEAT_PENALTY
                    )
                    
                    # Create game with visibility level
                    game = BeerGame(
                        agents, 
                        demand_pattern, 
                        visibility_level=VisibilityLevel(visibility)
                    )
                    
                    # Run game
                    while not game.is_complete():
                        game.step()
                    
                    # Get results
                    game_results = game.get_results()
                    visibility_results.append(game_results.summary())
                    
                    console.print(f"  Run {run+1}: Cost=${game_results.total_cost:.2f}, Service={game_results.service_level:.1%}")
                    
                except Exception as e:
                    console.print(f"[red]❌ Failed run {run+1} for {visibility}: {e}[/red]")
                
                progress.update(task, advance=1)
            
            results[visibility] = visibility_results
            console.print(f"[green]✅ Completed {visibility}: {len(visibility_results)}/{runs} successful runs[/green]")
    
    # Display visibility comparison
    display_visibility_comparison(results)


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


def display_experimental_results(results):
    """Display experimental results summary"""
    
    console.print("\n[bold blue]🧪 Canonical Benchmark Results Summary[/bold blue]")
    
    if not results:
        console.print("[red]No results to display[/red]")
        return
    
    console.print(f"\n📊 Results across {len(results)} experiments")
    
    # Basic summary statistics
    if results:
        avg_cost = sum(r['total_cost'] for r in results) / len(results)
        avg_service = sum(r['service_level'] for r in results) / len(results)
        console.print(f"Average cost: ${avg_cost:.2f}")
        console.print(f"Average service level: {avg_service:.1%}")
        
        # Count unique conditions tested
        unique_models = len(set(r['model'] for r in results))
        unique_memory = len(set(r['memory'] for r in results))  
        unique_visibility = len(set(r['visibility'] for r in results))
        
        console.print(f"🎯 Tested: {unique_models} models, {unique_memory} memory strategies, {unique_visibility} visibility levels")
        console.print(f"🎛️ All experiments used canonical settings: temp={CANONICAL_TEMPERATURE}, top_p={CANONICAL_TOP_P}")


def display_visibility_comparison(results):
    """Display visibility study comparison results"""
    
    console.print("\n[bold blue]👁️  Visibility Level Comparison[/bold blue]")
    
    # Check if we have any results
    valid_results = {k: v for k, v in results.items() if v}
    
    if not valid_results:
        console.print("[red]❌ No successful runs to compare[/red]")
        return
    
    table = Table()
    table.add_column("Visibility Level", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Avg Cost", style="green")
    table.add_column("Avg Service Level", style="blue")
    table.add_column("Avg Bullwhip", style="yellow")
    table.add_column("Std Dev Cost", style="red")
    
    descriptions = {
        'local': "Own state only (classic)",
        'adjacent': "Can see immediate neighbors (up/downstream)",
        'full': "Complete supply chain visibility"
    }
    
    # Sort results by average cost
    sorted_levels = []
    for level, runs in valid_results.items():
        if runs:
            avg_cost = sum(r['total_cost'] for r in runs) / len(runs)
            sorted_levels.append((avg_cost, level, runs))
    
    if not sorted_levels:
        console.print("[red]❌ No valid level results to display[/red]")
        return
        
    sorted_levels.sort()  # Sort by average cost
    
    for rank, (_, level, runs) in enumerate(sorted_levels, 1):
        avg_cost = sum(r['total_cost'] for r in runs) / len(runs)
        avg_service = sum(r['service_level'] for r in runs) / len(runs)
        avg_bullwhip = sum(r['bullwhip_ratio'] for r in runs) / len(runs)
        
        # Calculate standard deviation
        costs = [r['total_cost'] for r in runs]
        if len(costs) > 1:
            std_cost = (sum((c - avg_cost) ** 2 for c in costs) / len(costs)) ** 0.5
        else:
            std_cost = 0.0
        
        rank_emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}"
        
        table.add_row(
            f"{rank_emoji} {level}",
            descriptions.get(level, ""),
            f"${avg_cost:.2f}",
            f"{avg_service:.1%}",
            f"{avg_bullwhip:.2f}",
            f"±${std_cost:.2f}"
        )
    
    console.print(table)
    
    # Summary insights
    if len(sorted_levels) > 1:
        console.print("\n[bold]Visibility Impact Insights:[/bold]")
        best_level = sorted_levels[0][1]
        worst_level = sorted_levels[-1][1]
        
        console.print(f"🎯 Best visibility level: {best_level}")
        console.print(f"📉 Worst visibility level: {worst_level}")
        
        # Show cost differences
        best_cost = sorted_levels[0][0]
        worst_cost = sorted_levels[-1][0]
        improvement = ((worst_cost - best_cost) / worst_cost) * 100
        
        console.print(f"💰 Visibility impact: {improvement:.1f}% cost difference between best and worst")


def save_experimental_results(results, filename):
    """Save experimental results to CSV"""
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        console.print(f"[green]💾 Saved {len(results)} experimental results to {filename}[/green]")
    except ImportError:
        # Fallback to JSON if pandas not available
        import json
        with open(filename.replace('.csv', '.json'), 'w') as f:
            json.dump(results, f, indent=2)
        console.print(f"[green]💾 Saved {len(results)} experimental results to {filename.replace('.csv', '.json')}[/green]")


# Add the remaining CLI commands from the original (test-model, list-models, etc.)
@main.command()
@click.option('--model', '-m', default='llama3.2', help='Ollama model name')
def test_model(model: str):
    """Test a model with canonical settings"""
    
    if not test_ollama_connection():
        console.print("[red]❌ Cannot connect to Ollama server[/red]")
        return
    
    console.print(f"[blue]🧪 Testing model: {model} with canonical settings[/blue]")
    console.print(f"[blue]🎯 Settings: temp={CANONICAL_TEMPERATURE}, top_p={CANONICAL_TOP_P}[/blue]")
    
    try:
        # Create single agent for testing with canonical settings
        agent = OllamaAgent(
            Position.RETAILER, 
            model,
            temperature=CANONICAL_TEMPERATURE,
            top_p=CANONICAL_TOP_P,
            top_k=CANONICAL_TOP_K,
            repeat_penalty=CANONICAL_REPEAT_PENALTY
        )
        
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


if __name__ == "__main__":
    main()