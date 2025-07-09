"""
Enhanced command-line interface for SCM-Arena with complete baseline integration and concurrent execution.

FEATURES:
- Deterministic and non-deterministic seeding modes
- Full factorial experimental designs
- Canonical LLM settings for reproducibility
- Complete baseline agents comparison (Sterman, Newsvendor, Base-stock, etc.)
- Full factorial baseline study matching LLM experimental structure
- Database merging capabilities
- Comprehensive error handling and validation
- Concurrent execution with --runners flag for faster experiments
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import partial

from .beer_game.game import BeerGame, VisibilityLevel, create_classic_beer_game
from .beer_game.agents import Position, SimpleAgent, RandomAgent, OptimalAgent
from .models.ollama_client import OllamaAgent, test_ollama_connection, create_ollama_agents
from .evaluation.scenarios import DEMAND_PATTERNS

# FIXED: Import the seeding system
from .utils.seeding import ExperimentSeeder, get_seed_for_condition, DEFAULT_BASE_SEED

# Import baseline agents
from .beer_game.baseline_agents import (
    create_baseline_agent, create_baseline_agents_set, get_baseline_descriptions
)

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

# Thread-safe progress tracking
progress_lock = threading.Lock()

# CANONICAL BENCHMARK SETTINGS
CANONICAL_TEMPERATURE = 0.3    # Balanced decision-making (not too rigid, not too random)
CANONICAL_TOP_P = 0.9          # Standard nucleus sampling (industry default)
CANONICAL_TOP_K = 40           # Reasonable exploration window
CANONICAL_REPEAT_PENALTY = 1.1 # Slight anti-repetition bias


def run_single_experiment(experiment_params):
    """
    Run a single experiment - designed for concurrent execution.
    
    Returns:
        dict: Experiment result or None if failed
    """
    try:
        (condition, run_number, rounds, memory_windows, seeder, 
         save_database, tracker_class, db_path) = experiment_params
        
        model, mem, prompt_type, vis, scenario, game_mode = condition
        
        # Set up condition parameters
        prompt_settings = {'specific': False, 'neutral': True}
        neutral_prompt = prompt_settings[prompt_type]
        memory_window = memory_windows[mem]
        visibility_level = VisibilityLevel(vis)
        classic_mode = (game_mode == 'classic')
        
        # Generate deterministic seed for this condition
        seed = seeder.get_seed(
            model=model,
            memory=mem,
            prompt_type=prompt_type,
            visibility=vis,
            scenario=scenario,
            game_mode=game_mode,
            run_number=run_number
        )
        
        # Initialize thread-local database tracker if needed
        tracker = None
        if save_database and tracker_class:
            tracker = tracker_class(db_path)
            experiment_id = tracker.start_experiment(
                model_name=model,
                memory_strategy=mem,
                memory_window=memory_window,
                prompt_type=prompt_type,
                visibility_level=vis,
                scenario=scenario,
                game_mode=game_mode,
                rounds=rounds,
                run_number=run_number,
                temperature=CANONICAL_TEMPERATURE,
                top_p=CANONICAL_TOP_P,
                top_k=CANONICAL_TOP_K,
                repeat_penalty=CANONICAL_REPEAT_PENALTY,
                seed=seed,
                base_seed=seeder.base_seed,
                deterministic_seeding=seeder.deterministic
            )
        
        # Create agents with deterministic seed
        agents = create_ollama_agents(
            model, 
            neutral_prompt=neutral_prompt,
            memory_window=memory_window,
            temperature=CANONICAL_TEMPERATURE,
            top_p=CANONICAL_TOP_P,
            top_k=CANONICAL_TOP_K,
            repeat_penalty=CANONICAL_REPEAT_PENALTY,
            seed=seed
        )
        
        # Create game with potentially seeded scenarios
        if scenario == "random" and seeder.deterministic:
            from .evaluation.scenarios import generate_scenario_with_seed
            demand_pattern = generate_scenario_with_seed(scenario, rounds, seed=seed)
        else:
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
            tracker.close()  # Close connection in this thread
        
        result = {
            'model': model,
            'memory': mem,
            'memory_window': memory_window,
            'prompt_type': prompt_type,
            'visibility': vis,
            'scenario': scenario,
            'game_mode': game_mode,
            'run': run_number,
            'rounds': rounds,
            'temperature': CANONICAL_TEMPERATURE,
            'top_p': CANONICAL_TOP_P,
            'top_k': CANONICAL_TOP_K,
            'repeat_penalty': CANONICAL_REPEAT_PENALTY,
            'seed': seed,
            'base_seed': seeder.base_seed,
            'deterministic': seeder.deterministic,
            **summary
        }
        
        return result
        
    except Exception as e:
        # Return error information instead of None
        return {
            'error': str(e),
            'condition': condition,
            'run': run_number,
            'failed': True
        }


def run_single_baseline_experiment(experiment_params):
    """
    Run a single baseline experiment - designed for concurrent execution.
    """
    try:
        (condition, run_number, rounds, memory_windows, 
         save_database, tracker_class, db_path, experiment_count) = experiment_params
        
        agent_type, mem, prompt_type, vis, scenario, game_mode = condition
        
        # Set up experimental parameters
        memory_window = memory_windows[mem]
        classic_mode = (game_mode == 'classic')
        
        # Initialize thread-local database tracker if needed
        tracker = None
        if save_database and tracker_class:
            tracker = tracker_class(db_path)
            experiment_id = tracker.start_experiment(
                model_name=agent_type,  # Use agent type as "model"
                memory_strategy=mem,    # Store for consistency (though not used)
                memory_window=memory_window,
                prompt_type=prompt_type,  # Store for consistency (though not used)
                visibility_level=vis,    # Store for consistency (though not used)
                scenario=scenario,
                game_mode=game_mode,
                rounds=rounds,
                run_number=run_number,
                temperature=0.0,  # Not applicable for baseline agents
                top_p=0.0,
                top_k=0,
                repeat_penalty=0.0,
                seed=experiment_count,  # Use experiment count as seed for uniqueness
                base_seed=42,
                deterministic_seeding=False  # Not applicable for baseline agents
            )
        
        # Create baseline agents for all positions
        agents = {
            position: create_baseline_agent(agent_type, position)
            for position in Position
        }
        
        # Get demand pattern
        demand_pattern = DEMAND_PATTERNS.get(scenario, DEMAND_PATTERNS['classic'])[:rounds]
        
        # Create game
        if classic_mode:
            game = create_classic_beer_game(agents, demand_pattern)
            # NOTE: Baseline agents ignore visibility, but we store it for metadata
        else:
            game = BeerGame(agents, demand_pattern)
            # NOTE: Baseline agents ignore visibility, but we store it for metadata
        
        # Run game with data capture
        round_number = 0
        while not game.is_complete():
            round_number += 1
            state = game.step()
            
            # Capture round data if tracking enabled
            if tracker:
                agent_interactions = []
                for position, player in state.players.items():
                    agent_interactions.append({
                        'position': position.value,
                        'inventory': player.inventory,
                        'backlog': player.backlog,
                        'incoming_order': player.incoming_order,
                        'outgoing_order': player.outgoing_order,
                        'round_cost': player.period_cost,
                        'total_cost': player.total_cost,
                        'prompt': f'{agent_type}_baseline',  # Baseline identifier
                        'response': f'{agent_type}_decision',
                        'decision': player.outgoing_order,
                        'response_time_ms': 0.0
                    })
                
                # Create game state JSON with baseline metadata
                game_state_json = json.dumps({
                    "round": state.round,
                    "customer_demand": state.customer_demand,
                    "total_cost": state.total_cost,
                    "agent_type": agent_type,
                    "baseline_agent": True,
                    "memory_strategy": mem,  # Metadata only
                    "prompt_type": prompt_type,  # Metadata only  
                    "visibility_level": vis,  # Metadata only
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
        
        # Get results
        game_results = game.get_results()
        summary = game_results.summary()
        
        # Finish database tracking if enabled
        if tracker:
            tracker.finish_experiment(
                total_cost=summary['total_cost'],
                service_level=summary['service_level'],
                bullwhip_ratio=summary['bullwhip_ratio']
            )
            tracker.close()  # Close connection in this thread
        
        # Store result for optional CSV export
        result = {
            'agent_type': agent_type,
            'model_name': agent_type,  # For consistency with LLM data
            'memory_strategy': mem,
            'memory_window': memory_window,
            'prompt_type': prompt_type,
            'visibility_level': vis,
            'scenario': scenario,
            'game_mode': game_mode,
            'run': run_number,
            'rounds': rounds,
            'experiment_id': experiment_id if tracker else None,
            'baseline_agent': True,
            **summary
        }
        
        return result
        
    except Exception as e:
        return {
            'error': str(e),
            'condition': condition,
            'run': run_number,
            'failed': True
        }


def run_single_llm_vs_baseline_experiment(experiment_params):
    """
    Run a single LLM vs baseline experiment - designed for concurrent execution.
    """
    try:
        (agent_info, scenario, game_mode, run_number, rounds, memory_windows) = experiment_params
        
        agent_type = agent_info['type']
        
        if agent_type == 'llm':
            # LLM experiment
            model = agent_info['model']
            mem = agent_info['memory']
            vis = agent_info['visibility']
            memory_window = memory_windows[mem]
            visibility_level = VisibilityLevel(vis)
            classic_mode = (game_mode == 'classic')
            
            # Create LLM agents
            agents = create_ollama_agents(
                model,
                memory_window=memory_window,
                temperature=CANONICAL_TEMPERATURE,
                top_p=CANONICAL_TOP_P,
                top_k=CANONICAL_TOP_K,
                repeat_penalty=CANONICAL_REPEAT_PENALTY
            )
            
            # Get demand pattern
            demand_pattern = DEMAND_PATTERNS.get(scenario, DEMAND_PATTERNS['classic'])[:rounds]
            
            # Create game
            if classic_mode:
                game = create_classic_beer_game(agents, demand_pattern)
                game.visibility_level = visibility_level
            else:
                game = BeerGame(agents, demand_pattern, visibility_level=visibility_level)
            
            # Run game
            while not game.is_complete():
                game.step()
            
            # Get results
            game_results = game.get_results()
            summary = game_results.summary()
            
            # Store result
            result = {
                'agent_type': 'llm',
                'model': model,
                'memory': mem,
                'memory_window': memory_window,
                'visibility': vis,
                'scenario': scenario,
                'game_mode': game_mode,
                'run': run_number,
                'rounds': rounds,
                **summary
            }
            
        else:
            # Baseline experiment
            baseline_agent_type = agent_info['baseline_type']
            
            # Create baseline agents
            agents = {
                position: create_baseline_agent(baseline_agent_type, position)
                for position in Position
            }
            
            # Get demand pattern
            demand_pattern = DEMAND_PATTERNS.get(scenario, DEMAND_PATTERNS['classic'])[:rounds]
            
            # Create game
            classic_mode = (game_mode == 'classic')
            if classic_mode:
                game = create_classic_beer_game(agents, demand_pattern)
            else:
                game = BeerGame(agents, demand_pattern)
            
            # Run game
            while not game.is_complete():
                game.step()
            
            # Get results
            game_results = game.get_results()
            summary = game_results.summary()
            
            # Store result
            result = {
                'agent_type': baseline_agent_type,
                'model': 'baseline',
                'memory': 'baseline',
                'memory_window': None,
                'visibility': 'baseline',
                'scenario': scenario,
                'game_mode': game_mode,
                'run': run_number,
                'rounds': rounds,
                **summary
            }
        
        return result
        
    except Exception as e:
        return {
            'error': str(e),
            'agent_info': agent_info,
            'scenario': scenario,
            'game_mode': game_mode,
            'run': run_number,
            'failed': True
        }


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
@click.option('--base-seed', default=DEFAULT_BASE_SEED, help=f'Base seed for deterministic generation (default: {DEFAULT_BASE_SEED})')
@click.option('--deterministic/--fixed', default=True, help='Use deterministic seeding (default) vs fixed seed')
@click.option('--run-number', default=1, help='Run number for this condition (affects seed generation)')
@click.option('--plot', '-p', is_flag=True, help='Generate analysis plots')
@click.option('--save-analysis', help='Save complete analysis to directory')
@click.option('--save-database', is_flag=True, help='Save detailed data to database')
@click.option('--db-path', default='scm_arena_experiments.db', help='Database file path')
def run(model: str, scenario: str, rounds: int, verbose: bool, classic_mode: bool, 
        neutral_prompts: bool, memory: str, visibility: str, base_seed: int, 
        deterministic: bool, run_number: int, plot: bool, save_analysis: str, 
        save_database: bool, db_path: str):
    """Run a single Beer Game with specified conditions using deterministic seeding"""
    
    # Check Ollama connection
    if not test_ollama_connection():
        console.print("[red]❌ Cannot connect to Ollama server at http://localhost:11434[/red]")
        console.print("Make sure Ollama is running: [cyan]ollama serve[/cyan]")
        return
    
    # Initialize seeder
    seeder = ExperimentSeeder(base_seed=base_seed, deterministic=deterministic)
    
    # Generate seed for this specific condition
    prompt_type = "neutral" if neutral_prompts else "specific"
    game_mode = "classic" if classic_mode else "modern"
    
    seed = seeder.get_seed(
        model=model,
        memory=memory,
        prompt_type=prompt_type,
        visibility=visibility,
        scenario=scenario,
        game_mode=game_mode,
        run_number=run_number
    )
    
    console.print(f"[green]✅ Connected to Ollama server[/green]")
    console.print(f"[blue]🎯 Using canonical settings: temp={CANONICAL_TEMPERATURE}, top_p={CANONICAL_TOP_P}[/blue]")
    
    seeding_method = "deterministic" if deterministic else "randomized"
    console.print(f"[blue]🎲 {seeding_method.title()} seed: {seed} for {model}-{memory}-{prompt_type}-{visibility}-{scenario}-{game_mode}-run{run_number}[/blue]")
    
    # Convert parameters
    memory_windows = {'none': 0, 'short': 5, 'medium': 10, 'full': None}
    memory_window = memory_windows[memory]
    visibility_level = VisibilityLevel(visibility)
    
    # Create agents with determined seed
    try:
        agents = create_ollama_agents(
            model, 
            neutral_prompt=neutral_prompts,
            memory_window=memory_window,
            temperature=CANONICAL_TEMPERATURE,
            top_p=CANONICAL_TOP_P,
            top_k=CANONICAL_TOP_K,
            repeat_penalty=CANONICAL_REPEAT_PENALTY,
            seed=seed
        )
        
        memory_desc = f"{memory} memory ({memory_window if memory_window is not None else 'all'} decisions)"
        console.print(f"[green]✅ Created agents: {model} ({prompt_type} prompts, {memory_desc}, {visibility} visibility)[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Failed to create agents: {e}[/red]")
        return
    
    # Get demand pattern (potentially seeded for random scenarios)
    if scenario == "random" and deterministic:
        from .evaluation.scenarios import generate_scenario_with_seed
        demand_pattern = generate_scenario_with_seed(scenario, rounds, seed=seed)
    else:
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
    
    # Show seed info
    console.print(f"\n[cyan]🔄 Reproducibility: Use --base-seed {base_seed} --run-number {run_number} --{'deterministic' if deterministic else 'fixed'} to reproduce this exact result[/cyan]")
    
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
@click.option('--baseline-agents', '-b', multiple=True, 
              default=['sterman', 'newsvendor', 'basestock', 'reactive'],
              type=click.Choice(['sterman', 'newsvendor', 'basestock', 'reactive', 'movingavg']),
              help='Baseline agents to test')
@click.option('--scenarios', multiple=True, default=['classic', 'random'],
              type=click.Choice(['classic', 'random', 'shock', 'seasonal']),
              help='Scenarios to test')
@click.option('--game-modes', multiple=True, default=['modern'],
              type=click.Choice(['modern', 'classic']),
              help='Game modes to test')
@click.option('--runs', default=5, help='Runs per condition')
@click.option('--rounds', default=30, help='Rounds per game')
@click.option('--runners', default=1, help='Number of concurrent runners (default: 1, max recommended: 8)')
@click.option('--save-results', help='Save results to CSV file')
@click.option('--save-database', is_flag=True, help='Save to database')
@click.option('--db-path', default='baseline_study.db', help='Database path')
def baseline_study(baseline_agents: tuple, scenarios: tuple, game_modes: tuple,
                   runs: int, rounds: int, runners: int, save_results: str, 
                   save_database: bool, db_path: str):
    """
    Pure baseline agents study across scenarios and game modes with concurrent execution.
    
    Tests classical algorithms (no LLM) to establish performance benchmarks.
    Perfect for understanding algorithmic baselines before LLM comparison.
    """
    
    # Validate runners
    if runners < 1:
        runners = 1
    elif runners > 8:
        console.print(f"[yellow]⚠️ Limiting runners to 8 (requested: {runners})[/yellow]")
        runners = 8
    
    total_experiments = len(baseline_agents) * len(scenarios) * len(game_modes) * runs
    
    console.print(Panel(
        f"""[bold blue]🔬 Pure Baseline Agents Study (Concurrent)[/bold blue]

📊 Experimental Design:
• Baseline Agents: {len(baseline_agents)} ({', '.join(baseline_agents)})
• Scenarios: {len(scenarios)} ({', '.join(scenarios)})
• Game Modes: {len(game_modes)} ({', '.join(game_modes)})
• Runs per Condition: {runs}
• Rounds per Game: {rounds}

🎯 Total Experiments: {total_experiments}
⏱️  Sequential Time: {total_experiments * 0.5:.1f} minutes
🏃 Concurrent Time ({runners} runners): {(total_experiments * 0.5 / runners):.1f} minutes

This establishes algorithmic baselines across experimental conditions.""",
        title="Baseline Study Configuration"
    ))
    
    if not click.confirm("Proceed with concurrent baseline study?"):
        return
    
    # Initialize data capture
    tracker = None
    if save_database and ExperimentTracker:
        tracker = ExperimentTracker(db_path)
        console.print(f"[green]📊 Database tracking enabled: {db_path}[/green]")
    
    # Prepare experiment parameters
    experiment_params = []
    for scenario in scenarios:
        for game_mode in game_modes:
            for agent_type in baseline_agents:
                for run in range(runs):
                    params = (agent_type, scenario, game_mode, run + 1, rounds)
                    experiment_params.append(params)
    
    results = []
    failed_experiments = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task(f"Running baseline study with {runners} runners...", total=total_experiments)
        
        # Execute experiments concurrently
        with ThreadPoolExecutor(max_workers=runners) as executor:
            # Submit all experiments
            future_to_params = {}
            for params in experiment_params:
                agent_type, scenario, game_mode, run_number, rounds = params
                
                future = executor.submit(run_baseline_experiment_simple, 
                                       agent_type, scenario, game_mode, run_number, rounds,
                                       save_database, ExperimentTracker if save_database else None, db_path)
                future_to_params[future] = params
            
            # Collect results as they complete
            for future in as_completed(future_to_params):
                params = future_to_params[future]
                agent_type, scenario, game_mode, run_number, rounds = params
                
                try:
                    result = future.result()
                    
                    if result and not result.get('failed'):
                        results.append(result)
                        
                        with progress_lock:
                            progress.update(task, advance=1)
                            console.print(f"✅ {agent_type}-{scenario}-{game_mode} Run {run_number}: Cost=${result['total_cost']:.0f}")
                    else:
                        failed_experiments.append({
                            'agent_type': agent_type,
                            'scenario': scenario,
                            'game_mode': game_mode,
                            'run': run_number,
                            'error': result.get('error', 'Unknown error') if result else 'No result returned'
                        })
                        
                        with progress_lock:
                            progress.update(task, advance=1)
                            console.print(f"[red]❌ Failed: {agent_type}-{scenario}-{game_mode} Run {run_number}[/red]")
                            
                except Exception as e:
                    failed_experiments.append({
                        'agent_type': agent_type,
                        'scenario': scenario,
                        'game_mode': game_mode,
                        'run': run_number,
                        'error': str(e)
                    })
                    
                    with progress_lock:
                        progress.update(task, advance=1)
                        console.print(f"[red]❌ Exception: {agent_type}-{scenario}-{game_mode} Run {run_number}: {e}[/red]")
    
    # Display results
    console.print(f"\n[bold green]🎉 Concurrent Baseline Study Complete![/bold green]")
    console.print(f"✅ Successful experiments: {len(results)}")
    console.print(f"❌ Failed experiments: {len(failed_experiments)}")
    
    if results:
        display_baseline_study_results(results)
    
    # Save results if requested
    if save_results:
        save_experimental_results(results, save_results)
    
    # Close database tracker if used
    if tracker:
        tracker.close()
        console.print(f"[green]💾 Baseline study saved to database[/green]")


def run_baseline_experiment_simple(agent_type, scenario, game_mode, run_number, rounds,
                                 save_database, tracker_class, db_path):
    """Simple baseline experiment runner for concurrent execution"""
    try:
        # Start database tracking if enabled
        tracker = None
        if save_database and tracker_class:
            tracker = tracker_class(db_path)
            experiment_id = tracker.start_experiment(
                model_name=agent_type,
                memory_strategy="baseline",
                memory_window=None,
                prompt_type="baseline",
                visibility_level="local",
                scenario=scenario,
                game_mode=game_mode,
                rounds=rounds,
                run_number=run_number,
                temperature=0.0,
                top_p=0.0,
                top_k=0,
                repeat_penalty=0.0,
                seed=0,
                base_seed=0,
                deterministic_seeding=False
            )
        
        # Create baseline agents for all positions
        agents = {
            position: create_baseline_agent(agent_type, position)
            for position in Position
        }
        
        # Get demand pattern
        demand_pattern = DEMAND_PATTERNS.get(scenario, DEMAND_PATTERNS['classic'])[:rounds]
        
        # Create game
        classic_mode = (game_mode == 'classic')
        if classic_mode:
            game = create_classic_beer_game(agents, demand_pattern)
        else:
            game = BeerGame(agents, demand_pattern)
        
        # Run game with data capture
        round_number = 0
        while not game.is_complete():
            round_number += 1
            state = game.step()
            
            # Capture round data if tracking enabled
            if tracker:
                agent_interactions = []
                for position, player in state.players.items():
                    agent_interactions.append({
                        'position': position.value,
                        'inventory': player.inventory,
                        'backlog': player.backlog,
                        'incoming_order': player.incoming_order,
                        'outgoing_order': player.outgoing_order,
                        'round_cost': player.period_cost,
                        'total_cost': player.total_cost,
                        'prompt': '',
                        'response': f'{agent_type}_decision',
                        'decision': player.outgoing_order,
                        'response_time_ms': 0.0
                    })
                
                game_state_json = json.dumps({
                    "round": state.round,
                    "customer_demand": state.customer_demand,
                    "total_cost": state.total_cost,
                    "agent_type": agent_type,
                    "players": {pos.value: {
                        "inventory": player.inventory,
                        "backlog": player.backlog,
                        "incoming_order": player.incoming_order,
                        "outgoing_order": player.outgoing_order,
                        "period_cost": player.period_cost,
                        "total_cost": player.total_cost
                    } for pos, player in state.players.items()}
                })
                
                tracker.track_round(
                    round_number=state.round,
                    customer_demand=state.customer_demand,
                    total_system_cost=state.total_cost,
                    total_system_inventory=sum(p.inventory for p in state.players.values()),
                    total_system_backlog=sum(p.backlog for p in state.players.values()),
                    agent_interactions=agent_interactions,
                    game_state_json=game_state_json
                )
        
        # Get results
        game_results = game.get_results()
        summary = game_results.summary()
        
        # Finish database tracking if enabled
        if tracker:
            tracker.finish_experiment(
                total_cost=summary['total_cost'],
                service_level=summary['service_level'],
                bullwhip_ratio=summary['bullwhip_ratio']
            )
            tracker.close()
        
        # Store result
        result = {
            'agent_type': agent_type,
            'scenario': scenario,
            'game_mode': game_mode,
            'run': run_number,
            'rounds': rounds,
            **summary
        }
        
        return result
        
    except Exception as e:
        return {
            'error': str(e),
            'agent_type': agent_type,
            'scenario': scenario,
            'game_mode': game_mode,
            'run': run_number,
            'failed': True
        }


@main.command()
@click.option('--baseline-agents', '-b', multiple=True, 
              default=['sterman', 'newsvendor', 'basestock', 'reactive', 'movingavg'],
              type=click.Choice(['sterman', 'newsvendor', 'basestock', 'reactive', 'movingavg']),
              help='Baseline agents to test')
@click.option('--memory', multiple=True, default=['none', 'short', 'full'], 
              type=click.Choice(['none', 'short', 'medium', 'full']), 
              help='Memory strategies (for consistency with LLM study)')
@click.option('--prompts', multiple=True, default=['specific', 'neutral'], 
              type=click.Choice(['specific', 'neutral']), 
              help='Prompt types (for consistency with LLM study)')
@click.option('--visibility', multiple=True, default=['local', 'adjacent', 'full'], 
              type=click.Choice(['local', 'adjacent', 'full']), 
              help='Visibility levels (for consistency with LLM study)')
@click.option('--scenarios', multiple=True, default=['classic', 'random', 'shock'], 
              type=click.Choice(['classic', 'random', 'shock', 'seasonal']), 
              help='Scenarios to test')
@click.option('--game-modes', multiple=True, default=['modern', 'classic'], 
              type=click.Choice(['modern', 'classic']), 
              help='Game modes to test')
@click.option('--runs', default=20, help='Number of runs per condition (default: 20)')
@click.option('--rounds', default=52, help='Number of rounds per game (default: 52)')
@click.option('--runners', default=1, help='Number of concurrent runners (default: 1, max recommended: 8)')
@click.option('--db-path', default='baseline_full_factorial.db', help='Database file path')
@click.option('--save-results', help='Optional CSV export file')
def full_factorial_baseline(baseline_agents: tuple, memory: tuple, prompts: tuple, 
                           visibility: tuple, scenarios: tuple, game_modes: tuple,
                           runs: int, rounds: int, runners: int, db_path: str, save_results: str):
    """
    Run full factorial baseline study to match LLM experimental design with concurrent execution.
    
    Runs baseline agents across all experimental dimensions with same structure
    as LLM study for direct comparison and database merging.
    
    NOTE: Baseline agents ignore memory/prompt variations but data is structured
    consistently for analysis and merging with LLM results.
    """
    
    # Validate runners
    if runners < 1:
        runners = 1
    elif runners > 8:
        console.print(f"[yellow]⚠️ Limiting runners to 8 (requested: {runners})[/yellow]")
        runners = 8
    
    # Calculate total experiment scope
    total_conditions = len(baseline_agents) * len(memory) * len(prompts) * len(visibility) * len(scenarios) * len(game_modes)
    total_experiments = total_conditions * runs
    
    console.print(Panel(
        f"""[bold blue]🧪 Full Factorial Baseline Study (Concurrent)[/bold blue]

📊 Experimental Design (matching LLM study structure):
• Baseline Agents: {len(baseline_agents)} ({', '.join(baseline_agents)})
• Memory Strategies: {len(memory)} ({', '.join(memory)}) [metadata only]
• Prompt Types: {len(prompts)} ({', '.join(prompts)}) [metadata only]  
• Visibility Levels: {len(visibility)} ({', '.join(visibility)}) [metadata only]
• Scenarios: {len(scenarios)} ({', '.join(scenarios)})
• Game Modes: {len(game_modes)} ({', '.join(game_modes)})

🎯 Experimental Scope:
• Conditions per Agent: {len(memory) * len(prompts) * len(visibility) * len(scenarios) * len(game_modes)}
• Total Agent Types: {len(baseline_agents)}
• Runs per Condition: {runs}
• Rounds per Game: {rounds}
• Total Experiments: {total_experiments}

⏱️  Sequential Time: {total_experiments * 0.5:.0f}-{total_experiments * 1:.0f} minutes
🏃 Concurrent Time ({runners} runners): {(total_experiments * 0.5 / runners):.0f}-{(total_experiments * 1 / runners):.0f} minutes

💾 Database: {db_path}
🔄 Structure matches LLM study for easy merging
📈 Ready for direct baseline vs LLM comparison""",
        title="Full Factorial Baseline Configuration"
    ))
    
    console.print(f"\n[yellow]⚠️  Note: Baseline agents don't use memory/prompts but data structure matches LLM study[/yellow]")
    
    if not click.confirm("Proceed with concurrent full factorial baseline study?"):
        return
    
    # Initialize database tracking
    if not ExperimentTracker:
        console.print("[red]❌ Database tracking not available[/red]")
        return
    
    # Memory window mapping (for metadata consistency)
    memory_windows = {'none': 0, 'short': 5, 'medium': 10, 'full': None}
    
    # Generate all experimental conditions
    conditions = list(itertools.product(
        baseline_agents, memory, prompts, visibility, scenarios, game_modes
    ))
    
    # Prepare experiment parameters
    experiment_params = []
    experiment_count = 0
    for condition in conditions:
        for run in range(runs):
            experiment_count += 1
            params = (
                condition, 
                run + 1, 
                rounds, 
                memory_windows, 
                True,  # save_database
                ExperimentTracker, 
                db_path,
                experiment_count
            )
            experiment_params.append(params)
    
    results = []
    failed_experiments = []
    
    console.print(f"[green]📊 Database tracking enabled: {db_path}[/green]")
    console.print(f"[blue]🏃 Using {runners} concurrent runner{'s' if runners > 1 else ''}[/blue]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task(f"Running {total_experiments} baseline experiments with {runners} runners...", total=total_experiments)
        
        # Execute experiments concurrently
        with ThreadPoolExecutor(max_workers=runners) as executor:
            # Submit all experiments
            future_to_params = {
                executor.submit(run_single_baseline_experiment, params): params 
                for params in experiment_params
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_params):
                params = future_to_params[future]
                condition, run_number = params[0], params[1]
                
                try:
                    result = future.result()
                    
                    if result and not result.get('failed'):
                        results.append(result)
                        
                        # Thread-safe progress update
                        with progress_lock:
                            progress.update(task, advance=1)
                            agent_type = condition[0]
                            console.print(f"✅ {agent_type} Run {run_number}: Cost=${result['total_cost']:.0f}")
                    else:
                        failed_experiments.append({
                            'condition': condition,
                            'run': run_number,
                            'error': result.get('error', 'Unknown error') if result else 'No result returned'
                        })
                        
                        with progress_lock:
                            progress.update(task, advance=1)
                            console.print(f"[red]❌ Failed: {condition} Run {run_number}[/red]")
                            
                except Exception as e:
                    failed_experiments.append({
                        'condition': condition,
                        'run': run_number,
                        'error': str(e)
                    })
                    
                    with progress_lock:
                        progress.update(task, advance=1)
                        console.print(f"[red]❌ Exception: {condition} Run {run_number}: {e}[/red]")
    
    # Display completion summary
    console.print(f"\n[bold green]🎉 Concurrent Full Factorial Baseline Study Complete![/bold green]")
    console.print(f"✅ Successful experiments: {len(results)}")
    console.print(f"❌ Failed experiments: {len(failed_experiments)}")
    console.print(f"💾 Database: {db_path}")
    
    # Quick performance summary
    if results:
        agent_performance = {}
        for result in results:
            agent = result['agent_type']
            if agent not in agent_performance:
                agent_performance[agent] = []
            agent_performance[agent].append(result['total_cost'])
        
        console.print(f"\n📊 Quick Performance Summary:")
        for agent, costs in sorted(agent_performance.items(), key=lambda x: sum(x[1])/len(x[1])):
            avg_cost = sum(costs) / len(costs)
            console.print(f"  {agent}: ${avg_cost:.0f} avg cost ({len(costs)} experiments)")
    
    # Save CSV if requested
    if save_results:
        save_experimental_results(results, save_results)
    
    console.print(f"[green]💾 Full factorial baseline study saved to {db_path}[/green]")
    
    # Database merging instructions
    console.print(f"\n[blue]🔗 To merge with LLM database:[/blue]")
    console.print(f"[cyan]poetry run python -m scm_arena.cli merge-databases \\[/cyan]")
    console.print(f"[cyan]  --llm-db scm_arena_experiments.db \\[/cyan]")
    console.print(f"[cyan]  --baseline-db {db_path} \\[/cyan]")
    console.print(f"[cyan]  --output-db combined_study.db[/cyan]")


@main.command()
@click.option('--llm-model', '-m', default='llama3.2', help='LLM model for comparison')
@click.option('--baseline-agents', '-b', multiple=True, 
              default=['sterman', 'newsvendor'],
              type=click.Choice(['sterman', 'newsvendor', 'basestock', 'reactive', 'movingavg']),
              help='Baseline agents to include in comparison')
@click.option('--memory', multiple=True, default=['short'], 
              type=click.Choice(['none', 'short', 'medium', 'full']), 
              help='Memory strategies to test (LLM only)')
@click.option('--visibility', multiple=True, default=['local'], 
              type=click.Choice(['local', 'adjacent', 'full']), 
              help='Visibility levels to test (LLM only)')
@click.option('--scenarios', multiple=True, default=['classic', 'random'], 
              type=click.Choice(['classic', 'random', 'shock', 'seasonal']), 
              help='Scenarios to test')
@click.option('--game-modes', multiple=True, default=['modern'], 
              type=click.Choice(['modern', 'classic']), 
              help='Game modes to test')
@click.option('--runs', default=5, help='Number of runs per condition')
@click.option('--rounds', default=30, help='Number of rounds')
@click.option('--runners', default=1, help='Number of concurrent runners (default: 1, max recommended: 8)')
@click.option('--save-results', help='Save results to CSV file')
@click.option('--save-database', is_flag=True, help='Save to database')
@click.option('--db-path', default='llm_vs_baseline.db', help='Database path')
def llm_vs_baseline(llm_model: str, baseline_agents: tuple, memory: tuple, 
                    visibility: tuple, scenarios: tuple, game_modes: tuple,
                    runs: int, rounds: int, runners: int, save_results: str, 
                    save_database: bool, db_path: str):
    """
    Compare LLM against baseline agents across experimental conditions with concurrent execution.
    
    This is the core comparison for Paper 2: "LLM vs Classical Algorithms"
    Tests LLM across all memory/visibility combinations vs baseline agents
    in standard conditions.
    """
    
    if not test_ollama_connection():
        console.print("[red]❌ Cannot connect to Ollama server[/red]")
        return
    
    # Validate runners
    if runners < 1:
        runners = 1
    elif runners > 8:
        console.print(f"[yellow]⚠️ Limiting runners to 8 (requested: {runners})[/yellow]")
        runners = 8
    
    # Calculate experiment scope
    llm_conditions = len(memory) * len(visibility) * len(scenarios) * len(game_modes)
    baseline_conditions = len(scenarios) * len(game_modes)  # Baselines don't vary by memory/visibility
    total_experiments = (llm_conditions + len(baseline_agents) * baseline_conditions) * runs
    
    console.print(Panel(
        f"""[bold blue]🥊 LLM vs Baseline Agents Comparison (Concurrent)[/bold blue]

📊 Experimental Design:
• LLM Model: {llm_model}
• Baseline Agents: {len(baseline_agents)} ({', '.join(baseline_agents)})
• LLM Conditions: {llm_conditions} (memory × visibility × scenarios × modes)
• Baseline Conditions: {baseline_conditions} per agent (scenarios × modes)
• Runs per Condition: {runs}
• Concurrent Runners: {runners}

🎯 Total Experiments: {total_experiments}
⏱️  Sequential Time: {total_experiments * 2:.0f}-{total_experiments * 4:.0f} minutes
🏃 Concurrent Time: {(total_experiments * 2 / runners):.0f}-{(total_experiments * 4 / runners):.0f} minutes

🎛️ LLM uses canonical settings: temp={CANONICAL_TEMPERATURE}, top_p={CANONICAL_TOP_P}
📈 This tests LLM advantages across information architectures!""",
        title="LLM vs Baseline Study"
    ))
    
    if not click.confirm("Proceed with concurrent LLM vs baseline comparison?"):
        return
    
    # Initialize data capture
    tracker = None
    if save_database and ExperimentTracker:
        tracker = ExperimentTracker(db_path)
        console.print(f"[green]📊 Database tracking enabled: {db_path}[/green]")
    
    memory_windows = {'none': 0, 'short': 5, 'medium': 10, 'full': None}
    
    # Prepare experiment parameters
    experiment_params = []
    
    # LLM experiments
    for mem in memory:
        for vis in visibility:
            for scenario in scenarios:
                for game_mode in game_modes:
                    for run in range(runs):
                        agent_info = {
                            'type': 'llm',
                            'model': llm_model,
                            'memory': mem,
                            'visibility': vis
                        }
                        params = (agent_info, scenario, game_mode, run + 1, rounds, memory_windows)
                        experiment_params.append(params)
    
    # Baseline experiments
    for agent_type in baseline_agents:
        for scenario in scenarios:
            for game_mode in game_modes:
                for run in range(runs):
                    agent_info = {
                        'type': 'baseline',
                        'baseline_type': agent_type
                    }
                    params = (agent_info, scenario, game_mode, run + 1, rounds, memory_windows)
                    experiment_params.append(params)
    
    results = []
    failed_experiments = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task(f"Running LLM vs baseline study with {runners} runners...", total=total_experiments)
        
        # Execute experiments concurrently
        with ThreadPoolExecutor(max_workers=runners) as executor:
            # Submit all experiments
            future_to_params = {
                executor.submit(run_single_llm_vs_baseline_experiment, params): params 
                for params in experiment_params
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_params):
                params = future_to_params[future]
                agent_info, scenario, game_mode, run_number, rounds, memory_windows = params
                
                try:
                    result = future.result()
                    
                    if result and not result.get('failed'):
                        results.append(result)
                        
                        with progress_lock:
                            progress.update(task, advance=1)
                            agent_desc = f"{agent_info['model']}-{agent_info['memory']}-{agent_info['visibility']}" if agent_info['type'] == 'llm' else agent_info['baseline_type']
                            console.print(f"✅ {agent_desc}-{scenario}-{game_mode} Run {run_number}: Cost=${result['total_cost']:.0f}")
                    else:
                        failed_experiments.append({
                            'agent_info': agent_info,
                            'scenario': scenario,
                            'game_mode': game_mode,
                            'run': run_number,
                            'error': result.get('error', 'Unknown error') if result else 'No result returned'
                        })
                        
                        with progress_lock:
                            progress.update(task, advance=1)
                            console.print(f"[red]❌ Failed experiment[/red]")
                            
                except Exception as e:
                    failed_experiments.append({
                        'agent_info': agent_info,
                        'scenario': scenario,
                        'game_mode': game_mode,
                        'run': run_number,
                        'error': str(e)
                    })
                    
                    with progress_lock:
                        progress.update(task, advance=1)
                        console.print(f"[red]❌ Exception: {e}[/red]")
    
    # Display comparison results
    console.print(f"\n[bold green]🎉 Concurrent LLM vs Baseline Comparison Complete![/bold green]")
    console.print(f"✅ Successful experiments: {len(results)}")
    console.print(f"❌ Failed experiments: {len(failed_experiments)}")
    
    if results:
        display_llm_vs_baseline_results(results)
    
    # Save results if requested
    if save_results:
        save_experimental_results(results, save_results)
    
    # Close database tracker if used
    if tracker:
        tracker.close()
        console.print(f"[green]💾 LLM vs baseline comparison saved to database[/green]")


@main.command()
@click.option('--models', '-m', multiple=True, default=['llama3.2'], help='Models to test')
@click.option('--memory', multiple=True, default=['none', 'short', 'full'], 
              type=click.Choice(['none', 'short', 'medium', 'full']), help='Memory strategies')
@click.option('--prompts', multiple=True, default=['specific', 'neutral'], 
              type=click.Choice(['specific', 'neutral']), help='Prompt types')
@click.option('--visibility', multiple=True, default=['local', 'adjacent', 'full'], 
              type=click.Choice(['local', 'adjacent', 'full']), help='Visibility levels')
@click.option('--scenarios', multiple=True, default=['classic'], 
              type=click.Choice(['classic', 'random', 'shock', 'seasonal']), help='Scenarios to test')
@click.option('--game-modes', multiple=True, default=['modern', 'classic'], 
              type=click.Choice(['modern', 'classic']), help='Game mode settings')
@click.option('--runs', default=3, help='Number of runs per condition')
@click.option('--rounds', default=20, help='Rounds per game')
@click.option('--base-seed', default=DEFAULT_BASE_SEED, help=f'Base seed for deterministic generation (default: {DEFAULT_BASE_SEED})')
@click.option('--deterministic/--fixed', default=True, help='Use deterministic seeding (default) vs fixed seed')
@click.option('--runners', default=1, help='Number of concurrent runners (default: 1, max recommended: 8)')
@click.option('--save-results', help='Save results to CSV file')
@click.option('--save-database', is_flag=True, help='Save detailed data to database')
@click.option('--db-path', default='scm_arena_experiments.db', help='Database file path')
def experiment(models: tuple, memory: tuple, prompts: tuple, visibility: tuple, 
               scenarios: tuple, game_modes: tuple, runs: int, rounds: int, 
               base_seed: int, deterministic: bool, runners: int, save_results: str, 
               save_database: bool, db_path: str):
    """
    Run fully crossed experimental design with deterministic seeding and concurrent execution.
    
    Each experimental condition gets a unique, reproducible seed based on its parameters.
    Multiple runners can execute experiments in parallel for faster completion.
    """
    
    if not test_ollama_connection():
        console.print("[red]❌ Cannot connect to Ollama server[/red]")
        return
    
    # Validate runners
    if runners < 1:
        runners = 1
    elif runners > 8:  # Reasonable upper limit
        console.print(f"[yellow]⚠️ Limiting runners to 8 (requested: {runners})[/yellow]")
        runners = 8
    
    # Initialize seeder
    seeder = ExperimentSeeder(base_seed=base_seed, deterministic=deterministic)
    
    console.print(f"[blue]🎯 Using SCM-Arena canonical settings for all experiments:[/blue]")
    console.print(f"[blue]   Temperature: {CANONICAL_TEMPERATURE} | Top_P: {CANONICAL_TOP_P} | Top_K: {CANONICAL_TOP_K}[/blue]")
    
    seeding_method = "Hash-based per condition" if deterministic else "Randomized per condition"
    console.print(f"[blue]🎲 Seeding: {seeding_method} (base_seed={base_seed})[/blue]")
    console.print(f"[blue]🏃 Concurrent execution: {runners} runner{'s' if runners > 1 else ''}[/blue]")
    
    # Convert settings
    memory_windows = {'none': 0, 'short': 5, 'medium': 10, 'full': None}
    
    # Generate all experimental conditions
    conditions = list(itertools.product(
        models, memory, prompts, visibility, scenarios, game_modes
    ))
    
    total_experiments = len(conditions) * runs
    
    console.print(Panel(
        f"""[bold blue]🧪 SCM-Arena Concurrent Benchmark Study[/bold blue]
        
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
⏱️  Sequential Time: {total_experiments * 3:.0f}-{total_experiments * 6:.0f} minutes
🏃 Concurrent Time ({runners} runners): {(total_experiments * 3 / runners):.0f}-{(total_experiments * 6 / runners):.0f} minutes

🎛️ Canonical LLM Settings:
• Temperature: {CANONICAL_TEMPERATURE}
• Top_P: {CANONICAL_TOP_P}
• Top_K: {CANONICAL_TOP_K}
• Repeat Penalty: {CANONICAL_REPEAT_PENALTY}

🎲 Deterministic Seeding:
• Base Seed: {base_seed}
• Method: {seeding_method}
• Reproducible: ✅ Each condition gets consistent seed across runs
• Statistical: ✅ Different runs get different seeds for validity""",
        title="Concurrent Benchmark Configuration"
    ))
    
    if not click.confirm("Proceed with concurrent benchmark run?"):
        return
    
    # Prepare experiment parameters for all experiments
    experiment_params = []
    for condition in conditions:
        for run in range(runs):
            params = (
                condition, 
                run + 1, 
                rounds, 
                memory_windows, 
                seeder, 
                save_database, 
                ExperimentTracker if save_database else None, 
                db_path
            )
            experiment_params.append(params)
    
    results = []
    failed_experiments = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Running {total_experiments} experiments with {runners} runners...", total=total_experiments)
        
        # Execute experiments concurrently
        with ThreadPoolExecutor(max_workers=runners) as executor:
            # Submit all experiments
            future_to_params = {
                executor.submit(run_single_experiment, params): params 
                for params in experiment_params
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_params):
                params = future_to_params[future]
                condition, run_number = params[0], params[1]
                
                try:
                    result = future.result()
                    
                    if result and not result.get('failed'):
                        results.append(result)
                        
                        # Thread-safe progress update with result info
                        with progress_lock:
                            progress.update(task, advance=1)
                            model, mem, prompt_type, vis, scenario, game_mode = condition
                            console.print(f"✅ {model}-{mem}-{prompt_type}-{vis}-{scenario}-{game_mode} Run {run_number}: Cost=${result['total_cost']:.0f} (seed={result['seed']})")
                    else:
                        failed_experiments.append({
                            'condition': condition,
                            'run': run_number,
                            'error': result.get('error', 'Unknown error') if result else 'No result returned'
                        })
                        
                        with progress_lock:
                            progress.update(task, advance=1)
                            console.print(f"[red]❌ Failed: {condition} Run {run_number}: {result.get('error', 'Unknown error') if result else 'No result'}[/red]")
                            
                except Exception as e:
                    failed_experiments.append({
                        'condition': condition,
                        'run': run_number,
                        'error': str(e)
                    })
                    
                    with progress_lock:
                        progress.update(task, advance=1)
                        console.print(f"[red]❌ Exception: {condition} Run {run_number}: {e}[/red]")
    
    # Display results summary
    console.print(f"\n[bold green]🎉 Concurrent Benchmark Complete![/bold green]")
    console.print(f"✅ Successful experiments: {len(results)}")
    console.print(f"❌ Failed experiments: {len(failed_experiments)}")
    
    if failed_experiments:
        console.print(f"\n[yellow]⚠️ Failed experiment summary:[/yellow]")
        for failure in failed_experiments[:5]:  # Show first 5 failures
            console.print(f"  {failure['condition']} Run {failure['run']}: {failure['error']}")
        if len(failed_experiments) > 5:
            console.print(f"  ... and {len(failed_experiments) - 5} more failures")
    
    # Display experimental results with seeding info if we have results
    if results:
        display_experimental_results(results, seeder)
        
        # Save results if requested
        if save_results:
            save_experimental_results(results, save_results)
    
    console.print(f"[green]💾 Database saved with complete audit trail including deterministic seeds[/green]")


@main.command() 
@click.option('--model', '-m', default='llama3.2', help='Ollama model name')
@click.option('--scenario', '-s', default='classic', help='Demand scenario')
@click.option('--rounds', '-r', default=20, help='Number of rounds')
@click.option('--runs', default=3, help='Number of runs per condition')
@click.option('--base-seed', default=DEFAULT_BASE_SEED, help=f'Base seed for deterministic generation (default: {DEFAULT_BASE_SEED})')
@click.option('--deterministic/--fixed', default=True, help='Use deterministic seeding (default) vs fixed seed')
@click.option('--runners', default=1, help='Number of concurrent runners (default: 1, max recommended: 4)')
def visibility_study(model: str, scenario: str, rounds: int, runs: int, base_seed: int, deterministic: bool, runners: int):
    """Compare all visibility levels systematically using canonical settings with concurrent execution"""
    
    if not test_ollama_connection():
        console.print("[red]❌ Cannot connect to Ollama server[/red]")
        return
    
    # Validate runners
    if runners < 1:
        runners = 1
    elif runners > 4:
        console.print(f"[yellow]⚠️ Limiting runners to 4 for visibility study (requested: {runners})[/yellow]")
        runners = 4
    
    console.print(f"[blue]🎯 Using canonical settings: temp={CANONICAL_TEMPERATURE}, top_p={CANONICAL_TOP_P}[/blue]")
    seeding_method = "deterministic" if deterministic else "randomized"
    console.print(f"[blue]🎲 Seeding method: {seeding_method} (base_seed={base_seed})[/blue]")
    console.print(f"[blue]🏃 Using {runners} concurrent runner{'s' if runners > 1 else ''}[/blue]")
    
    visibility_levels = ['local', 'adjacent', 'full']
    demand_pattern = DEMAND_PATTERNS.get(scenario, DEMAND_PATTERNS['classic'])[:rounds]
    total_experiments = len(visibility_levels) * runs
    
    # Initialize seeder
    seeder = ExperimentSeeder(base_seed=base_seed, deterministic=deterministic)
    
    console.print(f"[blue]👁️  Visibility Study - Model: {model}, Scenario: {scenario}[/blue]")
    console.print(f"[blue]📊 Testing {len(visibility_levels)} visibility levels × {runs} runs = {total_experiments} games[/blue]")
    
    # Prepare experiment parameters
    experiment_params = []
    for visibility in visibility_levels:
        for run in range(runs):
            params = (model, visibility, scenario, rounds, run + 1, seeder)
            experiment_params.append(params)
    
    results = {}
    failed_experiments = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Running visibility study with {runners} runners...", total=total_experiments)
        
        # Execute experiments concurrently
        with ThreadPoolExecutor(max_workers=runners) as executor:
            # Submit all experiments
            future_to_params = {
                executor.submit(run_visibility_experiment, params): params 
                for params in experiment_params
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_params):
                params = future_to_params[future]
                model, visibility, scenario, rounds, run_number, seeder = params
                
                try:
                    result = future.result()
                    
                    if result and not result.get('failed'):
                        if visibility not in results:
                            results[visibility] = []
                        results[visibility].append(result)
                        
                        with progress_lock:
                            progress.update(task, advance=1)
                            console.print(f"✅ {visibility} Run {run_number}: Cost=${result['total_cost']:.2f}, Service={result['service_level']:.1%} (seed={result['seed']})")
                    else:
                        failed_experiments.append({
                            'visibility': visibility,
                            'run': run_number,
                            'error': result.get('error', 'Unknown error') if result else 'No result returned'
                        })
                        
                        with progress_lock:
                            progress.update(task, advance=1)
                            console.print(f"[red]❌ Failed: {visibility} Run {run_number}[/red]")
                            
                except Exception as e:
                    failed_experiments.append({
                        'visibility': visibility,
                        'run': run_number,
                        'error': str(e)
                    })
                    
                    with progress_lock:
                        progress.update(task, advance=1)
                        console.print(f"[red]❌ Exception: {visibility} Run {run_number}: {e}[/red]")
    
    # Display visibility comparison
    console.print(f"\n[bold green]🎉 Concurrent Visibility Study Complete![/bold green]")
    console.print(f"✅ Successful experiments: {sum(len(v) for v in results.values())}")
    console.print(f"❌ Failed experiments: {len(failed_experiments)}")
    
    if results:
        display_visibility_comparison(results)


def run_visibility_experiment(params):
    """Run a single visibility experiment for concurrent execution"""
    try:
        model, visibility, scenario, rounds, run_number, seeder = params
        
        # Generate seed for this specific condition
        seed = seeder.get_seed(
            model=model,
            memory="short",  # Default memory for visibility study
            prompt_type="specific",  # Default prompt
            visibility=visibility,
            scenario=scenario,
            game_mode="modern",  # Default mode
            run_number=run_number
        )
        
        # Create agents with canonical settings
        agents = create_ollama_agents(
            model_name=model,
            temperature=CANONICAL_TEMPERATURE,
            top_p=CANONICAL_TOP_P,
            top_k=CANONICAL_TOP_K,
            repeat_penalty=CANONICAL_REPEAT_PENALTY,
            seed=seed
        )
        
        # Get demand pattern
        demand_pattern = DEMAND_PATTERNS.get(scenario, DEMAND_PATTERNS['classic'])[:rounds]
        
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
        summary = game_results.summary()
        summary['seed'] = seed
        
        return summary
        
    except Exception as e:
        return {
            'error': str(e),
            'visibility': visibility,
            'run': run_number,
            'failed': True
        }


@main.command()
@click.option('--model', '-m', default='llama3.2', help='Ollama model name')
@click.option('--base-seed', default=DEFAULT_BASE_SEED, help=f'Base seed for deterministic generation (default: {DEFAULT_BASE_SEED})')
@click.option('--deterministic/--fixed', default=True, help='Use deterministic seeding (default) vs fixed seed')
def test_model(model: str, base_seed: int, deterministic: bool):
    """Test a model with canonical settings"""
    
    if not test_ollama_connection():
        console.print("[red]❌ Cannot connect to Ollama server[/red]")
        return
    
    console.print(f"[blue]🧪 Testing model: {model} with canonical settings[/blue]")
    seeding_method = "deterministic" if deterministic else "randomized" 
    console.print(f"[blue]🎯 Settings: temp={CANONICAL_TEMPERATURE}, top_p={CANONICAL_TOP_P}, seeding={seeding_method}[/blue]")
    
    try:
        # Generate seed for test
        if deterministic:
            seeder = ExperimentSeeder(base_seed=base_seed, deterministic=True)
            seed = seeder.get_seed("test", "short", "specific", "local", "classic", "modern", 1)
        else:
            seed = base_seed
            
        # Create single agent for testing with canonical settings
        agent = OllamaAgent(
            Position.RETAILER, 
            model,
            temperature=CANONICAL_TEMPERATURE,
            top_p=CANONICAL_TOP_P,
            top_k=CANONICAL_TOP_K,
            repeat_penalty=CANONICAL_REPEAT_PENALTY,
            seed=seed
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
        
        console.print(f"[green]✅ Model responded with order: {decision} (seed={seed})[/green]")
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
def baseline_agents():
    """List available baseline agents and their descriptions"""
    
    console.print("[bold blue]📋 Available Baseline Agents[/bold blue]")
    
    descriptions = get_baseline_descriptions()
    
    table = Table()
    table.add_column("Agent", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Reference", style="yellow")
    
    table.add_row(
        "sterman", 
        descriptions["sterman"],
        "Sterman (1989) Management Science"
    )
    table.add_row(
        "newsvendor", 
        descriptions["newsvendor"],
        "Classic Operations Research"
    )
    table.add_row(
        "basestock", 
        descriptions["basestock"],
        "Supply Chain Management Standard"
    )
    table.add_row(
        "reactive", 
        descriptions["reactive"],
        "Simple Order-Up-To Policy"
    )
    table.add_row(
        "movingavg", 
        descriptions["movingavg"],
        "Forecasting-Based Approach"
    )
    
    console.print(table)
    
    console.print(f"\n[blue]💡 Usage Examples:[/blue]")
    console.print("[cyan]# Pure baseline study (concurrent)[/cyan]")
    console.print("poetry run python -m scm_arena.cli baseline-study --runners 4")
    
    console.print("\n[cyan]# Full factorial baseline study (concurrent)[/cyan]")
    console.print("poetry run python -m scm_arena.cli full-factorial-baseline \\")
    console.print("  --runs 20 --rounds 52 --runners 6")
    
    console.print("\n[cyan]# LLM vs baseline comparison (concurrent)[/cyan]")
    console.print("poetry run python -m scm_arena.cli llm-vs-baseline \\")
    console.print("  --baseline-agents sterman newsvendor \\")
    console.print("  --memory short full --visibility local adjacent \\")
    console.print("  --runs 5 --runners 4 --save-results comparison.csv")


@main.command()
@click.option('--llm-db', required=True, help='Path to LLM experiments database')
@click.option('--baseline-db', required=True, help='Path to baseline experiments database') 
@click.option('--output-db', required=True, help='Path to combined output database')
def merge_databases(llm_db: str, baseline_db: str, output_db: str):
    """
    Merge LLM and baseline experiment databases into a combined database.
    
    This creates a unified database with both LLM and baseline results
    for comprehensive analysis and comparison.
    """
    
    console.print(f"[blue]🔗 Merging Databases[/blue]")
    console.print(f"  LLM Database: {llm_db}")
    console.print(f"  Baseline Database: {baseline_db}")
    console.print(f"  Output Database: {output_db}")
    
    try:
        import sqlite3
        import shutil
        
        # Copy LLM database as base
        shutil.copy2(llm_db, output_db)
        console.print(f"✅ Copied LLM database as base")
        
        # Connect to databases
        output_conn = sqlite3.connect(output_db)
        baseline_conn = sqlite3.connect(baseline_db)
        
        # Attach baseline database
        output_conn.execute(f"ATTACH DATABASE '{baseline_db}' AS baseline")
        
        # Get table names
        tables = ['experiments', 'rounds', 'agent_rounds', 'game_states']
        
        # Copy data from baseline database
        for table in tables:
            # Check if table exists in baseline database
            cursor = baseline_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", 
                (table,)
            )
            if cursor.fetchone():
                # Insert baseline data into output database
                output_conn.execute(f"INSERT INTO {table} SELECT * FROM baseline.{table}")
                console.print(f"✅ Merged {table} table")
        
        # Commit and close
        output_conn.commit()
        output_conn.close()
        baseline_conn.close()
        
        # Report final counts
        final_conn = sqlite3.connect(output_db)
        cursor = final_conn.execute("SELECT COUNT(*) FROM experiments")
        total_experiments = cursor.fetchone()[0]
        
        cursor = final_conn.execute("SELECT COUNT(*) FROM experiments WHERE model_name IN ('sterman', 'newsvendor', 'basestock', 'reactive', 'movingavg')")
        baseline_experiments = cursor.fetchone()[0]
        
        llm_experiments = total_experiments - baseline_experiments
        
        final_conn.close()
        
        console.print(f"\n[bold green]🎉 Database Merge Complete![/bold green]")
        console.print(f"📊 Combined Database: {output_db}")
        console.print(f"  Total Experiments: {total_experiments}")
        console.print(f"  LLM Experiments: {llm_experiments}")
        console.print(f"  Baseline Experiments: {baseline_experiments}")
        console.print(f"\n[blue]Ready for comprehensive LLM vs Baseline analysis![/blue]")
        
    except Exception as e:
        console.print(f"[red]❌ Database merge failed: {e}[/red]")
        console.print(f"Make sure both input databases exist and are accessible")


@main.command()
@click.option('--db-path', required=True, help='Database path to inspect')
def inspect_database(db_path: str):
    """Inspect database contents and show summary statistics"""
    
    console.print(f"[blue]🔍 Inspecting Database: {db_path}[/blue]")
    
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        
        # Check tables
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        console.print(f"📊 Tables: {', '.join(tables)}")
        
        if 'experiments' in tables:
            # Experiment counts
            cursor = conn.execute("SELECT COUNT(*) FROM experiments")
            total_experiments = cursor.fetchone()[0]
            console.print(f"📈 Total Experiments: {total_experiments}")
            
            # Agent type breakdown
            cursor = conn.execute("SELECT model_name, COUNT(*) FROM experiments GROUP BY model_name ORDER BY COUNT(*) DESC")
            agent_counts = cursor.fetchall()
            
            console.print(f"\n📊 Experiments by Agent Type:")
            for agent, count in agent_counts:
                console.print(f"  {agent}: {count}")
            
            # Scenario breakdown
            cursor = conn.execute("SELECT scenario, COUNT(*) FROM experiments GROUP BY scenario")
            scenario_counts = cursor.fetchall()
            
            console.print(f"\n🎯 Experiments by Scenario:")
            for scenario, count in scenario_counts:
                console.print(f"  {scenario}: {count}")
            
            # Game mode breakdown
            cursor = conn.execute("SELECT game_mode, COUNT(*) FROM experiments GROUP BY game_mode")
            mode_counts = cursor.fetchall()
            
            console.print(f"\n🎮 Experiments by Game Mode:")
            for mode, count in mode_counts:
                console.print(f"  {mode}: {count}")
        
        conn.close()
        console.print(f"\n✅ Database inspection complete")
        
    except Exception as e:
        console.print(f"[red]❌ Database inspection failed: {e}[/red]")


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


def display_baseline_study_results(results):
    """Display baseline study results"""
    
    console.print("\n[bold blue]🔬 Baseline Study Results[/bold blue]")
    
    if not results:
        console.print("[red]No results to display[/red]")
        return
    
    # Group results by agent type
    agent_performance = {}
    for result in results:
        agent = result['agent_type']
        if agent not in agent_performance:
            agent_performance[agent] = []
        agent_performance[agent].append(result['total_cost'])
    
    # Create performance table
    table = Table(title="Baseline Agent Performance")
    table.add_column("Agent Type", style="cyan")
    table.add_column("Avg Cost", style="green")
    table.add_column("Std Dev", style="yellow")
    table.add_column("Min Cost", style="blue")
    table.add_column("Max Cost", style="red")
    table.add_column("Experiments", style="white")
    
    # Calculate statistics
    agent_stats = []
    for agent, costs in agent_performance.items():
        mean_cost = sum(costs) / len(costs)
        std_cost = (sum((c - mean_cost) ** 2 for c in costs) / len(costs)) ** 0.5
        min_cost = min(costs)
        max_cost = max(costs)
        
        agent_stats.append({
            'agent': agent,
            'mean': mean_cost,
            'std': std_cost,
            'min': min_cost,
            'max': max_cost,
            'count': len(costs)
        })
    
    # Sort by mean cost
    agent_stats.sort(key=lambda x: x['mean'])
    
    # Add to table with ranking
    for rank, stats in enumerate(agent_stats, 1):
        rank_emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}"
        
        table.add_row(
            f"{rank_emoji} {stats['agent']}",
            f"${stats['mean']:.0f}",
            f"±{stats['std']:.0f}",
            f"${stats['min']:.0f}",
            f"${stats['max']:.0f}",
            str(stats['count'])
        )
    
    console.print(table)
    
    # Best performer analysis
    if agent_stats:
        best = agent_stats[0]
        console.print(f"\n[bold green]🏆 Best Performer: {best['agent']} (${best['mean']:.0f} average cost)[/bold green]")


def display_llm_vs_baseline_results(results):
    """Display LLM vs baseline comparison results"""
    
    console.print("\n[bold blue]🥊 LLM vs Baseline Results[/bold blue]")
    
    if not results:
        console.print("[red]No results to display[/red]")
        return
    
    # Separate LLM and baseline results
    llm_results = [r for r in results if r['agent_type'] == 'llm']
    baseline_results = [r for r in results if r['agent_type'] != 'llm']
    
    # Overall comparison table
    table = Table(title="Agent Type Performance")
    table.add_column("Agent Type", style="cyan")
    table.add_column("Avg Cost", style="green")
    table.add_column("Avg Service", style="blue")
    table.add_column("Avg Bullwhip", style="yellow")
    table.add_column("Experiments", style="white")
    
    # LLM performance (average across all conditions)
    if llm_results:
        llm_costs = [r['total_cost'] for r in llm_results]
        llm_service = [r['service_level'] for r in llm_results]
        llm_bullwhip = [r['bullwhip_ratio'] for r in llm_results]
        
        table.add_row(
            "🤖 LLM",
            f"${sum(llm_costs) / len(llm_costs):.0f}",
            f"{sum(llm_service) / len(llm_service):.1%}",
            f"{sum(llm_bullwhip) / len(llm_bullwhip):.2f}",
            str(len(llm_results))
        )
    
    # Baseline performance by agent type
    baseline_agents = {}
    for result in baseline_results:
        agent = result['agent_type']
        if agent not in baseline_agents:
            baseline_agents[agent] = []
        baseline_agents[agent].append(result)
    
    baseline_stats = []
    for agent, agent_results in baseline_agents.items():
        costs = [r['total_cost'] for r in agent_results]
        service = [r['service_level'] for r in agent_results]
        bullwhip = [r['bullwhip_ratio'] for r in agent_results]
        
        baseline_stats.append({
            'agent': agent,
            'avg_cost': sum(costs) / len(costs),
            'avg_service': sum(service) / len(service),
            'avg_bullwhip': sum(bullwhip) / len(bullwhip),
            'count': len(agent_results)
        })
    
    # Sort baseline agents by cost
    baseline_stats.sort(key=lambda x: x['avg_cost'])
    
    for rank, stats in enumerate(baseline_stats, 1):
        rank_emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}"
        
        table.add_row(
            f"{rank_emoji} {stats['agent']}",
            f"${stats['avg_cost']:.0f}",
            f"{stats['avg_service']:.1%}",
            f"{stats['avg_bullwhip']:.2f}",
            str(stats['count'])
        )
    
    console.print(table)
    
    # Advantage analysis
    if llm_results and baseline_results:
        llm_avg_cost = sum(r['total_cost'] for r in llm_results) / len(llm_results)
        best_baseline_cost = min(sum(costs) / len(costs) for costs in baseline_agents.values() if costs)
        
        advantage = (best_baseline_cost - llm_avg_cost) / best_baseline_cost * 100
        
        console.print(f"\n[bold green]🎯 LLM Performance Analysis:[/bold green]")
        console.print(f"• LLM average cost: ${llm_avg_cost:.0f}")
        console.print(f"• Best baseline cost: ${best_baseline_cost:.0f}")
        console.print(f"• LLM advantage: {advantage:.1f}% cost reduction")
        
        if advantage > 0:
            console.print(f"🎉 LLM outperforms all baseline agents!")
        else:
            console.print(f"🤔 Some baseline agents outperform LLM")


def display_experimental_results(results, seeder=None):
    """Display experimental results summary with seeding information"""
    
    console.print("\n[bold blue]🧪 Concurrent Benchmark Results Summary[/bold blue]")
    
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
        unique_game_modes = len(set(r['game_mode'] for r in results))
        unique_seeds = len(set(r['seed'] for r in results))
        
        console.print(f"🎯 Tested: {unique_models} models, {unique_memory} memory strategies, {unique_visibility} visibility levels, {unique_game_modes} game modes")
        console.print(f"🎛️ All experiments used canonical settings: temp={CANONICAL_TEMPERATURE}, top_p={CANONICAL_TOP_P}")
        console.print(f"🎲 Seeding: {unique_seeds} unique seeds generated")
        console.print(f"🏃 Concurrent execution reduced total runtime significantly")
        
        # Show seeder statistics if available
        if seeder:
            stats = seeder.get_seed_statistics()
            console.print(f"🔄 Seed collision rate: {stats.get('collision_rate', 0):.1%} (lower is better)")


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


if __name__ == "__main__":
    main()