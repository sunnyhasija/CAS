#!/usr/bin/env python3
# run_missing_direct.py
"""
Run missing experiments directly without CLI confirmations
Uses the actual game engine directly - no CLI interaction needed!
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from datetime import datetime
from pathlib import Path
import concurrent.futures
import pandas as pd
import sqlite3

from scm_arena.beer_game.game import BeerGame, VisibilityLevel
from scm_arena.beer_game.agents import Position
from scm_arena.models.ollama_client import create_ollama_agents, test_ollama_connection
from scm_arena.evaluation.scenarios import DEMAND_PATTERNS, generate_scenario_with_seed
from scm_arena.data_capture import ExperimentTracker
from scm_arena.utils.seeding import ExperimentSeeder

# Experimental configuration
ROUNDS = 52
RUNS = 20
BASE_SEED = 42
MODEL = "llama3.2"

# Missing experiments to run
MISSING_EXPERIMENTS = [
    # Random scenario experiments
    {"memory": "none", "visibility": "local", "scenario": "random", "id": "random_none_local"},
    {"memory": "none", "visibility": "adjacent", "scenario": "random", "id": "random_none_adjacent"},
    {"memory": "none", "visibility": "full", "scenario": "random", "id": "random_none_full"},
    {"memory": "short", "visibility": "local", "scenario": "random", "id": "random_short_local"},
    {"memory": "short", "visibility": "adjacent", "scenario": "random", "id": "random_short_adjacent"},
    {"memory": "short", "visibility": "full", "scenario": "random", "id": "random_short_full"},
    {"memory": "full", "visibility": "local", "scenario": "random", "id": "random_full_local"},
    {"memory": "full", "visibility": "adjacent", "scenario": "random", "id": "random_full_adjacent"},
    {"memory": "full", "visibility": "full", "scenario": "random", "id": "random_full_full"},
    # Adjacent visibility for shock/seasonal
    {"memory": "all", "visibility": "adjacent", "scenario": "shock", "id": "shock_adjacent_all"},
    {"memory": "all", "visibility": "adjacent", "scenario": "seasonal", "id": "seasonal_adjacent_all"},
]

def run_single_experiment(exp_config, memory, prompt_type, game_mode, run_number):
    """Run a single experiment"""
    try:
        # Set up seeding - use EXACT same approach as CLI
        seeder = ExperimentSeeder(base_seed=BASE_SEED, deterministic=True)
        
        # The CLI passes these exact parameter names to get_seed
        seed = seeder.get_seed(
            model=MODEL,
            memory=memory,
            prompt_type=prompt_type,  # CLI uses prompt_type, not prompt
            visibility=exp_config['visibility'],
            scenario=exp_config['scenario'],
            game_mode=game_mode,      # CLI uses game_mode, not mode
            run_number=run_number     # CLI uses run_number, not run
        )
        
        # Debug: Print seed info
        print(f"   Generated seed: {seed} for {memory}-{prompt_type}-{exp_config['visibility']}-{game_mode} Run {run_number}")
        
        # Convert to 32-bit for Ollama (it seems to require this now)
        seed_for_ollama = seed % (2**32)  # Modulo to get value in valid range
        print(f"   Ollama seed: {seed_for_ollama}")
        
        # Memory window mapping
        memory_windows = {'none': 0, 'short': 5, 'full': None}
        memory_window = memory_windows.get(memory, 5)
        
        # Create agents - use 32-bit seed for Ollama
        agents = create_ollama_agents(
            MODEL,
            neutral_prompt=(prompt_type == 'neutral'),
            memory_window=memory_window,
            temperature=0.3,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            seed=seed_for_ollama  # Use 32-bit seed for Ollama
        )
        
        # Get demand pattern - use 32-bit seed for consistency
        if exp_config['scenario'] == 'random':
            demand_pattern = generate_scenario_with_seed('random', ROUNDS, seed=seed_for_ollama)
        else:
            demand_pattern = DEMAND_PATTERNS[exp_config['scenario']][:ROUNDS]
        
        # Create game
        game = BeerGame(
            agents=agents,
            demand_pattern=demand_pattern,
            classic_mode=(game_mode == 'classic'),
            visibility_level=VisibilityLevel(exp_config['visibility'])
        )
        
        # Set up data tracking
        tracker = ExperimentTracker(f"missing_experiments/{exp_config['id']}.db")
        
        # Start experiment tracking
        experiment_id = tracker.start_experiment(
            model_name=MODEL,
            memory_strategy=memory,
            memory_window=memory_window,
            prompt_type=prompt_type,
            visibility_level=exp_config['visibility'],
            scenario=exp_config['scenario'],
            game_mode=game_mode,
            rounds=ROUNDS,
            run_number=run_number,
            temperature=0.3,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            seed=seed,  # Store original 64-bit seed for tracking
            base_seed=BASE_SEED,
            deterministic_seeding=True
        )
        
        # Run game
        round_num = 0
        while not game.is_complete():
            round_num += 1
            state = game.step()
            
            # Prepare agent interactions for tracking
            agent_interactions = []
            for position, player in state.players.items():
                agent = agents[position]
                
                # Get last interaction data if available
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
            
            # Track round
            tracker.track_round(
                round_number=round_num,
                customer_demand=state.customer_demand,
                total_system_cost=state.total_cost,
                total_system_inventory=sum(p.inventory for p in state.players.values()),
                total_system_backlog=sum(p.backlog for p in state.players.values()),
                agent_interactions=agent_interactions,
                game_state_json="{}"  # Simplified for speed
            )
        
        # Get results
        results = game.get_results()
        summary = results.summary()
        
        # Finish tracking
        tracker.finish_experiment(
            total_cost=summary['total_cost'],
            service_level=summary['service_level'],
            bullwhip_ratio=summary['bullwhip_ratio']
        )
        
        tracker.close()
        
        return {
            'success': True,
            'config': f"{memory}-{prompt_type}-{exp_config['visibility']}-{exp_config['scenario']}-{game_mode}",
            'run': run_number,
            'cost': summary['total_cost'],
            'service': summary['service_level']
        }
        
    except Exception as e:
        print(f"❌ Error in {memory}-{prompt_type}-{exp_config['visibility']}-{game_mode} Run {run_number}: {e}")
        return {'success': False, 'error': str(e)}

def run_experiment_batch(exp_config):
    """Run all experiments for a configuration"""
    batch_results = []
    start_time = datetime.now()
    
    print(f"\n🚀 Starting batch: {exp_config['id']}")
    print(f"   Scenario: {exp_config['scenario']}")
    print(f"   Visibility: {exp_config['visibility']}")
    
    # Determine memory strategies to run
    if exp_config['memory'] == 'all':
        memories = ['none', 'short', 'full']
    else:
        memories = [exp_config['memory']]
    
    total_experiments = len(memories) * 2 * 2 * RUNS  # memories × prompts × game_modes × runs
    completed = 0
    
    for memory in memories:
        for prompt_type in ['specific', 'neutral']:
            for game_mode in ['modern', 'classic']:
                for run in range(1, RUNS + 1):
                    result = run_single_experiment(exp_config, memory, prompt_type, game_mode, run)
                    batch_results.append(result)
                    completed += 1
                    
                    if result['success']:
                        print(f"   ✅ {result['config']} Run {result['run']}: Cost=${result['cost']:.0f}")
                    
                    # Progress update
                    if completed % 10 == 0:
                        elapsed = datetime.now() - start_time
                        rate = completed / (elapsed.total_seconds() / 3600)
                        print(f"   Progress: {completed}/{total_experiments} ({rate:.1f} exp/hour)")
    
    elapsed = datetime.now() - start_time
    success_count = sum(1 for r in batch_results if r['success'])
    
    print(f"\n✅ Batch {exp_config['id']} complete!")
    print(f"   Success: {success_count}/{len(batch_results)}")
    print(f"   Time: {elapsed}")
    
    return exp_config['id'], batch_results

def main():
    print("🚀 DIRECT MISSING EXPERIMENTS RUNNER")
    print("=" * 50)
    print("No CLI confirmations - running directly!")
    print()
    
    # Check Ollama connection
    if not test_ollama_connection():
        print("❌ Cannot connect to Ollama server!")
        print("Please start Ollama: ollama serve")
        return
    
    print("✅ Ollama server connected")
    
    # Create output directory
    Path("missing_experiments").mkdir(exist_ok=True)
    
    # Summary of what we're running
    total_random = 9 * 2 * 2 * RUNS  # 9 configs × 2 prompts × 2 modes × 20 runs = 720
    total_adjacent = 2 * 3 * 2 * 2 * RUNS  # 2 scenarios × 3 memories × 2 prompts × 2 modes × 20 runs = 480
    total_all = total_random + total_adjacent
    
    print(f"📊 Total experiments to run: {total_all}")
    print(f"   Random scenario: {total_random}")
    print(f"   Adjacent visibility: {total_adjacent}")
    print(f"   Max parallel: 10 (M4 Max optimized)")
    print()
    
    start_time = datetime.now()
    
    # Run experiments in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_config = {
            executor.submit(run_experiment_batch, config): config 
            for config in MISSING_EXPERIMENTS
        }
        
        completed_batches = 0
        for future in concurrent.futures.as_completed(future_to_config):
            batch_id, results = future.result()
            completed_batches += 1
            
            print(f"\n📊 {completed_batches}/{len(MISSING_EXPERIMENTS)} batches complete")
            
            elapsed = datetime.now() - start_time
            if completed_batches > 0:
                avg_time_per_batch = elapsed / completed_batches
                remaining_batches = len(MISSING_EXPERIMENTS) - completed_batches
                eta = avg_time_per_batch * remaining_batches
                print(f"   ETA: {eta}")
    
    total_time = datetime.now() - start_time
    print(f"\n🎉 ALL EXPERIMENTS COMPLETE!")
    print(f"Total time: {total_time}")
    
    # Quick verification
    print("\n📊 Verifying databases...")
    for config in MISSING_EXPERIMENTS:
        db_path = f"missing_experiments/{config['id']}.db"
        if Path(db_path).exists():
            conn = sqlite3.connect(db_path)
            count = conn.execute("SELECT COUNT(*) FROM experiments WHERE total_cost > 0").fetchone()[0]
            conn.close()
            print(f"   {config['id']}: {count} experiments")

if __name__ == '__main__':
    main()