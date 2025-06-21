#!/usr/bin/env python3
"""
Basic Beer Game example demonstrating SCM-Arena functionality.

This example shows how to:
1. Create agents (both algorithmic and LLM-based)
2. Set up and run a Beer Game
3. Analyze results and metrics
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from scm_arena.beer_game.game import BeerGame
from scm_arena.beer_game.agents import Position, SimpleAgent, OptimalAgent, RandomAgent
from scm_arena.models.ollama_client import OllamaAgent, test_ollama_connection
from scm_arena.evaluation.scenarios import DEMAND_PATTERNS


def run_simple_game():
    """Run a game with simple algorithmic agents"""
    print("🎮 Running Beer Game with Simple Agents")
    print("=" * 50)
    
    # Create simple agents for all positions
    agents = {
        Position.RETAILER: SimpleAgent(Position.RETAILER, target_inventory=15),
        Position.WHOLESALER: SimpleAgent(Position.WHOLESALER, target_inventory=12),
        Position.DISTRIBUTOR: SimpleAgent(Position.DISTRIBUTOR, target_inventory=12),
        Position.MANUFACTURER: SimpleAgent(Position.MANUFACTURER, target_inventory=10),
    }
    
    # Use classic demand pattern
    demand_pattern = DEMAND_PATTERNS["classic"][:20]  # First 20 rounds
    
    # Create and run game
    game = BeerGame(agents, demand_pattern)
    
    print(f"Starting game with demand pattern: {demand_pattern[:10]}...")
    print()
    
    round_number = 0
    while not game.is_complete():
        state = game.step()
        round_number += 1
        
        print(f"Round {round_number:2d}: Total Cost = ${state.total_cost:6.2f}, Customer Demand = {demand_pattern[round_number-1] if round_number <= len(demand_pattern) else 0}")
    
    # Get and display results
    results = game.get_results()
    print("\n" + "=" * 50)
    print("🏁 GAME RESULTS")
    print("=" * 50)
    
    summary = results.summary()
    print(f"Total Cost: ${summary['total_cost']:.2f}")
    print(f"Cost per Round: ${summary['cost_per_round']:.2f}")
    print(f"Bullwhip Ratio: {summary['bullwhip_ratio']:.2f}")
    print(f"Service Level: {summary['service_level']:.1%}")
    
    print("\nIndividual Costs:")
    for position in Position:
        cost = summary[f"{position.value}_cost"]
        percentage = (cost / summary['total_cost']) * 100
        print(f"  {position.value.title():12}: ${cost:6.2f} ({percentage:4.1f}%)")
    
    return results


def run_mixed_agents_game():
    """Run a game with different types of agents"""
    print("\n🎮 Running Beer Game with Mixed Agents")
    print("=" * 50)
    
    # Create different agent types for each position
    agents = {
        Position.RETAILER: OptimalAgent(Position.RETAILER),
        Position.WHOLESALER: SimpleAgent(Position.WHOLESALER),
        Position.DISTRIBUTOR: OptimalAgent(Position.DISTRIBUTOR),
        Position.MANUFACTURER: SimpleAgent(Position.MANUFACTURER),
    }
    
    # Use random demand pattern
    demand_pattern = DEMAND_PATTERNS["random"][:20]
    
    # Create and run game
    game = BeerGame(agents, demand_pattern)
    
    print("Agent types:")
    for pos, agent in agents.items():
        print(f"  {pos.value.title():12}: {agent.__class__.__name__}")
    print()
    
    round_number = 0
    while not game.is_complete():
        state = game.step()
        round_number += 1
        
        print(f"Round {round_number:2d}: Total Cost = ${state.total_cost:6.2f}")
    
    results = game.get_results()
    summary = results.summary()
    
    print(f"\nFinal Results: ${summary['total_cost']:.2f} total cost, {summary['bullwhip_ratio']:.2f} bullwhip ratio")
    return results


def run_ollama_game():
    """Run a game with Ollama LLM agents"""
    print("\n🤖 Running Beer Game with Ollama Agents")
    print("=" * 50)
    
    # Check if Ollama is available
    if not test_ollama_connection():
        print("❌ Ollama server not available at http://localhost:11434")
        print("   Make sure Ollama is running: 'ollama serve'")
        return None
    
    print("✅ Connected to Ollama server")
    
    try:
        # Create Ollama agents
        model_name = "llama3.2"  # Change this to your preferred model
        agents = {
            Position.RETAILER: OllamaAgent(Position.RETAILER, model_name, temperature=0.1),
            Position.WHOLESALER: OllamaAgent(Position.WHOLESALER, model_name, temperature=0.1),
            Position.DISTRIBUTOR: OllamaAgent(Position.DISTRIBUTOR, model_name, temperature=0.1),
            Position.MANUFACTURER: OllamaAgent(Position.MANUFACTURER, model_name, temperature=0.1),
        }
        
        print(f"✅ Created agents using model: {model_name}")
        
        # Use classic demand pattern for LLM test
        demand_pattern = DEMAND_PATTERNS["classic"][:15]  # Shorter for LLM test
        
        # Create and run game
        game = BeerGame(agents, demand_pattern)
        
        print(f"Starting LLM game with {len(demand_pattern)} rounds...")
        
        round_number = 0
        while not game.is_complete():
            print(f"  Processing round {round_number + 1}...", end=" ")
            state = game.step()
            round_number += 1
            print(f"Cost: ${state.total_cost:.2f}")
        
        results = game.get_results()
        summary = results.summary()
        
        print(f"\n🎉 LLM Game Complete!")
        print(f"   Total Cost: ${summary['total_cost']:.2f}")
        print(f"   Bullwhip Ratio: {summary['bullwhip_ratio']:.2f}")
        print(f"   Service Level: {summary['service_level']:.1%}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error running Ollama game: {e}")
        return None


def compare_agent_types():
    """Compare performance of different agent types"""
    print("\n📊 Comparing Agent Types")
    print("=" * 50)
    
    # Define agent configurations to test
    agent_configs = [
        ("Simple", lambda pos: SimpleAgent(pos)),
        ("Optimal", lambda pos: OptimalAgent(pos)),
        ("Random", lambda pos: RandomAgent(pos, max_order=15)),
    ]
    
    # Add Ollama if available
    if test_ollama_connection():
        agent_configs.append(("Ollama-llama3.2", lambda pos: OllamaAgent(pos, "llama3.2", temperature=0.1)))
    
    demand_pattern = DEMAND_PATTERNS["classic"][:15]
    results = {}
    
    for agent_name, agent_factory in agent_configs:
        try:
            print(f"Testing {agent_name} agents...", end=" ")
            
            # Create agents
            agents = {pos: agent_factory(pos) for pos in Position}
            
            # Run game
            game = BeerGame(agents, demand_pattern)
            while not game.is_complete():
                game.step()
            
            # Store results
            game_results = game.get_results()
            results[agent_name] = game_results.summary()
            
            print(f"Cost: ${results[agent_name]['total_cost']:.2f}")
            
        except Exception as e:
            print(f"Failed: {e}")
    
    # Display comparison
    print("\n🏆 COMPARISON RESULTS")
    print("=" * 50)
    print(f"{'Agent Type':<15} {'Total Cost':<12} {'Bullwhip':<10} {'Service Level':<12}")
    print("-" * 50)
    
    # Sort by total cost (lower is better)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['total_cost'])
    
    for i, (agent_name, summary) in enumerate(sorted_results):
        rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
        print(f"{rank} {agent_name:<12} ${summary['total_cost']:>8.2f}   {summary['bullwhip_ratio']:>6.2f}    {summary['service_level']:>8.1%}")


def main():
    """Run all examples"""
    print("🏭 SCM-Arena Basic Example")
    print("=" * 50)
    
    try:
        # Run examples in sequence
        run_simple_game()
        run_mixed_agents_game()
        run_ollama_game()
        compare_agent_types()
        
        print("\n✅ All examples completed successfully!")
        print("\nNext steps:")
        print("1. Try different demand scenarios: 'random', 'shock', 'seasonal'")
        print("2. Experiment with different Ollama models")
        print("3. Run longer games to see more coordination effects")
        print("4. Use the CLI: 'python -m scm_arena.cli run --help'")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()