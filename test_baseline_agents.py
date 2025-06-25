#!/usr/bin/env python3
"""
test_baseline_agents.py - Quick test of baseline agents

Run this to verify baseline agents work before full experiments.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_baseline_agents():
    """Test that all baseline agents can be created and make decisions"""
    
    print("🧪 Testing Baseline Agents")
    print("=" * 30)
    
    try:
        from scm_arena.beer_game.agents import Position
        from scm_arena.beer_game.baseline_agents import (
            StermanAgent, SimpleReactiveAgent, NewsvendorAgent, 
            BaseStockAgent, MovingAverageAgent, create_baseline_agent
        )
        
        # Test game state
        test_state = {
            "round": 5,
            "position": "retailer",
            "inventory": 8,
            "backlog": 2,
            "incoming_order": 6,
            "last_outgoing_order": 5,
            "round_cost": 12.0,
            "decision_history": [4, 5, 6, 5]
        }
        
        # Test each agent type
        agent_types = ['sterman', 'reactive', 'newsvendor', 'basestock', 'movingavg']
        
        for agent_type in agent_types:
            print(f"\n Testing {agent_type} agent...")
            
            try:
                # Create agent
                agent = create_baseline_agent(agent_type, Position.RETAILER)
                
                # Test decision making
                decision = agent.make_decision(test_state)
                
                print(f"  ✅ {agent_type}: Created successfully, decision = {decision}")
                
                # Test reset
                agent.reset()
                print(f"  ✅ {agent_type}: Reset successfully")
                
            except Exception as e:
                print(f"  ❌ {agent_type}: Failed with error: {e}")
                return False
        
        print(f"\n🎉 All baseline agents working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you created src/scm_arena/beer_game/baseline_agents.py")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False


def test_baseline_games():
    """Test running games with different baseline agents (no LLM needed)"""
    
    print("\n🎮 Testing Baseline Agents in Games")
    print("=" * 35)
    
    try:
        from scm_arena.beer_game.game import BeerGame
        from scm_arena.beer_game.agents import Position
        from scm_arena.beer_game.baseline_agents import create_baseline_agent
        from scm_arena.evaluation.scenarios import DEMAND_PATTERNS
        
        # Test different baseline agents
        agent_types = ['sterman', 'reactive', 'newsvendor', 'basestock']
        demand_pattern = DEMAND_PATTERNS['classic'][:15]  # 15 rounds
        
        results = {}
        
        for agent_type in agent_types:
            print(f"\nTesting {agent_type} agents in full game...")
            
            # Create agents for all positions
            agents = {
                position: create_baseline_agent(agent_type, position)
                for position in Position
            }
            
            # Run game
            game = BeerGame(agents, demand_pattern)
            
            rounds_completed = 0
            while not game.is_complete():
                game.step()
                rounds_completed += 1
            
            # Get results
            game_results = game.get_results()
            results[agent_type] = game_results
            
            print(f"  ✅ {agent_type}: {rounds_completed} rounds, Cost=${game_results.total_cost:.0f}, Service={game_results.service_level:.1%}")
        
        # Compare baseline agents
        print(f"\n📊 BASELINE AGENTS COMPARISON:")
        print(f"{'Agent':<12} {'Cost':<8} {'Service':<8} {'Bullwhip':<8}")
        print("-" * 40)
        
        sorted_results = sorted(results.items(), key=lambda x: x[1].total_cost)
        
        for i, (agent_type, result) in enumerate(sorted_results):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}"
            print(f"{rank} {agent_type:<9} ${result.total_cost:<7.0f} {result.service_level:<7.1%} {result.bullwhip_ratio:<7.2f}")
        
        print(f"\n🎉 All baseline agents completed successfully!")
        print(f"Best performer: {sorted_results[0][0]} (${sorted_results[0][1].total_cost:.0f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Game test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_baseline_agents()
    
    if success:
        test_baseline_games()
        print(f"\n🚀 Baseline agents are working perfectly!")
        print(f"Next step: Add CLI integration for comprehensive study")
    else:
        print(f"\n❌ Fix the baseline agents first")
        sys.exit(1)