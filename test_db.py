#!/usr/bin/env python3
"""
SCM-Arena Database Testing Script
Tests the fixed memory window logic by analyzing experimental data.

Usage: python test_db.py [database_path]
"""

import sqlite3
import pandas as pd
import json
import sys
from pathlib import Path
from typing import Dict, List, Any


def connect_to_db(db_path: str) -> sqlite3.Connection:
    """Connect to the SCM-Arena database"""
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    return conn


def test_basic_db_structure(conn: sqlite3.Connection) -> None:
    """Test basic database structure and content"""
    print("🔍 TESTING DATABASE STRUCTURE")
    print("=" * 50)
    
    # Check tables exist
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    expected_tables = ['experiments', 'rounds', 'agent_rounds', 'game_states']
    print(f"📊 Tables found: {tables}")
    
    for table in expected_tables:
        if table in tables:
            print(f"✅ {table} table exists")
        else:
            print(f"❌ {table} table missing")
    
    # Check row counts
    print(f"\n📈 Row counts:")
    for table in tables:
        cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {table}: {count} rows")
    
    print()


def test_memory_window_consistency(conn: sqlite3.Connection) -> None:
    """Test that memory window is applied consistently across visibility levels"""
    print("🧠 TESTING MEMORY WINDOW CONSISTENCY")
    print("=" * 50)
    
    # Get experiments grouped by memory strategy and visibility
    query = """
    SELECT memory_strategy, visibility_level, COUNT(*) as experiment_count,
           AVG(total_cost) as avg_cost, AVG(service_level) as avg_service
    FROM experiments 
    GROUP BY memory_strategy, visibility_level
    ORDER BY memory_strategy, visibility_level
    """
    
    df = pd.read_sql_query(query, conn)
    print("📊 Experiments by memory strategy and visibility:")
    print(df.to_string(index=False))
    print()
    
    # Test: Check that 'none' memory strategy shows different results from others
    none_results = df[df['memory_strategy'] == 'none']
    other_results = df[df['memory_strategy'] != 'none']
    
    if len(none_results) > 0 and len(other_results) > 0:
        none_avg_cost = none_results['avg_cost'].mean()
        other_avg_cost = other_results['avg_cost'].mean()
        
        print(f"💡 Memory Strategy Impact Analysis:")
        print(f"   'none' memory average cost: ${none_avg_cost:.2f}")
        print(f"   Other memory average cost: ${other_avg_cost:.2f}")
        print(f"   Difference: {abs(none_avg_cost - other_avg_cost):.2f}")
        
        if abs(none_avg_cost - other_avg_cost) > 50:  # Threshold for meaningful difference
            print(f"✅ Memory strategies show significant performance differences")
        else:
            print(f"⚠️  Memory strategies show minimal differences - check if bug is fixed")
    
    print()


def test_agent_decision_patterns(conn: sqlite3.Connection) -> None:
    """Test agent decision patterns for memory consistency"""
    print("🤖 TESTING AGENT DECISION PATTERNS")
    print("=" * 50)
    
    # Sample some agent interactions to check memory window application
    query = """
    SELECT e.memory_strategy, e.visibility_level, e.experiment_id, 
           ar.round_number, ar.position, ar.prompt_sent
    FROM experiments e
    JOIN agent_rounds ar ON e.experiment_id = ar.experiment_id
    WHERE ar.round_number = 5  -- Check round 5 where history should be visible
    AND ar.position = 'wholesaler'  -- Focus on wholesaler for adjacent visibility tests
    LIMIT 10
    """
    
    cursor = conn.execute(query)
    rows = cursor.fetchall()
    
    print(f"📝 Sample agent prompts (Round 5, Wholesaler position):")
    print()
    
    for row in rows:
        memory_strategy = row['memory_strategy']
        visibility_level = row['visibility_level']
        prompt = row['prompt_sent']
        
        print(f"🔬 Memory: {memory_strategy}, Visibility: {visibility_level}")
        print(f"Experiment ID: {row['experiment_id']}")
        
        # Check for memory-related content in prompts
        if 'decision_history' in prompt.lower() or 'order history' in prompt.lower():
            print(f"✅ Contains decision history information")
        else:
            print(f"❌ No decision history found in prompt")
        
        if memory_strategy == 'none' and ('recent' in prompt.lower() or 'history' in prompt.lower()):
            print(f"⚠️  WARNING: 'none' memory strategy but prompt contains history!")
        
        # Check visibility information
        if visibility_level == 'adjacent':
            if 'supply chain visibility' in prompt.lower() or 'partner' in prompt.lower():
                print(f"✅ Contains adjacent visibility information")
            else:
                print(f"❌ No adjacent visibility info found")
        
        print(f"📄 Prompt preview: {prompt[:200]}...")
        print("-" * 40)
    
    print()


def test_game_state_consistency(conn: sqlite3.Connection) -> None:
    """Test game state data for consistency"""
    print("🎮 TESTING GAME STATE CONSISTENCY")
    print("=" * 50)
    
    # Check a sample game state JSON for memory window application
    query = """
    SELECT e.memory_strategy, e.visibility_level, gs.game_state_json
    FROM experiments e
    JOIN game_states gs ON e.experiment_id = gs.experiment_id
    WHERE gs.round_number = 5
    LIMIT 3
    """
    
    cursor = conn.execute(query)
    rows = cursor.fetchall()
    
    for row in rows:
        memory_strategy = row['memory_strategy']
        visibility_level = row['visibility_level']
        
        try:
            game_state = json.loads(row['game_state_json'])
            print(f"🎯 Memory: {memory_strategy}, Visibility: {visibility_level}")
            
            # Check if decision histories are present and properly limited
            players = game_state.get('players', {})
            
            for position, player_data in players.items():
                decision_history = player_data.get('decision_history', [])
                print(f"  {position}: {len(decision_history)} decision history entries")
                
                # For 'none' memory, should be empty; for 'short', should be <= 5
                if memory_strategy == 'none' and len(decision_history) > 0:
                    print(f"    ⚠️  WARNING: 'none' memory but has {len(decision_history)} history entries")
                elif memory_strategy == 'short' and len(decision_history) > 5:
                    print(f"    ⚠️  WARNING: 'short' memory but has {len(decision_history)} history entries")
                else:
                    print(f"    ✅ Decision history length consistent with memory strategy")
            
            print()
            
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON in game state")
    
    print()


def test_cost_analysis_by_conditions(conn: sqlite3.Connection) -> None:
    """Analyze costs by experimental conditions to detect patterns"""
    print("💰 COST ANALYSIS BY CONDITIONS")
    print("=" * 50)
    
    # Compare costs across memory strategies for each visibility level
    query = """
    SELECT memory_strategy, visibility_level,
           COUNT(*) as runs,
           ROUND(AVG(total_cost), 2) as avg_cost,
           ROUND(MIN(total_cost), 2) as min_cost,
           ROUND(MAX(total_cost), 2) as max_cost,
           ROUND(AVG(service_level), 3) as avg_service,
           ROUND(AVG(bullwhip_ratio), 3) as avg_bullwhip
    FROM experiments
    GROUP BY memory_strategy, visibility_level
    ORDER BY visibility_level, memory_strategy
    """
    
    df = pd.read_sql_query(query, conn)
    
    print("📊 Performance metrics by condition:")
    print(df.to_string(index=False))
    print()
    
    # Look for unexpected patterns
    for visibility in df['visibility_level'].unique():
        print(f"🔍 Analysis for {visibility} visibility:")
        subset = df[df['visibility_level'] == visibility]
        
        cost_range = subset['avg_cost'].max() - subset['avg_cost'].min()
        service_range = subset['avg_service'].max() - subset['avg_service'].min()
        
        print(f"  Cost range: ${cost_range:.2f}")
        print(f"  Service level range: {service_range:.3f}")
        
        if cost_range > 100:  # Significant cost difference
            print(f"  ✅ Memory strategies show significant cost differences")
        else:
            print(f"  ⚠️  Memory strategies show minimal cost differences")
        
        print()


def run_comprehensive_test(db_path: str) -> None:
    """Run all database tests"""
    print("🧪 SCM-ARENA DATABASE TESTING SUITE")
    print("=" * 60)
    print(f"📁 Database: {db_path}")
    print()
    
    try:
        conn = connect_to_db(db_path)
        
        # Run all tests
        test_basic_db_structure(conn)
        test_memory_window_consistency(conn)
        test_agent_decision_patterns(conn)
        test_game_state_consistency(conn)
        test_cost_analysis_by_conditions(conn)
        
        print("🎉 DATABASE TESTING COMPLETE!")
        print()
        print("🔍 SUMMARY:")
        print("- Check for ⚠️  warnings that indicate potential bugs")
        print("- Look for significant differences between memory strategies")
        print("- Verify that 'none' memory shows no decision history")
        print("- Confirm adjacent visibility works without memory leakage")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
    
    finally:
        if 'conn' in locals():
            conn.close()


def main():
    """Main function"""
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        db_path = "test_fixed_memory.db"
    
    run_comprehensive_test(db_path)


if __name__ == "__main__":
    main()