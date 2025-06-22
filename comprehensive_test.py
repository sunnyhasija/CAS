#!/usr/bin/env python3
"""
Comprehensive SCM-Arena Testing Script
Tests all the critical fixes we implemented.

Usage: python comprehensive_test.py [database_path]
"""

import sqlite3
import pandas as pd
import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Any


def test_demand_scenario_consistency():
    """Test that all demand scenarios have consistent baseline levels"""
    print("🔍 TESTING DEMAND SCENARIO CONSISTENCY")
    print("=" * 50)
    
    try:
        # Import scenarios to test
        from src.scm_arena.evaluation.scenarios import DEMAND_PATTERNS, get_scenario_statistics
        
        stats = get_scenario_statistics()
        
        print("📊 Demand Scenario Statistics:")
        print(f"{'Scenario':<10} {'Mean':<6} {'Std':<6} {'Min':<4} {'Max':<4} {'Baseline Deviation':<18}")
        print("-" * 65)
        
        baseline = 4.0
        all_consistent = True
        
        for scenario, stat in stats.items():
            deviation = abs(stat['mean'] - baseline)
            status = "✅" if deviation <= 1.0 else "❌"
            
            if deviation > 1.0:
                all_consistent = False
            
            print(f"{scenario:<10} {stat['mean']:<6.1f} {stat['std']:<6.1f} {stat['min']:<4.0f} {stat['max']:<4.0f} {deviation:<6.1f} ({status})")
        
        if all_consistent:
            print("\n✅ All scenarios have consistent baseline demand (~4.0)")
        else:
            print("\n❌ Some scenarios have inconsistent baseline demand")
        
        # Show first 10 periods for verification
        print("\nFirst 10 periods preview:")
        for scenario, pattern in DEMAND_PATTERNS.items():
            print(f"  {scenario:>8}: {pattern[:10]}")
        
        return all_consistent
        
    except Exception as e:
        print(f"❌ Error testing demand scenarios: {e}")
        return False


def test_cli_experimental_defaults():
    """Test that CLI includes all experimental factors in defaults"""
    print("\n🔍 TESTING CLI EXPERIMENTAL DEFAULTS")
    print("=" * 50)
    
    try:
        # Check CLI source for default values
        cli_file = Path("src/scm_arena/cli.py")
        if not cli_file.exists():
            print("❌ CLI file not found")
            return False
        
        with open(cli_file, 'r') as f:
            cli_content = f.read()
        
        # Check visibility defaults
        if "default=['local', 'adjacent', 'full']" in cli_content:
            print("✅ Visibility defaults include all levels: local, adjacent, full")
            visibility_complete = True
        else:
            print("❌ Visibility defaults incomplete")
            visibility_complete = False
        
        # Check game mode defaults  
        if "default=['modern', 'classic']" in cli_content:
            print("✅ Game mode defaults include both: modern, classic")
            game_mode_complete = True
        else:
            print("❌ Game mode defaults incomplete")
            game_mode_complete = False
        
        # Check memory defaults
        if "default=['none', 'short', 'full']" in cli_content:
            print("✅ Memory defaults include: none, short, full")
            memory_complete = True
        else:
            print("❌ Memory defaults incomplete")
            memory_complete = False
        
        return visibility_complete and game_mode_complete and memory_complete
        
    except Exception as e:
        print(f"❌ Error testing CLI defaults: {e}")
        return False


def test_database_service_level_calculation(db_path: str):
    """Test service level calculation in database results"""
    print(f"\n🔍 TESTING SERVICE LEVEL CALCULATION")
    print("=" * 50)
    
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Get service level values from experiments
        query = """
        SELECT experiment_id, service_level, total_cost, bullwhip_ratio,
               memory_strategy, visibility_level, game_mode
        FROM experiments 
        ORDER BY service_level
        """
        
        df = pd.read_sql_query(query, conn)
        
        if len(df) == 0:
            print("❌ No experiments found in database")
            return False
        
        print(f"📊 Service Level Analysis ({len(df)} experiments):")
        print(f"  Range: {df['service_level'].min():.3f} - {df['service_level'].max():.3f}")
        print(f"  Mean: {df['service_level'].mean():.3f}")
        print(f"  Std: {df['service_level'].std():.3f}")
        
        # Check for reasonable service level values
        if df['service_level'].min() < 0 or df['service_level'].max() > 1:
            print("❌ Service levels outside valid range [0,1]")
            return False
        
        # Check for variation in service levels
        if df['service_level'].std() < 0.01:
            print("⚠️  Very low service level variation - check calculation")
        else:
            print("✅ Service levels show reasonable variation")
        
        # Show sample results
        print("\nSample service level results:")
        sample = df.sample(min(5, len(df)))
        for _, row in sample.iterrows():
            print(f"  {row['memory_strategy']}-{row['visibility_level']}-{row['game_mode']}: "
                  f"Service={row['service_level']:.3f}, Cost=${row['total_cost']:.0f}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error testing service level: {e}")
        return False


def test_database_bullwhip_calculation(db_path: str):
    """Test bullwhip ratio calculation in database results"""
    print(f"\n🔍 TESTING BULLWHIP RATIO CALCULATION")
    print("=" * 50)
    
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Get bullwhip values from experiments
        query = """
        SELECT experiment_id, bullwhip_ratio, service_level, total_cost,
               memory_strategy, visibility_level, game_mode
        FROM experiments 
        ORDER BY bullwhip_ratio
        """
        
        df = pd.read_sql_query(query, conn)
        
        if len(df) == 0:
            print("❌ No experiments found in database")
            return False
        
        print(f"📊 Bullwhip Ratio Analysis ({len(df)} experiments):")
        print(f"  Range: {df['bullwhip_ratio'].min():.3f} - {df['bullwhip_ratio'].max():.3f}")
        print(f"  Mean: {df['bullwhip_ratio'].mean():.3f}")
        print(f"  Std: {df['bullwhip_ratio'].std():.3f}")
        
        # Check for reasonable bullwhip values (should be >= 1.0)
        if df['bullwhip_ratio'].min() < 0.5:
            print("⚠️  Some bullwhip ratios < 0.5 - check calculation")
        
        # Check for variation in bullwhip ratios
        if df['bullwhip_ratio'].std() < 0.01:
            print("⚠️  Very low bullwhip variation - check calculation")
        else:
            print("✅ Bullwhip ratios show reasonable variation")
        
        # Show sample results
        print("\nSample bullwhip ratio results:")
        sample = df.sample(min(5, len(df)))
        for _, row in sample.iterrows():
            print(f"  {row['memory_strategy']}-{row['visibility_level']}-{row['game_mode']}: "
                  f"Bullwhip={row['bullwhip_ratio']:.3f}, Service={row['service_level']:.3f}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error testing bullwhip ratio: {e}")
        return False


def test_memory_window_consistency(db_path: str):
    """Test that memory window fix is working correctly"""
    print(f"\n🔍 TESTING MEMORY WINDOW CONSISTENCY")
    print("=" * 50)
    
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Check that different memory strategies produce different results
        query = """
        SELECT memory_strategy, visibility_level,
               COUNT(*) as experiments,
               AVG(total_cost) as avg_cost,
               AVG(service_level) as avg_service,
               AVG(bullwhip_ratio) as avg_bullwhip
        FROM experiments 
        GROUP BY memory_strategy, visibility_level
        ORDER BY memory_strategy, visibility_level
        """
        
        df = pd.read_sql_query(query, conn)
        
        if len(df) == 0:
            print("❌ No experiments found in database")
            return False
        
        print("📊 Memory Strategy Performance by Visibility:")
        print(f"{'Memory':<8} {'Visibility':<10} {'Experiments':<12} {'Avg Cost':<10} {'Avg Service':<12} {'Avg Bullwhip':<12}")
        print("-" * 80)
        
        memory_effects = {}
        
        for _, row in df.iterrows():
            print(f"{row['memory_strategy']:<8} {row['visibility_level']:<10} {row['experiments']:<12} "
                  f"${row['avg_cost']:<9.0f} {row['avg_service']:<12.3f} {row['avg_bullwhip']:<12.3f}")
            
            # Track memory strategy effects
            if row['visibility_level'] not in memory_effects:
                memory_effects[row['visibility_level']] = {}
            memory_effects[row['visibility_level']][row['memory_strategy']] = row['avg_cost']
        
        # Check for meaningful differences between memory strategies
        significant_differences = 0
        
        for visibility, strategies in memory_effects.items():
            if len(strategies) > 1:
                costs = list(strategies.values())
                cost_range = max(costs) - min(costs)
                print(f"\n{visibility} visibility cost range: ${cost_range:.0f}")
                
                if cost_range > 50:  # Threshold for meaningful difference
                    print(f"  ✅ Significant memory strategy differences in {visibility} visibility")
                    significant_differences += 1
                else:
                    print(f"  ⚠️  Minimal memory strategy differences in {visibility} visibility")
        
        conn.close()
        
        if significant_differences > 0:
            print(f"\n✅ Memory window fix working: {significant_differences} visibility levels show significant differences")
            return True
        else:
            print(f"\n⚠️  Memory strategies show minimal differences - may need investigation")
            return False
        
    except Exception as e:
        print(f"❌ Error testing memory window consistency: {e}")
        return False


def test_experimental_completeness(db_path: str):
    """Test that all experimental factors are represented"""
    print(f"\n🔍 TESTING EXPERIMENTAL COMPLETENESS")
    print("=" * 50)
    
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Check experimental factor coverage
        queries = {
            "Memory Strategies": "SELECT DISTINCT memory_strategy FROM experiments ORDER BY memory_strategy",
            "Visibility Levels": "SELECT DISTINCT visibility_level FROM experiments ORDER BY visibility_level", 
            "Game Modes": "SELECT DISTINCT game_mode FROM experiments ORDER BY game_mode",
            "Prompt Types": "SELECT DISTINCT prompt_type FROM experiments ORDER BY prompt_type",
            "Scenarios": "SELECT DISTINCT scenario FROM experiments ORDER BY scenario"
        }
        
        expected = {
            "Memory Strategies": ["full", "none", "short"],
            "Visibility Levels": ["adjacent", "full", "local"],
            "Game Modes": ["classic", "modern"],
            "Prompt Types": ["neutral", "specific"],
            "Scenarios": ["classic"]
        }
        
        all_complete = True
        
        for factor, query in queries.items():
            df = pd.read_sql_query(query, conn)
            found = df.iloc[:, 0].tolist()
            expected_values = expected[factor]
            
            print(f"\n{factor}:")
            print(f"  Found: {found}")
            print(f"  Expected: {expected_values}")
            
            missing = set(expected_values) - set(found)
            extra = set(found) - set(expected_values)
            
            if missing:
                print(f"  ❌ Missing: {missing}")
                all_complete = False
            else:
                print(f"  ✅ Complete coverage")
            
            if extra:
                print(f"  ℹ️  Extra: {extra}")
        
        # Check total experimental combinations
        combination_query = """
        SELECT memory_strategy, visibility_level, game_mode, prompt_type, scenario,
               COUNT(*) as runs
        FROM experiments 
        GROUP BY memory_strategy, visibility_level, game_mode, prompt_type, scenario
        ORDER BY memory_strategy, visibility_level, game_mode
        """
        
        df = pd.read_sql_query(combination_query, conn)
        unique_combinations = len(df)
        
        print(f"\n📊 Experimental Combinations:")
        print(f"  Total unique combinations: {unique_combinations}")
        print(f"  Total experiment runs: {df['runs'].sum()}")
        print(f"  Runs per combination: {df['runs'].mean():.1f} ± {df['runs'].std():.1f}")
        
        conn.close()
        
        if all_complete:
            print(f"\n✅ All experimental factors have complete coverage")
        else:
            print(f"\n❌ Some experimental factors are missing")
        
        return all_complete
        
    except Exception as e:
        print(f"❌ Error testing experimental completeness: {e}")
        return False


def run_comprehensive_test(db_path: str = "test_fixed_memory.db"):
    """Run all tests and provide summary"""
    print("🧪 COMPREHENSIVE SCM-ARENA TESTING SUITE")
    print("=" * 60)
    print(f"📁 Testing with database: {db_path}")
    print()
    
    # Track test results
    test_results = {}
    
    # Run all tests
    test_results["Demand Scenario Consistency"] = test_demand_scenario_consistency()
    test_results["CLI Experimental Defaults"] = test_cli_experimental_defaults()
    
    # Database tests (only if database exists)
    if Path(db_path).exists():
        test_results["Service Level Calculation"] = test_database_service_level_calculation(db_path)
        test_results["Bullwhip Ratio Calculation"] = test_database_bullwhip_calculation(db_path)
        test_results["Memory Window Consistency"] = test_memory_window_consistency(db_path)
        test_results["Experimental Completeness"] = test_experimental_completeness(db_path)
    else:
        print(f"\n⚠️  Database {db_path} not found - skipping database tests")
        print("   Run an experiment first to generate test data")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Your fixes are working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for issues.")
    
    return passed == total


def main():
    """Main function"""
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        db_path = "test_fixed_memory.db"
    
    success = run_comprehensive_test(db_path)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()