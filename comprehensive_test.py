#!/usr/bin/env python3
"""
Comprehensive SCM-Arena Testing Script
Tests all the critical fixes we implemented.

UPDATED: Now includes deterministic seeding system tests.

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


def test_deterministic_seeding_system():
    """Test that deterministic seeding system is working correctly"""
    print("\n🔍 TESTING DETERMINISTIC SEEDING SYSTEM")
    print("=" * 50)
    
    try:
        # Test seeding utilities import
        from src.scm_arena.utils.seeding import ExperimentSeeder, deterministic_seed, get_seed_for_condition
        print("✅ Seeding utilities imported successfully")
        
        # Test basic deterministic seeding
        condition = ("llama3.2", "short", "specific", "local", "classic", "modern", 1)
        seed1 = deterministic_seed(*condition)
        seed2 = deterministic_seed(*condition)
        
        if seed1 == seed2:
            print("✅ Deterministic seeding: Same conditions produce identical seeds")
            deterministic_test = True
        else:
            print("❌ Deterministic seeding: Same conditions produce different seeds")
            deterministic_test = False
        
        # Test that different conditions produce different seeds
        condition2 = ("llama3.2", "full", "specific", "local", "classic", "modern", 1)
        seed3 = deterministic_seed(*condition2)
        
        if seed1 != seed3:
            print("✅ Unique seeding: Different conditions produce different seeds")
            unique_test = True
        else:
            print("❌ Unique seeding: Different conditions produce same seed")
            unique_test = False
        
        # Test ExperimentSeeder class
        seeder = ExperimentSeeder(base_seed=42, deterministic=True)
        seed4 = seeder.get_seed(*condition)
        seed5 = seeder.get_seed(*condition)
        
        if seed4 == seed5:
            print("✅ ExperimentSeeder: Consistent seed generation")
            seeder_test = True
        else:
            print("❌ ExperimentSeeder: Inconsistent seed generation")
            seeder_test = False
        
        # Test scenario integration
        from src.scm_arena.evaluation.scenarios import generate_scenario_with_seed
        
        random1 = generate_scenario_with_seed("random", 5, seed=12345)
        random2 = generate_scenario_with_seed("random", 5, seed=12345)
        
        if random1 == random2:
            print("✅ Scenario seeding: Same seed produces identical random scenarios")
            scenario_test = True
        else:
            print("❌ Scenario seeding: Same seed produces different random scenarios")
            scenario_test = False
        
        return deterministic_test and unique_test and seeder_test and scenario_test
        
    except ImportError as e:
        print(f"❌ Seeding system not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing seeding system: {e}")
        return False


def test_cli_experimental_defaults():
    """Test that CLI includes all experimental factors in defaults"""
    print("\n🔍 TESTING CLI EXPERIMENTAL DEFAULTS")
    print("=" * 50)
    
    try:
        # Check CLI source for default values and seeding options
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
        
        # Check seeding options
        seeding_options = ["--base-seed", "--deterministic"]
        seeding_found = sum(1 for option in seeding_options if option in cli_content)
        
        if seeding_found == len(seeding_options):
            print("✅ CLI includes deterministic seeding options")
            seeding_complete = True
        else:
            print(f"❌ CLI missing seeding options: found {seeding_found}/{len(seeding_options)}")
            seeding_complete = False
        
        return visibility_complete and game_mode_complete and memory_complete and seeding_complete
        
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
               memory_strategy, visibility_level, game_mode, seed, base_seed, deterministic_seeding
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
        
        # Check seeding information if available
        if 'seed' in df.columns and 'deterministic_seeding' in df.columns:
            deterministic_count = df['deterministic_seeding'].sum()
            unique_seeds = df['seed'].nunique()
            print(f"📊 Seeding Info: {deterministic_count} deterministic experiments, {unique_seeds} unique seeds")
        
        # Show sample results
        print("\nSample service level results:")
        sample = df.sample(min(5, len(df)))
        for _, row in sample.iterrows():
            seed_info = f", seed={row.get('seed', 'N/A')}" if 'seed' in row else ""
            print(f"  {row['memory_strategy']}-{row['visibility_level']}-{row['game_mode']}: "
                  f"Service={row['service_level']:.3f}, Cost=${row['total_cost']:.0f}{seed_info}")
        
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
        
        # Check seeding information if available
        seeding_query = """
        SELECT deterministic_seeding, base_seed, COUNT(*) as count
        FROM experiments 
        GROUP BY deterministic_seeding, base_seed
        """
        
        try:
            seeding_df = pd.read_sql_query(seeding_query, conn)
            print(f"\n📊 Seeding Information:")
            for _, row in seeding_df.iterrows():
                method = "Deterministic" if row['deterministic_seeding'] else "Fixed"
                print(f"  {method} (base_seed={row['base_seed']}): {row['count']} experiments")
        except Exception:
            print(f"\n⚠️  Seeding information not available in database")
        
        conn.close()
        
        if all_complete:
            print(f"\n✅ All experimental factors have complete coverage")
        else:
            print(f"\n❌ Some experimental factors are missing")
        
        return all_complete
        
    except Exception as e:
        print(f"❌ Error testing experimental completeness: {e}")
        return False


def test_database_seeding_integration(db_path: str):
    """Test that database properly tracks seeding information"""
    print(f"\n🔍 TESTING DATABASE SEEDING INTEGRATION")
    print("=" * 50)
    
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Check if seeding columns exist
        cursor = conn.execute("PRAGMA table_info(experiments)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        required_seeding_columns = ['seed', 'base_seed', 'deterministic_seeding']
        missing_columns = [col for col in required_seeding_columns if col not in column_names]
        
        if missing_columns:
            print(f"❌ Database missing seeding columns: {missing_columns}")
            return False
        else:
            print("✅ Database includes all seeding columns")
        
        # Check seeding data quality
        query = """
        SELECT seed, base_seed, deterministic_seeding, COUNT(*) as count
        FROM experiments 
        GROUP BY seed, base_seed, deterministic_seeding
        ORDER BY count DESC
        LIMIT 10
        """
        
        df = pd.read_sql_query(query, conn)
        
        if len(df) == 0:
            print("❌ No seeding data found in database")
            return False
        
        print(f"\n📊 Seeding Data Analysis:")
        print(f"{'Seed':<12} {'Base Seed':<10} {'Deterministic':<12} {'Count':<6}")
        print("-" * 45)
        
        deterministic_experiments = 0
        unique_seeds = set()
        
        for _, row in df.iterrows():
            det_str = "Yes" if row['deterministic_seeding'] else "No"
            print(f"{row['seed']:<12} {row['base_seed']:<10} {det_str:<12} {row['count']:<6}")
            
            if row['deterministic_seeding']:
                deterministic_experiments += row['count']
            unique_seeds.add(row['seed'])
        
        # Get total experiments
        total_query = "SELECT COUNT(*) as total FROM experiments"
        total_df = pd.read_sql_query(total_query, conn)
        total_experiments = total_df['total'].iloc[0]
        
        print(f"\n📊 Seeding Summary:")
        print(f"  Total experiments: {total_experiments}")
        print(f"  Deterministic experiments: {deterministic_experiments}")
        print(f"  Unique seeds: {len(unique_seeds)}")
        print(f"  Deterministic percentage: {(deterministic_experiments/total_experiments)*100:.1f}%")
        
        conn.close()
        
        seeding_integration_test = len(unique_seeds) > 1 or deterministic_experiments > 0
        if seeding_integration_test:
            print("✅ Database seeding integration working")
        else:
            print("❌ Database seeding integration not working")
        
        return seeding_integration_test
        
    except Exception as e:
        print(f"❌ Error testing database seeding integration: {e}")
        return False


def run_comprehensive_test(db_path: str = "test_fixed_memory.db"):
    """Run all tests and provide summary"""
    print("🧪 COMPREHENSIVE SCM-ARENA TESTING SUITE (WITH DETERMINISTIC SEEDING)")
    print("=" * 80)
    print(f"📁 Testing with database: {db_path}")
    print()
    
    # Track test results
    test_results = {}
    
    # Run all tests
    test_results["Demand Scenario Consistency"] = test_demand_scenario_consistency()
    test_results["Deterministic Seeding System"] = test_deterministic_seeding_system()
    test_results["CLI Experimental Defaults"] = test_cli_experimental_defaults()
    
    # Database tests (only if database exists)
    if Path(db_path).exists():
        test_results["Service Level Calculation"] = test_database_service_level_calculation(db_path)
        test_results["Bullwhip Ratio Calculation"] = test_database_bullwhip_calculation(db_path)
        test_results["Memory Window Consistency"] = test_memory_window_consistency(db_path)
        test_results["Experimental Completeness"] = test_experimental_completeness(db_path)
        test_results["Database Seeding Integration"] = test_database_seeding_integration(db_path)
    else:
        print(f"\n⚠️  Database {db_path} not found - skipping database tests")
        print("   Run an experiment first to generate test data")
    
    # Summary
    print("\n" + "=" * 80)
    print("🎯 TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<35} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Your fixes and deterministic seeding are working correctly.")
        print("🎯 Ready for reproducible scientific benchmarking!")
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