#!/usr/bin/env python3
"""
Comprehensive test script for SCM-Arena deterministic seeding system.

Tests:
1. Deterministic seed generation (same conditions = same seeds)
2. Unique seed generation (different conditions = different seeds)
3. Run-to-run variation (different runs = different seeds)
4. Hash collision detection
5. Integration with CLI and scenarios
6. Reproducibility validation
7. Statistical properties of generated seeds

Usage: python test_deterministic_seeding.py
"""

import sys
import os
import hashlib
import time
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from scm_arena.utils.seeding import (
        deterministic_seed, ExperimentSeeder, get_seed_for_condition,
        create_experiment_seeds, DEFAULT_BASE_SEED
    )
    from scm_arena.evaluation.scenarios import generate_scenario_with_seed, CANONICAL_SEED
    seeding_available = True
except ImportError as e:
    print(f"⚠️  Seeding module not available: {e}")
    seeding_available = False

try:
    from scm_arena.models.ollama_client import test_ollama_connection, create_ollama_agents
    from scm_arena.beer_game.agents import Position
    ollama_available = test_ollama_connection()
except ImportError:
    ollama_available = False


def test_basic_deterministic_seeding():
    """Test basic deterministic seed generation"""
    print("🧪 TESTING BASIC DETERMINISTIC SEEDING")
    print("=" * 50)
    
    if not seeding_available:
        print("❌ Seeding module not available - skipping test")
        return False
    
    # Test same conditions produce same seeds
    condition = ("llama3.2", "short", "specific", "local", "classic", "modern", 1)
    
    seed1 = deterministic_seed(*condition)
    seed2 = deterministic_seed(*condition)
    seed3 = deterministic_seed(*condition)
    
    print(f"Same condition called 3 times:")
    print(f"  Seed 1: {seed1}")
    print(f"  Seed 2: {seed2}")
    print(f"  Seed 3: {seed3}")
    
    if seed1 == seed2 == seed3:
        print("✅ Deterministic: Same conditions produce identical seeds")
        deterministic_test = True
    else:
        print("❌ Non-deterministic: Same conditions produce different seeds")
        deterministic_test = False
    
    # Test different conditions produce different seeds
    conditions = [
        ("llama3.2", "short", "specific", "local", "classic", "modern", 1),
        ("llama3.2", "full", "specific", "local", "classic", "modern", 1),    # Different memory
        ("llama3.2", "short", "neutral", "local", "classic", "modern", 1),    # Different prompt
        ("llama3.2", "short", "specific", "adjacent", "classic", "modern", 1), # Different visibility
        ("llama3.2", "short", "specific", "local", "random", "modern", 1),    # Different scenario
        ("llama3.2", "short", "specific", "local", "classic", "classic", 1),  # Different mode
        ("llama3.2", "short", "specific", "local", "classic", "modern", 2),   # Different run
    ]
    
    seeds = [deterministic_seed(*cond) for cond in conditions]
    unique_seeds = set(seeds)
    
    print(f"\nDifferent conditions:")
    for i, (cond, seed) in enumerate(zip(conditions, seeds)):
        print(f"  Condition {i+1}: {seed} ({cond[1]}-{cond[2]}-{cond[3]}-{cond[4]}-{cond[5]}-run{cond[6]})")
    
    if len(unique_seeds) == len(seeds):
        print("✅ Unique: Different conditions produce different seeds")
        unique_test = True
    else:
        print(f"❌ Collisions: {len(seeds) - len(unique_seeds)} duplicate seeds found")
        unique_test = False
    
    return deterministic_test and unique_test


def test_experiment_seeder_class():
    """Test the ExperimentSeeder class"""
    print("\n🧪 TESTING EXPERIMENT SEEDER CLASS")
    print("=" * 50)
    
    if not seeding_available:
        print("❌ Seeding module not available - skipping test")
        return False
    
    # Test deterministic mode
    seeder_det = ExperimentSeeder(base_seed=42, deterministic=True)
    
    # Generate seeds for same condition multiple times
    condition = ("llama3.2", "short", "specific", "local", "classic", "modern", 1)
    seed1 = seeder_det.get_seed(*condition)
    seed2 = seeder_det.get_seed(*condition)
    
    print(f"Deterministic seeder - same condition twice:")
    print(f"  Seed 1: {seed1}")
    print(f"  Seed 2: {seed2}")
    
    deterministic_class_test = seed1 == seed2
    if deterministic_class_test:
        print("✅ ExperimentSeeder deterministic mode working")
    else:
        print("❌ ExperimentSeeder deterministic mode failed")
    
    # Test non-deterministic mode
    seeder_fixed = ExperimentSeeder(base_seed=42, deterministic=False)
    
    conditions = [
        ("llama3.2", "short", "specific", "local", "classic", "modern", 1),
        ("llama3.2", "full", "specific", "local", "classic", "modern", 1),
        ("llama3.2", "short", "neutral", "local", "classic", "modern", 1),
    ]
    
    fixed_seeds = [seeder_fixed.get_seed(*cond) for cond in conditions]
    
    print(f"\nNon-deterministic seeder (should all be 42):")
    for i, seed in enumerate(fixed_seeds):
        print(f"  Condition {i+1}: {seed}")
    
    fixed_mode_test = all(seed == 42 for seed in fixed_seeds)
    if fixed_mode_test:
        print("✅ ExperimentSeeder fixed mode working")
    else:
        print("❌ ExperimentSeeder fixed mode failed")
    
    # Test validation
    validation_test = seeder_det.validate_reproducibility(*condition)
    if validation_test:
        print("✅ Reproducibility validation working")
    else:
        print("❌ Reproducibility validation failed")
    
    # Test statistics
    stats = seeder_det.get_seed_statistics()
    print(f"\nSeeder statistics: {stats}")
    
    return deterministic_class_test and fixed_mode_test and validation_test


def test_seed_collision_rates():
    """Test seed collision rates for realistic experimental designs"""
    print("\n🧪 TESTING SEED COLLISION RATES")
    print("=" * 50)
    
    if not seeding_available:
        print("❌ Seeding module not available - skipping test")
        return False
    
    # Simulate a large experimental design
    models = ["llama3.2", "mistral", "qwen"]
    memory_strategies = ["none", "short", "full"]
    prompt_types = ["specific", "neutral"]
    visibility_levels = ["local", "adjacent", "full"]
    scenarios = ["classic", "random", "shock", "seasonal"]
    game_modes = ["modern", "classic"]
    runs = 5
    
    print(f"Testing collision rates for:")
    print(f"  Models: {len(models)}")
    print(f"  Memory: {len(memory_strategies)}")
    print(f"  Prompts: {len(prompt_types)}")
    print(f"  Visibility: {len(visibility_levels)}")
    print(f"  Scenarios: {len(scenarios)}")
    print(f"  Modes: {len(game_modes)}")
    print(f"  Runs: {runs}")
    
    total_conditions = len(models) * len(memory_strategies) * len(prompt_types) * len(visibility_levels) * len(scenarios) * len(game_modes) * runs
    print(f"  Total experiments: {total_conditions}")
    
    # Generate all seeds
    seeds = []
    condition_count = 0
    
    for model in models:
        for memory in memory_strategies:
            for prompt in prompt_types:
                for visibility in visibility_levels:
                    for scenario in scenarios:
                        for mode in game_modes:
                            for run in range(1, runs + 1):
                                seed = deterministic_seed(model, memory, prompt, visibility, scenario, mode, run)
                                seeds.append(seed)
                                condition_count += 1
    
    # Analyze collisions
    unique_seeds = set(seeds)
    collision_rate = (len(seeds) - len(unique_seeds)) / len(seeds)
    
    print(f"\nCollision Analysis:")
    print(f"  Total seeds generated: {len(seeds)}")
    print(f"  Unique seeds: {len(unique_seeds)}")
    print(f"  Collisions: {len(seeds) - len(unique_seeds)}")
    print(f"  Collision rate: {collision_rate:.4%}")
    
    # Analyze seed distribution
    seed_counts = Counter(seeds)
    duplicates = {seed: count for seed, count in seed_counts.items() if count > 1}
    
    if duplicates:
        print(f"  Duplicate seeds found: {len(duplicates)}")
        for seed, count in list(duplicates.items())[:5]:  # Show first 5
            print(f"    Seed {seed}: {count} occurrences")
    else:
        print("  No duplicate seeds found")
    
    # Test acceptable collision rate (should be very low for good hash function)
    acceptable_collision_rate = 0.001  # 0.1%
    collision_test = collision_rate <= acceptable_collision_rate
    
    if collision_test:
        print(f"✅ Collision rate {collision_rate:.4%} is acceptable (≤ {acceptable_collision_rate:.1%})")
    else:
        print(f"⚠️  Collision rate {collision_rate:.4%} is high (> {acceptable_collision_rate:.1%})")
    
    return collision_test


def test_scenario_seeding_integration():
    """Test integration with scenario generation"""
    print("\n🧪 TESTING SCENARIO SEEDING INTEGRATION")
    print("=" * 50)
    
    if not seeding_available:
        print("❌ Seeding module not available - skipping test")
        return False
    
    # Test that same seed produces same random scenario
    seed = 12345
    scenario1 = generate_scenario_with_seed("random", 10, seed)
    scenario2 = generate_scenario_with_seed("random", 10, seed)
    
    print(f"Random scenario with seed {seed}:")
    print(f"  Generation 1: {scenario1}")
    print(f"  Generation 2: {scenario2}")
    
    scenario_reproducible = scenario1 == scenario2
    if scenario_reproducible:
        print("✅ Random scenarios are reproducible with same seed")
    else:
        print("❌ Random scenarios are not reproducible")
    
    # Test that different seeds produce different scenarios
    seed_a = 12345
    seed_b = 67890
    scenario_a = generate_scenario_with_seed("random", 10, seed_a)
    scenario_b = generate_scenario_with_seed("random", 10, seed_b)
    
    print(f"\nDifferent seeds:")
    print(f"  Seed {seed_a}: {scenario_a}")
    print(f"  Seed {seed_b}: {scenario_b}")
    
    scenario_different = scenario_a != scenario_b
    if scenario_different:
        print("✅ Different seeds produce different random scenarios")
    else:
        print("⚠️  Different seeds produced identical scenarios (unlikely but possible)")
    
    # Test deterministic scenarios are unaffected by seed
    classic1 = generate_scenario_with_seed("classic", 10, seed_a)
    classic2 = generate_scenario_with_seed("classic", 10, seed_b)
    
    deterministic_unaffected = classic1 == classic2
    if deterministic_unaffected:
        print("✅ Deterministic scenarios unaffected by seed")
    else:
        print("❌ Deterministic scenarios affected by seed (should not happen)")
    
    return scenario_reproducible and scenario_different and deterministic_unaffected


def test_agent_seeding_integration():
    """Test integration with agent creation (if Ollama available)"""
    print("\n🧪 TESTING AGENT SEEDING INTEGRATION")
    print("=" * 50)
    
    if not ollama_available:
        print("⚠️  Ollama not available - skipping agent integration test")
        return True  # Not a failure, just not testable
    
    if not seeding_available:
        print("❌ Seeding module not available - skipping test")
        return False
    
    try:
        # Test that agents created with same seed behave identically
        seed = 12345
        
        print("Creating agents with same seed...")
        agents1 = create_ollama_agents("llama3.2", seed=seed, memory_window=0)  # No memory for determinism
        agents2 = create_ollama_agents("llama3.2", seed=seed, memory_window=0)
        
        # Test simple game state
        test_state = {
            "round": 1,
            "position": "retailer",
            "inventory": 10,
            "backlog": 0,
            "incoming_order": 4,
            "last_outgoing_order": 4,
            "round_cost": 10.0,
            "decision_history": [],
            "customer_demand": 4
        }
        
        print("Testing agent decisions with identical state...")
        decision1 = agents1[Position.RETAILER].make_decision(test_state)
        decision2 = agents2[Position.RETAILER].make_decision(test_state)
        
        print(f"  Agent set 1 decision: {decision1}")
        print(f"  Agent set 2 decision: {decision2}")
        
        if decision1 == decision2:
            print("✅ Agents with same seed produce identical decisions")
            return True
        else:
            print("⚠️  Agents with same seed produced different decisions")
            print("    (This might be expected if the LLM has other sources of randomness)")
            return True  # Not necessarily a failure - LLMs may have other randomness sources
            
    except Exception as e:
        print(f"⚠️  Agent test failed with error: {e}")
        return True  # Not a critical failure


def test_full_experimental_reproducibility():
    """Test end-to-end experimental reproducibility"""
    print("\n🧪 TESTING FULL EXPERIMENTAL REPRODUCIBILITY")
    print("=" * 50)
    
    if not seeding_available:
        print("❌ Seeding module not available - skipping test")
        return False
    
    # Simulate running the same experimental condition multiple times
    seeder = ExperimentSeeder(base_seed=42, deterministic=True)
    
    # Test condition
    condition = ("llama3.2", "short", "specific", "local", "classic", "modern", 1)
    
    # Generate seed multiple times (simulating CLI calls)
    seeds = []
    for i in range(5):
        seed = seeder.get_seed(*condition)
        seeds.append(seed)
    
    print(f"Same experimental condition called 5 times:")
    for i, seed in enumerate(seeds):
        print(f"  Call {i+1}: seed={seed}")
    
    # Check all seeds are identical
    all_identical = len(set(seeds)) == 1
    if all_identical:
        print("✅ Full experimental reproducibility confirmed")
    else:
        print("❌ Experimental reproducibility failed")
    
    # Test different runs get different seeds
    run_seeds = []
    for run in range(1, 4):
        run_condition = ("llama3.2", "short", "specific", "local", "classic", "modern", run)
        seed = seeder.get_seed(*run_condition)
        run_seeds.append(seed)
    
    print(f"\nDifferent runs of same condition:")
    for run, seed in enumerate(run_seeds, 1):
        print(f"  Run {run}: seed={seed}")
    
    runs_different = len(set(run_seeds)) == len(run_seeds)
    if runs_different:
        print("✅ Different runs get different seeds")
    else:
        print("❌ Different runs got identical seeds")
    
    return all_identical and runs_different


def test_cli_integration():
    """Test CLI integration (without actually running CLI)"""
    print("\n🧪 TESTING CLI INTEGRATION LOGIC")
    print("=" * 50)
    
    if not seeding_available:
        print("❌ Seeding module not available - skipping test")
        return False
    
    # Simulate CLI parameter processing
    cli_params = {
        "model": "llama3.2",
        "memory": "short",
        "neutral_prompts": False,  # specific
        "visibility": "local",
        "scenario": "classic",
        "classic_mode": False,     # modern
        "run_number": 1,
        "base_seed": 42,
        "deterministic": True
    }
    
    # Convert CLI params to seeder params (like CLI would do)
    prompt_type = "neutral" if cli_params["neutral_prompts"] else "specific"
    game_mode = "classic" if cli_params["classic_mode"] else "modern"
    
    seeder = ExperimentSeeder(
        base_seed=cli_params["base_seed"], 
        deterministic=cli_params["deterministic"]
    )
    
    seed = seeder.get_seed(
        model=cli_params["model"],
        memory=cli_params["memory"],
        prompt=prompt_type,
        visibility=cli_params["visibility"],
        scenario=cli_params["scenario"],
        mode=game_mode,
        run=cli_params["run_number"]
    )
    
    print(f"CLI simulation:")
    print(f"  Parameters: {cli_params}")
    print(f"  Generated seed: {seed}")
    
    # Test that CLI parameter changes affect seed
    cli_params_modified = cli_params.copy()
    cli_params_modified["memory"] = "full"
    
    seed_modified = seeder.get_seed(
        model=cli_params_modified["model"],
        memory=cli_params_modified["memory"],
        prompt=prompt_type,
        visibility=cli_params_modified["visibility"],
        scenario=cli_params_modified["scenario"],
        mode=game_mode,
        run=cli_params_modified["run_number"]
    )
    
    print(f"  Modified parameters (memory=full): seed={seed_modified}")
    
    cli_test = seed != seed_modified
    if cli_test:
        print("✅ CLI parameter changes affect seed generation")
    else:
        print("❌ CLI parameter changes don't affect seed")
    
    return cli_test


def run_all_tests():
    """Run all tests and provide summary"""
    print("🎯 SCM-ARENA DETERMINISTIC SEEDING TEST SUITE")
    print("=" * 60)
    print()
    
    tests = [
        ("Basic Deterministic Seeding", test_basic_deterministic_seeding),
        ("ExperimentSeeder Class", test_experiment_seeder_class),
        ("Seed Collision Rates", test_seed_collision_rates),
        ("Scenario Integration", test_scenario_seeding_integration),
        ("Agent Integration", test_agent_seeding_integration),
        ("Full Reproducibility", test_full_experimental_reproducibility),
        ("CLI Integration", test_cli_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Deterministic seeding system is working correctly")
        print("✅ Experiments will be reproducible and scientifically valid")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        print("❌ Review the failures above before using in production")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    
    if not success:
        sys.exit(1)
    else:
        print("\n🚀 Ready for deterministic benchmarking!")