"""
Predefined demand scenarios for Beer Game evaluation with FIXED demand consistency.

CRITICAL FIX: All scenarios now use consistent baseline demand levels for comparability.
- Classic scenario: steady state = 4 units (matches historical Beer Game)
- Random scenario: mean = 4.0 units (matches classic baseline)  
- Shock scenario: base = 4 units (consistent with other scenarios)
- Seasonal scenario: base = 4.0 units (consistent baseline)

This ensures scenarios are comparable for cross-analysis and benchmarking.

UPDATED: Now uses deterministic seeding system instead of hardcoded seed.
"""

import numpy as np
from typing import List, Dict, Any

# Import deterministic seeding
try:
    from ..utils.seeding import DEFAULT_BASE_SEED
    CANONICAL_SEED = DEFAULT_BASE_SEED  # Use same default as seeding system
except ImportError:
    # Fallback if seeding module not available
    CANONICAL_SEED = 42


def generate_classic_demand(length: int = 50) -> List[int]:
    """
    Generate the classic Beer Game demand pattern.
    
    Standard academic Beer Game pattern:
    - Periods 1-4: 4 units (steady state)
    - Periods 5-12: 8 units (step increase) 
    - Periods 13+: 4 units (return to steady state)
    """
    pattern = []
    pattern.extend([4] * 4)      # Initial steady state
    pattern.extend([8] * 8)      # Step increase
    remaining = length - len(pattern)
    pattern.extend([4] * remaining)  # Return to steady state
    return pattern[:length]


def generate_random_demand(length: int = 50, mean: float = 4.0, std: float = 2.0, seed: int = CANONICAL_SEED) -> List[int]:
    """
    Generate random demand following normal distribution.
    
    FIXED: Now uses mean=4.0 to match classic scenario baseline for comparability.
    UPDATED: Uses deterministic seeding system default.
    
    Args:
        length: Number of periods
        mean: Mean demand (FIXED: 4.0 to match classic baseline)
        std: Standard deviation of demand
        seed: Random seed for reproducibility (default: from seeding system)
    """
    np.random.seed(seed % (2**32))
    demands = np.random.normal(mean, std, length)
    demands = np.maximum(1, demands).astype(int)  # Ensure minimum demand of 1
    return demands.tolist()


def generate_shock_demand(length: int = 50, base_demand: int = 4, shock_magnitude: int = 12, shock_frequency: int = 10, seed: int = CANONICAL_SEED) -> List[int]:
    """
    Generate demand with periodic shocks.
    
    FIXED: Now uses base_demand=4 to match classic scenario baseline.
    UPDATED: Can use seed for shock timing variation.
    
    Args:
        length: Number of periods
        base_demand: Baseline demand level (FIXED: 4 to match classic)
        shock_magnitude: Size of demand shocks
        shock_frequency: Periods between shocks
        seed: Random seed for shock timing variation (optional)
    """
    # Use seed to add slight variation to shock timing if desired
    if seed != CANONICAL_SEED:
        np.random.seed(seed % (2**32))
        pattern = [base_demand] * length
        for i in range(shock_frequency, length, shock_frequency):
            if i < length:
                # Add slight randomness to shock timing (±1 period)
                actual_shock_period = min(length - 1, max(0, i + np.random.randint(-1, 2)))
                pattern[actual_shock_period] = shock_magnitude
    else:
        # Standard deterministic shock pattern
        pattern = [base_demand] * length
        for i in range(shock_frequency, length, shock_frequency):
            if i < length:
                pattern[i] = shock_magnitude
    
    return pattern


def generate_seasonal_demand(length: int = 50, base_demand: float = 4.0, amplitude: float = 3.0, period: int = 12, seed: int = CANONICAL_SEED) -> List[int]:
    """
    Generate seasonal demand with cyclical pattern.
    
    FIXED: Now uses base_demand=4.0 to match classic scenario baseline.
    UPDATED: Can add noise based on seed.
    
    Args:
        length: Number of periods
        base_demand: Average demand level (FIXED: 4.0 to match classic)
        amplitude: Seasonal variation amplitude
        period: Length of seasonal cycle (default 12 = annual)
        seed: Random seed for adding realistic noise (optional)
    """
    if seed != CANONICAL_SEED:
        np.random.seed(seed % (2**32))
        add_noise = True
    else:
        add_noise = False
    
    pattern = []
    for i in range(length):
        seasonal_factor = amplitude * np.sin(2 * np.pi * i / period)
        
        # Add small amount of noise for realism if using non-default seed
        if add_noise:
            noise = np.random.normal(0, 0.5)
        else:
            noise = 0
            
        demand = base_demand + seasonal_factor + noise
        demand = max(1, int(round(demand)))  # Ensure minimum demand of 1
        pattern.append(demand)
    return pattern


# UPDATED: Predefined scenarios now use deterministic seeding system default
DEMAND_PATTERNS: Dict[str, List[int]] = {
    "classic": generate_classic_demand(50),
    "random": generate_random_demand(50, mean=4.0, seed=CANONICAL_SEED),  # Uses seeding system default
    "shock": generate_shock_demand(50, base_demand=4, seed=CANONICAL_SEED),  # Deterministic shocks
    "seasonal": generate_seasonal_demand(50, base_demand=4.0, seed=CANONICAL_SEED),  # Deterministic seasonal
}


def get_scenario_description(scenario_name: str) -> str:
    """Get description of a demand scenario"""
    descriptions = {
        "classic": "Classic step change: 4→8→4 units. Tests response to sudden demand shifts.",
        "random": "Random demand (mean=4.0, std=2.0). Tests handling of stochastic demand around classic baseline.",
        "shock": "Periodic demand shocks (base=4, shock=12 every 10 periods). Tests recovery from sudden spikes.",
        "seasonal": "Cyclical seasonal pattern (base=4.0, amplitude=3.0). Tests learning of repeating patterns.",
    }
    return descriptions.get(scenario_name, "Unknown scenario")


def generate_scenario_with_seed(scenario_name: str, length: int = 50, seed: int = CANONICAL_SEED) -> List[int]:
    """
    Generate a specific scenario with a custom seed for experimental control.
    
    UPDATED: Now integrates with deterministic seeding system.
    
    Args:
        scenario_name: Name of scenario ('classic', 'random', 'shock', 'seasonal')
        length: Number of periods
        seed: Custom seed for this generation (from deterministic seeding system)
        
    Returns:
        List of demand values
    """
    if scenario_name == "classic":
        return generate_classic_demand(length)
    elif scenario_name == "random":
        return generate_random_demand(length, mean=4.0, seed=seed)
    elif scenario_name == "shock":
        return generate_shock_demand(length, base_demand=4, seed=seed)
    elif scenario_name == "seasonal":
        return generate_seasonal_demand(length, base_demand=4.0, seed=seed)
    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")


def validate_scenario_consistency() -> Dict[str, float]:
    """
    Validate that all scenarios have consistent baseline demand levels for comparability.
    
    Returns:
        Dictionary with average demand levels for each scenario
    """
    averages = {}
    
    for scenario_name, pattern in DEMAND_PATTERNS.items():
        avg_demand = sum(pattern) / len(pattern)
        averages[scenario_name] = avg_demand
    
    return averages


def get_scenario_statistics() -> Dict[str, Dict[str, float]]:
    """
    Get comprehensive statistics for all demand scenarios.
    
    Returns:
        Dictionary with statistics for each scenario
    """
    stats = {}
    
    for scenario_name, pattern in DEMAND_PATTERNS.items():
        pattern_array = np.array(pattern)
        
        stats[scenario_name] = {
            "mean": float(np.mean(pattern_array)),
            "std": float(np.std(pattern_array)),
            "min": float(np.min(pattern_array)),
            "max": float(np.max(pattern_array)),
            "range": float(np.max(pattern_array) - np.min(pattern_array)),
            "length": len(pattern)
        }
    
    return stats


def get_reproducibility_info() -> Dict[str, Any]:
    """
    Get information about scenario reproducibility settings.
    
    UPDATED: Now reflects deterministic seeding system integration.
    
    Returns:
        Dictionary with reproducibility information
    """
    return {
        "default_seed": CANONICAL_SEED,
        "seeding_system": "deterministic",
        "scenarios_with_seed": ["random", "shock", "seasonal"],
        "deterministic_scenarios": ["classic"],
        "reproducible": True,
        "description": "Scenarios use deterministic seeding system for condition-specific reproducibility"
    }


def test_scenario_determinism():
    """
    Test that scenarios are properly deterministic with seeding system.
    
    NEW: Test function for validating scenario behavior with deterministic seeding.
    """
    print("Testing scenario determinism with seeding system...")
    
    # Test same seed produces same results
    seed = 12345
    random1 = generate_scenario_with_seed("random", 10, seed)
    random2 = generate_scenario_with_seed("random", 10, seed)
    
    assert random1 == random2, "Same seed should produce identical random scenarios"
    print(f"✅ Same seed produces identical results: {random1}")
    
    # Test different seeds produce different results
    seed_a, seed_b = 12345, 67890
    random_a = generate_scenario_with_seed("random", 10, seed_a)
    random_b = generate_scenario_with_seed("random", 10, seed_b)
    
    # Note: Different seeds SHOULD produce different results, but it's not guaranteed
    # for small samples, so we just check and report
    if random_a != random_b:
        print(f"✅ Different seeds produce different results")
    else:
        print(f"⚠️  Different seeds produced same results (rare but possible)")
    
    # Test deterministic scenarios are unaffected
    classic_a = generate_scenario_with_seed("classic", 10, seed_a)
    classic_b = generate_scenario_with_seed("classic", 10, seed_b)
    
    assert classic_a == classic_b, "Classic scenario should be unaffected by seed"
    print(f"✅ Classic scenario unaffected by seed: {classic_a}")
    
    print("All scenario determinism tests passed!")


# Validation check - ensure scenarios are comparable
_scenario_averages = validate_scenario_consistency()
_baseline_demand = 4.0

# Verify consistency (within reasonable tolerance)
for scenario, avg in _scenario_averages.items():
    if abs(avg - _baseline_demand) > 1.0:  # Allow 1 unit tolerance
        import warnings
        warnings.warn(f"Scenario '{scenario}' has average demand {avg:.2f} which deviates significantly from baseline {_baseline_demand}. This may affect cross-scenario comparability.")


if __name__ == "__main__":
    # Demo and validation
    print("=== SCM-Arena Demand Scenarios (DETERMINISTIC SEEDING) ===")
    print()
    
    # Show reproducibility info
    repro_info = get_reproducibility_info()
    print(f"🎲 Default Seed: {repro_info['default_seed']}")
    print(f"🔧 Seeding System: {repro_info['seeding_system']}")
    print(f"🔄 Scenarios with Seeding: {repro_info['scenarios_with_seed']}")
    print(f"📐 Deterministic Scenarios: {repro_info['deterministic_scenarios']}")
    print()
    
    # Show scenario statistics
    stats = get_scenario_statistics()
    
    print("Scenario Comparison:")
    print(f"{'Scenario':<10} {'Mean':<6} {'Std':<6} {'Min':<4} {'Max':<4} {'Range':<6} {'Seeded':<8}")
    print("-" * 60)
    
    for scenario, stat in stats.items():
        seeded = "✅" if scenario in repro_info['scenarios_with_seed'] else "N/A"
        print(f"{scenario:<10} {stat['mean']:<6.1f} {stat['std']:<6.1f} {stat['min']:<4.0f} {stat['max']:<4.0f} {stat['range']:<6.0f} {seeded}")
    
    print()
    print("✅ FIXED: All scenarios now use baseline demand ≈ 4.0 for comparability")
    print("✅ DETERMINISTIC: Scenarios use deterministic seeding system")
    print("✅ Cross-scenario analysis is now valid and meaningful")
    
    # Show first 20 periods of each scenario
    print("\nFirst 20 periods of each scenario:")
    for scenario_name, pattern in DEMAND_PATTERNS.items():
        print(f"{scenario_name:>8}: {pattern[:20]}")
    
    print()
    print("Scenario descriptions:")
    for scenario in DEMAND_PATTERNS.keys():
        print(f"• {scenario}: {get_scenario_description(scenario)}")
    
    # Test determinism
    print("\nTesting deterministic seeding integration...")
    try:
        test_scenario_determinism()
    except Exception as e:
        print(f"❌ Determinism test failed: {e}")
    
    print("\n🎯 Scenarios ready for deterministic benchmarking!")