"""
Predefined demand scenarios for Beer Game evaluation with FIXED demand consistency.

CRITICAL FIX: All scenarios now use consistent baseline demand levels for comparability.
- Classic scenario: steady state = 4 units (matches historical Beer Game)
- Random scenario: mean = 4.0 units (matches classic baseline)  
- Shock scenario: base = 4 units (consistent with other scenarios)
- Seasonal scenario: base = 4.0 units (consistent baseline)

This ensures scenarios are comparable for cross-analysis and benchmarking.
"""

import numpy as np
from typing import List, Dict


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


def generate_random_demand(length: int = 50, mean: float = 4.0, std: float = 2.0, seed: int = 42) -> List[int]:
    """
    Generate random demand following normal distribution.
    
    FIXED: Now uses mean=4.0 to match classic scenario baseline for comparability.
    
    Args:
        length: Number of periods
        mean: Mean demand (FIXED: 4.0 to match classic baseline)
        std: Standard deviation of demand
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    demands = np.random.normal(mean, std, length)
    demands = np.maximum(1, demands).astype(int)  # Ensure minimum demand of 1
    return demands.tolist()


def generate_shock_demand(length: int = 50, base_demand: int = 4, shock_magnitude: int = 12, shock_frequency: int = 10) -> List[int]:
    """
    Generate demand with periodic shocks.
    
    FIXED: Now uses base_demand=4 to match classic scenario baseline.
    
    Args:
        length: Number of periods
        base_demand: Baseline demand level (FIXED: 4 to match classic)
        shock_magnitude: Size of demand shocks
        shock_frequency: Periods between shocks
    """
    pattern = [base_demand] * length
    for i in range(shock_frequency, length, shock_frequency):
        if i < length:
            pattern[i] = shock_magnitude
    return pattern


def generate_seasonal_demand(length: int = 50, base_demand: float = 4.0, amplitude: float = 3.0, period: int = 12) -> List[int]:
    """
    Generate seasonal demand with cyclical pattern.
    
    FIXED: Now uses base_demand=4.0 to match classic scenario baseline.
    
    Args:
        length: Number of periods
        base_demand: Average demand level (FIXED: 4.0 to match classic)
        amplitude: Seasonal variation amplitude
        period: Length of seasonal cycle (default 12 = annual)
    """
    pattern = []
    for i in range(length):
        seasonal_factor = amplitude * np.sin(2 * np.pi * i / period)
        demand = base_demand + seasonal_factor
        demand = max(1, int(round(demand)))  # Ensure minimum demand of 1
        pattern.append(demand)
    return pattern


# FIXED: Predefined scenarios now use consistent baseline demand levels
DEMAND_PATTERNS: Dict[str, List[int]] = {
    "classic": generate_classic_demand(50),
    "random": generate_random_demand(50, mean=4.0),  # FIXED: mean=4.0 matches classic
    "shock": generate_shock_demand(50, base_demand=4),  # FIXED: base=4 matches classic
    "seasonal": generate_seasonal_demand(50, base_demand=4.0),  # FIXED: base=4.0 matches classic
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
    print("=== SCM-Arena Demand Scenarios (FIXED for Consistency) ===")
    print()
    
    # Show scenario statistics
    stats = get_scenario_statistics()
    
    print("Scenario Comparison:")
    print(f"{'Scenario':<10} {'Mean':<6} {'Std':<6} {'Min':<4} {'Max':<4} {'Range':<6}")
    print("-" * 45)
    
    for scenario, stat in stats.items():
        print(f"{scenario:<10} {stat['mean']:<6.1f} {stat['std']:<6.1f} {stat['min']:<4.0f} {stat['max']:<4.0f} {stat['range']:<6.0f}")
    
    print()
    print("✅ FIXED: All scenarios now use baseline demand ≈ 4.0 for comparability")
    print("✅ Cross-scenario analysis is now valid and meaningful")
    
    # Show first 20 periods of each scenario
    print("\nFirst 20 periods of each scenario:")
    for scenario_name, pattern in DEMAND_PATTERNS.items():
        print(f"{scenario_name:>8}: {pattern[:20]}")
    
    print()
    print("Scenario descriptions:")
    for scenario in DEMAND_PATTERNS.keys():
        print(f"• {scenario}: {get_scenario_description(scenario)}")