"""
Predefined demand scenarios for Beer Game evaluation.

Contains various demand patterns used to test different aspects of
supply chain coordination and LLM performance.
"""

import numpy as np
from typing import List, Dict


def generate_classic_demand(length: int = 50) -> List[int]:
    """
    Generate the classic Beer Game demand pattern.
    
    The original pattern: 4 units for several weeks, then jump to 8 units,
    then back to 4 units. This tests response to step changes.
    
    Args:
        length: Total length of demand pattern
        
    Returns:
        List of demand values
    """
    pattern = []
    
    # Initial steady state (4 units)
    pattern.extend([4] * 4)
    
    # Step increase (8 units) 
    pattern.extend([8] * 8)
    
    # Return to steady state (4 units)
    remaining = length - len(pattern)
    pattern.extend([4] * remaining)
    
    return pattern[:length]


def generate_random_demand(length: int = 50, mean: float = 6.0, std: float = 2.0, seed: int = 42) -> List[int]:
    """
    Generate random demand following normal distribution.
    
    Tests ability to handle stochastic demand with no clear pattern.
    
    Args:
        length: Total length of demand pattern
        mean: Mean demand per period
        std: Standard deviation of demand
        seed: Random seed for reproducibility
        
    Returns:
        List of demand values
    """
    np.random.seed(seed)
    demands = np.random.normal(mean, std, length)
    
    # Ensure non-negative integers
    demands = np.maximum(0, demands).astype(int)
    
    return demands.tolist()


def generate_shock_demand(length: int = 50, base_demand: int = 4, shock_magnitude: int = 12, shock_frequency: int = 10) -> List[int]:
    """
    Generate demand with periodic shocks.
    
    Tests recovery and adaptation to sudden demand spikes.
    
    Args:
        length: Total length of demand pattern
        base_demand: Normal demand level
        shock_magnitude: Size of demand shocks
        shock_frequency: How often shocks occur (every N periods)
        
    Returns:
        List of demand values
    """
    pattern = [base_demand] * length
    
    # Add shocks at regular intervals
    for i in range(shock_frequency, length, shock_frequency):
        if i < length:
            pattern[i] = shock_magnitude
            
    return pattern


def generate_seasonal_demand(length: int = 50, base_demand: float = 6.0, amplitude: float = 3.0, period: int = 12) -> List[int]:
    """
    Generate seasonal demand with cyclical pattern.
    
    Tests ability to learn and adapt to repeating patterns.
    
    Args:
        length: Total length of demand pattern
        base_demand: Average demand level
        amplitude: Size of seasonal variation
        period: Length of seasonal cycle
        
    Returns:
        List of demand values
    """
    pattern = []
    
    for i in range(length):
        # Sinusoidal seasonal pattern
        seasonal_factor = amplitude * np.sin(2 * np.pi * i / period)
        demand = base_demand + seasonal_factor
        
        # Ensure non-negative integers
        demand = max(0, int(round(demand)))
        pattern.append(demand)
    
    return pattern


def generate_trend_demand(length: int = 50, initial_demand: int = 4, trend_rate: float = 0.1) -> List[int]:
    """
    Generate demand with linear trend.
    
    Tests ability to adapt to gradually changing demand levels.
    
    Args:
        length: Total length of demand pattern
        initial_demand: Starting demand level
        trend_rate: Rate of demand increase per period
        
    Returns:
        List of demand values
    """
    pattern = []
    
    for i in range(length):
        demand = initial_demand + (trend_rate * i)
        demand = max(0, int(round(demand)))
        pattern.append(demand)
    
    return pattern


def generate_complex_demand(length: int = 50, seed: int = 42) -> List[int]:
    """
    Generate complex demand combining multiple patterns.
    
    Combines trend, seasonality, and random noise for realistic scenarios.
    
    Args:
        length: Total length of demand pattern
        seed: Random seed for reproducibility
        
    Returns:
        List of demand values
    """
    np.random.seed(seed)
    pattern = []
    
    base_demand = 6.0
    trend_rate = 0.05
    seasonal_amplitude = 2.0
    seasonal_period = 8
    noise_std = 1.0
    
    for i in range(length):
        # Base demand with trend
        trend_component = base_demand + (trend_rate * i)
        
        # Seasonal component
        seasonal_component = seasonal_amplitude * np.sin(2 * np.pi * i / seasonal_period)
        
        # Random noise
        noise_component = np.random.normal(0, noise_std)
        
        # Combine components
        demand = trend_component + seasonal_component + noise_component
        demand = max(0, int(round(demand)))
        pattern.append(demand)
    
    return pattern


# Predefined scenarios for easy access
DEMAND_PATTERNS: Dict[str, List[int]] = {
    "classic": generate_classic_demand(50),
    "random": generate_random_demand(50),
    "shock": generate_shock_demand(50),
    "seasonal": generate_seasonal_demand(50),
    "trend": generate_trend_demand(50),
    "complex": generate_complex_demand(50),
}


# Short scenarios for quick testing
SHORT_PATTERNS: Dict[str, List[int]] = {
    "classic_short": generate_classic_demand(20),
    "random_short": generate_random_demand(20),
    "shock_short": generate_shock_demand(20),
    "seasonal_short": generate_seasonal_demand(20),
}


def get_scenario_description(scenario_name: str) -> str:
    """Get description of a demand scenario"""
    descriptions = {
        "classic": "Classic step change: 4→8→4 units. Tests response to sudden demand shifts.",
        "random": "Random demand (normal distribution). Tests handling of stochastic demand.",
        "shock": "Periodic demand shocks. Tests recovery from sudden spikes.",
        "seasonal": "Cyclical seasonal pattern. Tests learning of repeating patterns.",
        "trend": "Linear demand growth. Tests adaptation to gradual changes.",
        "complex": "Combined trend, seasonality, and noise. Tests real-world scenarios.",
    }
    
    return descriptions.get(scenario_name, "Unknown scenario")


def get_all_scenarios() -> Dict[str, Dict]:
    """Get all scenarios with metadata"""
    scenarios = {}
    
    for name, pattern in DEMAND_PATTERNS.items():
        scenarios[name] = {
            "name": name,
            "pattern": pattern,
            "description": get_scenario_description(name),
            "length": len(pattern),
            "mean_demand": np.mean(pattern),
            "std_demand": np.std(pattern),
            "min_demand": min(pattern),
            "max_demand": max(pattern),
        }
    
    return scenarios


if __name__ == "__main__":
    # Demo all scenarios
    import matplotlib.pyplot as plt
    
    scenarios = get_all_scenarios()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (name, info) in enumerate(scenarios.items()):
        if i < len(axes):
            axes[i].plot(info["pattern"])
            axes[i].set_title(f"{name.title()} Demand")
            axes[i].set_xlabel("Round")
            axes[i].set_ylabel("Demand")
            axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print scenario summaries
    print("\nDemand Scenario Summary:")
    print("=" * 60)
    
    for name, info in scenarios.items():
        print(f"\n{name.upper()}:")
        print(f"  Description: {info['description']}")
        print(f"  Length: {info['length']} rounds")
        print(f"  Mean demand: {info['mean_demand']:.1f}")
        print(f"  Std deviation: {info['std_demand']:.1f}")
        print(f"  Range: {info['min_demand']} - {info['max_demand']}")
        print(f"  Pattern preview: {info['pattern'][:10]}...")