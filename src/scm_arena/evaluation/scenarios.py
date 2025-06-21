"""
Predefined demand scenarios for Beer Game evaluation.
"""

import numpy as np
from typing import List, Dict


def generate_classic_demand(length: int = 50) -> List[int]:
    """Generate the classic Beer Game demand pattern."""
    pattern = []
    pattern.extend([4] * 4)      # Initial steady state
    pattern.extend([8] * 8)      # Step increase
    remaining = length - len(pattern)
    pattern.extend([4] * remaining)  # Return to steady state
    return pattern[:length]


def generate_random_demand(length: int = 50, mean: float = 6.0, std: float = 2.0, seed: int = 42) -> List[int]:
    """Generate random demand following normal distribution."""
    np.random.seed(seed)
    demands = np.random.normal(mean, std, length)
    demands = np.maximum(0, demands).astype(int)
    return demands.tolist()


def generate_shock_demand(length: int = 50, base_demand: int = 4, shock_magnitude: int = 12, shock_frequency: int = 10) -> List[int]:
    """Generate demand with periodic shocks."""
    pattern = [base_demand] * length
    for i in range(shock_frequency, length, shock_frequency):
        if i < length:
            pattern[i] = shock_magnitude
    return pattern


def generate_seasonal_demand(length: int = 50, base_demand: float = 6.0, amplitude: float = 3.0, period: int = 12) -> List[int]:
    """Generate seasonal demand with cyclical pattern."""
    pattern = []
    for i in range(length):
        seasonal_factor = amplitude * np.sin(2 * np.pi * i / period)
        demand = base_demand + seasonal_factor
        demand = max(0, int(round(demand)))
        pattern.append(demand)
    return pattern


# Predefined scenarios for easy access
DEMAND_PATTERNS: Dict[str, List[int]] = {
    "classic": generate_classic_demand(50),
    "random": generate_random_demand(50),
    "shock": generate_shock_demand(50),
    "seasonal": generate_seasonal_demand(50),
}


def get_scenario_description(scenario_name: str) -> str:
    """Get description of a demand scenario"""
    descriptions = {
        "classic": "Classic step change: 4→8→4 units. Tests response to sudden demand shifts.",
        "random": "Random demand (normal distribution). Tests handling of stochastic demand.",
        "shock": "Periodic demand shocks. Tests recovery from sudden spikes.",
        "seasonal": "Cyclical seasonal pattern. Tests learning of repeating patterns.",
    }
    return descriptions.get(scenario_name, "Unknown scenario")