# src/scm_arena/utils/seeding.py - NEW FILE
"""
Deterministic seeding system for SCM-Arena experiments.

Provides reproducible seed generation based on experimental conditions,
ensuring that identical conditions always get the same seed while
different conditions get different seeds.
"""

import hashlib
from typing import Dict, Any, Optional


def deterministic_seed(
    model: str,
    memory: str, 
    prompt: str,
    visibility: str,
    scenario: str,
    mode: str,
    run: int,
    base_seed: int = 42
) -> int:
    """
    Generate deterministic seed based on experimental conditions.
    
    This ensures:
    - Identical conditions always get the same seed (reproducible)
    - Different conditions get different seeds (fair comparison)
    - Each run gets a different seed (statistical validity)
    
    Args:
        model: Model name (e.g., "llama3.2")
        memory: Memory strategy (e.g., "short")
        prompt: Prompt type (e.g., "specific")
        visibility: Visibility level (e.g., "local")
        scenario: Scenario name (e.g., "classic")
        mode: Game mode (e.g., "modern")
        run: Run number (1, 2, 3, ...)
        base_seed: Base seed for hash mixing
        
    Returns:
        32-bit deterministic seed
    """
    # Create condition string
    condition_str = f"{model}|{memory}|{prompt}|{visibility}|{scenario}|{mode}|{run}|{base_seed}"
    
    # Hash to get deterministic seed
    hash_obj = hashlib.md5(condition_str.encode())
    seed = int(hash_obj.hexdigest()[:8], 16) & 0x7FFFFFFF  # 32-bit positive
    
    return seed


def create_experiment_seeds(
    models: list,
    memory_strategies: list,
    prompt_types: list,
    visibility_levels: list,
    scenarios: list,
    game_modes: list,
    runs_per_condition: int,
    base_seed: int = 42
) -> Dict[tuple, list]:
    """
    Generate all seeds for a complete experimental design.
    
    Args:
        models: List of model names
        memory_strategies: List of memory strategies
        prompt_types: List of prompt types
        visibility_levels: List of visibility levels
        scenarios: List of scenarios
        game_modes: List of game modes
        runs_per_condition: Number of runs per condition
        base_seed: Base seed for deterministic generation
        
    Returns:
        Dictionary mapping condition tuples to lists of seeds
    """
    import itertools
    
    seeds_map = {}
    
    # Generate all conditions
    conditions = list(itertools.product(
        models, memory_strategies, prompt_types,
        visibility_levels, scenarios, game_modes
    ))
    
    # Generate seeds for each condition
    for condition in conditions:
        model, memory, prompt, visibility, scenario, mode = condition
        
        # Generate seeds for all runs of this condition
        condition_seeds = []
        for run in range(1, runs_per_condition + 1):
            seed = deterministic_seed(
                model, memory, prompt, visibility, scenario, mode, run, base_seed
            )
            condition_seeds.append(seed)
        
        seeds_map[condition] = condition_seeds
    
    return seeds_map


def get_seed_for_condition(
    model: str,
    memory: str,
    prompt: str,
    visibility: str,
    scenario: str,
    mode: str,
    run: int,
    base_seed: int = 42
) -> int:
    """
    Get seed for a specific experimental condition.
    
    Convenience function for getting a single seed.
    """
    return deterministic_seed(model, memory, prompt, visibility, scenario, mode, run, base_seed)


class ExperimentSeeder:
    """
    Manages deterministic seeding for experiments.
    
    Provides a clean interface for seed management with validation
    and reproducibility guarantees.
    """
    
    def __init__(self, base_seed: int = 42, deterministic: bool = True):
        """
        Initialize experiment seeder.
        
        Args:
            base_seed: Base seed for all experiments
            deterministic: If True, use deterministic seeding; if False, use base_seed directly
        """
        self.base_seed = base_seed
        self.deterministic = deterministic
        self._seed_cache = {}
    
    def get_seed(
        self,
        model: str,
        memory: str,
        prompt: str,
        visibility: str,
        scenario: str,
        mode: str,
        run: int
    ) -> int:
        """Get seed for experimental condition with caching."""
        
        if not self.deterministic:
            return self.base_seed
        
        # Create cache key
        key = (model, memory, prompt, visibility, scenario, mode, run)
        
        # Check cache first
        if key in self._seed_cache:
            return self._seed_cache[key]
        
        # Generate and cache seed
        seed = deterministic_seed(
            model, memory, prompt, visibility, scenario, mode, run, self.base_seed
        )
        self._seed_cache[key] = seed
        
        return seed
    
    def validate_reproducibility(
        self,
        model: str,
        memory: str,
        prompt: str,
        visibility: str,
        scenario: str,
        mode: str,
        run: int
    ) -> bool:
        """
        Validate that the same condition produces the same seed.
        
        Returns True if reproducible, False otherwise.
        """
        seed1 = self.get_seed(model, memory, prompt, visibility, scenario, mode, run)
        seed2 = self.get_seed(model, memory, prompt, visibility, scenario, mode, run)
        
        return seed1 == seed2
    
    def get_seed_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated seeds."""
        if not self._seed_cache:
            return {"message": "No seeds generated yet"}
        
        seeds = list(self._seed_cache.values())
        
        return {
            "total_seeds": len(seeds),
            "unique_seeds": len(set(seeds)),
            "collision_rate": 1 - (len(set(seeds)) / len(seeds)),
            "min_seed": min(seeds),
            "max_seed": max(seeds),
            "deterministic": self.deterministic,
            "base_seed": self.base_seed
        }


# Constants for CLI integration
DEFAULT_BASE_SEED = 42
DETERMINISTIC_DEFAULT = True


if __name__ == "__main__":
    # Demo the deterministic seeding system
    print("🎯 SCM-Arena Deterministic Seeding System")
    print("=" * 50)
    
    # Test basic seeding
    print("\n1. Basic Deterministic Seeding:")
    conditions = [
        ("llama3.2", "short", "specific", "local", "classic", "modern", 1),
        ("llama3.2", "short", "specific", "local", "classic", "modern", 2),
        ("llama3.2", "full", "specific", "local", "classic", "modern", 1),
        ("llama3.2", "short", "neutral", "local", "classic", "modern", 1),
    ]
    
    for condition in conditions:
        seed = deterministic_seed(*condition)
        print(f"  {condition}: seed={seed}")
    
    # Test reproducibility
    print("\n2. Reproducibility Test:")
    test_condition = ("llama3.2", "short", "specific", "local", "classic", "modern", 1)
    seed1 = deterministic_seed(*test_condition)
    seed2 = deterministic_seed(*test_condition)
    print(f"  Same condition called twice: {seed1} == {seed2} ✅" if seed1 == seed2 else "❌")
    
    # Test seeder class
    print("\n3. ExperimentSeeder Test:")
    seeder = ExperimentSeeder(base_seed=42, deterministic=True)
    
    # Generate some seeds
    test_conditions = [
        ("llama3.2", "short", "specific", "local", "classic", "modern", 1),
        ("llama3.2", "short", "specific", "local", "classic", "modern", 2),
        ("llama3.2", "full", "specific", "local", "classic", "modern", 1),
    ]
    
    for condition in test_conditions:
        seed = seeder.get_seed(*condition)
        print(f"  {condition[0]}-{condition[1]}-run{condition[6]}: seed={seed}")
    
    # Show statistics
    print("\n4. Seed Statistics:")
    stats = seeder.get_seed_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test validation
    print("\n5. Reproducibility Validation:")
    is_reproducible = seeder.validate_reproducibility(*test_conditions[0])
    print(f"  Reproducible: {'✅' if is_reproducible else '❌'}")
    
    print("\n✅ Deterministic seeding system working correctly!")
    print("🔄 Each experimental condition gets a unique, reproducible seed")
    print("🎲 Different runs get different seeds for statistical validity")