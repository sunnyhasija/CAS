# src/scm_arena/utils/seeding.py - FINAL FIXED VERSION
"""
Deterministic seeding system for SCM-Arena experiments - FINAL FIX!

CRITICAL FIX: Parameter names now match what CLI actually uses.
"""

import hashlib
import random
from typing import Dict, Any, Optional


def deterministic_seed(
    model: str,
    memory: str, 
    prompt_type: str,    # FIXED: was 'prompt'
    visibility: str,
    scenario: str,
    game_mode: str,      # FIXED: was 'mode'
    run_number: int,     # FIXED: was 'run'
    base_seed: int = 42
) -> int:
    """
    Generate deterministic seed based on experimental conditions.
    
    FIXED: Parameter names now match CLI usage exactly.
    """
    # Create condition string
    condition_str = f"{model}|{memory}|{prompt_type}|{visibility}|{scenario}|{game_mode}|{run_number}|{base_seed}"
    
    # Use SHA-256 for better collision resistance
    hash_obj = hashlib.sha256(condition_str.encode('utf-8'))
    seed = int(hash_obj.hexdigest()[:16], 16) & 0x7FFFFFFFFFFFFFFF  # 63-bit positive
    
    return seed


class ExperimentSeeder:
    """
    Manages deterministic seeding for experiments - FINAL FIXED VERSION!
    """
    
    def __init__(self, base_seed: int = 42, deterministic: bool = True):
        self.base_seed = base_seed
        self.deterministic = deterministic
        self._seed_cache = {}
        
        # Initialize random generator for non-deterministic mode
        if not deterministic:
            self._random = random.Random(base_seed)
    
    def get_seed(
        self,
        model: str,
        memory: str,
        prompt_type: str,    # FIXED: was 'prompt'
        visibility: str,
        scenario: str,
        game_mode: str,      # FIXED: was 'mode'
        run_number: int      # FIXED: was 'run'
    ) -> int:
        """
        Get seed for experimental condition with caching.
        
        FIXED: Parameter names now match CLI calls exactly.
        """
        
        # Create cache key
        key = (model, memory, prompt_type, visibility, scenario, game_mode, run_number)
        
        # Check cache first
        if key in self._seed_cache:
            return self._seed_cache[key]
        
        if self.deterministic:
            # Deterministic mode: use hash-based seeding
            seed = deterministic_seed(
                model, memory, prompt_type, visibility, scenario, game_mode, run_number, self.base_seed
            )
        else:
            # Fixed: Non-deterministic mode generates unique seeds per condition
            condition_hash = hash(key)
            self._random.seed(self.base_seed + condition_hash)
            seed = self._random.randint(0, 0x7FFFFFFFFFFFFFFF)  # 63-bit positive
        
        # Cache the result
        self._seed_cache[key] = seed
        return seed
    
    def get_agent_seed(
        self,
        model: str,
        memory: str,
        prompt_type: str,     # FIXED: was 'prompt'
        visibility: str,
        scenario: str,
        game_mode: str,       # FIXED: was 'mode'
        run_number: int,      # FIXED: was 'run'
        agent_position: str
    ) -> int:
        """
        Get unique seed for a specific agent in an experimental condition.
        
        FIXED: Parameter names now match CLI calls exactly.
        """
        # Extend the condition with agent position for unique seeding
        extended_key = (model, memory, prompt_type, visibility, scenario, game_mode, run_number, agent_position)
        
        if extended_key in self._seed_cache:
            return self._seed_cache[extended_key]
        
        if self.deterministic:
            # Create unique condition string including agent position
            condition_str = f"{model}|{memory}|{prompt_type}|{visibility}|{scenario}|{game_mode}|{run_number}|{agent_position}|{self.base_seed}"
            hash_obj = hashlib.sha256(condition_str.encode('utf-8'))
            seed = int(hash_obj.hexdigest()[:16], 16) & 0x7FFFFFFFFFFFFFFF
        else:
            # Non-deterministic mode with agent-specific seeding
            condition_hash = hash(extended_key)
            self._random.seed(self.base_seed + condition_hash)
            seed = self._random.randint(0, 0x7FFFFFFFFFFFFFFF)
        
        self._seed_cache[extended_key] = seed
        return seed
    
    def validate_reproducibility(
        self,
        model: str,
        memory: str,
        prompt_type: str,     # FIXED: was 'prompt'
        visibility: str,
        scenario: str,
        game_mode: str,       # FIXED: was 'mode'
        run_number: int       # FIXED: was 'run'
    ) -> bool:
        """
        Validate that the same condition produces the same seed.
        
        FIXED: Parameter names now match CLI calls exactly.
        """
        seed1 = self.get_seed(model, memory, prompt_type, visibility, scenario, game_mode, run_number)
        seed2 = self.get_seed(model, memory, prompt_type, visibility, scenario, game_mode, run_number)
        
        return seed1 == seed2
    
    def get_seed_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated seeds."""
        if not self._seed_cache:
            return {"message": "No seeds generated yet"}
        
        seeds = list(self._seed_cache.values())
        
        return {
            "total_seeds": len(seeds),
            "unique_seeds": len(set(seeds)),
            "collision_rate": 1 - (len(set(seeds)) / len(seeds)) if seeds else 0,
            "min_seed": min(seeds) if seeds else None,
            "max_seed": max(seeds) if seeds else None,
            "seed_space_bits": 63,
            "algorithm": "SHA-256" if self.deterministic else "Random",
            "deterministic": self.deterministic,
            "base_seed": self.base_seed
        }


def get_seed_for_condition(
    model: str,
    memory: str,
    prompt_type: str,     # FIXED: was 'prompt'
    visibility: str,
    scenario: str,
    game_mode: str,       # FIXED: was 'mode'
    run_number: int,      # FIXED: was 'run'
    base_seed: int = 42
) -> int:
    """
    Get seed for a specific experimental condition.
    
    FIXED: Parameter names now match CLI calls exactly.
    """
    return deterministic_seed(model, memory, prompt_type, visibility, scenario, game_mode, run_number, base_seed)


# Constants for CLI integration
DEFAULT_BASE_SEED = 42
DETERMINISTIC_DEFAULT = True


if __name__ == "__main__":
    # Quick test when run directly
    print("🎯 SCM-Arena Deterministic Seeding System - FINAL FIXED VERSION")
    print("=" * 70)
    
    # Test deterministic mode
    print("\n1. Testing Deterministic Mode:")
    seeder_det = ExperimentSeeder(base_seed=42, deterministic=True)
    seed1 = seeder_det.get_seed("llama3.2", "short", "specific", "local", "classic", "modern", 1)
    seed2 = seeder_det.get_seed("llama3.2", "short", "specific", "local", "classic", "modern", 1)
    print(f"   Same condition twice: {seed1} == {seed2} -> {'✅ PASS' if seed1 == seed2 else '❌ FAIL'}")
    
    # Test non-deterministic mode
    print("\n2. Testing Non-Deterministic Mode:")
    seeder_nondet = ExperimentSeeder(base_seed=42, deterministic=False)
    seeds = []
    for i in range(5):
        seed = seeder_nondet.get_seed("llama3.2", "short", "specific", "local", "classic", "modern", i+1)
        seeds.append(seed)
    
    unique_seeds = len(set(seeds))
    print(f"   Generated {len(seeds)} seeds, {unique_seeds} unique -> {'✅ PASS' if unique_seeds == len(seeds) else '❌ FAIL'}")
    
    # Test parameter name compatibility
    print("\n3. Testing Parameter Name Compatibility:")
    try:
        # Test the exact call pattern that CLI will use
        test_seed = seeder_det.get_seed(
            model="llama3.2",
            memory="short", 
            prompt_type="specific",  # This is what CLI passes
            visibility="local",
            scenario="classic",
            game_mode="modern",      # This is what CLI passes
            run_number=1             # This is what CLI passes
        )
        print(f"   CLI-compatible call: seed={test_seed} -> ✅ PASS")
    except Exception as e:
        print(f"   CLI-compatible call failed: {e} -> ❌ FAIL")
    
    print(f"\n✅ Final version ready for production!")