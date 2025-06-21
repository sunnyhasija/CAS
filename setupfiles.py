#!/usr/bin/env python3
"""
Setup script to populate all SCM-Arena files with the correct content.
Run this once to set up the complete project structure.
"""

import os
from pathlib import Path

def write_file(filepath, content):
    """Write content to file, creating directories if needed"""
    file_path = Path(filepath)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Created {filepath}")

def setup_project():
    """Set up all project files"""
    
    # pyproject.toml
    pyproject_content = '''[tool.poetry]
name = "scm-arena"
version = "0.1.0"
description = "Supply Chain Management LLM Benchmark Platform - Beer Game Edition"
authors = ["Your Name <your.email@example.com>"]
readme = "README.md"
packages = [{include = "scm_arena", from = "src"}]

[tool.poetry.dependencies]
python = "^3.9"
requests = "^2.31.0"
pydantic = "^2.0.0"
numpy = "^1.24.0"
pandas = "^2.0.0"
matplotlib = "^3.7.0"
plotly = "^5.15.0"
click = "^8.1.0"
rich = "^13.0.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
pytest-cov = "^4.1.0"
black = "^23.7.0"
isort = "^5.12.0"
mypy = "^1.5.0"
flake8 = "^6.0.0"

[tool.poetry.scripts]
scm-arena = "scm_arena.cli:main"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.black]
line-length = 88
target-version = ['py39']

[tool.isort]
profile = "black"
multi_line_output = 3

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true'''
    
    # src/scm_arena/__init__.py
    main_init_content = '''"""
SCM-Arena: Supply Chain Management LLM Benchmark Platform

A standardized benchmark for evaluating Large Language Models on 
supply chain management coordination tasks using the Beer Game simulation.
"""

__version__ = "0.1.0"
__author__ = "SCM-Arena Contributors"

from .beer_game.game import BeerGame, GameResults, GameState
from .beer_game.agents import Agent, Position, SimpleAgent, OptimalAgent, RandomAgent
from .models.ollama_client import OllamaAgent, create_ollama_agents, test_ollama_connection

__all__ = [
    "BeerGame",
    "GameResults", 
    "GameState",
    "Agent",
    "Position",
    "SimpleAgent",
    "OptimalAgent", 
    "RandomAgent",
    "OllamaAgent",
    "create_ollama_agents",
    "test_ollama_connection",
]'''

    # src/scm_arena/beer_game/__init__.py
    beer_game_init_content = '''"""Beer Game simulation components"""

from .game import BeerGame, GameResults, GameState
from .agents import Agent, Position, SimpleAgent, OptimalAgent, RandomAgent

__all__ = [
    "BeerGame",
    "GameResults",
    "GameState", 
    "Agent",
    "Position",
    "SimpleAgent",
    "OptimalAgent",
    "RandomAgent",
]'''

    # src/scm_arena/models/__init__.py
    models_init_content = '''"""
Model integrations for SCM-Arena.

This module provides interfaces to various LLM providers
and model hosting services for Beer Game agents.
"""

from .ollama_client import OllamaAgent, create_ollama_agents, test_ollama_connection

__all__ = [
    "OllamaAgent",
    "create_ollama_agents", 
    "test_ollama_connection",
]'''

    # src/scm_arena/evaluation/__init__.py
    evaluation_init_content = '''"""Evaluation components for SCM-Arena"""

from .scenarios import DEMAND_PATTERNS, get_scenario_description

__all__ = [
    "DEMAND_PATTERNS",
    "get_scenario_description",
]'''

    print("🚀 Setting up SCM-Arena project files...")
    
    # Write all files
    write_file("pyproject.toml", pyproject_content)
    write_file("src/scm_arena/__init__.py", main_init_content)
    write_file("src/scm_arena/beer_game/__init__.py", beer_game_init_content)
    write_file("src/scm_arena/models/__init__.py", models_init_content)
    write_file("src/scm_arena/evaluation/__init__.py", evaluation_init_content)
    
    print("\n🎯 Next steps:")
    print("1. I need to provide you with the remaining core files")
    print("2. Run: poetry install")
    print("3. Test: poetry run python -m scm_arena.cli test-model")
    print("\nRunning this script now...")

if __name__ == "__main__":
    setup_project()
'''

# The remaining files are too large for a single script, so I'll provide them separately
BEER_GAME_AGENTS_CONTENT = '''"""
Agent interfaces for Beer Game players.

Defines the abstract Agent class and specific implementations for different
types of agents (human, LLM, algorithmic, etc.)
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional
import random


class Position(Enum):
    """Supply chain positions in the Beer Game"""
    RETAILER = "retailer"
    WHOLESALER = "wholesaler" 
    DISTRIBUTOR = "distributor"
    MANUFACTURER = "manufacturer"
    
    @classmethod
    def get_supply_chain_order(cls) -> List['Position']:
        """Get positions in supply chain order (downstream to upstream)"""
        return [cls.RETAILER, cls.WHOLESALER, cls.DISTRIBUTOR, cls.MANUFACTURER]
    
    def get_upstream_position(self) -> Optional['Position']:
        """Get the upstream position (who this position orders from)"""
        order = self.get_supply_chain_order()
        try:
            current_idx = order.index(self)
            if current_idx < len(order) - 1:
                return order[current_idx + 1]
        except ValueError:
            pass
        return None
    
    def get_downstream_position(self) -> Optional['Position']:
        """Get the downstream position (who orders from this position)"""
        order = self.get_supply_chain_order()
        try:
            current_idx = order.index(self)
            if current_idx > 0:
                return order[current_idx - 1]
        except ValueError:
            pass
        return None


class Agent(ABC):
    """
    Abstract base class for Beer Game agents.
    
    Each agent represents one player in the supply chain and must decide
    how many units to order each round based on their observable state.
    """
    
    def __init__(self, position: Position, name: Optional[str] = None):
        """
        Initialize agent.
        
        Args:
            position: Supply chain position (retailer, wholesaler, etc.)
            name: Optional human-readable name for the agent
        """
        self.position = position
        self.name = name or f"{position.value}_agent"
        
    @abstractmethod
    def make_decision(self, game_state: Dict[str, Any]) -> int:
        """
        Make an ordering decision based on current game state.
        
        Args:
            game_state: Observable state dictionary containing:
                - round: Current round number
                - position: Agent's position in supply chain
                - inventory: Current inventory level
                - backlog: Current backlog (unfulfilled orders)
                - incoming_order: Most recent order from downstream
                - last_outgoing_order: Last order placed upstream
                - round_cost: Cost incurred this round
                - decision_history: List of previous order decisions
                - customer_demand: Customer demand (only for retailer)
        
        Returns:
            Order quantity (non-negative integer)
        """
        pass
    
    def get_position(self) -> Position:
        """Get agent's position in supply chain"""
        return self.position
    
    def get_name(self) -> str:
        """Get agent's name"""
        return self.name
    
    def reset(self) -> None:
        """Reset agent state (called between games)"""
        pass


class SimpleAgent(Agent):
    """
    Simple algorithmic agent that uses basic inventory management rules.
    
    This serves as a baseline/benchmark agent that follows simple heuristics:
    - Order = Recent demand + (Target inventory - Current inventory)
    - Uses exponential smoothing to estimate demand
    """
    
    def __init__(
        self, 
        position: Position, 
        target_inventory: int = 12,
        smoothing_factor: float = 0.3,
        name: Optional[str] = None
    ):
        """
        Initialize simple agent.
        
        Args:
            position: Supply chain position
            target_inventory: Target inventory level to maintain
            smoothing_factor: Alpha for exponential smoothing (0-1)
            name: Optional agent name
        """
        super().__init__(position, name)
        self.target_inventory = target_inventory
        self.smoothing_factor = smoothing_factor
        self.demand_estimate = 4.0  # Initial estimate
        
    def make_decision(self, game_state: Dict[str, Any]) -> int:
        """Make decision using simple inventory management rules"""
        current_inventory = game_state["inventory"]
        incoming_order = game_state["incoming_order"]
        backlog = game_state["backlog"]
        
        # Update demand estimate using exponential smoothing
        if game_state["round"] > 1:
            actual_demand = incoming_order
            self.demand_estimate = (
                self.smoothing_factor * actual_demand + 
                (1 - self.smoothing_factor) * self.demand_estimate
            )
        
        # Calculate order quantity
        inventory_gap = self.target_inventory - current_inventory
        safety_stock = max(0, backlog)  # Extra stock to clear backlog
        
        order = max(0, int(self.demand_estimate + inventory_gap + safety_stock))
        
        return order
    
    def reset(self) -> None:
        """Reset demand estimate"""
        self.demand_estimate = 4.0


class RandomAgent(Agent):
    """Random agent for testing purposes"""
    
    def __init__(self, position: Position, min_order: int = 0, max_order: int = 20, name: Optional[str] = None):
        super().__init__(position, name)
        self.min_order = min_order
        self.max_order = max_order
        
    def make_decision(self, game_state: Dict[str, Any]) -> int:
        """Make random decision within bounds"""
        return random.randint(self.min_order, self.max_order)


class OptimalAgent(Agent):
    """
    Agent that uses more sophisticated optimization strategies.
    
    This implements a more advanced baseline using:
    - Demand forecasting with trend detection
    - Safety stock calculations
    - Bullwhip effect mitigation
    """
    
    def __init__(
        self, 
        position: Position,
        service_level_target: float = 0.95,
        forecast_horizon: int = 4,
        name: Optional[str] = None
    ):
        super().__init__(position, name)
        self.service_level_target = service_level_target
        self.forecast_horizon = forecast_horizon
        self.demand_history: List[int] = []
        
    def make_decision(self, game_state: Dict[str, Any]) -> int:
        """Make decision using demand forecasting and safety stock"""
        incoming_order = game_state["incoming_order"]
        current_inventory = game_state["inventory"]
        backlog = game_state["backlog"]
        
        # Update demand history
        if game_state["round"] > 1:
            self.demand_history.append(incoming_order)
            
        # Keep only recent history
        if len(self.demand_history) > 10:
            self.demand_history = self.demand_history[-10:]
        
        # Forecast demand
        if len(self.demand_history) >= 2:
            # Simple trend-adjusted forecast
            recent_avg = sum(self.demand_history[-3:]) / min(3, len(self.demand_history))
            trend = 0
            if len(self.demand_history) >= 4:
                early_avg = sum(self.demand_history[-6:-3]) / 3
                trend = recent_avg - early_avg
            forecast = recent_avg + trend
        else:
            forecast = 4.0  # Initial estimate
        
        # Calculate safety stock (simple approximation)
        if len(self.demand_history) >= 2:
            demand_std = (sum((d - forecast) ** 2 for d in self.demand_history) / len(self.demand_history)) ** 0.5
            safety_stock = 1.65 * demand_std  # ~95% service level
        else:
            safety_stock = 2.0
        
        # Calculate order quantity
        expected_demand = forecast * 2  # 2-week lead time
        target_inventory = expected_demand + safety_stock
        
        # Account for pipeline inventory (simplified)
        pipeline_estimate = game_state["last_outgoing_order"] * 2
        
        effective_inventory = current_inventory + pipeline_estimate
        order = max(0, int(target_inventory - effective_inventory + forecast + backlog))
        
        return order
    
    def reset(self) -> None:
        """Reset demand history"""
        self.demand_history = []