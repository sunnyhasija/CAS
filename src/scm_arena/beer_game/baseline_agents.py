# src/scm_arena/beer_game/baseline_agents.py
"""
Baseline agents for Beer Game benchmarking.

Implements canonical behavioral and rule-based agents from the literature
for comparison against LLM performance.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from .agents import Agent, Position


class StermanAgent(Agent):
    """
    Sterman (1989) canonical behavioral heuristic for Beer Game.
    
    Implements the classic order-up-to policy with behavioral parameters:
    Order = Demand + α × Backlog + β × (Desired_Inventory - Current_Inventory)
    
    Based on:
    Sterman, J. D. (1989). Modeling managerial behavior: Misperceptions of feedback 
    in a dynamic decision making experiment. Management Science, 35(3), 321-339.
    """
    
    def __init__(
        self, 
        position: Position,
        desired_inventory: int = 12,
        backlog_weight: float = 0.5,
        inventory_weight: float = 0.5,
        demand_smoothing: float = 0.3,
        name: Optional[str] = None
    ):
        super().__init__(position, name or f"sterman_{position.value}")
        self.desired_inventory = desired_inventory
        self.backlog_weight = backlog_weight
        self.inventory_weight = inventory_weight
        self.demand_smoothing = demand_smoothing
        
        # State tracking
        self.demand_estimate = 4.0  # Initial Beer Game demand
        self.decision_history: List[int] = []
        
    def make_decision(self, game_state: Dict[str, Any]) -> int:
        """Make ordering decision using Sterman (1989) heuristic"""
        # Extract game state
        current_inventory = game_state["inventory"]
        backlog = game_state["backlog"]
        incoming_order = game_state["incoming_order"]
        
        # Update demand estimate using exponential smoothing
        if game_state["round"] > 1:
            self.demand_estimate = (
                self.demand_smoothing * incoming_order + 
                (1 - self.demand_smoothing) * self.demand_estimate
            )
        
        # Sterman (1989) canonical formula
        demand_component = self.demand_estimate
        backlog_component = self.backlog_weight * backlog
        inventory_component = self.inventory_weight * (self.desired_inventory - current_inventory)
        
        # Total order quantity
        order = demand_component + backlog_component + inventory_component
        
        # Ensure non-negative integer
        order = max(0, int(round(order)))
        
        # Store decision
        self.decision_history.append(order)
        
        return order
    
    def reset(self) -> None:
        """Reset agent state between games"""
        self.demand_estimate = 4.0
        self.decision_history = []


class SimpleReactiveAgent(Agent):
    """
    Simple reactive agent - no memory, pure order-up-to policy.
    Works across all visibility/memory conditions.
    """
    
    def __init__(self, position: Position, target_inventory: int = 12, name: Optional[str] = None):
        super().__init__(position, name or f"reactive_{position.value}")
        self.target_inventory = target_inventory
    
    def make_decision(self, game_state: Dict[str, Any]) -> int:
        """Pure reactive: Order = Demand + Inventory_Gap"""
        incoming_order = game_state["incoming_order"]
        inventory = game_state["inventory"]
        backlog = game_state["backlog"]
        
        # Simple order-up-to policy
        inventory_gap = max(0, self.target_inventory - inventory)
        order = incoming_order + inventory_gap + backlog
        
        return max(0, int(order))


class NewsvendorAgent(Agent):
    """
    Newsvendor model agent - classic operations research baseline.
    Optimal single-period inventory decision under uncertainty.
    """
    
    def __init__(self, position: Position, service_level: float = 0.95, name: Optional[str] = None):
        super().__init__(position, name or f"newsvendor_{position.value}")
        self.service_level = service_level
        self.demand_history = []
        
    def make_decision(self, game_state: Dict[str, Any]) -> int:
        """Newsvendor optimal policy"""
        incoming_order = game_state["incoming_order"]
        inventory = game_state["inventory"]
        backlog = game_state["backlog"]
        
        # Update demand history
        self.demand_history.append(incoming_order)
        
        # Estimate demand parameters
        if len(self.demand_history) >= 3:
            mean_demand = sum(self.demand_history[-10:]) / min(10, len(self.demand_history))
            recent_demands = self.demand_history[-5:] if len(self.demand_history) >= 5 else self.demand_history
            if len(recent_demands) > 1:
                variance = sum((d - mean_demand) ** 2 for d in recent_demands) / len(recent_demands)
                std_demand = max(1.0, variance ** 0.5)
            else:
                std_demand = 2.0
        else:
            mean_demand = 4.0
            std_demand = 2.0
        
        # Newsvendor critical ratio: Cu/(Cu+Co) = 2/(2+1) = 0.67
        critical_ratio = 2.0 / (2.0 + 1.0)
        
        # Normal approximation z-score
        if critical_ratio >= 0.95:
            z_score = 1.65
        elif critical_ratio >= 0.90:
            z_score = 1.28
        elif critical_ratio >= 0.67:
            z_score = 0.44
        else:
            z_score = 0.0
        
        # Optimal order quantity
        optimal_stock = mean_demand + z_score * std_demand
        current_position = inventory - backlog
        order = max(0, int(optimal_stock - current_position))
        
        return order
    
    def reset(self):
        self.demand_history = []


class BaseStockAgent(Agent):
    """
    Base-stock policy agent - classic supply chain baseline.
    Order up to a fixed base stock level.
    """
    
    def __init__(self, position: Position, base_stock_level: int = None, name: Optional[str] = None):
        super().__init__(position, name or f"basestock_{position.value}")
        
        # Position-specific base stock levels
        if base_stock_level is None:
            base_stock_levels = {
                Position.RETAILER: 12,
                Position.WHOLESALER: 16, 
                Position.DISTRIBUTOR: 20,
                Position.MANUFACTURER: 24
            }
            self.base_stock_level = base_stock_levels[position]
        else:
            self.base_stock_level = base_stock_level
    
    def make_decision(self, game_state: Dict[str, Any]) -> int:
        """Order-up-to base stock level"""
        inventory = game_state["inventory"]
        backlog = game_state["backlog"]
        
        # Calculate inventory position
        inventory_position = inventory - backlog
        
        # Order up to base stock level
        order = max(0, self.base_stock_level - inventory_position)
        
        return order


class MovingAverageAgent(Agent):
    """
    Moving average forecasting agent.
    Uses simple moving average for demand forecasting.
    """
    
    def __init__(self, position: Position, window_size: int = 5, safety_factor: float = 1.5, name: Optional[str] = None):
        super().__init__(position, name or f"movingavg_{position.value}")
        self.window_size = window_size
        self.safety_factor = safety_factor
        self.demand_history = []
    
    def make_decision(self, game_state: Dict[str, Any]) -> int:
        """Forecast-based ordering using moving average"""
        incoming_order = game_state["incoming_order"]
        inventory = game_state["inventory"]
        backlog = game_state["backlog"]
        
        # Update demand history
        self.demand_history.append(incoming_order)
        
        # Forecast using moving average
        if len(self.demand_history) >= self.window_size:
            forecast = sum(self.demand_history[-self.window_size:]) / self.window_size
        else:
            forecast = sum(self.demand_history) / len(self.demand_history) if self.demand_history else 4.0
        
        # Safety stock based on forecast
        safety_stock = self.safety_factor * forecast
        
        # Order quantity
        target_inventory = forecast + safety_stock
        current_position = inventory - backlog
        order = max(0, int(target_inventory - current_position + forecast))
        
        return order
    
    def reset(self):
        self.demand_history = []


def create_baseline_agent(agent_type: str, position: Position, **kwargs) -> Agent:
    """Factory function to create baseline agents"""
    
    if agent_type == 'sterman':
        return StermanAgent(position, **kwargs)
    elif agent_type == 'reactive':
        return SimpleReactiveAgent(position, **kwargs)
    elif agent_type == 'newsvendor':
        return NewsvendorAgent(position, **kwargs)
    elif agent_type == 'basestock':
        return BaseStockAgent(position, **kwargs)
    elif agent_type == 'movingavg':
        return MovingAverageAgent(position, **kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def create_baseline_agents_set(agent_type: str, **kwargs) -> Dict[Position, Agent]:
    """Create baseline agents for all positions"""
    return {
        position: create_baseline_agent(agent_type, position, **kwargs)
        for position in Position
    }


def get_baseline_descriptions() -> Dict[str, str]:
    """Get descriptions of baseline agents"""
    return {
        "sterman": "Sterman (1989) behavioral heuristic - canonical Beer Game baseline",
        "reactive": "Simple reactive agent - order-up-to policy, no memory",
        "newsvendor": "Newsvendor model - optimal single-period inventory under uncertainty", 
        "basestock": "Base-stock policy - classic supply chain control",
        "movingavg": "Moving average forecasting - simple demand prediction"
    }