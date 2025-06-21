"""
Core Beer Game simulation engine.

The Beer Game is a classic supply chain coordination game where 4 agents
(Retailer, Wholesaler, Distributor, Manufacturer) must coordinate to minimize
total system cost while dealing with information delays and uncertainty.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from .agents import Agent, Position


class GamePhase(Enum):
    """Current phase of the game"""
    SETUP = "setup"
    RUNNING = "running"
    COMPLETED = "completed"


@dataclass
class PlayerState:
    """State of a single player in the supply chain"""
    position: Position
    inventory: int
    backlog: int  # Unfulfilled orders
    incoming_order: int  # Order from downstream player
    outgoing_order: int  # Order to upstream player
    cost: float
    decisions: List[int] = field(default_factory=list)  # History of order decisions


@dataclass
class GameState:
    """Complete state of the Beer Game at a given round"""
    round: int
    phase: GamePhase
    players: Dict[Position, PlayerState]
    customer_demand: int
    total_cost: float
    
    def get_player_state(self, position: Position) -> PlayerState:
        """Get state for a specific player position"""
        return self.players[position]
    
    def get_observable_state(self, position: Position) -> Dict:
        """Get the observable state for a player (what they can see)"""
        player = self.players[position]
        
        # Base information every player can see
        observable = {
            "round": self.round,
            "position": position.value,
            "inventory": player.inventory,
            "backlog": player.backlog,
            "incoming_order": player.incoming_order,
            "last_outgoing_order": player.outgoing_order,
            "round_cost": player.cost,
            "decision_history": player.decisions.copy(),
        }
        
        # Add customer demand for retailer
        if position == Position.RETAILER:
            observable["customer_demand"] = self.customer_demand
            
        return observable


@dataclass
class GameResults:
    """Final results of a completed Beer Game"""
    total_cost: float
    individual_costs: Dict[Position, float]
    bullwhip_ratio: float
    service_level: float
    rounds_played: int
    demand_pattern: List[int]
    final_inventories: Dict[Position, int]
    
    def summary(self) -> Dict:
        """Get a summary of key metrics"""
        return {
            "total_cost": self.total_cost,
            "cost_per_round": self.total_cost / self.rounds_played,
            "bullwhip_ratio": self.bullwhip_ratio,
            "service_level": self.service_level,
            "retailer_cost": self.individual_costs[Position.RETAILER],
            "wholesaler_cost": self.individual_costs[Position.WHOLESALER],
            "distributor_cost": self.individual_costs[Position.DISTRIBUTOR],
            "manufacturer_cost": self.individual_costs[Position.MANUFACTURER],
        }


class BeerGame:
    """
    Core Beer Game simulation engine.
    
    Implements the classic 4-tier supply chain coordination game with:
    - 2-week shipping delays
    - 2-week information delays  
    - $1/unit holding cost, $2/unit backorder cost
    - Weekly decision cycles
    """
    
    def __init__(
        self,
        agents: Dict[Position, Agent],
        demand_pattern: List[int],
        initial_inventory: int = 12,
        initial_backlog: int = 0,
        holding_cost: float = 1.0,
        backorder_cost: float = 2.0,
        max_rounds: int = 50
    ):
        """
        Initialize a new Beer Game.
        
        Args:
            agents: Dictionary mapping positions to agent instances
            demand_pattern: List of customer demands for each round
            initial_inventory: Starting inventory for each player
            initial_backlog: Starting backlog for each player
            holding_cost: Cost per unit of inventory held
            backorder_cost: Cost per unit of backlog
            max_rounds: Maximum number of rounds to play
        """
        self.agents = agents
        self.demand_pattern = demand_pattern
        self.holding_cost = holding_cost
        self.backorder_cost = backorder_cost
        self.max_rounds = max_rounds
        
        # Initialize game state
        self.current_round = 0
        self.phase = GamePhase.SETUP
        self.total_cost = 0.0
        
        # Initialize player states
        self.players = {
            position: PlayerState(
                position=position,
                inventory=initial_inventory,
                backlog=initial_backlog,
                incoming_order=4,  # Initial steady state
                outgoing_order=4,
                cost=0.0
            )
            for position in Position
        }
        
        # Shipping and information delay queues (2 weeks each)
        self.shipping_delays = {
            Position.RETAILER: [4, 4],  # From wholesaler
            Position.WHOLESALER: [4, 4],  # From distributor  
            Position.DISTRIBUTOR: [4, 4],  # From manufacturer
            Position.MANUFACTURER: [4, 4],  # Production
        }
        
        self.order_delays = {
            Position.WHOLESALER: [4, 4],  # From retailer
            Position.DISTRIBUTOR: [4, 4],  # From wholesaler
            Position.MANUFACTURER: [4, 4],  # From distributor
        }
        
        # Game history
        self.history: List[GameState] = []
        
    def get_current_state(self) -> GameState:
        """Get the current game state"""
        return GameState(
            round=self.current_round,
            phase=self.phase,
            players=self.players.copy(),
            customer_demand=self._get_current_demand(),
            total_cost=self.total_cost
        )
    
    def step(self) -> GameState:
        """
        Execute one round of the Beer Game.
        
        Returns:
            Current game state after the step
        """
        if self.phase != GamePhase.RUNNING and self.phase != GamePhase.SETUP:
            raise ValueError("Cannot step a completed game")
            
        if self.phase == GamePhase.SETUP:
            self.phase = GamePhase.RUNNING
            
        self.current_round += 1
        
        # 1. Receive shipments (from delay queues)
        self._process_shipments()
        
        # 2. Fill orders and calculate costs
        self._fill_orders()
        
        # 3. Get decisions from agents
        current_state = self.get_current_state()
        decisions = self._get_agent_decisions(current_state)
        
        # 4. Process new orders through delay queues
        self._process_new_orders(decisions)
        
        # 5. Update history
        final_state = self.get_current_state()
        self.history.append(final_state)
        
        # 6. Check if game is complete
        if self.current_round >= len(self.demand_pattern) or self.current_round >= self.max_rounds:
            self.phase = GamePhase.COMPLETED
            
        return final_state
    
    def is_complete(self) -> bool:
        """Check if the game has finished"""
        return self.phase == GamePhase.COMPLETED
    
    def get_results(self) -> GameResults:
        """Get final game results (only available after completion)"""
        if not self.is_complete():
            raise ValueError("Game not yet complete")
            
        # Calculate individual costs
        individual_costs = {
            position: sum(state.players[position].cost for state in self.history)
            for position in Position
        }
        
        # Calculate bullwhip ratio
        bullwhip_ratio = self._calculate_bullwhip_ratio()
        
        # Calculate service level
        service_level = self._calculate_service_level()
        
        # Get final inventories
        final_inventories = {
            position: self.players[position].inventory
            for position in Position
        }
        
        return GameResults(
            total_cost=self.total_cost,
            individual_costs=individual_costs,
            bullwhip_ratio=bullwhip_ratio,
            service_level=service_level,
            rounds_played=len(self.history),
            demand_pattern=self.demand_pattern[:len(self.history)],
            final_inventories=final_inventories
        )
    
    def _get_current_demand(self) -> int:
        """Get customer demand for current round"""
        if self.current_round <= 0 or self.current_round > len(self.demand_pattern):
            return 0
        return self.demand_pattern[self.current_round - 1]
    
    def _process_shipments(self) -> None:
        """Process incoming shipments from delay queues"""
        for position in Position:
            if position in self.shipping_delays:
                # Receive shipment that was sent 2 rounds ago
                shipment = self.shipping_delays[position].pop(0)
                self.players[position].inventory += shipment
    
    def _fill_orders(self) -> None:
        """Fill orders and calculate costs for current round"""
        # Start with customer demand to retailer
        current_demand = self._get_current_demand()
        
        for position in Position:
            player = self.players[position]
            
            # Determine demand on this player
            if position == Position.RETAILER:
                demand = current_demand
            else:
                demand = player.incoming_order
            
            # Fill orders from available inventory
            fulfilled = min(demand + player.backlog, player.inventory)
            player.inventory -= fulfilled
            
            # Update backlog
            total_demand = demand + player.backlog
            player.backlog = max(0, total_demand - fulfilled)
            
            # Calculate costs
            holding_cost = player.inventory * self.holding_cost
            backorder_cost = player.backlog * self.backorder_cost
            player.cost = holding_cost + backorder_cost
            self.total_cost += player.cost
    
    def _get_agent_decisions(self, state: GameState) -> Dict[Position, int]:
        """Get order decisions from all agents"""
        decisions = {}
        
        for position, agent in self.agents.items():
            observable_state = state.get_observable_state(position)
            decision = agent.make_decision(observable_state)
            decisions[position] = decision
            
            # Store decision in player history
            self.players[position].decisions.append(decision)
            self.players[position].outgoing_order = decision
            
        return decisions
    
    def _process_new_orders(self, decisions: Dict[Position, int]) -> None:
        """Process new orders through information and shipping delay queues"""
        # Update order delays (information flow)
        if Position.RETAILER in decisions:
            self.order_delays[Position.WHOLESALER].append(decisions[Position.RETAILER])
            
        if Position.WHOLESALER in decisions:
            self.order_delays[Position.DISTRIBUTOR].append(decisions[Position.WHOLESALER])
            
        if Position.DISTRIBUTOR in decisions:
            self.order_delays[Position.MANUFACTURER].append(decisions[Position.DISTRIBUTOR])
        
        # Process orders that arrive this round (2-round delay)
        for downstream, upstream in [
            (Position.RETAILER, Position.WHOLESALER),
            (Position.WHOLESALER, Position.DISTRIBUTOR), 
            (Position.DISTRIBUTOR, Position.MANUFACTURER)
        ]:
            if upstream in self.order_delays:
                arriving_order = self.order_delays[upstream].pop(0)
                self.players[upstream].incoming_order = arriving_order
        
        # Update shipping delays (production/shipping)
        for position in Position:
            if position in decisions:
                self.shipping_delays[position].append(decisions[position])
    
    def _calculate_bullwhip_ratio(self) -> float:
        """Calculate bullwhip effect (variance amplification up the supply chain)"""
        if len(self.history) < 4:
            return 1.0
            
        # Get order variance at each level
        retailer_orders = [state.players[Position.RETAILER].outgoing_order for state in self.history]
        manufacturer_orders = [state.players[Position.MANUFACTURER].outgoing_order for state in self.history]
        
        retailer_var = np.var(retailer_orders) if len(retailer_orders) > 1 else 1.0
        manufacturer_var = np.var(manufacturer_orders) if len(manufacturer_orders) > 1 else 1.0
        
        # Avoid division by zero
        if retailer_var == 0:
            return 1.0 if manufacturer_var == 0 else float('inf')
            
        return manufacturer_var / retailer_var
    
    def _calculate_service_level(self) -> float:
        """Calculate overall service level (orders fulfilled / orders received)"""
        if not self.history:
            return 0.0
            
        total_demand = sum(self._get_demand_for_round(i) for i in range(1, len(self.history) + 1))
        total_backlog = sum(state.players[Position.RETAILER].backlog for state in self.history)
        
        if total_demand == 0:
            return 1.0
            
        fulfilled = total_demand - total_backlog
        return max(0.0, fulfilled / total_demand)
    
    def _get_demand_for_round(self, round_num: int) -> int:
        """Get demand for a specific round"""
        if round_num <= 0 or round_num > len(self.demand_pattern):
            return 0
        return self.demand_pattern[round_num - 1]