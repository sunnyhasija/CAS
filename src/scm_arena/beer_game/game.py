

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import copy

from .agents import Agent, Position


# Standard academic cost parameters
HOLDING_COST_PER_UNIT = 1.0    # $1 per unit per period
BACKORDER_COST_PER_UNIT = 2.0  # $2 per unit per period


class VisibilityLevel(Enum):
    """Information visibility levels in the supply chain"""
    LOCAL = "local"              # See only own state (classic)
    ADJACENT = "adjacent"        # See own + immediate upstream/downstream
    FULL = "full"               # See entire supply chain state


@dataclass
class PlayerState:
    """State of a single player in the supply chain"""
    position: Position
    inventory: int = 12           # Starting inventory
    backlog: int = 0             # Unfulfilled orders
    incoming_order: int = 4      # Order from downstream
    outgoing_order: int = 4      # Order to upstream
    
    # Shipping delays (2-period delay)
    shipping_delay_1: int = 4    # Arriving next period
    shipping_delay_2: int = 4    # Arriving in 2 periods
    
    # Order delays (2-period delay)
    order_delay_1: int = 4       # Order placed 1 period ago
    order_delay_2: int = 4       # Order placed 2 periods ago
    
    # Cost tracking
    period_cost: float = 0.0
    total_cost: float = 0.0
    cost: float = 0.0  # Alias for compatibility
    
    # Decision history
    decision_history: List[int] = None
    
    # FIXED: Service level tracking
    period_demand: int = 0       # Demand received this period
    period_fulfilled: int = 0    # Demand fulfilled this period
    
    def __post_init__(self):
        if self.decision_history is None:
            self.decision_history = []


@dataclass
class GameState:
    """Complete state of the Beer Game at one point in time"""
    round: int
    players: Dict[Position, PlayerState]
    customer_demand: int
    total_cost: float
    is_complete: bool = False
    
    def to_agent_view(self, position: Position, visibility: VisibilityLevel = VisibilityLevel.LOCAL, 
                      state_history: List['GameState'] = None, memory_window: Optional[int] = None) -> Dict[str, Any]:
        """
        Convert to agent-visible state based on visibility level and history.
        
        FIXED: Now consistently applies memory_window to both own and partner histories.
        
        Args:
            position: Agent's position
            visibility: Information visibility level
            state_history: Historical game states for visibility
            memory_window: Memory window for decision histories (None=all, 0=none, N=last N decisions)
        """
        player = self.players[position]
        
        # Helper function to apply memory window consistently
        def apply_memory_window(decision_history: List[int]) -> List[int]:
            """Apply memory window to decision history consistently"""
            if memory_window == 0:
                return []  # No history
            elif memory_window is None:
                return decision_history.copy()  # Full history
            elif memory_window > 0:
                return decision_history[-memory_window:] if decision_history else []  # Limited history
            else:
                return []  # Invalid memory window
        
        # Base state (always visible)
        state = {
            "round": self.round,
            "position": position.value,
            "inventory": player.inventory,
            "backlog": player.backlog,
            "incoming_order": player.incoming_order,
            "last_outgoing_order": player.outgoing_order,
            "round_cost": player.period_cost,
            "decision_history": apply_memory_window(player.decision_history),  # FIXED: Apply memory window
        }
        
        # Add customer demand for retailer
        if position == Position.RETAILER:
            state["customer_demand"] = self.customer_demand
        
        # Add visibility-based information
        if visibility != VisibilityLevel.LOCAL:
            visible_positions = self._get_visible_positions(position, visibility)
            
            if visible_positions:
                # Current state visibility - FIXED: Apply memory window to partner histories
                state["visible_supply_chain"] = {}
                for vis_pos in visible_positions:
                    vis_player = self.players[vis_pos]
                    state["visible_supply_chain"][vis_pos.value] = {
                        "inventory": vis_player.inventory,
                        "backlog": vis_player.backlog,
                        "incoming_order": vis_player.incoming_order,
                        "outgoing_order": vis_player.outgoing_order,
                        "cost": vis_player.period_cost,
                        "decision_history": apply_memory_window(vis_player.decision_history)  # FIXED: Consistent memory window
                    }
                
                # Historical visibility if state_history provided - FIXED: Apply memory window
                if state_history:
                    state["visible_history"] = self._get_visible_history(
                        position, visible_positions, state_history, memory_window
                    )
        
        # Add system-wide metrics for full visibility
        if visibility == VisibilityLevel.FULL:
            state["system_metrics"] = {
                "total_system_cost": self.total_cost,
                "total_system_inventory": sum(p.inventory for p in self.players.values()),
                "total_system_backlog": sum(p.backlog for p in self.players.values()),
                "customer_demand": self.customer_demand
            }
            
        return state
    
    def _get_visible_positions(self, position: Position, visibility: VisibilityLevel) -> List[Position]:
        """Get positions visible to given position based on visibility level"""
        if visibility == VisibilityLevel.LOCAL:
            return []
        
        elif visibility == VisibilityLevel.ADJACENT:
            positions = []
            downstream = position.get_downstream_position()
            upstream = position.get_upstream_position()
            if downstream:
                positions.append(downstream)
            if upstream:
                positions.append(upstream)
            return positions
                
        elif visibility == VisibilityLevel.FULL:
            # All other positions
            return [pos for pos in Position if pos != position]
        
        return []
    
    def _get_visible_history(self, position: Position, visible_positions: List[Position], 
                           state_history: List['GameState'], memory_window: Optional[int] = None) -> Dict[str, List[Dict]]:
        """
        Get historical states for visible positions.
        
        FIXED: Now respects memory_window parameter for historical data.
        FIXED: Added proper error handling instead of silent data loss.
        
        Args:
            position: Agent's position
            visible_positions: Positions visible to the agent
            state_history: Historical game states
            memory_window: Memory window for historical data (None=all, 0=none, N=last N rounds)
        """
        history = {}
        
        for vis_pos in visible_positions:
            position_history = []
            for past_state in state_history:
                # FIXED: Proper error handling instead of silent skip
                if vis_pos not in past_state.players:
                    raise ValueError(f"Position {vis_pos} missing from historical state round {past_state.round}")
                
                past_player = past_state.players[vis_pos]
                position_history.append({
                    "round": past_state.round,
                    "inventory": past_player.inventory,
                    "backlog": past_player.backlog,
                    "incoming_order": past_player.incoming_order,
                    "outgoing_order": past_player.outgoing_order,
                    "cost": past_player.period_cost
                })
            
            # FIXED: Apply memory window to historical data consistently
            if memory_window == 0:
                position_history = []  # No history
            elif memory_window is not None and memory_window > 0:
                position_history = position_history[-memory_window:]  # Limited history
            # If memory_window is None, keep full history
            
            history[vis_pos.value] = position_history
        
        return history


@dataclass
class GameResults:
    """Final results and metrics from a completed game"""
    total_cost: float
    individual_costs: Dict[Position, float]
    service_level: float
    bullwhip_ratio: float
    total_rounds: int
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        return {
            "total_cost": self.total_cost,
            "cost_per_round": self.total_cost / self.total_rounds,
            "service_level": self.service_level,
            "bullwhip_ratio": self.bullwhip_ratio,
            **{f"{pos.value}_cost": cost for pos, cost in self.individual_costs.items()}
        }


class BeerGame:
    """
    Beer Game simulation engine implementing the classic supply chain coordination game.
    
    Uses standard academic cost structure:
    - Holding cost: $1 per unit per period
    - Backorder cost: $2 per unit per period
    
    Classic mode implements MIT classroom version:
    - 2-period information delay + 2-period shipping delay = 4-period total delay
    
    FIXED: Service level calculation now tracks period-by-period fulfillment
    FIXED: Bullwhip calculation uses Lee, Padmanabhan & Whang (1997) methodology
    """
    
    def __init__(
        self, 
        agents: Dict[Position, Agent], 
        demand_pattern: List[int],
        classic_mode: bool = False,
        visibility_level: VisibilityLevel = VisibilityLevel.LOCAL,
        holding_cost: float = HOLDING_COST_PER_UNIT,
        backorder_cost: float = BACKORDER_COST_PER_UNIT
    ):
        """
        Initialize Beer Game.
        
        Args:
            agents: Dictionary mapping positions to agent instances
            demand_pattern: List of customer demands for each round
            classic_mode: If True, use classic 2-period information delays
            visibility_level: Information visibility level across supply chain
            holding_cost: Cost per unit of inventory per period
            backorder_cost: Cost per unit of backlog per period
        """
        self.agents = agents
        self.demand_pattern = demand_pattern
        self.classic_mode = classic_mode
        self.visibility_level = visibility_level
        self.holding_cost = holding_cost
        self.backorder_cost = backorder_cost
        
        # Validate agents
        required_positions = set(Position)
        provided_positions = set(agents.keys())
        if required_positions != provided_positions:
            missing = required_positions - provided_positions
            extra = provided_positions - required_positions
            raise ValueError(f"Invalid agents. Missing: {missing}, Extra: {extra}")
        
        # Initialize game state
        self.round = 0
        self.state_history: List[GameState] = []
        self.current_state = self._initialize_game_state()
        
    def _initialize_game_state(self) -> GameState:
        """Initialize starting state with standard parameters"""
        players = {}
        
        for position in Position:
            players[position] = PlayerState(
                position=position,
                inventory=12,           # Standard starting inventory
                backlog=0,
                incoming_order=4,       # Standard starting order
                outgoing_order=4,
                shipping_delay_1=4,     # Standard pipeline
                shipping_delay_2=4,
                order_delay_1=4,        # Standard order pipeline  
                order_delay_2=4,
                decision_history=[]
            )
        
        return GameState(
            round=0,
            players=players,
            customer_demand=4,  # Standard starting demand
            total_cost=0.0
        )
    
    def step(self) -> GameState:
        """Execute one round of the game"""
        if self.is_complete():
            return self.current_state
            
        self.round += 1
        
        # Get customer demand for this round
        if self.round <= len(self.demand_pattern):
            customer_demand = self.demand_pattern[self.round - 1]
        else:
            # Use last demand if pattern is exhausted
            customer_demand = self.demand_pattern[-1]
        
        # Phase 1: Receive shipments and update inventory
        self._process_shipments()
        
        # Phase 2: Fill orders and update backlogs - FIXED: Now tracks service level properly
        self._fill_orders(customer_demand)
        
        # Phase 3: Get agent decisions - FIXED: Pass memory window consistently
        decisions = self._get_agent_decisions()
        
        # Phase 4: Update order pipeline
        self._update_order_pipeline(decisions)
        
        # Phase 5: Calculate costs
        self._calculate_costs()
        
        # Update state
        self.current_state.round = self.round
        self.current_state.customer_demand = customer_demand
        
        # Store history
        self.state_history.append(copy.deepcopy(self.current_state))
        
        return self.current_state
    
    def _process_shipments(self):
        """Process incoming shipments (advance shipping pipeline)"""
        for position in Position:
            player = self.current_state.players[position]
            
            # Receive shipment from shipping_delay_1
            player.inventory += player.shipping_delay_1
            
            # Advance pipeline
            player.shipping_delay_1 = player.shipping_delay_2
            player.shipping_delay_2 = 0  # Will be filled by upstream orders
    
    def _fill_orders(self, customer_demand: int):
        """
        Fill orders from downstream and update backlogs.
        
        CRITICAL BUG FIX: Modern mode now properly propagates actual orders
        instead of fulfilled amounts to ensure fair comparison with classic mode.
        
        FIXED: Now properly tracks service level metrics period-by-period.
        """
        # Start with customer demand at retailer
        demand = customer_demand
        
        for position in Position.get_supply_chain_order():
            player = self.current_state.players[position]
            
            # Update incoming order (this round's demand)
            player.incoming_order = demand
            
            # FIXED: Track period demand for service level calculation
            player.period_demand = demand
            
            # Total demand = incoming order + existing backlog
            total_demand = player.incoming_order + player.backlog
            
            # Fill what we can from inventory
            fulfilled = min(total_demand, player.inventory)
            player.inventory -= fulfilled
            
            # Update backlog
            player.backlog = total_demand - fulfilled
            
            # FIXED: Track period fulfillment for service level calculation
            player.period_fulfilled = min(player.incoming_order, fulfilled)
            
            # FIXED: Demand propagation logic
            if self.classic_mode and position != Position.RETAILER:
                # Classic mode: Use historical order information (delayed but accurate)
                demand = player.order_delay_2
            else:
                # FIXED Modern mode: Use actual incoming order, NOT fulfilled amount
                # This preserves demand signal integrity through the supply chain
                demand = player.incoming_order
    
    def _get_agent_decisions(self) -> Dict[Position, int]:
        """Get ordering decisions from all agents - FIXED: Pass memory window consistently"""
        decisions = {}
        
        for position in Position:
            agent = self.agents[position]
            
            # FIXED: Get memory window from agent if available
            memory_window = getattr(agent, 'memory_window', None)
            
            agent_state = self.current_state.to_agent_view(
                position, self.visibility_level, self.state_history, memory_window
            )
            
            try:
                decision = agent.make_decision(agent_state)
                # Validate decision
                if not isinstance(decision, int) or decision < 0:
                    raise ValueError(f"Invalid decision: {decision}")
                decisions[position] = decision
            except Exception as e:
                print(f"Warning: Agent {agent.get_name()} failed to make decision: {e}")
                # Fallback decision
                decisions[position] = agent_state["incoming_order"]
        
        return decisions
    
    def _update_order_pipeline(self, decisions: Dict[Position, int]):
        """Update order pipeline with new decisions"""
        for position in Position:
            player = self.current_state.players[position]
            decision = decisions[position]
            
            # Update decision history
            player.decision_history.append(decision)
            player.outgoing_order = decision
            
            # Advance order pipeline for classic mode delays
            player.order_delay_2 = player.order_delay_1
            player.order_delay_1 = decision
            
            # Update upstream shipping pipeline
            upstream_pos = position.get_upstream_position()
            if upstream_pos:
                upstream_player = self.current_state.players[upstream_pos]
                if self.classic_mode:
                    # Classic mode: 2-period shipping delay
                    upstream_player.shipping_delay_2 = decision
                else:
                    # Modern mode: 1-period shipping delay
                    upstream_player.shipping_delay_1 = decision
            # Manufacturer has infinite supply
            elif position == Position.MANUFACTURER:
                if self.classic_mode:
                    player.shipping_delay_2 = decision
                else:
                    player.shipping_delay_1 = decision
    
    def _calculate_costs(self):
        """Calculate costs for this round"""
        total_round_cost = 0.0
        
        for position in Position:
            player = self.current_state.players[position]
            
            # Calculate individual costs
            holding_cost = max(0, player.inventory) * self.holding_cost
            backorder_cost = player.backlog * self.backorder_cost
            
            player.period_cost = holding_cost + backorder_cost
            player.total_cost += player.period_cost
            player.cost = player.total_cost  # Compatibility alias
            total_round_cost += player.period_cost
        
        self.current_state.total_cost += total_round_cost
    
    def is_complete(self) -> bool:
        """Check if game is complete"""
        return self.round >= len(self.demand_pattern)
    
    def get_results(self) -> GameResults:
        """Get final game results and metrics"""
        if not self.is_complete():
            raise ValueError("Game is not complete yet")
        
        # Calculate individual costs
        individual_costs = {}
        total_cost = 0.0
        
        for position in Position:
            cost = self.current_state.players[position].total_cost
            individual_costs[position] = cost
            total_cost += cost
        
        # FIXED: Calculate service level properly using period-by-period fulfillment
        service_level = self._calculate_service_level()
        
        # FIXED: Calculate bullwhip ratio using Lee, Padmanabhan & Whang (1997) methodology
        bullwhip_ratio = self._calculate_bullwhip_ratio()
        
        return GameResults(
            total_cost=total_cost,
            individual_costs=individual_costs,
            service_level=service_level,
            bullwhip_ratio=bullwhip_ratio,
            total_rounds=len(self.state_history)
        )
    
    def _calculate_service_level(self) -> float:
        """
        FIXED: Calculate service level as period-by-period fulfillment rate.
        
        Service level = Total demand fulfilled / Total demand received
        Calculated properly without double-counting cumulative backlog.
        """
        total_demand = 0
        total_fulfilled = 0
        
        for state in self.state_history:
            for position in Position:
                player = state.players[position]
                # Use period-specific demand and fulfillment tracking
                if hasattr(player, 'period_demand') and hasattr(player, 'period_fulfilled'):
                    total_demand += player.period_demand
                    total_fulfilled += player.period_fulfilled
                else:
                    # Fallback for states without period tracking (shouldn't happen with fixed code)
                    total_demand += player.incoming_order
                    # Estimate fulfillment as incoming order minus increase in backlog
                    if len(self.state_history) > 1:
                        prev_idx = self.state_history.index(state) - 1
                        if prev_idx >= 0:
                            prev_player = self.state_history[prev_idx].players[position]
                            backlog_increase = player.backlog - prev_player.backlog
                            estimated_fulfilled = max(0, player.incoming_order - backlog_increase)
                            total_fulfilled += estimated_fulfilled
                        else:
                            total_fulfilled += max(0, player.incoming_order - player.backlog)
                    else:
                        total_fulfilled += max(0, player.incoming_order - player.backlog)
        
        return total_fulfilled / max(1, total_demand)
    
    def _calculate_bullwhip_ratio(self) -> float:
        """
        FIXED: Calculate bullwhip effect using Lee, Padmanabhan & Whang (1997) methodology.
        
        Bullwhip_i = Var(O^out_i) / Var(O^in_i)
        
        Where:
        - O^in_i = orders received by stage i (its effective demand) 
        - O^out_i = orders sent upstream by stage i
        
        Returns the ratio of manufacturer outgoing order variance to retailer incoming order variance
        (variance amplification from downstream to upstream).
        """
        if len(self.state_history) < 3:
            return 1.0
        
        # Collect order sequences: incoming orders (demand) and outgoing orders (response)
        incoming_orders_by_position = {pos: [] for pos in Position}
        outgoing_orders_by_position = {pos: [] for pos in Position}
        
        for state in self.state_history:
            for position in Position:
                player = state.players[position]
                incoming_orders_by_position[position].append(player.incoming_order)
                outgoing_orders_by_position[position].append(player.outgoing_order)
        
        # Calculate variances for incoming and outgoing orders
        def calculate_variance(orders: List[int]) -> float:
            if len(orders) <= 1:
                return 0.001  # Avoid division by zero
            mean = sum(orders) / len(orders)
            variance = sum((x - mean) ** 2 for x in orders) / len(orders)
            return max(variance, 0.001)  # Avoid division by zero
        
        # Calculate bullwhip ratio using Lee, Padmanabhan & Whang methodology:
        # Ratio of upstream outgoing variance to downstream incoming variance
        
        # Retailer incoming orders = customer demand variance (baseline)
        retailer_incoming_var = calculate_variance(incoming_orders_by_position[Position.RETAILER])
        
        # Manufacturer outgoing orders = most upstream variance amplification
        manufacturer_outgoing_var = calculate_variance(outgoing_orders_by_position[Position.MANUFACTURER])
        
        # Bullwhip ratio = variance amplification from customer demand to manufacturer response
        bullwhip_ratio = manufacturer_outgoing_var / retailer_incoming_var
        
        return bullwhip_ratio
    
    def get_state_history(self) -> List[GameState]:
        """Get complete game state history"""
        return self.state_history.copy()


def create_classic_beer_game(
    agents: Dict[Position, Agent], 
    demand_pattern: List[int]
) -> BeerGame:
    """
    Create Beer Game with classic 1960s settings (MIT classroom version).
    
    - 2-period information delays
    - 2-period shipping delays  
    - Total: 4-period delay (2 info + 2 transportation)
    - $1 holding : $2 backorder costs
    """
    return BeerGame(
        agents=agents,
        demand_pattern=demand_pattern,
        classic_mode=True,
        holding_cost=HOLDING_COST_PER_UNIT,
        backorder_cost=BACKORDER_COST_PER_UNIT
    )


def create_modern_beer_game(
    agents: Dict[Position, Agent], 
    demand_pattern: List[int]
) -> BeerGame:
    """
    Create Beer Game with modern settings.
    
    - Instant information flow (FIXED: now preserves actual orders)
    - 1-period shipping delays
    - $1 holding : $2 backorder costs
    """
    return BeerGame(
        agents=agents,
        demand_pattern=demand_pattern,
        classic_mode=False,
        holding_cost=HOLDING_COST_PER_UNIT,
        backorder_cost=BACKORDER_COST_PER_UNIT
    )