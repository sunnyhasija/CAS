"""
Ollama client integration for Beer Game agents with cost-focused prompts.

This module provides an agent that uses Ollama-hosted LLMs to make
supply chain decisions with realistic cost constraints and no game references.
"""

import json
import requests
import time
from typing import Dict, Any, Optional, List, Tuple
from ..beer_game.agents import Agent, Position


class OllamaAgent(Agent):
    """
    LLM agent using Ollama API for supply chain decisions.
    
    Uses cost-focused prompts that avoid game references and emphasize
    the fundamental trade-off between inventory and stockout costs.
    """
    
    def __init__(
        self,
        position: Position,
        model_name: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_retries: int = 3,
        timeout: float = 30.0,
        neutral_prompt: bool = False,
        memory_window: Optional[int] = 5,
        name: Optional[str] = None
    ):
        """
        Initialize Ollama agent.
        
        Args:
            position: Supply chain position
            model_name: Name of Ollama model to use
            base_url: Ollama server URL
            temperature: LLM temperature (0.0 = deterministic, 1.0 = creative)
            max_retries: Maximum API retry attempts
            timeout: Request timeout in seconds
            neutral_prompt: Use neutral prompts instead of position-specific ones
            memory_window: Number of past decisions to include (None = all, 0 = none)
            name: Optional agent name
        """
        super().__init__(position, name or f"ollama_{model_name}_{position.value}")
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self.neutral_prompt = neutral_prompt
        self.memory_window = memory_window
        
        # API endpoints
        self.chat_url = f"{self.base_url}/api/chat"
        
        # Set up system prompt
        if self.neutral_prompt:
            self.system_prompt = self._create_neutral_cost_focused_prompt()
        else:
            self.system_prompt = self._create_cost_focused_system_prompt()
        
        # Request session for connection pooling
        self.session = requests.Session()
        
        # Initialize interaction tracking
        self._last_interaction = {}
        
    def _create_cost_focused_system_prompt(self) -> str:
        """Create position-specific system prompt focused on cost minimization"""
        
        position_prompts = {
            Position.RETAILER: """You are the RETAILER in a consumer goods supply chain. Your role is to serve end customers while minimizing total operational costs.

BUSINESS OBJECTIVE: Minimize total costs while maintaining reasonable customer service.

COST STRUCTURE:
- Inventory holding cost: $1 per unit per period (warehouse, insurance, obsolescence)
- Stockout penalty: $2 per unit of unmet customer demand (lost sales, expediting)
- Working capital constraint: Excess inventory ties up cash and reduces profitability

BUSINESS PRIORITIES:
1. Optimize inventory levels - avoid both stockouts AND excess inventory
2. Respond intelligently to demand changes without overreacting
3. Maintain customer service while controlling costs
4. Use demand visibility advantage to make informed decisions

DECISION FRAMEWORK:
- Analyze recent customer demand patterns and trends
- Right-size inventory to balance service vs. cost
- Ask: Is this demand change temporary or permanent?
- Consider: What's the cost of being wrong in either direction?

Your success = Low total costs + Adequate customer service""",

            Position.WHOLESALER: """You are the WHOLESALER in a consumer goods supply chain. Your role is to efficiently serve retailers while minimizing distribution costs.

BUSINESS OBJECTIVE: Minimize total costs while providing reliable service to retail partners.

COST STRUCTURE:
- Inventory holding cost: $1 per unit per period (warehousing, handling, capital)
- Stockout penalty: $2 per unit of unmet retailer orders (lost sales, relationships)
- Working capital constraint: Inventory investment must generate returns

BUSINESS PRIORITIES:
1. Optimize inventory investment for maximum efficiency
2. Provide reliable service to retailers without costly overstocking
3. Smooth retailer demand volatility through intelligent buffering
4. Balance service reliability with cost control

DECISION FRAMEWORK:
- Anticipate retailer needs based on ordering patterns
- Maintain cost-effective safety stock levels
- Distinguish between temporary spikes and sustained demand changes
- Consider: What's the minimum inventory needed for reliable service?

Your success = Low total costs + Reliable retailer service""",

            Position.DISTRIBUTOR: """You are the DISTRIBUTOR in a consumer goods supply chain. Your role is to coordinate regional supply networks while minimizing logistics costs.

BUSINESS OBJECTIVE: Minimize total costs while ensuring supply network reliability.

COST STRUCTURE:
- Inventory holding cost: $1 per unit per period (storage, handling, opportunity cost)
- Stockout penalty: $2 per unit of unmet orders (disruption, expediting)
- Working capital constraint: Inventory ties up significant capital

BUSINESS PRIORITIES:
1. Optimize inventory investment across the distribution network
2. Coordinate between wholesalers and manufacturing efficiently
3. Minimize logistics costs through intelligent planning
4. Balance network reliability with cost control

DECISION FRAMEWORK:
- Think strategically about supply chain flow and bottlenecks
- Consider transportation costs and batch economics
- Smooth demand signals while avoiding costly overreaction
- Plan for delivery constraints and lead time variability

Your success = Low total costs + Network reliability""",

            Position.MANUFACTURER: """You are the MANUFACTURER in a consumer goods supply chain. Your role is to plan production efficiently while minimizing total manufacturing costs.

BUSINESS OBJECTIVE: Minimize total costs while meeting downstream demand through optimal production planning.

COST STRUCTURE:
- Inventory holding cost: $1 per unit per period (storage, obsolescence, capital)
- Stockout penalty: $2 per unit of unmet distributor orders (lost sales, disruption)
- Production volatility increases manufacturing costs and inefficiency

BUSINESS PRIORITIES:
1. Optimize production schedules for cost efficiency
2. Balance production smoothing with demand responsiveness
3. Minimize inventory while ensuring adequate supply
4. Plan production runs for economies of scale

DECISION FRAMEWORK:
- Plan production considering lead times and capacity constraints
- Smooth production to reduce manufacturing costs
- Avoid both costly stockouts AND expensive overproduction
- Consider: Is demand volatility real or just noise in the system?

Your success = Low total costs + Efficient production + Adequate supply"""
        }

        base_prompt = f"""
{position_prompts[self.position]}

SYSTEM DYNAMICS:
- Each period, decide how many units to order from your upstream supplier
- Information delays exist in the system - you see orders with delay
- Shipping takes time - orders don't arrive immediately
- Goal: Minimize YOUR total costs while maintaining supply chain performance

RESPONSE FORMAT:
You must respond with a JSON object containing only your order decision:
{{"order": [integer_value]}}

The order must be a non-negative integer. Provide no other text outside the JSON."""

        return base_prompt

    def _create_neutral_cost_focused_prompt(self) -> str:
        """Create neutral system prompt focused on cost optimization"""
        return f"""You are the {self.position.value.upper()} in a consumer goods supply chain coordination system.

OBJECTIVE: Minimize your total costs while maintaining adequate supply chain performance.

COST STRUCTURE:
- Inventory holding cost: $1 per unit per period
- Stockout/backlog penalty: $2 per unit of unmet orders  
- Goal: Find the optimal balance between these competing costs

SYSTEM DYNAMICS:
- Information and shipping delays exist in the supply chain
- Your decisions affect both your costs and overall system performance
- Excessive inventory wastes capital; insufficient inventory creates shortages

DECISION PROCESS:
1. Analyze your current inventory and backlog situation
2. Consider incoming orders from your downstream customer
3. Estimate appropriate order quantity balancing costs and service
4. Avoid both expensive stockouts and costly excess inventory

RESPONSE FORMAT:
You must respond with a JSON object containing only your order decision:
{{"order": [integer_value]}}

The order must be a non-negative integer. Provide no other text outside the JSON."""

    def make_decision(self, game_state: Dict[str, Any]) -> int:
        """
        Make ordering decision using Ollama LLM.
        
        Args:
            game_state: Current observable game state
            
        Returns:
            Order quantity (non-negative integer)
        """
        decision, interaction_data = self._make_decision_with_tracking(game_state)
        # Store interaction data for potential retrieval
        self._last_interaction = interaction_data
        return decision
    
    def _make_decision_with_tracking(self, game_state: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """
        Make ordering decision and return decision + interaction data.
        
        Args:
            game_state: Current observable game state
            
        Returns:
            Tuple of (order_quantity, interaction_data)
        """
        start_time = time.time()
        
        # Create prompt from game state
        user_prompt = self._create_user_prompt(game_state)
        
        interaction_data = {
            'prompt': user_prompt,
            'response': '',
            'decision': 0,
            'response_time_ms': 0.0,
            'success': False
        }
        
        # Get LLM response with retries
        for attempt in range(self.max_retries):
            try:
                response = self._call_ollama(user_prompt)
                interaction_data['response'] = response
                
                order = self._parse_response(response)
                
                # Validate order
                if isinstance(order, int) and order >= 0:
                    interaction_data['decision'] = order
                    interaction_data['success'] = True
                    interaction_data['response_time_ms'] = (time.time() - start_time) * 1000
                    return order, interaction_data
                else:
                    raise ValueError(f"Invalid order value: {order}")
                    
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {self.name}: {e}")
                if attempt == self.max_retries - 1:
                    # Fallback to simple heuristic
                    print(f"Using fallback decision for {self.name}")
                    fallback_order = self._fallback_decision(game_state)
                    interaction_data['decision'] = fallback_order
                    interaction_data['response'] = f"FALLBACK: {e}"
                    interaction_data['response_time_ms'] = (time.time() - start_time) * 1000
                    return fallback_order, interaction_data
                time.sleep(1)  # Brief delay before retry
        
        # Should not reach here, but safety fallback
        fallback_order = self._fallback_decision(game_state)
        interaction_data['decision'] = fallback_order
        interaction_data['response'] = "FALLBACK: Maximum retries exceeded"
        interaction_data['response_time_ms'] = (time.time() - start_time) * 1000
        return fallback_order, interaction_data
    
    def get_last_interaction(self) -> Dict[str, Any]:
        """Get interaction data from last decision"""
        return getattr(self, '_last_interaction', {})
    
    def _create_user_prompt(self, game_state: Dict[str, Any]) -> str:
        """Create user prompt from game state with configurable memory"""
        prompt_parts = [
            f"PERIOD {game_state['round']}",
            f"Position: {game_state['position'].upper()}",
            f"Current inventory: {game_state['inventory']} units",
            f"Current backlog: {game_state['backlog']} units", 
            f"Incoming order: {game_state['incoming_order']} units",
            f"Last order placed: {game_state['last_outgoing_order']} units",
            f"Period cost: ${game_state['round_cost']:.2f}",
        ]
        
        # Add customer demand for retailer
        if self.position == Position.RETAILER and "customer_demand" in game_state:
            prompt_parts.append(f"Customer demand: {game_state['customer_demand']} units")
        
        # Add decision history based on memory window setting
        if game_state.get("decision_history") and self.memory_window != 0:
            full_history = game_state["decision_history"]
            
            if self.memory_window is None:
                # Full memory - all decisions
                history = full_history
                memory_type = "complete order history"
            elif self.memory_window > 0:
                # Limited memory window
                history = full_history[-self.memory_window:]
                memory_type = f"recent {len(history)} order(s)"
            
            if history:
                prompt_parts.append(f"Your {memory_type}: {history}")
        
        # Add memory context to decision prompt
        if self.memory_window == 0:
            prompt_parts.append("\nMake your order decision based on current period information only.")
        elif self.memory_window is None:
            prompt_parts.append("\nConsider your complete order history when making this decision.")
        else:
            prompt_parts.append(f"\nConsider your recent {self.memory_window} order(s) when making this decision.")
        
        # Add visibility information to prompt if available
        if "visible_supply_chain" in game_state:
            prompt_parts.append("\nSUPPLY CHAIN VISIBILITY:")
            for pos, info in game_state["visible_supply_chain"].items():
                prompt_parts.append(f"{pos.title()}: Inventory={info['inventory']}, Backlog={info['backlog']}, Cost=${info['cost']:.2f}")
        
        if "visible_history" in game_state:
            prompt_parts.append("\nPARTNER HISTORY:")
            for pos, history in game_state["visible_history"].items():
                recent_history = history[-3:] if len(history) > 3 else history  # Show last 3 rounds
                prompt_parts.append(f"{pos.title()} recent: {[f'R{h['round']}:Inv{h['inventory']},Order{h['outgoing_order']}' for h in recent_history]}")
        
        if "system_metrics" in game_state:
            metrics = game_state["system_metrics"]
            prompt_parts.append(f"\nSYSTEM METRICS:")
            prompt_parts.append(f"Total system cost: ${metrics['total_system_cost']:.2f}")
            prompt_parts.append(f"Total system inventory: {metrics['total_system_inventory']} units")
            prompt_parts.append(f"Total system backlog: {metrics['total_system_backlog']} units")
        
        prompt_parts.append("How many units should you order this period to minimize your total costs?")
        
        return "\n".join(prompt_parts)
    
    def _call_ollama(self, user_prompt: str) -> str:
        """Make API call to Ollama"""
        # Prepare request payload
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": 0.9,
                "num_predict": 100,  # Limit response length
            }
        }
        
        # Make request
        response = self.session.post(
            self.chat_url,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        if "message" in result and "content" in result["message"]:
            return result["message"]["content"]
        else:
            raise ValueError(f"Unexpected Ollama response format: {result}")
    
    def _parse_response(self, response: str) -> int:
        """Parse LLM response to extract order decision"""
        # Try to find JSON in response
        response = response.strip()
        
        # Look for JSON object
        try:
            # Try parsing entire response as JSON
            data = json.loads(response)
            if isinstance(data, dict) and "order" in data:
                return int(data["order"])
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from text
        import re
        json_match = re.search(r'\{[^}]*"order"[^}]*\}', response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                if "order" in data:
                    return int(data["order"])
            except (json.JSONDecodeError, ValueError):
                pass
        
        # Try to extract number from text
        number_match = re.search(r'\b(\d+)\b', response)
        if number_match:
            return int(number_match.group(1))
        
        raise ValueError(f"Could not parse order from response: {response}")
    
    def _fallback_decision(self, game_state: Dict[str, Any]) -> int:
        """Simple fallback decision when LLM fails"""
        # Basic heuristic: order = recent demand + inventory gap
        incoming_order = game_state["incoming_order"]
        inventory = game_state["inventory"]
        backlog = game_state["backlog"]
        
        target_inventory = 12  # Basic target
        inventory_gap = max(0, target_inventory - inventory)
        safety_buffer = backlog  # Clear backlog
        
        order = max(0, incoming_order + inventory_gap + safety_buffer)
        return min(order, 50)  # Cap at reasonable maximum
    
    def check_connection(self) -> bool:
        """Check if Ollama server is accessible"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False
    
    def list_available_models(self) -> List[str]:
        """Get list of available models from Ollama"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10.0)
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            print(f"Failed to list models: {e}")
            return []
    
    def __del__(self):
        """Clean up session on deletion"""
        if hasattr(self, 'session'):
            self.session.close()


def create_ollama_agents(
    model_name: str = "llama3.2",
    base_url: str = "http://localhost:11434",
    neutral_prompt: bool = False,
    memory_window: Optional[int] = 5,
    **kwargs
) -> Dict[Position, OllamaAgent]:
    """
    Create a full set of Ollama agents for all supply chain positions.
    
    Args:
        model_name: Ollama model to use
        base_url: Ollama server URL
        neutral_prompt: Use neutral prompts instead of position-specific ones
        memory_window: Number of past decisions to include (None = all, 0 = none, 5 = default)
        **kwargs: Additional arguments passed to OllamaAgent
        
    Returns:
        Dictionary mapping positions to agent instances
    """
    return {
        position: OllamaAgent(
            position, 
            model_name, 
            base_url, 
            neutral_prompt=neutral_prompt,
            memory_window=memory_window,
            **kwargs
        )
        for position in Position
    }


def test_ollama_connection(base_url: str = "http://localhost:11434") -> bool:
    """
    Test connection to Ollama server.
    
    Args:
        base_url: Ollama server URL
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False