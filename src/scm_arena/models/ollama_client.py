"""
Ollama client integration for Beer Game agents.

This module provides an agent that uses Ollama-hosted LLMs to make
supply chain decisions in the Beer Game.
"""

import json
import requests
import time
from typing import Dict, Any, Optional, List
from ..beer_game.agents import Agent, Position


class OllamaAgent(Agent):
    """
    LLM agent using Ollama API for Beer Game decisions.
    
    This agent sends the current game state to an Ollama-hosted LLM
    and parses the response to extract an order decision.
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
            self.system_prompt = self._create_neutral_system_prompt()
        else:
            self.system_prompt = self._create_default_system_prompt()
        
        # Request session for connection pooling
        self.session = requests.Session()
        
    def _create_default_system_prompt(self) -> str:
        """Create position-specific system prompt based on business realities"""
        
        position_prompts = {
            Position.RETAILER: """You are the RETAILER in a beer supply chain. Your primary objective is to fulfill ALL customer demand to maximize sales revenue and customer satisfaction.

BUSINESS PRIORITIES:
1. NEVER lose a sale - stockouts directly hurt your revenue and customer relationships
2. Minimize holding costs, but service level is more important than inventory costs  
3. You see actual customer demand - use this critical information advantage
4. React quickly to demand changes - customers won't wait

DECISION FRAMEWORK:
- Analyze customer demand patterns carefully
- Order enough to avoid stockouts, even if it means higher inventory
- Consider demand seasonality and trends
- Balance: High service level > Low inventory costs

Your success = Customer satisfaction + Sales maximization""",

            Position.WHOLESALER: """You are the WHOLESALER in a beer supply chain. Your role is to efficiently serve retailers while optimizing your distribution operations.

BUSINESS PRIORITIES:
1. Maintain high fill rates to retailers - they depend on you for their success
2. Optimize warehouse utilization and inventory turns
3. Smooth out retailer demand volatility through intelligent buffering  
4. Build reliable supply relationships upstream and downstream

DECISION FRAMEWORK:
- Anticipate retailer needs based on their ordering patterns
- Maintain safety stock for reliable service but avoid excessive inventory
- Look for trends in retailer orders - are they seasonal or one-time spikes?
- Balance: Reliable service to retailers + Operational efficiency

Your success = Retailer satisfaction + Operational efficiency""",

            Position.DISTRIBUTOR: """You are the DISTRIBUTOR in a beer supply chain. Your role is to coordinate regional supply networks and optimize logistics across your territory.

BUSINESS PRIORITIES:
1. Ensure consistent supply availability across your distribution network
2. Minimize transportation and storage costs through efficient planning
3. Coordinate between multiple wholesalers and manufacturing capacity
4. Plan for regional demand variations and logistics constraints

DECISION FRAMEWORK:
- Think strategically about supply chain flow and bottlenecks
- Consider transportation lead times and batch economics
- Smooth demand signals between wholesalers and manufacturer
- Plan for capacity constraints and delivery schedules

Your success = Network reliability + Logistics efficiency""",

            Position.MANUFACTURER: """You are the MANUFACTURER in a beer supply chain. Your role is to plan production efficiently while meeting downstream demand through optimal capacity management.

BUSINESS PRIORITIES:
1. Optimize production efficiency and capacity utilization
2. Plan production runs for economies of scale and cost minimization
3. Ensure adequate supply to meet distributor demands
4. Balance production smoothing with demand responsiveness

DECISION FRAMEWORK:
- Think in terms of production batches and capacity constraints
- Consider brewing lead times and production scheduling
- Avoid production volatility - smooth operations reduce costs
- Plan for demand patterns but don't overreact to short-term spikes

Your success = Production efficiency + Demand fulfillment"""
        }

        base_prompt = f"""
{position_prompts[self.position]}

GAME RULES:
- Each round, decide how many units to order from your upstream supplier
- Costs: $1 per unit held in inventory, $2 per unit of backlog (unfulfilled orders)
- There are shipping delays in the system
- Goal: Optimize YOUR position's performance while supporting overall supply chain success

RESPONSE FORMAT:
You must respond with a JSON object containing only your order decision:
{{"order": [integer_value]}}

The order must be a non-negative integer. Provide no other text outside the JSON."""

        return base_prompt
    
    def _create_neutral_system_prompt(self) -> str:
        """Create neutral system prompt without position-specific biasing"""
        return f"""You are the {self.position.value.upper()} in a supply chain coordination simulation.

OBJECTIVE: Minimize your costs while maintaining good supply chain performance.

GAME RULES:
- Each round, decide how many units to order from your upstream supplier
- Costs: $1 per unit held in inventory, $2 per unit of backlog (unfulfilled orders)  
- There are shipping delays in the system
- Goal: Balance inventory costs with service level

DECISION PROCESS:
1. Analyze your current inventory and backlog levels
2. Consider incoming orders from your downstream partner
3. Estimate appropriate order quantity for next round
4. Balance avoiding stockouts with minimizing excess inventory

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
        # Create prompt from game state
        user_prompt = self._create_user_prompt(game_state)
        
        # Get LLM response with retries
        for attempt in range(self.max_retries):
            try:
                response = self._call_ollama(user_prompt)
                order = self._parse_response(response)
                
                # Validate order
                if isinstance(order, int) and order >= 0:
                    return order
                else:
                    raise ValueError(f"Invalid order value: {order}")
                    
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {self.name}: {e}")
                if attempt == self.max_retries - 1:
                    # Fallback to simple heuristic
                    print(f"Using fallback decision for {self.name}")
                    return self._fallback_decision(game_state)
                time.sleep(1)  # Brief delay before retry
        
        # Should not reach here, but safety fallback
        return self._fallback_decision(game_state)
    
    def _create_user_prompt(self, game_state: Dict[str, Any]) -> str:
        """Create user prompt from game state with configurable memory"""
        prompt_parts = [
            f"ROUND {game_state['round']}",
            f"Position: {game_state['position'].upper()}",
            f"Current inventory: {game_state['inventory']} units",
            f"Current backlog: {game_state['backlog']} units", 
            f"Incoming order: {game_state['incoming_order']} units",
            f"Last order placed: {game_state['last_outgoing_order']} units",
            f"Round cost: ${game_state['round_cost']:.2f}",
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
            prompt_parts.append("\nMake your order decision based on current round information only.")
        elif self.memory_window is None:
            prompt_parts.append("\nConsider your complete order history when making this decision.")
        else:
            prompt_parts.append(f"\nConsider your recent {self.memory_window} order(s) when making this decision.")
        
        prompt_parts.append("How many units should you order this round?")
        
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