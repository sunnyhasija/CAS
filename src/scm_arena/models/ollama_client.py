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
        system_prompt: Optional[str] = None,
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
            system_prompt: Custom system prompt (uses default if None)
            name: Optional agent name
        """
        super().__init__(position, name or f"ollama_{model_name}_{position.value}")
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        
        # API endpoints
        self.chat_url = f"{self.base_url}/api/chat"
        self.generate_url = f"{self.base_url}/api/generate"
        
        # Set up system prompt
        self.system_prompt = system_prompt or self._create_default_system_prompt()
        
        # Request session for connection pooling
        self.session = requests.Session()
        
    def _create_default_system_prompt(self) -> str:
        """Create default system prompt for the agent's position"""
        position_context = {
            Position.RETAILER: "You serve customers directly and must balance having enough inventory to meet demand while minimizing holding costs.",
            Position.WHOLESALER: "You supply retailers and must anticipate their ordering patterns while managing your own inventory efficiently.", 
            Position.DISTRIBUTOR: "You coordinate between wholesalers and manufacturers, helping smooth demand variability in the supply chain.",
            Position.MANUFACTURER: "You produce goods for the entire supply chain and must plan production to meet downstream demand."
        }
        
        return f"""You are playing the Beer Game, a supply chain coordination simulation. You are the {self.position.value.upper()} in a 4-tier supply chain (Retailer → Wholesaler → Distributor → Manufacturer).

ROLE: {position_context[self.position]}

GAME RULES:
- Each round, you must decide how many units to order from your upstream supplier
- There is a 2-week delay for both information (orders) and shipments
- Costs: $1 per unit held in inventory, $2 per unit of backlog (unfulfilled orders)
- Goal: Minimize total supply chain cost through effective coordination

DECISION PROCESS:
1. Analyze current inventory, backlog, and incoming orders
2. Consider the delays in the system
3. Estimate future demand based on available information
4. Place an order that balances service level and cost

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
        """Create user prompt from game state"""
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
        
        # Add decision history if available
        if game_state.get("decision_history"):
            history = game_state["decision_history"][-5:]  # Last 5 decisions
            prompt_parts.append(f"Recent order history: {history}")
        
        prompt_parts.append("\nHow many units should you order this round?")
        
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
    **kwargs
) -> Dict[Position, OllamaAgent]:
    """
    Create a full set of Ollama agents for all supply chain positions.
    
    Args:
        model_name: Ollama model to use
        base_url: Ollama server URL
        **kwargs: Additional arguments passed to OllamaAgent
        
    Returns:
        Dictionary mapping positions to agent instances
    """
    return {
        position: OllamaAgent(position, model_name, base_url, **kwargs)
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