"""
SCM-Arena: Supply Chain Management LLM Benchmark Platform
A standardized benchmark for evaluating Large Language Models on 
supply chain management coordination tasks using the Beer Game simulation.
"""

__version__ = "0.1.0"
__author__ = "SCM-Arena Contributors"

from .beer_game.game import BeerGame, GameResults, GameState
from .beer_game.agents import Agent, Position, SimpleAgent, OptimalAgent, RandomAgent
from .models.ollama_client import OllamaAgent, create_ollama_agents, test_ollama_connection
from .data_capture import ExperimentTracker, ExperimentDatabase

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
    "ExperimentTracker",
    "ExperimentDatabase",
]
