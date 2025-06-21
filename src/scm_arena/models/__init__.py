"""
Model integrations for SCM-Arena.

This module provides interfaces to various LLM providers
and model hosting services for Beer Game agents.
"""

from .ollama_client import OllamaAgent, create_ollama_agents, test_ollama_connection

__all__ = [
    "OllamaAgent",
    "create_ollama_agents", 
    "test_ollama_connection",
]