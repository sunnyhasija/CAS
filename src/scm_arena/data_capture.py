"""
Comprehensive data capture system for SCM-Arena experiments.

Captures all experimental data including:
- Game states every round
- Agent prompts and responses
- Experimental conditions
- Performance metrics
"""

import sqlite3
import json
import uuid
import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ExperimentMetadata:
    """Metadata for a single experimental run"""
    experiment_id: str
    model_name: str
    memory_strategy: str
    memory_window: Optional[int]
    prompt_type: str  # 'specific' or 'neutral'
    visibility_level: str
    scenario: str
    game_mode: str  # 'modern' or 'classic'
    rounds: int
    run_number: int
    timestamp: str
    total_cost: float
    service_level: float
    bullwhip_ratio: float


@dataclass
class RoundData:
    """Data captured for a single round"""
    experiment_id: str
    round_number: int
    customer_demand: int
    total_system_cost: float
    total_system_inventory: int
    total_system_backlog: int


@dataclass
class AgentRoundData:
    """Data for a single agent in a single round"""
    experiment_id: str
    round_number: int
    position: str
    inventory: int
    backlog: int
    incoming_order: int
    outgoing_order: int
    round_cost: float
    total_cost: float
    prompt_sent: str
    llm_response: str
    decision: int
    decision_time_ms: float


class ExperimentDatabase:
    """SQLite database for storing experimental data"""
    
    def __init__(self, db_path: str = "scm_arena_experiments.db"):
        """Initialize database connection and create tables if needed"""
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        
        # Experiments table - one row per experimental run
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                memory_strategy TEXT NOT NULL,
                memory_window INTEGER,
                prompt_type TEXT NOT NULL,
                visibility_level TEXT NOT NULL,
                scenario TEXT NOT NULL,
                game_mode TEXT NOT NULL,
                rounds INTEGER NOT NULL,
                run_number INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                total_cost REAL NOT NULL,
                service_level REAL NOT NULL,
                bullwhip_ratio REAL NOT NULL
            )
        """)
        
        # Rounds table - one row per round per experiment
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS rounds (
                experiment_id TEXT NOT NULL,
                round_number INTEGER NOT NULL,
                customer_demand INTEGER NOT NULL,
                total_system_cost REAL NOT NULL,
                total_system_inventory INTEGER NOT NULL,
                total_system_backlog INTEGER NOT NULL,
                PRIMARY KEY (experiment_id, round_number),
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
            )
        """)
        
        # Agent rounds table - one row per agent per round per experiment
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_rounds (
                experiment_id TEXT NOT NULL,
                round_number INTEGER NOT NULL,
                position TEXT NOT NULL,
                inventory INTEGER NOT NULL,
                backlog INTEGER NOT NULL,
                incoming_order INTEGER NOT NULL,
                outgoing_order INTEGER NOT NULL,
                round_cost REAL NOT NULL,
                total_cost REAL NOT NULL,
                prompt_sent TEXT NOT NULL,
                llm_response TEXT NOT NULL,
                decision INTEGER NOT NULL,
                decision_time_ms REAL NOT NULL,
                PRIMARY KEY (experiment_id, round_number, position),
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
            )
        """)
        
        # Game states table - full JSON state for each round
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS game_states (
                experiment_id TEXT NOT NULL,
                round_number INTEGER NOT NULL,
                game_state_json TEXT NOT NULL,
                PRIMARY KEY (experiment_id, round_number),
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
            )
        """)
        
        self.conn.commit()
    
    def start_experiment(self, metadata: ExperimentMetadata) -> str:
        """Start a new experiment and return experiment ID"""
        
        self.conn.execute("""
            INSERT INTO experiments (
                experiment_id, model_name, memory_strategy, memory_window,
                prompt_type, visibility_level, scenario, game_mode,
                rounds, run_number, timestamp, total_cost, service_level, bullwhip_ratio
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metadata.experiment_id, metadata.model_name, metadata.memory_strategy,
            metadata.memory_window, metadata.prompt_type, metadata.visibility_level,
            metadata.scenario, metadata.game_mode, metadata.rounds, metadata.run_number,
            metadata.timestamp, metadata.total_cost, metadata.service_level, metadata.bullwhip_ratio
        ))
        
        self.conn.commit()
        return metadata.experiment_id
    
    def save_round_data(self, round_data: RoundData, agent_data: List[AgentRoundData], 
                       game_state_json: str):
        """Save complete round data including game state"""
        
        # Save round summary
        self.conn.execute("""
            INSERT INTO rounds (
                experiment_id, round_number, customer_demand,
                total_system_cost, total_system_inventory, total_system_backlog
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            round_data.experiment_id, round_data.round_number, round_data.customer_demand,
            round_data.total_system_cost, round_data.total_system_inventory, round_data.total_system_backlog
        ))
        
        # Save agent data
        for agent in agent_data:
            self.conn.execute("""
                INSERT INTO agent_rounds (
                    experiment_id, round_number, position, inventory, backlog,
                    incoming_order, outgoing_order, round_cost, total_cost,
                    prompt_sent, llm_response, decision, decision_time_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                agent.experiment_id, agent.round_number, agent.position,
                agent.inventory, agent.backlog, agent.incoming_order, agent.outgoing_order,
                agent.round_cost, agent.total_cost, agent.prompt_sent, agent.llm_response,
                agent.decision, agent.decision_time_ms
            ))
        
        # Save complete game state as JSON
        self.conn.execute("""
            INSERT INTO game_states (experiment_id, round_number, game_state_json)
            VALUES (?, ?, ?)
        """, (round_data.experiment_id, round_data.round_number, game_state_json))
        
        self.conn.commit()
    
    def update_experiment_results(self, experiment_id: str, total_cost: float, 
                                 service_level: float, bullwhip_ratio: float):
        """Update experiment with final results"""
        
        self.conn.execute("""
            UPDATE experiments 
            SET total_cost = ?, service_level = ?, bullwhip_ratio = ?
            WHERE experiment_id = ?
        """, (total_cost, service_level, bullwhip_ratio, experiment_id))
        
        self.conn.commit()
    
    def close(self):
        """Close database connection"""
        self.conn.close()


class ExperimentTracker:
    """High-level interface for tracking experiments"""
    
    def __init__(self, db_path: str = "scm_arena_experiments.db"):
        self.db = ExperimentDatabase(db_path)
        self.current_experiment_id = None
    
    def start_experiment(self, model_name: str, memory_strategy: str, memory_window: Optional[int],
                        prompt_type: str, visibility_level: str, scenario: str,
                        game_mode: str, rounds: int, run_number: int) -> str:
        """Start tracking a new experiment"""
        
        self.current_experiment_id = str(uuid.uuid4())
        
        metadata = ExperimentMetadata(
            experiment_id=self.current_experiment_id,
            model_name=model_name,
            memory_strategy=memory_strategy,
            memory_window=memory_window,
            prompt_type=prompt_type,
            visibility_level=visibility_level,
            scenario=scenario,
            game_mode=game_mode,
            rounds=rounds,
            run_number=run_number,
            timestamp=datetime.datetime.now().isoformat(),
            total_cost=0.0,  # Will be updated later
            service_level=0.0,
            bullwhip_ratio=0.0
        )
        
        return self.db.start_experiment(metadata)
    
    def track_round(self, round_number: int, customer_demand: int, total_system_cost: float,
                   total_system_inventory: int, total_system_backlog: int, 
                   agent_interactions: List[Dict[str, Any]], game_state_json: str):
        """Track a complete round including all agent interactions"""
        
        if not self.current_experiment_id:
            raise ValueError("No active experiment. Call start_experiment first.")
        
        # Prepare round data
        round_data = RoundData(
            experiment_id=self.current_experiment_id,
            round_number=round_number,
            customer_demand=customer_demand,
            total_system_cost=total_system_cost,
            total_system_inventory=total_system_inventory,
            total_system_backlog=total_system_backlog
        )
        
        # Prepare agent data
        agent_data = []
        for interaction in agent_interactions:
            agent_round = AgentRoundData(
                experiment_id=self.current_experiment_id,
                round_number=round_number,
                position=interaction['position'],
                inventory=interaction['inventory'],
                backlog=interaction['backlog'],
                incoming_order=interaction['incoming_order'],
                outgoing_order=interaction['outgoing_order'],
                round_cost=interaction['round_cost'],
                total_cost=interaction['total_cost'],
                prompt_sent=interaction.get('prompt', ''),
                llm_response=interaction.get('response', ''),
                decision=interaction.get('decision', 0),
                decision_time_ms=interaction.get('response_time_ms', 0.0)
            )
            agent_data.append(agent_round)
        
        # Save to database
        self.db.save_round_data(round_data, agent_data, game_state_json)
    
    def finish_experiment(self, total_cost: float, service_level: float, bullwhip_ratio: float):
        """Finish tracking experiment with final results"""
        
        if not self.current_experiment_id:
            raise ValueError("No active experiment to finish.")
        
        self.db.update_experiment_results(self.current_experiment_id, total_cost, service_level, bullwhip_ratio)
        experiment_id = self.current_experiment_id
        self.current_experiment_id = None
        
        return experiment_id
    
    def close(self):
        """Close database connection"""
        self.db.close()