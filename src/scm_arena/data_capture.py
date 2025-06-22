"""
Comprehensive data capture system for SCM-Arena experiments with canonical settings support.

MAJOR UPDATE: Added support for tracking canonical LLM settings (temperature, top_p, etc.)
for complete reproducibility and benchmark compliance verification.

UPDATED: Now uses deterministic seeding instead of hardcoded seed.
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
    """Metadata for a single experimental run with canonical settings"""
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
    # Canonical LLM settings
    temperature: float
    top_p: float
    top_k: int
    repeat_penalty: float
    seed: int
    base_seed: int  # NEW: Track base seed for reproducibility
    deterministic_seeding: bool  # NEW: Track seeding method


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
    """SQLite database for storing experimental data with deterministic seeding"""
    
    def __init__(self, db_path: str = "scm_arena_experiments.db"):
        """Initialize database connection and create tables if needed"""
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        
        # Experiments table - one row per experimental run (UPDATED with deterministic seeding fields)
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
                bullwhip_ratio REAL NOT NULL,
                temperature REAL NOT NULL,
                top_p REAL NOT NULL,
                top_k INTEGER NOT NULL,
                repeat_penalty REAL NOT NULL,
                seed INTEGER NOT NULL,
                base_seed INTEGER NOT NULL,
                deterministic_seeding BOOLEAN NOT NULL
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
                rounds, run_number, timestamp, total_cost, service_level, bullwhip_ratio,
                temperature, top_p, top_k, repeat_penalty, seed, base_seed, deterministic_seeding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metadata.experiment_id, metadata.model_name, metadata.memory_strategy,
            metadata.memory_window, metadata.prompt_type, metadata.visibility_level,
            metadata.scenario, metadata.game_mode, metadata.rounds, metadata.run_number,
            metadata.timestamp, metadata.total_cost, metadata.service_level, metadata.bullwhip_ratio,
            metadata.temperature, metadata.top_p, metadata.top_k, metadata.repeat_penalty, 
            metadata.seed, metadata.base_seed, metadata.deterministic_seeding
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
    """High-level interface for tracking experiments with deterministic seeding"""
    
    def __init__(self, db_path: str = "scm_arena_experiments.db"):
        self.db = ExperimentDatabase(db_path)
        self.current_experiment_id = None
    
    def start_experiment(self, model_name: str, memory_strategy: str, memory_window: Optional[int],
                        prompt_type: str, visibility_level: str, scenario: str,
                        game_mode: str, rounds: int, run_number: int,
                        temperature: float = 0.3, top_p: float = 0.9, 
                        top_k: int = 40, repeat_penalty: float = 1.1, 
                        seed: int = 42, base_seed: int = 42, 
                        deterministic_seeding: bool = True) -> str:
        """Start tracking a new experiment with deterministic seeding support"""
        
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
            bullwhip_ratio=0.0,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            seed=seed,
            base_seed=base_seed,
            deterministic_seeding=deterministic_seeding
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
    
    def get_canonical_settings_summary(self) -> Dict[str, Any]:
        """Get summary of canonical settings usage in database (including seeding info)"""
        try:
            cursor = self.db.conn.execute("""
                SELECT 
                    temperature, top_p, top_k, repeat_penalty, 
                    base_seed, deterministic_seeding,
                    COUNT(*) as experiment_count
                FROM experiments 
                GROUP BY temperature, top_p, top_k, repeat_penalty, base_seed, deterministic_seeding
                ORDER BY experiment_count DESC
            """)
            
            results = cursor.fetchall()
            settings_summary = []
            
            for row in results:
                settings_summary.append({
                    'temperature': row[0],
                    'top_p': row[1], 
                    'top_k': row[2],
                    'repeat_penalty': row[3],
                    'base_seed': row[4],
                    'deterministic_seeding': bool(row[5]),
                    'experiment_count': row[6]
                })
            
            return {
                'canonical_settings_usage': settings_summary,
                'total_experiments': sum(s['experiment_count'] for s in settings_summary)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_reproducibility_report(self) -> Dict[str, Any]:
        """Get detailed reproducibility report including seeding methodology"""
        try:
            # Check seeding method usage
            cursor = self.db.conn.execute("""
                SELECT deterministic_seeding, base_seed, COUNT(*) as count
                FROM experiments 
                GROUP BY deterministic_seeding, base_seed
                ORDER BY count DESC
            """)
            
            seeding_usage = cursor.fetchall()
            
            # Check for identical experimental conditions with deterministic seeding
            cursor = self.db.conn.execute("""
                SELECT 
                    model_name, memory_strategy, prompt_type, visibility_level, 
                    scenario, game_mode, base_seed, deterministic_seeding,
                    COUNT(*) as replications,
                    COUNT(DISTINCT seed) as unique_seeds
                FROM experiments 
                WHERE deterministic_seeding = 1
                GROUP BY model_name, memory_strategy, prompt_type, visibility_level, 
                         scenario, game_mode, base_seed, deterministic_seeding
                HAVING COUNT(*) > 1
                ORDER BY replications DESC
            """)
            
            replicated_conditions = cursor.fetchall()
            
            # Check seed uniqueness within deterministic experiments
            cursor = self.db.conn.execute("""
                SELECT COUNT(*) as total_seeds, COUNT(DISTINCT seed) as unique_seeds
                FROM experiments 
                WHERE deterministic_seeding = 1
            """)
            
            seed_stats = cursor.fetchone()
            
            return {
                'seeding_usage': [
                    {
                        'deterministic': bool(row[0]), 
                        'base_seed': row[1], 
                        'experiments': row[2]
                    } for row in seeding_usage
                ],
                'replicated_conditions': len(replicated_conditions),
                'seed_uniqueness': {
                    'total_seeds': seed_stats[0] if seed_stats else 0,
                    'unique_seeds': seed_stats[1] if seed_stats else 0,
                    'collision_rate': (1 - (seed_stats[1] / seed_stats[0])) if seed_stats and seed_stats[0] > 0 else 0
                },
                'reproducible': len(replicated_conditions) > 0,
                'deterministic_experiments': sum(row[2] for row in seeding_usage if row[0])
            }
            
        except Exception as e:
            return {'error': str(e)}