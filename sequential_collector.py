#!/usr/bin/env python3
"""
Sequential SCM-Arena Data Collector
===================================

Runs experiments one at a time - slower but reliable.
No parallel processing complexity, just steady data collection.
"""

import os
import sys
import time
import json
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path

# Configuration
MODELS = ["phi4:latest", "qwen3:8b"]  # Two models for comparison
MEMORY_STRATEGIES = ["none", "short", "full"]
PROMPT_TYPES = ["specific", "neutral"]
VISIBILITY_LEVELS = ["local", "adjacent", "full"]
SCENARIOS = ["classic", "random", "shock", "seasonal"]
GAME_MODES = ["modern", "classic"]
RUNS_PER_CONDITION = 20
ROUNDS_PER_GAME = 52
BASE_SEED = 42

# Database setup
OUTPUT_DB = "scm_arena_sequential_experiments.db"
PROGRESS_FILE = "sequential_progress.json"

class SequentialCollector:
    """Simple sequential experiment collector"""
    
    def __init__(self):
        self.completed_experiments = set()
        self.failed_experiments = set()
        self.load_progress()
        self.total_experiments = self.calculate_total_experiments()
        
    def calculate_total_experiments(self):
        """Calculate total number of experiments"""
        return (len(MODELS) * len(MEMORY_STRATEGIES) * len(PROMPT_TYPES) * 
                len(VISIBILITY_LEVELS) * len(SCENARIOS) * len(GAME_MODES) * 
                RUNS_PER_CONDITION)
    
    def load_progress(self):
        """Load existing progress"""
        if os.path.exists(PROGRESS_FILE):
            try:
                with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.completed_experiments = set(data.get('completed', []))
                    self.failed_experiments = set(data.get('failed', []))
                print(f"📈 Loaded progress: {len(self.completed_experiments)} completed, {len(self.failed_experiments)} failed")
            except Exception as e:
                print(f"⚠️ Could not load progress: {e}")
    
    def save_progress(self, experiment_id, success, duration=None):
        """Save progress to file"""
        if success:
            self.completed_experiments.add(experiment_id)
            self.failed_experiments.discard(experiment_id)
        else:
            self.failed_experiments.add(experiment_id)
            self.completed_experiments.discard(experiment_id)
        
        try:
            data = {
                'completed': list(self.completed_experiments),
                'failed': list(self.failed_experiments),
                'last_updated': time.time(),
                'last_experiment': experiment_id,
                'last_duration': duration
            }
            with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save progress: {e}")
    
    def run_single_experiment(self, model, memory, prompt_type, visibility, scenario, game_mode, run_num):
        """Run a single experiment"""
        
        experiment_id = f"{model}-{memory}-{prompt_type}-{visibility}-{scenario}-{game_mode}-{run_num}"
        
        if experiment_id in self.completed_experiments:
            return True, "Already completed"
        
        print(f"\n🎯 Running: {experiment_id}")
        print(f"   Progress: {len(self.completed_experiments)}/{self.total_experiments} ({len(self.completed_experiments)/self.total_experiments*100:.1f}%)")
        
        # Build command
        cmd = [
            "poetry", "run", "python", "-m", "scm_arena.cli", "experiment",
            "--models", model,
            "--memory", memory,
            "--prompts", prompt_type,
            "--visibility", visibility,
            "--scenarios", scenario,
            "--game-modes", game_mode,
            "--runs", "1",
            "--rounds", str(ROUNDS_PER_GAME),
            "--base-seed", str(BASE_SEED),
            "--deterministic",
            "--save-database",
            "--db-path", OUTPUT_DB
        ]
        
        start_time = time.time()
        
        try:
            # Set environment for Ollama connection
            env = os.environ.copy()
            if 'OLLAMA_HOST' not in env:
                # Try to detect if we're in WSL2 or Windows
                try:
                    with open('/proc/version', 'r') as f:
                        if 'microsoft' in f.read().lower():
                            # WSL2 - connect to Windows Ollama
                            env['OLLAMA_HOST'] = 'http://172.20.128.1:11434'
                except:
                    # Probably native Linux or Windows
                    pass
            
            # Run experiment
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                env=env,
                cwd=os.getcwd()
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"   ✅ SUCCESS ({duration:.1f}s)")
                self.save_progress(experiment_id, True, duration)
                return True, f"Completed in {duration:.1f}s"
            else:
                print(f"   ❌ FAILED ({duration:.1f}s)")
                if result.stderr:
                    print(f"   Error: {result.stderr[:200]}")
                self.save_progress(experiment_id, False, duration)
                return False, f"Failed: {result.stderr[:100]}"
                
        except subprocess.TimeoutExpired:
            print(f"   ⏰ TIMEOUT")
            self.save_progress(experiment_id, False)
            return False, "Timeout after 5 minutes"
        except Exception as e:
            print(f"   💥 EXCEPTION: {e}")
            self.save_progress(experiment_id, False)
            return False, str(e)
    
    def run_all_experiments(self):
        """Run all experiments sequentially"""
        
        print("🎯 Sequential SCM-Arena Data Collector")
        print("=" * 50)
        print(f"Models: {', '.join(MODELS)}")
        print(f"Total experiments: {self.total_experiments:,}")
        print(f"Completed: {len(self.completed_experiments):,}")
        print(f"Failed: {len(self.failed_experiments):,}")
        print(f"Remaining: {self.total_experiments - len(self.completed_experiments) - len(self.failed_experiments):,}")
        print(f"Database: {OUTPUT_DB}")
        print()
        
        # Check database
        if os.path.exists(OUTPUT_DB):
            try:
                conn = sqlite3.connect(OUTPUT_DB)
                cursor = conn.execute("SELECT COUNT(*) FROM experiments")
                db_count = cursor.fetchone()[0]
                print(f"📊 Database contains: {db_count:,} experiments")
                conn.close()
            except Exception as e:
                print(f"⚠️ Database check failed: {e}")
        
        # Calculate estimates
        remaining = self.total_experiments - len(self.completed_experiments)
        if remaining == 0:
            print("🎉 All experiments completed!")
            return
        
        estimated_hours = remaining * 2 / 60  # Assume ~2 minutes per experiment
        print(f"⏱️ Estimated time: {estimated_hours:.1f} hours")
        print()
        
        # Confirm
        response = input(f"🚀 Start sequential collection of {remaining} experiments? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
        
        # Run experiments
        start_time = time.time()
        completed_this_session = 0
        failed_this_session = 0
        
        try:
            for model in MODELS:
                print(f"\n🤖 Starting model: {model}")
                print("-" * 30)
                
                for memory in MEMORY_STRATEGIES:
                    for prompt_type in PROMPT_TYPES:
                        for visibility in VISIBILITY_LEVELS:
                            for scenario in SCENARIOS:
                                for game_mode in GAME_MODES:
                                    for run_num in range(1, RUNS_PER_CONDITION + 1):
                                        
                                        success, message = self.run_single_experiment(
                                            model, memory, prompt_type, visibility,
                                            scenario, game_mode, run_num
                                        )
                                        
                                        if success:
                                            completed_this_session += 1
                                        else:
                                            failed_this_session += 1
                                        
                                        # Progress update every 10 experiments
                                        if (completed_this_session + failed_this_session) % 10 == 0:
                                            self.print_progress_update(start_time, completed_this_session, failed_this_session)
        
        except KeyboardInterrupt:
            print("\n\n🛑 Interrupted by user")
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n🏁 SESSION COMPLETE")
        print("=" * 30)
        print(f"⏱️ Session time: {total_time/3600:.1f} hours")
        print(f"✅ Completed this session: {completed_this_session}")
        print(f"❌ Failed this session: {failed_this_session}")
        print(f"📊 Total completed: {len(self.completed_experiments):,}/{self.total_experiments:,}")
        
        # Check final database
        if os.path.exists(OUTPUT_DB):
            try:
                conn = sqlite3.connect(OUTPUT_DB)
                cursor = conn.execute("SELECT COUNT(*) FROM experiments")
                db_count = cursor.fetchone()[0]
                print(f"💾 Database contains: {db_count:,} experiments")
                conn.close()
            except:
                pass
    
    def print_progress_update(self, start_time, completed_session, failed_session):
        """Print progress update"""
        elapsed_hours = (time.time() - start_time) / 3600
        total_completed = len(self.completed_experiments)
        total_failed = len(self.failed_experiments)
        
        if elapsed_hours > 0:
            rate_per_hour = (completed_session + failed_session) / elapsed_hours
            remaining = self.total_experiments - total_completed - total_failed
            eta_hours = remaining / rate_per_hour if rate_per_hour > 0 else 0
            
            print(f"\n📊 PROGRESS UPDATE:")
            print(f"   Session: {completed_session} completed, {failed_session} failed")
            print(f"   Total: {total_completed:,}/{self.total_experiments:,} ({total_completed/self.total_experiments*100:.1f}%)")
            print(f"   Rate: {rate_per_hour:.1f} experiments/hour")
            print(f"   ETA: {eta_hours:.1f} hours")
            print(f"   Success rate: {total_completed/(total_completed+total_failed)*100:.1f}%" if (total_completed+total_failed) > 0 else "   Success rate: N/A")

def main():
    """Main function"""
    
    collector = SequentialCollector()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "status":
            # Show current status
            total = collector.total_experiments
            completed = len(collector.completed_experiments)
            failed = len(collector.failed_experiments)
            
            print(f"📊 Sequential Collector Status")
            print(f"Models: {', '.join(MODELS)}")
            print(f"Progress: {completed:,}/{total:,} ({completed/total*100:.1f}%)")
            print(f"Failed: {failed:,}")
            print(f"Database: {OUTPUT_DB}")
            
            if os.path.exists(OUTPUT_DB):
                try:
                    conn = sqlite3.connect(OUTPUT_DB)
                    cursor = conn.execute("SELECT COUNT(*) FROM experiments")
                    db_count = cursor.fetchone()[0]
                    print(f"DB experiments: {db_count:,}")
                    conn.close()
                except:
                    print("DB experiments: Could not check")
            
        elif command == "test":
            # Test single experiment
            print("🧪 Testing single experiment...")
            success, message = collector.run_single_experiment(
                "phi4:latest", "none", "neutral", "local", "classic", "modern", 999
            )
            print(f"Test result: {'SUCCESS' if success else 'FAILED'} - {message}")
            
        else:
            print("Usage: python sequential_collector.py [status|test]")
    
    else:
        # Run full collection
        collector.run_all_experiments()

if __name__ == "__main__":
    main()