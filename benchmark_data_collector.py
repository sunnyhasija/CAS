#!/usr/bin/env python3
"""
SCM-Arena Benchmark Data Collector
==================================

Comprehensive benchmark data collection for full factorial experimental design.
Optimized for high-performance hardware (16 cores, 128GB RAM).

EXPERIMENTAL DESIGN:
- Models × 3 memory × 2 prompts × 3 visibility × 4 scenarios × 2 game modes × 20 runs
- Auto-generates model-specific file names for organization
- ~150,000+ total simulation rounds per model

FEATURES:
- Parallel execution with optimal batching
- Model-specific file naming for organization
- Automatic progress tracking and resumption
- Resource monitoring and throttling
- Comprehensive error handling and recovery
- Real-time performance metrics
- Database integrity verification
"""

import subprocess
import threading
import time
import json
import sys
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import psutil
import signal
import platform

# Configuration constants
MODELS = ["phi4:latest"]  # Add more models here: ["phi4:latest", "llama3.2", "qwen2.5"]
MEMORY_STRATEGIES = ["none", "short", "full"]
PROMPT_TYPES = ["specific", "neutral"]
VISIBILITY_LEVELS = ["local", "adjacent", "full"]
SCENARIOS = ["classic", "random", "shock", "seasonal"]
GAME_MODES = ["modern", "classic"]
RUNS_PER_CONDITION = 20
ROUNDS_PER_GAME = 52

# Hardware optimization
MAX_PARALLEL_JOBS = 8  # Conservative for LLM workload
BASE_SEED = 42

def get_model_safe_name(model_name: str) -> str:
    """Convert model name to filesystem-safe string"""
    return model_name.replace(":", "_").replace("/", "_")

def get_models_string(models: List[str]) -> str:
    """Get string representation for multiple models"""
    if len(models) == 1:
        return get_model_safe_name(models[0])
    else:
        return f"multi_model_{len(models)}_models"

# Dynamic file naming based on models
MODELS_STRING = get_models_string(MODELS)
OUTPUT_DB = f"scm_arena_benchmark_{MODELS_STRING}.db"
PROGRESS_FILE = f"benchmark_progress_{MODELS_STRING}.json"

@dataclass
class ExperimentCondition:
    """Single experimental condition"""
    model: str
    memory: str
    prompt_type: str
    visibility: str
    scenario: str
    game_mode: str
    run_number: int
    
    def to_identifier(self) -> str:
        """Create unique identifier for this condition"""
        return f"{self.model}-{self.memory}-{self.prompt_type}-{self.visibility}-{self.scenario}-{self.game_mode}-{self.run_number}"
    
    def to_cli_args(self) -> List[str]:
        """Convert to CLI arguments"""
        return [
            "poetry", "run", "python", "-m", "scm_arena.cli", "experiment",
            "--models", self.model,
            "--memory", self.memory,
            "--prompts", self.prompt_type,
            "--visibility", self.visibility,
            "--scenarios", self.scenario,
            "--game-modes", self.game_mode,
            "--runs", "1",  # Single run per job for better parallelization
            "--rounds", str(ROUNDS_PER_GAME),
            "--base-seed", str(BASE_SEED),
            "--deterministic",
            "--save-database",
            "--db-path", OUTPUT_DB
        ]

class ProgressTracker:
    """Track and persist experimental progress"""
    
    def __init__(self, progress_file: str = PROGRESS_FILE):
        self.progress_file = progress_file
        self.completed_experiments = set()
        self.failed_experiments = set()
        self.start_time = time.time()
        self.experiment_times = []
        self.lock = threading.Lock()
        self.load_progress()
    
    def load_progress(self):
        """Load existing progress if available"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    self.completed_experiments = set(data.get('completed', []))
                    self.failed_experiments = set(data.get('failed', []))
                    self.experiment_times = data.get('times', [])
                print(f"📈 Loaded progress: {len(self.completed_experiments)} completed, {len(self.failed_experiments)} failed")
            except Exception as e:
                print(f"⚠️ Could not load progress: {e}")
    
    def save_progress(self):
        """Save current progress"""
        try:
            data = {
                'completed': list(self.completed_experiments),
                'failed': list(self.failed_experiments),
                'times': self.experiment_times,
                'last_updated': time.time()
            }
            with open(self.progress_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save progress: {e}")
    
    def mark_completed(self, experiment_id: str, duration: float):
        """Mark experiment as completed"""
        with self.lock:
            self.completed_experiments.add(experiment_id)
            self.failed_experiments.discard(experiment_id)  # Remove from failed if present
            self.experiment_times.append(duration)
            self.save_progress()
    
    def mark_failed(self, experiment_id: str):
        """Mark experiment as failed"""
        with self.lock:
            self.failed_experiments.add(experiment_id)
            self.save_progress()
    
    def is_completed(self, experiment_id: str) -> bool:
        """Check if experiment is already completed"""
        return experiment_id in self.completed_experiments
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current progress statistics"""
        total_time = time.time() - self.start_time
        completed_count = len(self.completed_experiments)
        failed_count = len(self.failed_experiments)
        total_target = len(MODELS) * 144 * RUNS_PER_CONDITION
        
        if self.experiment_times:
            avg_time = sum(self.experiment_times) / len(self.experiment_times)
            est_total_remaining = (total_target - completed_count) * avg_time
        else:
            avg_time = 0
            est_total_remaining = 0
        
        return {
            'models': MODELS,
            'models_string': MODELS_STRING,
            'completed': completed_count,
            'failed': failed_count,
            'total_experiments': total_target,
            'completion_rate': completed_count / total_target,
            'total_runtime': total_time,
            'avg_experiment_time': avg_time,
            'estimated_remaining': est_total_remaining
        }

class ResourceMonitor:
    """Monitor system resources during execution"""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.max_cpu_usage = 0
        self.max_memory_usage = 0
        self.ollama_process = None
    
    def start_monitoring(self):
        """Start resource monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Find Ollama process
        for proc in psutil.process_iter(['pid', 'name']):
            if 'ollama' in proc.info['name'].lower():
                self.ollama_process = psutil.Process(proc.info['pid'])
                break
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Monitor system resources"""
        while self.monitoring:
            try:
                # System metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                self.max_cpu_usage = max(self.max_cpu_usage, cpu_percent)
                self.max_memory_usage = max(self.max_memory_usage, memory.percent)
                
                # Throttle if system is overloaded
                if cpu_percent > 90 or memory.percent > 90:
                    print(f"⚠️ High resource usage: CPU {cpu_percent:.1f}%, Memory {memory.percent:.1f}%")
                    time.sleep(5)  # Brief pause
                
                time.sleep(10)  # Monitor every 10 seconds
            except Exception as e:
                print(f"⚠️ Monitoring error: {e}")
                break
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics"""
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        ollama_stats = {}
        if self.ollama_process:
            try:
                ollama_stats = {
                    'cpu_percent': self.ollama_process.cpu_percent(),
                    'memory_mb': self.ollama_process.memory_info().rss / (1024 * 1024)
                }
            except:
                ollama_stats = {'status': 'not_accessible'}
        
        return {
            'current_cpu': cpu,
            'current_memory': memory.percent,
            'max_cpu': self.max_cpu_usage,
            'max_memory': self.max_memory_usage,
            'available_memory_gb': memory.available / (1024**3),
            'ollama': ollama_stats
        }

class BenchmarkDataCollector:
    """Main benchmark data collector with parallel execution and model-specific organization"""
    
    def __init__(self):
        self.progress = ProgressTracker()
        self.monitor = ResourceMonitor()
        self.shutdown_requested = False
        self.total_experiments = len(MODELS) * 144 * RUNS_PER_CONDITION
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n🛑 Shutdown signal received. Completing current experiments...")
        self.shutdown_requested = True
    
    def test_model_connection(self, model: str) -> bool:
        """Test model connection with improved subprocess handling"""
        try:
            print(f"🔍 Testing {model} connection...")
            
            # Use consistent subprocess approach with UTF-8 encoding
            cmd = ["poetry", "run", "python", "-m", "scm_arena.cli", "test-model", "--model", model]
            cmd_str = " ".join(cmd)
            
            # Set environment variables for UTF-8 support on Windows
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONUTF8'] = '1'
            
            result = subprocess.run(
                cmd_str,
                capture_output=True,
                text=True,
                timeout=60,  # Increased timeout
                cwd=os.getcwd(),
                shell=True,  # Use shell for consistent behavior
                env=env,     # Set UTF-8 encoding
                encoding='utf-8',  # Explicit encoding
                errors='replace'   # Handle encoding errors gracefully
            )
            
            if result.returncode == 0:
                print(f"✅ {model} connection verified")
                return True
            else:
                print(f"❌ {model} test failed!")
                print(f"   Return code: {result.returncode}")
                if result.stdout:
                    print(f"   STDOUT: {result.stdout}")
                if result.stderr:
                    print(f"   STDERR: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"❌ {model} test timed out after 60 seconds")
            return False
        except Exception as e:
            print(f"❌ Failed to test {model}: {e}")
            return False
    
    def generate_all_conditions(self) -> List[ExperimentCondition]:
        """Generate all experimental conditions"""
        conditions = []
        
        for model in MODELS:
            for memory in MEMORY_STRATEGIES:
                for prompt_type in PROMPT_TYPES:
                    for visibility in VISIBILITY_LEVELS:
                        for scenario in SCENARIOS:
                            for game_mode in GAME_MODES:
                                for run in range(1, RUNS_PER_CONDITION + 1):
                                    condition = ExperimentCondition(
                                        model=model,
                                        memory=memory,
                                        prompt_type=prompt_type,
                                        visibility=visibility,
                                        scenario=scenario,
                                        game_mode=game_mode,
                                        run_number=run
                                    )
                                    conditions.append(condition)
        
        return conditions
    
    def filter_pending_conditions(self, all_conditions: List[ExperimentCondition]) -> List[ExperimentCondition]:
        """Filter out already completed conditions"""
        pending = []
        for condition in all_conditions:
            if not self.progress.is_completed(condition.to_identifier()):
                pending.append(condition)
        
        print(f"📋 Total conditions: {len(all_conditions)}")
        print(f"✅ Completed: {len(all_conditions) - len(pending)}")
        print(f"⏳ Pending: {len(pending)}")
        
        return pending
    
    def run_single_experiment(self, condition: ExperimentCondition) -> Tuple[bool, str]:
        """Run a single experimental condition"""
        experiment_id = condition.to_identifier()
        start_time = time.time()
        
        try:
            # Build CLI command
            cmd = condition.to_cli_args()
            
            # Run experiment with consistent subprocess handling and UTF-8 support
            cmd_str = " ".join(cmd)
            
            # Set environment variables for UTF-8 support on Windows
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONUTF8'] = '1'
            
            result = subprocess.run(
                cmd_str,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=os.getcwd(),
                shell=True,
                env=env,
                encoding='utf-8',
                errors='replace'
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                self.progress.mark_completed(experiment_id, duration)
                return True, f"Completed in {duration:.1f}s"
            else:
                error_msg = f"Exit code {result.returncode}: {result.stderr[:200]}"
                self.progress.mark_failed(experiment_id)
                return False, error_msg
                
        except subprocess.TimeoutExpired:
            self.progress.mark_failed(experiment_id)
            return False, "Timeout after 5 minutes"
        except Exception as e:
            self.progress.mark_failed(experiment_id)
            return False, str(e)
    
    def run_batch_parallel(self, conditions: List[ExperimentCondition]) -> None:
        """Run experiments in parallel batches"""
        print(f"🚀 Starting parallel execution with {MAX_PARALLEL_JOBS} workers")
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        completed_count = 0
        failed_count = 0
        
        try:
            with ThreadPoolExecutor(max_workers=MAX_PARALLEL_JOBS) as executor:
                # Submit all experiments
                future_to_condition = {
                    executor.submit(self.run_single_experiment, condition): condition
                    for condition in conditions
                }
                
                # Process completed experiments
                for future in as_completed(future_to_condition):
                    if self.shutdown_requested:
                        print("🛑 Shutdown requested, stopping new experiments...")
                        break
                    
                    condition = future_to_condition[future]
                    success, message = future.result()
                    
                    if success:
                        completed_count += 1
                        print(f"✅ {condition.to_identifier()}: {message}")
                    else:
                        failed_count += 1
                        print(f"❌ {condition.to_identifier()}: {message}")
                    
                    # Progress update every 10 experiments
                    if (completed_count + failed_count) % 10 == 0:
                        self._print_progress_update()
        
        finally:
            self.monitor.stop_monitoring()
            
        print(f"\n📊 Batch completed: {completed_count} succeeded, {failed_count} failed")
    
    def _print_progress_update(self):
        """Print current progress and resource stats"""
        stats = self.progress.get_stats()
        resource_stats = self.monitor.get_stats()
        
        print(f"\n📈 PROGRESS UPDATE:")
        print(f"   Completed: {stats['completed']}/2880 ({stats['completion_rate']:.1%})")
        print(f"   Failed: {stats['failed']}")
        print(f"   Avg time per experiment: {stats['avg_experiment_time']:.1f}s")
        print(f"   Estimated remaining: {stats['estimated_remaining']/3600:.1f} hours")
        print(f"   CPU: {resource_stats['current_cpu']:.1f}% (max: {resource_stats['max_cpu']:.1f}%)")
        print(f"   Memory: {resource_stats['current_memory']:.1f}% (max: {resource_stats['max_memory']:.1f}%)")
        print()
    
    def run_full_study(self):
        """Run the complete benchmark data collection"""
        print("🧪 SCM-Arena Benchmark Data Collector")
        print("=" * 60)
        print(f"🤖 Models: {', '.join(MODELS)}")
        print(f"📊 Total experiments: {self.total_experiments:,}")
        print(f"🎯 Conditions per model: 144")
        print(f"🔄 Runs per condition: {RUNS_PER_CONDITION}")
        print(f"🎮 Rounds per game: {ROUNDS_PER_GAME}")
        print(f"🚀 Parallel workers: {MAX_PARALLEL_JOBS}")
        print(f"💾 Database: {OUTPUT_DB}")
        print(f"📁 Progress file: {PROGRESS_FILE}")
        print()
        
        # Check all models with improved testing
        print("🔍 Testing model connections...")
        all_models_ok = True
        for model in MODELS:
            if not self.test_model_connection(model):
                all_models_ok = False
        
        if not all_models_ok:
            print("\n❌ One or more models failed testing!")
            print("Make sure Ollama is running and all models are available")
            return False
        
        # Generate all conditions
        all_conditions = self.generate_all_conditions()
        pending_conditions = self.filter_pending_conditions(all_conditions)
        
        if not pending_conditions:
            print("🎉 All experiments already completed!")
            self._print_final_summary()
            return True
        
        print(f"⏱️ Estimated time: {len(pending_conditions) * 3 / MAX_PARALLEL_JOBS / 3600:.1f} hours")
        
        # Confirm before starting
        response = input(f"\n🚀 Start benchmark data collection for {MODELS_STRING}? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            return False
        
        # Run the study
        start_time = time.time()
        
        try:
            self.run_batch_parallel(pending_conditions)
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
        
        total_time = time.time() - start_time
        print(f"\n⏱️ Total execution time: {total_time/3600:.1f} hours")
        
        self._print_final_summary()
        return True
    
    def _print_final_summary(self):
        """Print final summary and statistics"""
        stats = self.progress.get_stats()
        
        print(f"\n🏁 BENCHMARK DATA COLLECTION SUMMARY")
        print("=" * 50)
        print(f"🤖 Models: {', '.join(stats['models'])}")
        print(f"✅ Completed experiments: {stats['completed']:,}")
        print(f"❌ Failed experiments: {stats['failed']:,}")
        if (stats['completed'] + stats['failed']) > 0:
            print(f"📊 Success rate: {stats['completed']/(stats['completed']+stats['failed']):.1%}")
        print(f"💾 Database: {OUTPUT_DB}")
        print(f"📁 Progress: {PROGRESS_FILE}")
        
        # Check database
        if os.path.exists(OUTPUT_DB):
            try:
                conn = sqlite3.connect(OUTPUT_DB)
                cursor = conn.execute("SELECT COUNT(*) FROM experiments")
                db_count = cursor.fetchone()[0]
                print(f"📈 Database contains: {db_count:,} experiments")
                
                # Show breakdown by model
                cursor = conn.execute("SELECT model_name, COUNT(*) FROM experiments GROUP BY model_name")
                model_counts = cursor.fetchall()
                print("📋 Per model breakdown:")
                for model, count in model_counts:
                    print(f"   {model}: {count:,} experiments")
                
                conn.close()
            except Exception as e:
                print(f"⚠️ Could not check database: {e}")
        
        if stats['completed'] >= self.total_experiments:
            print(f"\n🎉 BENCHMARK DATA COLLECTION COMPLETE!")
            print("Ready for Complex Adaptive Systems analysis!")
            print(f"Use database: {OUTPUT_DB}")
        else:
            print(f"\n📋 {self.total_experiments - stats['completed']:,} experiments remaining")
            print("Run script again to resume from where it left off")

def main():
    """Main entry point"""
    collector = BenchmarkDataCollector()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "status":
            # Show current progress
            stats = collector.progress.get_stats()
            print(f"🤖 Models: {', '.join(stats['models'])}")
            print(f"📊 Progress: {stats['completed']:,}/{stats['total_experiments']:,} ({stats['completion_rate']:.1%})")
            print(f"❌ Failed: {stats['failed']:,}")
            print(f"⏱️ Avg time: {stats['avg_experiment_time']:.1f}s")
            print(f"🕒 Estimated remaining: {stats['estimated_remaining']/3600:.1f} hours")
            print(f"💾 Database: {OUTPUT_DB}")
            print(f"📁 Progress: {PROGRESS_FILE}")
            
        elif command == "reset":
            # Reset progress (use with caution!)
            if input(f"⚠️ Reset all progress for {MODELS_STRING}? [y/N]: ").lower() == 'y':
                if os.path.exists(PROGRESS_FILE):
                    os.remove(PROGRESS_FILE)
                print("🔄 Progress reset")
            
        elif command == "verify":
            # Verify database integrity
            if os.path.exists(OUTPUT_DB):
                conn = sqlite3.connect(OUTPUT_DB)
                cursor = conn.execute("SELECT COUNT(*) FROM experiments")
                count = cursor.fetchone()[0]
                print(f"💾 Database: {OUTPUT_DB}")
                print(f"📈 Contains {count:,} experiments")
                
                # Check for each model/condition
                cursor = conn.execute("""
                    SELECT model_name, memory_strategy, prompt_type, visibility_level, 
                           scenario, game_mode, COUNT(*) as count
                    FROM experiments 
                    GROUP BY model_name, memory_strategy, prompt_type, visibility_level, 
                             scenario, game_mode
                    ORDER BY model_name, count DESC
                """)
                print(f"\nCondition completeness:")
                for row in cursor.fetchall():
                    expected = RUNS_PER_CONDITION
                    actual = row[6]
                    status = "✅" if actual >= expected else f"❌ ({actual}/{expected})"
                    model = row[0]
                    condition = '-'.join(row[1:6])
                    print(f"  {model}: {condition}: {status}")
                
                conn.close()
            else:
                print(f"❌ Database not found: {OUTPUT_DB}")
        
        else:
            print("Usage: python benchmark_data_collector.py [status|reset|verify]")
    
    else:
        # Run the full study
        collector.run_full_study()

if __name__ == "__main__":
    main()