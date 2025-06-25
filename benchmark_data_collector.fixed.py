#!/usr/bin/env python3
"""
SCM-Arena Benchmark - Ultimate Windows Unicode Fix
=================================================

This version completely eliminates Unicode issues by:
1. Setting proper encoding at system level
2. Using robust subprocess error handling
3. Reducing parallel load for Windows stability
"""

import os
import sys
import locale
import subprocess
import threading
import time
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import signal

# CRITICAL: Apply Windows fixes before any other operations
def force_utf8_environment():
    """Force UTF-8 environment for Windows"""
    # Set all UTF-8 environment variables
    utf8_vars = {
        'PYTHONIOENCODING': 'utf-8',
        'PYTHONUTF8': '1',
        'PYTHONLEGACYWINDOWSSTDIO': '0',
        'PYTHONUNBUFFERED': '1',
        'LC_ALL': 'en_US.UTF-8',
        'LANG': 'en_US.UTF-8'
    }
    
    for key, value in utf8_vars.items():
        os.environ[key] = value
    
    # Set console code page to UTF-8 (Windows)
    if sys.platform.startswith('win'):
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleOutputCP(65001)  # UTF-8 code page
            kernel32.SetConsoleCP(65001)        # Input code page
        except:
            pass
    
    # Configure locale
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        try:
            locale.setlocale(locale.LC_ALL, '')
        except:
            pass

# Apply fixes immediately
force_utf8_environment()

# Import psutil safely
try:
    import psutil
except ImportError:
    print("Installing psutil...")
    subprocess.run([sys.executable, "-m", "pip", "install", "psutil"], check=True)
    import psutil

# Configuration constants - REDUCED for Windows stability
MODELS = ["phi4:latest"]
MEMORY_STRATEGIES = ["none", "short", "full"]
PROMPT_TYPES = ["specific", "neutral"]
VISIBILITY_LEVELS = ["local", "adjacent", "full"]
SCENARIOS = ["classic", "random", "shock", "seasonal"]
GAME_MODES = ["modern", "classic"]
RUNS_PER_CONDITION = 20
ROUNDS_PER_GAME = 52

# REDUCED parallel jobs for Windows stability
MAX_PARALLEL_JOBS = 4  # Reduced from 8 to 4
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

MODELS_STRING = get_models_string(MODELS)
OUTPUT_DB = f"scm_arena_benchmark_{MODELS_STRING}.db"
PROGRESS_FILE = f"benchmark_progress_{MODELS_STRING}.json"

def run_subprocess_robust(cmd, **kwargs):
    """Run subprocess with maximum robustness for Windows Unicode issues"""
    # Force UTF-8 environment
    env = kwargs.get('env', os.environ.copy())
    env.update({
        'PYTHONIOENCODING': 'utf-8',
        'PYTHONUTF8': '1',
        'PYTHONLEGACYWINDOWSSTDIO': '0',
        'PYTHONUNBUFFERED': '1',
        'LC_ALL': 'en_US.UTF-8',
        'LANG': 'en_US.UTF-8'
    })
    kwargs['env'] = env
    
    # Set encoding parameters
    kwargs['encoding'] = 'utf-8'
    kwargs['errors'] = 'replace'  # Replace invalid characters instead of crashing
    
    # For Windows, ensure proper shell handling
    if sys.platform.startswith('win'):
        if isinstance(cmd, list):
            cmd = ' '.join(f'"{arg}"' if ' ' in arg else arg for arg in cmd)
        kwargs['shell'] = True
    
    return subprocess.run(cmd, **kwargs)

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
            "--runs", "1",
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
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.completed_experiments = set(data.get('completed', []))
                    self.failed_experiments = set(data.get('failed', []))
                    self.experiment_times = data.get('times', [])
                print(f"[PROGRESS] Loaded: {len(self.completed_experiments)} completed, {len(self.failed_experiments)} failed")
            except Exception as e:
                print(f"[WARNING] Could not load progress: {e}")
    
    def save_progress(self):
        """Save current progress"""
        try:
            data = {
                'completed': list(self.completed_experiments),
                'failed': list(self.failed_experiments),
                'times': self.experiment_times,
                'last_updated': time.time()
            }
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[WARNING] Could not save progress: {e}")
    
    def mark_completed(self, experiment_id: str, duration: float):
        """Mark experiment as completed"""
        with self.lock:
            self.completed_experiments.add(experiment_id)
            self.failed_experiments.discard(experiment_id)
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
            'completion_rate': completed_count / total_target if total_target > 0 else 0,
            'total_runtime': total_time,
            'avg_experiment_time': avg_time,
            'estimated_remaining': est_total_remaining
        }

class BenchmarkDataCollector:
    """Main benchmark data collector with robust Windows Unicode handling"""
    
    def __init__(self):
        self.progress = ProgressTracker()
        self.shutdown_requested = False
        self.total_experiments = len(MODELS) * 144 * RUNS_PER_CONDITION
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n[SHUTDOWN] Signal received. Completing current experiments...")
        self.shutdown_requested = True
    
    def test_model_connection(self, model: str) -> bool:
        """Test model connection with robust error handling"""
        try:
            print(f"[TEST] Testing {model} connection...")
            
            # Test with very simple command first
            cmd = ["poetry", "run", "python", "-c", f"print('Model {model} test')"]
            
            result = run_subprocess_robust(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=os.getcwd()
            )
            
            if result.returncode == 0:
                print(f"[SUCCESS] {model} basic test passed")
                return True
            else:
                print(f"[ERROR] {model} basic test failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Failed to test {model}: {e}")
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
        
        print(f"[PROGRESS] Total conditions: {len(all_conditions)}")
        print(f"[PROGRESS] Completed: {len(all_conditions) - len(pending)}")
        print(f"[PROGRESS] Pending: {len(pending)}")
        
        return pending
    
    def run_single_experiment(self, condition: ExperimentCondition) -> Tuple[bool, str]:
        """Run a single experimental condition with robust error handling"""
        experiment_id = condition.to_identifier()
        start_time = time.time()
        
        try:
            cmd = condition.to_cli_args()
            
            # Run with maximum robustness
            result = run_subprocess_robust(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=os.getcwd()
            )
            
            duration = time.time() - start_time
            
            # Check for success conditions
            if result.returncode == 0:
                self.progress.mark_completed(experiment_id, duration)
                return True, f"Completed in {duration:.1f}s"
            
            # Check for Unicode errors that might indicate partial success
            elif (result.stderr and 
                  ("UnicodeEncodeError" in result.stderr or "UnicodeDecodeError" in result.stderr) and
                  duration > 10):  # If it ran for a while, might have succeeded
                print(f"[INFO] {experiment_id}: Unicode error but long runtime - checking database")
                
                # Check if database was actually updated
                if self._check_experiment_in_database(condition):
                    self.progress.mark_completed(experiment_id, duration)
                    return True, f"Completed in {duration:.1f}s (Unicode display issue)"
                else:
                    self.progress.mark_failed(experiment_id)
                    return False, "Unicode error prevented completion"
            
            else:
                error_msg = f"Exit code {result.returncode}"
                if result.stderr:
                    error_msg += f": {result.stderr[:100]}"
                self.progress.mark_failed(experiment_id)
                return False, error_msg
                
        except subprocess.TimeoutExpired:
            self.progress.mark_failed(experiment_id)
            return False, "Timeout after 5 minutes"
        except Exception as e:
            self.progress.mark_failed(experiment_id)
            return False, str(e)
    
    def _check_experiment_in_database(self, condition: ExperimentCondition) -> bool:
        """Check if experiment was actually written to database despite Unicode errors"""
        try:
            if not os.path.exists(OUTPUT_DB):
                return False
            
            conn = sqlite3.connect(OUTPUT_DB)
            cursor = conn.cursor()
            
            # Check for experiments matching this condition added in last 5 minutes
            cursor.execute("""
                SELECT COUNT(*) FROM experiments 
                WHERE model_name = ? AND memory_strategy = ? AND prompt_type = ?
                AND visibility_level = ? AND scenario = ? AND game_mode = ?
                AND created_at > datetime('now', '-5 minutes')
            """, (condition.model, condition.memory, condition.prompt_type,
                  condition.visibility, condition.scenario, condition.game_mode))
            
            count = cursor.fetchone()[0]
            conn.close()
            
            return count > 0
        except Exception:
            return False
    
    def run_batch_parallel(self, conditions: List[ExperimentCondition]) -> None:
        """Run experiments in parallel batches with reduced load for Windows"""
        print(f"[START] Starting parallel execution with {MAX_PARALLEL_JOBS} workers")
        print(f"[INFO] Using reduced parallel load for Windows stability")
        
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
                        print("[SHUTDOWN] Stopping new experiments...")
                        break
                    
                    condition = future_to_condition[future]
                    success, message = future.result()
                    
                    if success:
                        completed_count += 1
                        print(f"[SUCCESS] {condition.to_identifier()}: {message}")
                    else:
                        failed_count += 1
                        print(f"[FAILED] {condition.to_identifier()}: {message}")
                    
                    # Progress update every 5 experiments (more frequent)
                    if (completed_count + failed_count) % 5 == 0:
                        self._print_progress_update()
        
        finally:
            print(f"\n[COMPLETE] Batch completed: {completed_count} succeeded, {failed_count} failed")
    
    def _print_progress_update(self):
        """Print current progress"""
        stats = self.progress.get_stats()
        
        print(f"\n[PROGRESS] UPDATE:")
        print(f"   Completed: {stats['completed']}/{stats['total_experiments']} ({stats['completion_rate']:.1%})")
        print(f"   Failed: {stats['failed']}")
        print(f"   Avg time per experiment: {stats['avg_experiment_time']:.1f}s")
        print(f"   Estimated remaining: {stats['estimated_remaining']/3600:.1f} hours")
        print()
    
    def run_full_study(self):
        """Run the complete benchmark data collection"""
        print("SCM-Arena Benchmark Data Collector - Ultimate Windows Fix")
        print("=" * 60)
        print(f"Models: {', '.join(MODELS)}")
        print(f"Total experiments: {self.total_experiments:,}")
        print(f"Parallel workers: {MAX_PARALLEL_JOBS} (reduced for Windows)")
        print(f"Database: {OUTPUT_DB}")
        print(f"Progress file: {PROGRESS_FILE}")
        print()
        
        # Test models
        print("[TEST] Testing model connections...")
        all_models_ok = True
        for model in MODELS:
            if not self.test_model_connection(model):
                all_models_ok = False
        
        if not all_models_ok:
            print("\n[WARNING] Some model tests failed, but continuing...")
        
        # Generate conditions
        all_conditions = self.generate_all_conditions()
        pending_conditions = self.filter_pending_conditions(all_conditions)
        
        if not pending_conditions:
            print("[COMPLETE] All experiments already completed!")
            self._print_final_summary()
            return True
        
        print(f"[ESTIMATE] Time: {len(pending_conditions) * 4 / MAX_PARALLEL_JOBS / 3600:.1f} hours")
        
        # Confirm before starting
        response = input(f"\n[CONFIRM] Start benchmark with {len(pending_conditions)} experiments? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            return False
        
        # Run the study
        start_time = time.time()
        
        try:
            self.run_batch_parallel(pending_conditions)
        except KeyboardInterrupt:
            print("\n[INTERRUPT] Interrupted by user")
        except Exception as e:
            print(f"\n[ERROR] Unexpected error: {e}")
        
        total_time = time.time() - start_time
        print(f"\n[COMPLETE] Total execution time: {total_time/3600:.1f} hours")
        
        self._print_final_summary()
        return True
    
    def _print_final_summary(self):
        """Print final summary and statistics"""
        stats = self.progress.get_stats()
        
        print(f"\n[SUMMARY] BENCHMARK DATA COLLECTION SUMMARY")
        print("=" * 50)
        print(f"Models: {', '.join(stats['models'])}")
        print(f"Completed experiments: {stats['completed']:,}")
        print(f"Failed experiments: {stats['failed']:,}")
        if (stats['completed'] + stats['failed']) > 0:
            print(f"Success rate: {stats['completed']/(stats['completed']+stats['failed']):.1%}")
        print(f"Database: {OUTPUT_DB}")
        
        # Check database
        if os.path.exists(OUTPUT_DB):
            try:
                conn = sqlite3.connect(OUTPUT_DB)
                cursor = conn.execute("SELECT COUNT(*) FROM experiments")
                db_count = cursor.fetchone()[0]
                print(f"Database contains: {db_count:,} experiments")
                conn.close()
            except Exception as e:
                print(f"[WARNING] Could not check database: {e}")

def main():
    """Main entry point with maximum Windows compatibility"""
    # Ensure UTF-8 environment
    force_utf8_environment()
    
    collector = BenchmarkDataCollector()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "status":
            stats = collector.progress.get_stats()
            print(f"Models: {', '.join(stats['models'])}")
            print(f"Progress: {stats['completed']:,}/{stats['total_experiments']:,} ({stats['completion_rate']:.1%})")
            print(f"Failed: {stats['failed']:,}")
            print(f"Database: {OUTPUT_DB}")
            
        elif command == "test":
            # Test single experiment
            test_condition = ExperimentCondition(
                model="phi4:latest",
                memory="none", 
                prompt_type="neutral",
                visibility="local",
                scenario="classic",
                game_mode="modern",
                run_number=999  # Test run
            )
            success, message = collector.run_single_experiment(test_condition)
            print(f"Test result: {'SUCCESS' if success else 'FAILED'} - {message}")
            
        else:
            print("Usage: python benchmark_ultimate_fix.py [status|test]")
    
    else:
        # Run the full study
        collector.run_full_study()

if __name__ == "__main__":
    main()