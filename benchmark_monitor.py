#!/usr/bin/env python3
"""
SCM-Arena Benchmark Monitor Dashboard
====================================
Live monitoring dashboard for benchmark data collection.
Shows real-time progress, resource usage, and performance metrics.
Auto-detects model-specific file names for organization.
"""
import time
import json
import sqlite3
import os
import psutil
from datetime import datetime, timedelta
import threading
from collections import deque
import glob

def detect_model_files():
    """Auto-detect model-specific files"""
    # Look for progress files
    progress_files = glob.glob("benchmark_progress_*.json")
    db_files = glob.glob("scm_arena_benchmark_*.db")
    
    models_data = {}
    
    for progress_file in progress_files:
        # Extract model string from filename
        model_string = progress_file.replace("benchmark_progress_", "").replace(".json", "")
        db_file = f"scm_arena_benchmark_{model_string}.db"
        
        if os.path.exists(db_file):
            models_data[model_string] = {
                'progress_file': progress_file,
                'db_file': db_file
            }
    
    return models_data

class BenchmarkMonitor:
    """Real-time monitoring of benchmark data collection"""
    
    def __init__(self, model_string=None):
        # Auto-detect files if not specified
        if model_string is None:
            models_data = detect_model_files()
            if len(models_data) == 1:
                model_string = list(models_data.keys())[0]
                print(f"🔍 Auto-detected model: {model_string}")
            elif len(models_data) > 1:
                print("🔍 Multiple models detected:")
                for i, ms in enumerate(models_data.keys(), 1):
                    print(f"  {i}. {ms}")
                choice = input("Select model number: ")
                try:
                    model_string = list(models_data.keys())[int(choice) - 1]
                except:
                    model_string = list(models_data.keys())[0]
            else:
                print("❌ No benchmark files found")
                model_string = "phi4_latest"  # Default
        
        self.model_string = model_string
        self.progress_file = f"benchmark_progress_{model_string}.json"
        self.db_path = f"scm_arena_benchmark_{model_string}.db"
        self.running = False
        
        # Performance tracking
        self.completion_history = deque(maxlen=60)  # Last 60 data points
        self.resource_history = deque(maxlen=60)
        
        print(f"📊 Monitoring: {model_string}")
        print(f"📁 Progress file: {self.progress_file}")
        print(f"💾 Database: {self.db_path}")
    
    def get_current_stats(self):
        """Get current study statistics"""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'model_string': self.model_string,
            'progress': self._get_progress_stats(),
            'database': self._get_database_stats(),
            'resources': self._get_resource_stats(),
            'performance': self._get_performance_stats()
        }
        return stats
    
    def _get_progress_stats(self):
        """Get progress from JSON file"""
        if not os.path.exists(self.progress_file):
            return {'completed': 0, 'failed': 0, 'times': [], 'models': []}
        
        try:
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
                total_target = data.get('total_target', 2880)  # Get dynamic total
                completed_count = len(data.get('completed', []))
                
                return {
                    'models': data.get('models', []),
                    'completed': completed_count,
                    'failed': len(data.get('failed', [])),
                    'total_target': total_target,
                    'completion_rate': completed_count / total_target if total_target > 0 else 0,
                    'avg_time': sum(data.get('times', [])) / max(1, len(data.get('times', [])))
                }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_database_stats(self):
        """Get statistics from database"""
        if not os.path.exists(self.db_path):
            return {'total_experiments': 0, 'unique_conditions': 0}
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Count total experiments
            cursor = conn.execute("SELECT COUNT(*) FROM experiments")
            total_experiments = cursor.fetchone()[0]
            
            # Count unique conditions
            cursor = conn.execute("""
                SELECT COUNT(DISTINCT memory_strategy || '-' || prompt_type || '-' || 
                       visibility_level || '-' || scenario || '-' || game_mode) 
                FROM experiments
            """)
            unique_conditions = cursor.fetchone()[0]
            
            # Recent experiments (last hour)
            one_hour_ago = datetime.now() - timedelta(hours=1)
            cursor = conn.execute("""
                SELECT COUNT(*) FROM experiments 
                WHERE timestamp > ?
            """, (one_hour_ago.isoformat(),))
            recent_experiments = cursor.fetchone()[0]
            
            # Model breakdown
            cursor = conn.execute("""
                SELECT model_name, COUNT(*) as count
                FROM experiments 
                GROUP BY model_name
            """)
            model_breakdown = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Average costs by condition type
            cursor = conn.execute("""
                SELECT game_mode, AVG(total_cost) as avg_cost, COUNT(*) as count
                FROM experiments 
                GROUP BY game_mode
            """)
            cost_by_mode = {row[0]: {'avg_cost': row[1], 'count': row[2]} for row in cursor.fetchall()}
            
            conn.close()
            
            return {
                'total_experiments': total_experiments,
                'unique_conditions': unique_conditions,
                'recent_experiments': recent_experiments,
                'model_breakdown': model_breakdown,
                'cost_by_mode': cost_by_mode
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_resource_stats(self):
        """Get current system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            # Find Ollama processes
            ollama_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
                try:
                    if 'ollama' in proc.info['name'].lower():
                        ollama_processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cpu_percent': proc.info['cpu_percent'],
                            'memory_mb': proc.info['memory_info'].rss / (1024 * 1024)
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_free_gb': disk.free / (1024**3),
                'ollama_processes': ollama_processes
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_performance_stats(self):
        """Calculate performance metrics"""
        if len(self.completion_history) < 2:
            return {'experiments_per_hour': 0, 'eta_hours': 0}
        
        # Calculate completion rate from recent history
        if len(self.completion_history) >= 2:
            recent_points = list(self.completion_history)[-2:]
            time_diff = (datetime.fromisoformat(recent_points[-1]['timestamp']) - 
                        datetime.fromisoformat(recent_points[0]['timestamp'])).total_seconds() / 3600
            completion_diff = recent_points[-1]['completed'] - recent_points[0]['completed']
            experiments_per_hour = completion_diff / max(0.1, time_diff)
        else:
            experiments_per_hour = 0
        
        progress = self._get_progress_stats()
        total_target = progress.get('total_target', 2880)
        remaining = total_target - progress.get('completed', 0)
        eta_hours = remaining / max(1, experiments_per_hour) if experiments_per_hour > 0 else 0
        
        return {
            'experiments_per_hour': experiments_per_hour,
            'eta_hours': eta_hours,
            'estimated_completion': (datetime.now() + timedelta(hours=eta_hours)).isoformat() if eta_hours > 0 else None
        }
    
    def start_monitoring(self, update_interval=30):
        """Start monitoring loop"""
        self.running = True
        
        def monitor_loop():
            while self.running:
                try:
                    stats = self.get_current_stats()
                    
                    # Store history
                    self.completion_history.append({
                        'timestamp': stats['timestamp'],
                        'completed': stats['progress'].get('completed', 0)
                    })
                    
                    self.resource_history.append({
                        'timestamp': stats['timestamp'],
                        'cpu': stats['resources'].get('cpu_percent', 0),
                        'memory': stats['resources'].get('memory_percent', 0)
                    })
                    
                    # Print dashboard
                    self._print_dashboard(stats)
                    
                    time.sleep(update_interval)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"⚠️ Monitoring error: {e}")
                    time.sleep(5)
        
        monitor_thread = threading.Thread(target=monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        return monitor_thread
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
    
    def _print_dashboard(self, stats):
        """Print real-time dashboard"""
        os.system('clear' if os.name == 'posix' else 'cls')  # Clear screen
        
        print(f"🧪 SCM-Arena Benchmark Monitor - {stats['model_string']}")
        print("=" * 70)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Progress section
        progress = stats['progress']
        print("📊 BENCHMARK PROGRESS")
        print("-" * 30)
        
        if 'error' not in progress:
            completed = progress['completed']
            failed = progress['failed']
            total = progress['total_target']
            rate = progress['completion_rate']
            models = progress.get('models', [])
            
            print(f"🤖 Models: {', '.join(models) if models else 'Unknown'}")
            print(f"✅ Completed: {completed:,}/{total:,} ({rate:.1%})")
            print(f"❌ Failed: {failed:,}")
            print(f"⏱️  Avg time per experiment: {progress['avg_time']:.1f}s")
            
            # Progress bar
            bar_width = 50
            filled = int(bar_width * rate)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(f"[{bar}] {rate:.1%}")
        else:
            print(f"❌ Progress error: {progress['error']}")
        
        print()
        
        # Performance section
        perf = stats['performance']
        print("⚡ PERFORMANCE METRICS")
        print("-" * 30)
        print(f"🚀 Experiments/hour: {perf['experiments_per_hour']:.1f}")
        print(f"🕒 ETA: {perf['eta_hours']:.1f} hours")
        if perf.get('estimated_completion'):
            completion_time = datetime.fromisoformat(perf['estimated_completion'])
            print(f"🏁 Est. completion: {completion_time.strftime('%Y-%m-%d %H:%M')}")
        print()
        
        # Resource section
        resources = stats['resources']
        print("💻 SYSTEM RESOURCES")
        print("-" * 30)
        
        if 'error' not in resources:
            print(f"🖥️  CPU: {resources['cpu_percent']:.1f}%")
            print(f"🧠 Memory: {resources['memory_percent']:.1f}% ({resources['memory_available_gb']:.1f}GB free)")
            print(f"💾 Disk: {resources['disk_free_gb']:.1f}GB free")
            
            if resources['ollama_processes']:
                print(f"🦙 Ollama processes: {len(resources['ollama_processes'])}")
                for proc in resources['ollama_processes']:
                    print(f"   PID {proc['pid']}: {proc['memory_mb']:.0f}MB")
        else:
            print(f"❌ Resource error: {resources['error']}")
        
        print()
        
        # Database section
        db = stats['database']
        print("🗄️  DATABASE STATUS")
        print("-" * 30)
        
        if 'error' not in db:
            print(f"📈 Total experiments: {db['total_experiments']:,}")
            print(f"🎯 Unique conditions: {db['unique_conditions']}")
            print(f"🕐 Recent (1h): {db['recent_experiments']}")
            
            if 'model_breakdown' in db and db['model_breakdown']:
                print("🤖 Per model:")
                for model, count in db['model_breakdown'].items():
                    print(f"   {model}: {count:,}")
            
            if 'cost_by_mode' in db and db['cost_by_mode']:
                print("💰 Avg costs:")
                for mode, data in db['cost_by_mode'].items():
                    avg_cost = data.get('avg_cost', 0)
                    count = data.get('count', 0)
                    if avg_cost and count:
                        print(f"   {mode}: ${avg_cost:.0f} ({count} exp)")
        else:
            print(f"❌ Database error: {db['error']}")
        
        print()
        print("Press Ctrl+C to stop monitoring")

def main():
    """Main monitoring function"""
    # Auto-detect or let user choose
    models_data = detect_model_files()
    
    if not models_data:
        print("❌ No benchmark files found!")
        print("Run the benchmark data collector first.")
        return
    
    if len(models_data) == 1:
        model_string = list(models_data.keys())[0]
    else:
        print("🔍 Multiple benchmark studies detected:")
        model_list = list(models_data.keys())
        for i, ms in enumerate(model_list, 1):
            print(f"  {i}. {ms}")
        
        try:
            choice = int(input("Select which to monitor: ")) - 1
            model_string = model_list[choice]
        except:
            model_string = model_list[0]
    
    monitor = BenchmarkMonitor(model_string)
    
    print(f"🔍 Starting Monitor for {model_string}...")
    print("Press Ctrl+C to stop")
    
    try:
        monitor_thread = monitor.start_monitoring(update_interval=10)
        monitor_thread.join()
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped")
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()