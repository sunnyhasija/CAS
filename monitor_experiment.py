#!/usr/bin/env python3
"""
SCM Arena Experiment Progress Monitor

Real-time monitoring of running experiments from a separate terminal.
Displays progress by condition with completion rates and ETA estimates.

Usage:
    python monitor_experiments.py --db-path /mnt/e/scm_arena_phi4_qwen3_full_factorial_20250625_8pm.db
"""

import sqlite3
import time
import argparse
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
import os

console = Console()

class ExperimentMonitor:
    def __init__(self, db_path):
        self.db_path = db_path
        self.start_time = None
        self.schema_checked = False
        self.completion_column = None
        self.start_column = None
        
    def connect_db(self):
        """Connect to the experiment database"""
        if not os.path.exists(self.db_path):
            return None
        try:
            return sqlite3.connect(self.db_path)
        except Exception as e:
            console.print(f"[red]Database connection error: {e}[/red]")
            return None
    
    def inspect_schema(self):
        """Inspect the database schema to understand available columns"""
        conn = self.connect_db()
        if not conn:
            return False
            
        try:
            # Get table schema
            cursor = conn.execute("PRAGMA table_info(experiments)")
            columns = [row[1] for row in cursor.fetchall()]
            
            console.print(f"[blue]Available columns: {', '.join(columns)}[/blue]")
            
            # Determine completion indicator
            if 'end_time' in columns:
                self.completion_column = 'end_time IS NOT NULL'
                self.start_column = 'start_time'
            elif 'completed_at' in columns:
                self.completion_column = 'completed_at IS NOT NULL'
                self.start_column = 'created_at'
            elif 'finished' in columns:
                self.completion_column = 'finished = 1'
                self.start_column = 'start_time'
            elif 'status' in columns:
                self.completion_column = "status = 'completed'"
                self.start_column = 'start_time'
            else:
                # Check if we have total_cost as completion indicator
                if 'total_cost' in columns:
                    self.completion_column = 'total_cost IS NOT NULL'
                    # Determine start time column
                    if 'timestamp' in columns:
                        self.start_column = 'timestamp'
                    elif 'start_time' in columns:
                        self.start_column = 'start_time'
                    elif 'created_at' in columns:
                        self.start_column = 'created_at'
                    else:
                        console.print("[yellow]Cannot find timestamp column[/yellow]")
                        conn.close()
                        return False
                else:
                    console.print("[yellow]Cannot determine completion status from schema[/yellow]")
                    conn.close()
                    return False
            
            self.schema_checked = True
            conn.close()
            return True
            
        except Exception as e:
            console.print(f"[red]Schema inspection error: {e}[/red]")
            conn.close()
            return False
    
    def get_experiment_progress(self):
        """Get current progress of all experimental conditions"""
        if not self.schema_checked:
            if not self.inspect_schema():
                return None, None
        
        conn = self.connect_db()
        if not conn:
            return None, None
            
        try:
            # Get total experiments and completed experiments by condition
            query = f"""
            SELECT 
                model_name,
                memory_strategy,
                prompt_type,
                visibility_level,
                scenario,
                game_mode,
                COUNT(*) as total_runs,
                SUM(CASE WHEN {self.completion_column} THEN 1 ELSE 0 END) as completed_runs,
                AVG(CASE WHEN {self.completion_column} THEN total_cost ELSE NULL END) as avg_cost,
                MIN({self.start_column}) as first_start
            FROM experiments 
            GROUP BY model_name, memory_strategy, prompt_type, visibility_level, scenario, game_mode
            ORDER BY model_name, memory_strategy, prompt_type, visibility_level, scenario, game_mode
            """
            
            cursor = conn.execute(query)
            conditions = cursor.fetchall()
            
            # Calculate expected total based on experimental design
            # Get unique experimental conditions and their expected run counts
            design_query = """
            SELECT 
                COUNT(DISTINCT model_name) as num_models,
                COUNT(DISTINCT memory_strategy) as num_memory,
                COUNT(DISTINCT prompt_type) as num_prompts,
                COUNT(DISTINCT visibility_level) as num_visibility,
                COUNT(DISTINCT scenario) as num_scenarios,
                COUNT(DISTINCT game_mode) as num_modes,
                MAX(run_number) as max_runs
            FROM experiments
            """
            
            cursor = conn.execute(design_query)
            design_stats = cursor.fetchone()
            
            if design_stats and all(x is not None for x in design_stats[:-1]):
                num_models, num_memory, num_prompts, num_visibility, num_scenarios, num_modes, max_runs = design_stats
                expected_total = num_models * num_memory * num_prompts * num_visibility * num_scenarios * num_modes * (max_runs or 20)
            else:
                expected_total = None
            
            # Get current database stats
            current_query = f"""
            SELECT 
                COUNT(*) as current_experiments,
                SUM(CASE WHEN {self.completion_column} THEN 1 ELSE 0 END) as completed_experiments,
                AVG(CASE WHEN {self.completion_column} THEN total_cost ELSE NULL END) as overall_avg_cost,
                COUNT(DISTINCT model_name) as num_models,
                MIN({self.start_column}) as experiment_start
            FROM experiments
            """
            
            cursor = conn.execute(current_query)
            current_stats = cursor.fetchone()
            
            # Combine stats: (current_in_db, completed, avg_cost, num_models, start_time, expected_total)
            if current_stats:
                combined_stats = current_stats + (expected_total,)
            else:
                combined_stats = None
            
            conn.close()
            return conditions, combined_stats
            
        except Exception as e:
            conn.close()
            console.print(f"[red]Database query error: {e}[/red]")
            return None, None
    
    def calculate_eta(self, completed, total, start_time_str):
        """Calculate estimated time to completion"""
        if not completed or not start_time_str or completed == 0:
            return "Unknown"
            
        try:
            # Parse the datetime string - handle different formats
            if 'T' in start_time_str:
                start_time = datetime.fromisoformat(start_time_str.replace('Z', ''))
            else:
                start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
                
            elapsed = datetime.now() - start_time
            rate = completed / elapsed.total_seconds()  # experiments per second
            remaining = total - completed
            
            if rate <= 0 or remaining <= 0:
                return "Complete" if remaining <= 0 else "Unknown"
                
            eta_seconds = remaining / rate
            eta = timedelta(seconds=eta_seconds)
            
            # Format ETA nicely
            if eta.days > 0:
                return f"{eta.days}d {eta.seconds//3600}h"
            elif eta.seconds > 3600:
                return f"{eta.seconds//3600}h {(eta.seconds%3600)//60}m"
            elif eta.seconds > 60:
                return f"{eta.seconds//60}m"
            else:
                return f"{eta.seconds}s"
                
        except Exception as e:
            return "Unknown"
    
    def create_progress_display(self, conditions, overall_stats):
        """Create a rich display showing progress"""
        
        if not conditions or not overall_stats:
            return Panel("[red]No data available - database may not exist or be empty[/red]", title="Experiment Monitor")
        
        current_experiments, completed_experiments, overall_avg_cost, num_models, experiment_start, expected_total = overall_stats
        
        # Use expected total if available, otherwise fall back to current count
        total_experiments = expected_total if expected_total else current_experiments
        
        # Calculate overall progress
        overall_progress = completed_experiments / total_experiments if total_experiments > 0 else 0
        
        # Format average cost
        cost_str = f"${overall_avg_cost:.0f}" if overall_avg_cost else "N/A"
        
        # Create progress status text
        if expected_total:
            progress_text = f"[green]{completed_experiments:,} / {expected_total:,}[/green] ([yellow]{overall_progress:.1%}[/yellow])"
            db_text = f"DB Entries: [blue]{current_experiments:,}[/blue]"
        else:
            progress_text = f"[green]{completed_experiments:,} / {current_experiments:,}[/green] ([yellow]{overall_progress:.1%}[/yellow])"
            db_text = "[yellow]Expected total: Unknown[/yellow]"
        
        # Overall progress info
        overall_info = f"""[bold blue]📊 Overall Progress[/bold blue]
Completed: {progress_text}
{db_text}
Models: [cyan]{num_models}[/cyan]
Average Cost: [green]{cost_str}[/green]
ETA: [yellow]{self.calculate_eta(completed_experiments, total_experiments, experiment_start)}[/yellow]
Last Updated: [blue]{datetime.now().strftime('%H:%M:%S')}[/blue]"""
        
        # Create progress table
        table = Table(title="Progress by Experimental Condition", show_lines=True)
        table.add_column("Model", style="cyan", width=12)
        table.add_column("Memory", style="blue", width=8) 
        table.add_column("Prompt", style="green", width=8)
        table.add_column("Visibility", style="yellow", width=10)
        table.add_column("Scenario", style="magenta", width=10)
        table.add_column("Mode", style="red", width=8)
        table.add_column("Progress", style="white", width=15)
        table.add_column("Avg Cost", style="green", width=10)
        table.add_column("Status", style="white", width=12)
        
        for condition in conditions:
            (model, memory, prompt, visibility, scenario, game_mode, 
             total_runs, completed_runs, avg_cost, first_start) = condition
            
            progress_pct = completed_runs / total_runs if total_runs > 0 else 0
            
            # Create progress bar
            bar_width = 10
            filled = int(progress_pct * bar_width)
            progress_bar = f"[{'█' * filled}{'░' * (bar_width - filled)}] {completed_runs}/{total_runs}"
            
            # Status indicator
            if completed_runs == 0:
                status = "⏳ Waiting"
            elif completed_runs == total_runs:
                status = "✅ Complete"
            else:
                status = "🔄 Running"
            
            # Cost formatting
            cost_display = f"${avg_cost:.0f}" if avg_cost else "N/A"
            
            # Truncate long names
            model_short = model[:12] if model else "Unknown"
            
            table.add_row(
                model_short,
                memory or "N/A",
                prompt or "N/A", 
                visibility or "N/A",
                scenario or "N/A",
                game_mode or "N/A",
                progress_bar,
                cost_display,
                status
            )
        
        # Combine overall info and table
        layout = Layout()
        layout.split_column(
            Layout(Panel(overall_info, title="Experiment Status"), size=8),
            Layout(table)
        )
        
        return layout
    
    def run_monitor(self, refresh_interval=30):
        """Run the monitoring loop"""
        console.print(f"[bold blue]🔍 SCM Arena Experiment Monitor[/bold blue]")
        console.print(f"📁 Database: {self.db_path}")
        console.print(f"🔄 Refresh: Every {refresh_interval}s (Ctrl+C to exit)")
        console.print()
        
        with Live(console=console, refresh_per_second=0.5, screen=False) as live:
            while True:
                try:
                    conditions, overall_stats = self.get_experiment_progress()
                    
                    if conditions is None:
                        display = Panel("[red]Cannot connect to database or no data available[/red]", title="Error")
                    else:
                        display = self.create_progress_display(conditions, overall_stats)
                    
                    live.update(display)
                    time.sleep(refresh_interval)
                    
                except KeyboardInterrupt:
                    console.print("\n[yellow]Monitor stopped by user[/yellow]")
                    break
                except Exception as e:
                    console.print(f"\n[red]Error: {e}[/red]")
                    time.sleep(5)


def quick_status(db_path):
    """Quick status check without live updates"""
    if not os.path.exists(db_path):
        console.print(f"[red]Database not found: {db_path}[/red]")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Get basic table info
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        console.print(f"[blue]Tables: {', '.join(tables)}[/blue]")
        
        if 'experiments' in tables:
            # Get column info
            cursor = conn.execute("PRAGMA table_info(experiments)")
            columns = [row[1] for row in cursor.fetchall()]
            console.print(f"[blue]Columns: {', '.join(columns)}[/blue]")
            
            # Get basic counts
            cursor = conn.execute("SELECT COUNT(*) FROM experiments")
            current_total = cursor.fetchone()[0]
            
            # Calculate expected total from experimental design
            design_query = """
            SELECT 
                COUNT(DISTINCT model_name) as num_models,
                COUNT(DISTINCT memory_strategy) as num_memory,
                COUNT(DISTINCT prompt_type) as num_prompts,
                COUNT(DISTINCT visibility_level) as num_visibility,
                COUNT(DISTINCT scenario) as num_scenarios,
                COUNT(DISTINCT game_mode) as num_modes,
                MAX(run_number) as max_runs
            FROM experiments
            """
            
            cursor = conn.execute(design_query)
            design_stats = cursor.fetchone()
            
            if design_stats and all(x is not None for x in design_stats[:-1]):
                num_models, num_memory, num_prompts, num_visibility, num_scenarios, num_modes, max_runs = design_stats
                expected_total = num_models * num_memory * num_prompts * num_visibility * num_scenarios * num_modes * (max_runs or 20)
                console.print(f"[green]Expected total experiments: {expected_total:,}[/green]")
                console.print(f"[blue]Experimental design: {num_models} models × {num_memory} memory × {num_prompts} prompts × {num_visibility} visibility × {num_scenarios} scenarios × {num_modes} modes × {max_runs or 20} runs[/blue]")
            else:
                expected_total = None
                console.print("[yellow]Cannot calculate expected total - incomplete design in database[/yellow]")
            
            # Try to determine completion
            if 'total_cost' in columns:
                cursor = conn.execute("SELECT COUNT(*) FROM experiments WHERE total_cost IS NOT NULL")
                completed = cursor.fetchone()[0]
            else:
                completed = "Unknown"
            
            console.print(f"[green]Current database entries: {current_total:,}[/green]")
            console.print(f"[green]Completed experiments: {completed}[/green]")
            
            if expected_total and isinstance(completed, int):
                progress = completed / expected_total
                console.print(f"[yellow]Overall progress: {progress:.1%}[/yellow]")
            
            # Show some sample data
            cursor = conn.execute("SELECT * FROM experiments LIMIT 3")
            samples = cursor.fetchall()
            console.print(f"[yellow]Sample rows: {len(samples)}[/yellow]")
            for i, row in enumerate(samples):
                console.print(f"  Row {i+1}: {row[:5]}...")  # Show first 5 columns
        
        conn.close()
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def main():
    parser = argparse.ArgumentParser(description="Monitor SCM Arena experiment progress")
    parser.add_argument("--db-path", required=True, help="Path to experiment database")
    parser.add_argument("--refresh", type=int, default=30, help="Refresh interval in seconds (default: 30)")
    parser.add_argument("--quick", action="store_true", help="Quick status check only")
    
    args = parser.parse_args()
    
    # Check if database exists
    if not os.path.exists(args.db_path):
        console.print(f"[red]Database not found: {args.db_path}[/red]")
        console.print("[yellow]Make sure your experiment is running and creating the database[/yellow]")
        return
    
    if args.quick:
        quick_status(args.db_path)
    else:
        monitor = ExperimentMonitor(args.db_path)
        monitor.run_monitor(args.refresh)


if __name__ == "__main__":
    main()