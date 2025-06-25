#!/usr/bin/env python3
# merge_missing_experiments.py - FIXED VERSION
"""
Merge the missing experiments into the main dataset
Creates a complete full factorial dataset
"""

import sqlite3
import pandas as pd
import glob
import shutil
import gc
import time
from datetime import datetime
from pathlib import Path

def analyze_missing_experiments():
    """Analyze what we have in the missing experiments folder"""
    print("📊 ANALYZING MISSING EXPERIMENTS")
    print("=" * 60)
    
    missing_dbs = glob.glob('missing_experiments_copy/*.db')
    
    total_missing = 0
    details = []
    
    for db_path in missing_dbs:
        if Path(db_path).exists() and Path(db_path).stat().st_size > 40000:  # Skip empty DBs
            conn = None
            try:
                conn = sqlite3.connect(db_path)
                
                # Get experiment count and details
                stats = pd.read_sql_query("""
                    SELECT 
                        scenario,
                        memory_strategy,
                        visibility_level,
                        COUNT(*) as count,
                        AVG(total_cost) as avg_cost,
                        MIN(total_cost) as min_cost,
                        MAX(total_cost) as max_cost
                    FROM experiments 
                    WHERE total_cost > 0
                    GROUP BY scenario, memory_strategy, visibility_level
                """, conn)
                
                if len(stats) > 0:
                    db_name = Path(db_path).name
                    completed = stats['count'].sum()
                    total_missing += completed
                    
                    details.append({
                        'database': db_name,
                        'completed': completed,
                        'scenario': stats['scenario'].iloc[0],
                        'memory': stats['memory_strategy'].iloc[0] if len(stats) == 1 else 'multiple',
                        'visibility': stats['visibility_level'].iloc[0] if len(stats) == 1 else 'multiple'
                    })
                    
                    print(f"\n📁 {db_name}")
                    print(f"   Completed: {completed} experiments")
                    print(f"   Scenario: {stats['scenario'].iloc[0]}")
                    if len(stats) > 1:
                        print("   Details:")
                        for _, row in stats.iterrows():
                            print(f"     {row['memory_strategy']}-{row['visibility_level']}: {row['count']} experiments")
                
            except Exception as e:
                print(f"⚠️  Error reading {Path(db_path).name}: {e}")
            finally:
                if conn:
                    conn.close()
                    conn = None
                # Force cleanup
                gc.collect()
                time.sleep(0.1)  # Give SQLite time to release locks
    
    print(f"\n📊 Total missing experiments ready to merge: {total_missing}")
    
    return details, total_missing

def merge_all_databases(existing_merged='full_factorial_merged.db', 
                       output_path='full_factorial_complete.db',
                       backup_existing=True):
    """Merge missing experiments with existing merged database"""
    
    print("\n🔄 MERGING DATABASES")
    print("=" * 60)
    
    # Check existing merged database
    if not Path(existing_merged).exists():
        print(f"❌ Existing merged database not found: {existing_merged}")
        return False
    
    # Create backup of existing
    if backup_existing:
        backup_path = f"{existing_merged}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(existing_merged, backup_path)
        print(f"✅ Backed up existing database to: {backup_path}")
    
    # Copy existing to new complete database
    if Path(output_path).exists():
        Path(output_path).unlink()
    shutil.copy2(existing_merged, output_path)
    print(f"✅ Created new complete database: {output_path}")
    
    # Open connection to complete database
    complete_conn = None
    try:
        complete_conn = sqlite3.connect(output_path)
        
        # Get count of existing experiments
        existing_count = complete_conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
        print(f"📊 Starting with {existing_count} existing experiments")
        
        # Merge each missing experiment database
        missing_dbs = glob.glob('missing_experiments/*.db')
        added_total = 0
        
        for db_path in missing_dbs:
            if Path(db_path).stat().st_size > 40000:  # Skip empty DBs
                attached = False
                try:
                    print(f"\n🔄 Merging {Path(db_path).name}...")
                    
                    # Attach the database with a unique name
                    attach_name = f"missing_db_{hash(db_path) % 10000}"
                    complete_conn.execute(f"ATTACH DATABASE '{db_path}' AS {attach_name}")
                    attached = True
                    
                    # Copy experiments
                    added = complete_conn.execute(f"""
                        INSERT INTO main.experiments 
                        SELECT * FROM {attach_name}.experiments 
                        WHERE total_cost > 0
                    """).rowcount
                    
                    print(f"   Added {added} experiments")
                    added_total += added
                    
                    # Copy rounds
                    rounds_added = complete_conn.execute(f"""
                        INSERT INTO main.rounds
                        SELECT r.* FROM {attach_name}.rounds r
                        INNER JOIN {attach_name}.experiments e ON r.experiment_id = e.experiment_id
                        WHERE e.total_cost > 0
                    """).rowcount
                    
                    print(f"   Added {rounds_added} rounds")
                    
                    # Copy agent_rounds
                    agent_rounds_added = complete_conn.execute(f"""
                        INSERT INTO main.agent_rounds
                        SELECT ar.* FROM {attach_name}.agent_rounds ar
                        INNER JOIN {attach_name}.experiments e ON ar.experiment_id = e.experiment_id
                        WHERE e.total_cost > 0
                    """).rowcount
                    
                    print(f"   Added {agent_rounds_added} agent rounds")
                    
                    # Copy game_states if exists
                    try:
                        game_states_added = complete_conn.execute(f"""
                            INSERT INTO main.game_states
                            SELECT gs.* FROM {attach_name}.game_states gs
                            INNER JOIN {attach_name}.experiments e ON gs.experiment_id = e.experiment_id
                            WHERE e.total_cost > 0
                        """).rowcount
                        print(f"   Added {game_states_added} game states")
                    except:
                        pass  # game_states might not exist in all databases
                    
                    # Commit the transaction
                    complete_conn.commit()
                    
                except Exception as e:
                    print(f"⚠️  Error merging {Path(db_path).name}: {e}")
                    # Rollback any partial changes
                    complete_conn.rollback()
                finally:
                    # Always detach, even if there was an error
                    if attached:
                        try:
                            complete_conn.execute(f"DETACH DATABASE {attach_name}")
                        except:
                            pass
                    # Force cleanup after each database
                    gc.collect()
                    time.sleep(0.1)
        
        print(f"\n✅ Total new experiments added: {added_total}")
        
        # Verify and show final statistics
        print("\n📊 FINAL DATABASE STATISTICS")
        print("=" * 60)
        
        # Check for duplicates
        duplicates = complete_conn.execute("""
            SELECT COUNT(*) FROM (
                SELECT model_name, memory_strategy, prompt_type, visibility_level,
                       scenario, game_mode, run_number, COUNT(*) as count
                FROM experiments
                GROUP BY model_name, memory_strategy, prompt_type, visibility_level,
                         scenario, game_mode, run_number
                HAVING count > 1
            )
        """).fetchone()[0]
        
        if duplicates > 0:
            print(f"⚠️  Found {duplicates} duplicate conditions - cleaning...")
            # Remove duplicates keeping the first occurrence
            complete_conn.execute("""
                DELETE FROM experiments 
                WHERE rowid NOT IN (
                    SELECT MIN(rowid) 
                    FROM experiments 
                    GROUP BY model_name, memory_strategy, prompt_type, visibility_level,
                             scenario, game_mode, run_number
                )
            """)
            complete_conn.commit()
            print("✅ Duplicates removed")
        
        # Final statistics
        final_stats = pd.read_sql_query("""
            SELECT 
                scenario,
                COUNT(DISTINCT memory_strategy || '-' || visibility_level || '-' || prompt_type || '-' || game_mode) as conditions,
                COUNT(*) as experiments,
                ROUND(AVG(total_cost), 2) as avg_cost,
                ROUND(AVG(service_level), 3) as avg_service
            FROM experiments
            GROUP BY scenario
            ORDER BY scenario
        """, complete_conn)
        
        print("\nScenario Summary:")
        print(final_stats.to_string(index=False))
        
        # Total summary
        total_final = complete_conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
        print(f"\n🎉 MERGE COMPLETE!")
        print(f"Total experiments in complete database: {total_final}")
        print(f"Increase from original: {total_final - existing_count} experiments")
        
        # Export to CSV
        csv_path = output_path.replace('.db', '.csv')
        all_experiments = pd.read_sql_query("SELECT * FROM experiments ORDER BY scenario, memory_strategy", complete_conn)
        all_experiments.to_csv(csv_path, index=False)
        print(f"\n📄 Exported to CSV: {csv_path}")
        
        # Coverage check
        print("\n📊 COVERAGE CHECK")
        coverage = pd.read_sql_query("""
            SELECT 
                scenario,
                memory_strategy,
                visibility_level,
                COUNT(*) as count
            FROM experiments
            GROUP BY scenario, memory_strategy, visibility_level
            ORDER BY scenario, memory_strategy, visibility_level
        """, complete_conn)
        
        # Expected counts (assuming 2 prompts × 2 game_modes × 20 runs = 80 per combination)
        expected_per_combo = 2 * 2 * 20  # 80
        
        print(f"\nExpected per combination: {expected_per_combo} experiments")
        print("\nActual coverage:")
        
        # Pivot for better display
        coverage_pivot = coverage.pivot_table(
            index=['scenario', 'memory_strategy'], 
            columns='visibility_level', 
            values='count',
            fill_value=0
        )
        print(coverage_pivot)
        
        return True
        
    except Exception as e:
        print(f"❌ Fatal error during merge: {e}")
        return False
    finally:
        if complete_conn:
            complete_conn.close()
        # Final cleanup
        gc.collect()

def main():
    """Main function to run the merge process"""
    print("🚀 SCM-ARENA MISSING EXPERIMENTS MERGER")
    print("This will merge your in-progress missing experiments")
    print("with your existing merged database\n")
    
    # First analyze what we have
    details, total_missing = analyze_missing_experiments()
    
    if total_missing == 0:
        print("\n❌ No completed experiments found in missing_experiments/")
        print("Wait for some experiments to complete before merging.")
        return
    
    print(f"\n🎯 Ready to merge {total_missing} new experiments")
    response = input("Proceed with merge? (y/n): ")
    
    if response.lower() != 'y':
        print("Merge cancelled.")
        return
    
    # Run the merge
    success = merge_all_databases()
    
    if success:
        print("\n✅ SUCCESS! Your dataset is now more complete.")
        print("\n💡 TIPS:")
        print("1. Run this script again when more experiments complete")
        print("2. The original databases are preserved")
        print("3. Use 'full_factorial_complete.db' for analysis")
        print("4. Check 'full_factorial_complete.csv' for easy loading")

if __name__ == '__main__':
    main()