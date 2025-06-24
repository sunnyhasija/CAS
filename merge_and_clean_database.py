#!/usr/bin/env python3
# merge_and_clean_databases.py
"""
Merge all batch databases into one clean, deduplicated database
Handles the duplication issue from the original runs
"""

import sqlite3
import pandas as pd
import shutil
from datetime import datetime
from pathlib import Path

def analyze_database(db_path):
    """Analyze a database to understand its contents"""
    conn = sqlite3.connect(db_path)
    
    # Get basic counts
    total_experiments = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    completed_experiments = conn.execute("SELECT COUNT(*) FROM experiments WHERE total_cost > 0").fetchone()[0]
    total_rounds = conn.execute("SELECT COUNT(*) FROM rounds").fetchone()[0]
    
    # Get scenario breakdown
    scenario_breakdown = pd.read_sql_query("""
        SELECT scenario, COUNT(*) as count 
        FROM experiments 
        WHERE total_cost > 0 
        GROUP BY scenario
    """, conn)
    
    conn.close()
    
    return {
        'path': db_path,
        'total_experiments': total_experiments,
        'completed_experiments': completed_experiments,
        'incomplete_experiments': total_experiments - completed_experiments,
        'total_rounds': total_rounds,
        'scenarios': scenario_breakdown.to_dict('records')
    }

def merge_databases(output_db_path='full_factorial_merged.db', preserve_originals=True):
    """
    Merge all batch databases into a single clean database
    
    Args:
        output_db_path: Path for the merged database
        preserve_originals: If True, keeps original databases intact (default: True)
    """
    
    print("🔄 MERGING AND CLEANING EXPERIMENT DATABASES")
    print("=" * 60)
    
    if preserve_originals:
        print("✅ Original databases will be preserved for verification")
    
    # Find all batch databases
    batch_dbs = [
        'full_factorial_20250622_025052/batch_1_Memory_None.db',
        'full_factorial_20250622_025052/batch_2_Memory_Short.db',
        'full_factorial_20250622_025052/batch_3_Memory_Full.db',
        'full_factorial_20250622_025052/batch_4_Validation.db'
    ]
    
    # Check which databases exist
    existing_dbs = [db for db in batch_dbs if Path(db).exists()]
    
    if not existing_dbs:
        print("❌ No batch databases found!")
        return
    
    print(f"Found {len(existing_dbs)} databases to merge\n")
    
    # Analyze each database
    print("📊 ANALYZING DATABASES:")
    print("-" * 60)
    
    total_completed = 0
    all_analyses = []
    
    for db_path in existing_dbs:
        analysis = analyze_database(db_path)
        all_analyses.append(analysis)
        
        print(f"\n📁 {Path(db_path).name}")
        print(f"   Total records: {analysis['total_experiments']}")
        print(f"   Completed: {analysis['completed_experiments']}")
        print(f"   Incomplete (duplicates): {analysis['incomplete_experiments']}")
        print(f"   Scenarios: ", end="")
        for scenario in analysis['scenarios']:
            print(f"{scenario['scenario']}({scenario['count']}) ", end="")
        print()
        
        total_completed += analysis['completed_experiments']
    
    print(f"\n📊 TOTAL COMPLETED EXPERIMENTS: {total_completed}")
    print("=" * 60)
    
    # Create merged database
    print(f"\n🔄 Creating merged database: {output_db_path}")
    
    # Handle existing merged database
    if Path(output_db_path).exists():
        backup_path = f"{output_db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(output_db_path, backup_path)
        print(f"   Backed up existing database to: {backup_path}")
        Path(output_db_path).unlink()
    
    # Create new merged database
    merged_conn = sqlite3.connect(output_db_path)
    
    # Create tables in merged database
    first_db_conn = sqlite3.connect(existing_dbs[0])
    
    # Get schema from first database
    schema_queries = first_db_conn.execute("""
        SELECT sql FROM sqlite_master 
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
    """).fetchall()
    
    # Create tables in merged database
    for (sql,) in schema_queries:
        if sql:
            merged_conn.execute(sql)
    
    first_db_conn.close()
    merged_conn.commit()
    
    # Merge data from all databases
    print("\n🔄 Merging data...")
    
    tables_to_merge = ['experiments', 'rounds', 'agent_rounds', 'game_states']
    
    for table in tables_to_merge:
        print(f"\n   Merging table: {table}")
        total_rows = 0
        
        for db_path in existing_dbs:
            conn = sqlite3.connect(db_path)
            
            # Only merge completed experiments
            if table == 'experiments':
                query = f"SELECT * FROM {table} WHERE total_cost > 0"
            else:
                # For other tables, only include rows for completed experiments
                query = f"""
                    SELECT t.* FROM {table} t
                    INNER JOIN experiments e ON t.experiment_id = e.experiment_id
                    WHERE e.total_cost > 0
                """
            
            df = pd.read_sql_query(query, conn)
            
            if len(df) > 0:
                df.to_sql(table, merged_conn, if_exists='append', index=False)
                total_rows += len(df)
                print(f"      Added {len(df)} rows from {Path(db_path).name}")
            
            conn.close()
        
        print(f"      Total rows merged: {total_rows}")
    
    # Create indexes for performance
    print("\n🔧 Creating indexes...")
    merged_conn.execute("CREATE INDEX IF NOT EXISTS idx_exp_scenario ON experiments(scenario)")
    merged_conn.execute("CREATE INDEX IF NOT EXISTS idx_exp_memory ON experiments(memory_strategy)")
    merged_conn.execute("CREATE INDEX IF NOT EXISTS idx_exp_visibility ON experiments(visibility_level)")
    merged_conn.execute("CREATE INDEX IF NOT EXISTS idx_rounds_exp ON rounds(experiment_id)")
    merged_conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_exp ON agent_rounds(experiment_id)")
    
    merged_conn.commit()
    
    # Verify the merge
    print("\n✅ VERIFICATION:")
    print("-" * 60)
    
    # Check for duplicates
    duplicate_check = merged_conn.execute("""
        SELECT COUNT(*) as duplicates FROM (
            SELECT model_name, memory_strategy, prompt_type, visibility_level, 
                   scenario, game_mode, run_number, COUNT(*) as count
            FROM experiments
            GROUP BY model_name, memory_strategy, prompt_type, visibility_level, 
                     scenario, game_mode, run_number
            HAVING count > 1
        )
    """).fetchone()[0]
    
    if duplicate_check == 0:
        print("✅ No duplicate experiments found")
    else:
        print(f"⚠️  Found {duplicate_check} duplicate experiment conditions")
    
    # Final statistics
    final_stats = pd.read_sql_query("""
        SELECT 
            scenario,
            memory_strategy,
            COUNT(*) as experiments,
            AVG(total_cost) as avg_cost,
            AVG(service_level) as avg_service,
            AVG(bullwhip_ratio) as avg_bullwhip
        FROM experiments
        GROUP BY scenario, memory_strategy
        ORDER BY scenario, memory_strategy
    """, merged_conn)
    
    print("\n📊 FINAL MERGED DATABASE STATISTICS:")
    print(final_stats.to_string(index=False))
    
    # Check for incomplete/duplicate entries for reporting
    print("\n📋 VERIFICATION DATA (preserved in original databases):")
    
    for db_path in existing_dbs:
        conn = sqlite3.connect(db_path)
        
        # Check for experiments with seed data
        seed_check = conn.execute("""
            SELECT COUNT(*) FROM experiments 
            WHERE seed IS NOT NULL AND deterministic_seeding IS NOT NULL
        """).fetchone()[0]
        
        # Check for duplicate experiment records
        duplicates = conn.execute("""
            SELECT memory_strategy, visibility_level, scenario, game_mode, 
                   COUNT(*) as count, MIN(total_cost) as min_cost, MAX(total_cost) as max_cost
            FROM experiments
            GROUP BY memory_strategy, visibility_level, scenario, game_mode
            HAVING COUNT(*) > 40
            LIMIT 5
        """).fetchall()
        
        if seed_check > 0 or duplicates:
            print(f"\n   {Path(db_path).name}:")
            if seed_check > 0:
                print(f"      ✓ Contains {seed_check} experiments with seeding data")
            if duplicates:
                print(f"      ✓ Contains duplicate records for verification")
                for dup in duplicates[:2]:  # Show first 2
                    print(f"        - {dup[0]}-{dup[1]}-{dup[2]}-{dup[3]}: {dup[4]} records")
        
        conn.close()
    
    print("\n💡 Original databases contain important verification data:")
    print("   - Seeding information for reproducibility")
    print("   - Duplicate records showing the echo y issue")
    print("   - Complete audit trail of all attempts")
    
    # Final summary
    scenario_totals = pd.read_sql_query("""
        SELECT scenario, COUNT(*) as total_experiments
        FROM experiments
        GROUP BY scenario
        ORDER BY scenario
    """, merged_conn)
    
    print("\n📊 TOTAL EXPERIMENTS BY SCENARIO:")
    for _, row in scenario_totals.iterrows():
        print(f"   {row['scenario']}: {row['total_experiments']} experiments")
    
    total_experiments = merged_conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    print(f"\n🎉 MERGE COMPLETE!")
    print(f"   Total experiments in merged database: {total_experiments}")
    print(f"   Saved to: {output_db_path}")
    
    # Also export to CSV for easy analysis
    csv_path = output_db_path.replace('.db', '.csv')
    experiments_df = pd.read_sql_query("SELECT * FROM experiments ORDER BY scenario, memory_strategy", merged_conn)
    experiments_df.to_csv(csv_path, index=False)
    print(f"   Also exported to CSV: {csv_path}")
    
    merged_conn.close()
    
    return output_db_path

def create_analysis_ready_dataset(merged_db_path='full_factorial_merged.db'):
    """Create a clean dataset ready for analysis"""
    
    print("\n📊 CREATING ANALYSIS-READY DATASET")
    print("=" * 60)
    
    conn = sqlite3.connect(merged_db_path)
    
    # Load experiments
    experiments_df = pd.read_sql_query("SELECT * FROM experiments", conn)
    
    # Add derived columns for easier analysis
    experiments_df['total_cost_per_round'] = experiments_df['total_cost'] / experiments_df['rounds']
    experiments_df['cost_category'] = pd.qcut(experiments_df['total_cost'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
    experiments_df['service_category'] = pd.qcut(experiments_df['service_level'], q=4, labels=['Poor', 'Fair', 'Good', 'Excellent'])
    
    # Save enhanced dataset
    experiments_df.to_csv('experiments_analysis_ready.csv', index=False)
    
    print("✅ Created analysis-ready dataset: experiments_analysis_ready.csv")
    print(f"   Shape: {experiments_df.shape}")
    print(f"   Columns: {', '.join(experiments_df.columns)}")
    
    conn.close()

if __name__ == '__main__':
    print("🚀 SCM-Arena Database Merge Tool")
    print("This tool creates a clean merged database while preserving originals")
    print()
    
    # Run the merge
    merged_db = merge_databases(preserve_originals=True)
    
    if merged_db:
        # Create analysis-ready dataset
        create_analysis_ready_dataset(merged_db)
        
        print("\n✅ SUMMARY:")
        print("1. Original databases: Preserved for verification/reproducibility")
        print("2. Merged database: Contains only completed experiments")
        print("3. Analysis files: Ready for your research")
        
        print("\n🎯 NEXT STEPS:")
        print("1. Run your missing experiments with: python run_missing_safe.py")
        print("2. After those complete, re-run this script to merge everything")
        print("3. Your final dataset will have all 2,880 experiments!")
        print("\n💡 TIP: Keep the original databases for:")
        print("   - Verifying deterministic seeding worked correctly")
        print("   - Understanding the duplication issue")
        print("   - Full audit trail for reproducibility")