#!/usr/bin/env python3
"""
Database Merger Script
Merges multiple SCM-Arena databases while handling overlaps and conflicts
"""

import sqlite3
import pandas as pd
import os
from datetime import datetime

def merge_databases(primary_db_path: str, secondary_db_path: str, output_db_path: str = None):
    """
    Merge two SCM-Arena databases.
    
    Args:
        primary_db_path: Main database (e.g., full_factorial_complete.db.backup)
        secondary_db_path: Database to merge in (e.g., baseline_full_factorial.db)
        output_db_path: Output merged database
    """
    
    if output_db_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_db_path = f"merged_database_{timestamp}.db"
    
    print("🔗 Database Merger Script")
    print("=" * 35)
    print(f"📁 Primary DB:   {primary_db_path}")
    print(f"📁 Secondary DB: {secondary_db_path}")
    print(f"📁 Output DB:    {output_db_path}")
    
    # Check input files exist
    for db_path in [primary_db_path, secondary_db_path]:
        if not os.path.exists(db_path):
            print(f"❌ Database not found: {db_path}")
            return None
    
    # Connect to databases
    primary_conn = sqlite3.connect(primary_db_path)
    secondary_conn = sqlite3.connect(secondary_db_path)
    
    try:
        # Load experiments from both databases
        print("\n📊 Analyzing databases...")
        
        primary_experiments = pd.read_sql_query("SELECT * FROM experiments", primary_conn)
        secondary_experiments = pd.read_sql_query("SELECT * FROM experiments", secondary_conn)
        
        print(f"   Primary database: {len(primary_experiments)} experiments")
        print(f"   Secondary database: {len(secondary_experiments)} experiments")
        
        # Show model breakdown
        print(f"\n📈 Primary DB Models:")
        primary_models = primary_experiments['model_name'].value_counts()
        for model, count in primary_models.items():
            print(f"   {model}: {count}")
        
        print(f"\n📈 Secondary DB Models:")
        secondary_models = secondary_experiments['model_name'].value_counts()
        for model, count in secondary_models.items():
            print(f"   {model}: {count}")
        
        # Check for experiment_id overlaps
        primary_ids = set(primary_experiments['experiment_id'])
        secondary_ids = set(secondary_experiments['experiment_id'])
        overlapping_ids = primary_ids.intersection(secondary_ids)
        
        print(f"\n🔍 Overlap Analysis:")
        print(f"   Overlapping experiment IDs: {len(overlapping_ids)}")
        
        if len(overlapping_ids) > 0:
            print(f"   First few overlaps: {list(overlapping_ids)[:5]}")
            
            # Check if overlapping experiments are actually the same
            overlap_primary = primary_experiments[primary_experiments['experiment_id'].isin(overlapping_ids)]
            overlap_secondary = secondary_experiments[secondary_experiments['experiment_id'].isin(overlapping_ids)]
            
            # Simple check: same model names?
            if len(overlap_primary) > 0 and len(overlap_secondary) > 0:
                primary_models_overlap = set(overlap_primary['model_name'])
                secondary_models_overlap = set(overlap_secondary['model_name'])
                print(f"   Primary overlap models: {primary_models_overlap}")
                print(f"   Secondary overlap models: {secondary_models_overlap}")
        
        # Strategy: Keep primary database as base, add non-overlapping from secondary
        print(f"\n🔄 Merging strategy:")
        print(f"   1. Keep ALL primary database experiments")
        print(f"   2. Add secondary experiments with unique experiment_ids")
        print(f"   3. For overlaps, keep primary version")
        
        # Filter secondary to only non-overlapping experiments
        secondary_unique = secondary_experiments[~secondary_experiments['experiment_id'].isin(primary_ids)]
        print(f"   Secondary unique experiments to add: {len(secondary_unique)}")
        
        # Combine experiments
        merged_experiments = pd.concat([primary_experiments, secondary_unique], ignore_index=True)
        print(f"   Total merged experiments: {len(merged_experiments)}")
        
        # Show final model breakdown
        print(f"\n📈 Merged DB Models:")
        merged_models = merged_experiments['model_name'].value_counts()
        for model, count in merged_models.items():
            print(f"   {model}: {count}")
        
        # Create output database by copying primary
        print(f"\n🗄️ Creating merged database...")
        import shutil
        shutil.copy2(primary_db_path, output_db_path)
        
        # Connect to output database
        output_conn = sqlite3.connect(output_db_path)
        
        # Get experiment IDs to add
        add_experiment_ids = set(secondary_unique['experiment_id'])
        
        if len(add_experiment_ids) > 0:
            # Add experiments
            secondary_unique.to_sql('experiments', output_conn, if_exists='append', index=False)
            print(f"   ✅ Added {len(secondary_unique)} experiments")
            
            # Add related data for new experiments
            tables_to_merge = ['rounds', 'agent_rounds', 'game_states']
            
            for table in tables_to_merge:
                try:
                    # Check if table exists in secondary database
                    secondary_table_data = pd.read_sql_query(f"SELECT * FROM {table}", secondary_conn)
                    
                    if 'experiment_id' in secondary_table_data.columns:
                        # Filter to only new experiment IDs
                        table_to_add = secondary_table_data[
                            secondary_table_data['experiment_id'].isin(add_experiment_ids)
                        ]
                        
                        if len(table_to_add) > 0:
                            table_to_add.to_sql(table, output_conn, if_exists='append', index=False)
                            print(f"   ✅ Added {len(table_to_add):,} records to {table}")
                        else:
                            print(f"   ⚠️ No records to add to {table}")
                    else:
                        print(f"   ⚠️ {table} has no experiment_id column")
                        
                except Exception as e:
                    print(f"   ❌ Error merging {table}: {e}")
        
        # Commit and close
        output_conn.commit()
        output_conn.close()
        
        print(f"\n🎉 Database merge complete!")
        print(f"📁 Merged database: {output_db_path}")
        print(f"📊 Final stats:")
        print(f"   Total experiments: {len(merged_experiments)}")
        print(f"   From primary: {len(primary_experiments)}")
        print(f"   From secondary: {len(secondary_unique)}")
        
        return output_db_path
        
    except Exception as e:
        print(f"❌ Error during merge: {e}")
        raise
    finally:
        primary_conn.close()
        secondary_conn.close()


def main():
    """Main merge function."""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python database_merger.py <primary_db> <secondary_db> [output_db]")
        print("Example: python database_merger.py full_factorial_complete.db.backup baseline_full_factorial.db merged.db")
        sys.exit(1)
    
    primary_db = sys.argv[1]
    secondary_db = sys.argv[2]
    output_db = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        merged_db_path = merge_databases(primary_db, secondary_db, output_db)
        if merged_db_path:
            print(f"\n🚀 Ready for deduplication!")
            print(f"   Next step: python simple_deduplication.py {merged_db_path}")
    except Exception as e:
        print(f"❌ Merge failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()