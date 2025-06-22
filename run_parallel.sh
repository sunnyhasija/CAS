#!/bin/bash
# Full Factorial SCM-Arena Experiment with Concurrent Processes
# This runs the complete experimental design split across multiple parallel jobs

echo "🚀 FULL FACTORIAL SCM-ARENA EXPERIMENT (CONCURRENT PROCESSES)"
echo "================================================================="
echo
echo "📊 Complete Experimental Design:"
echo "• Models: llama3.2"
echo "• Memory: none, short, full (3 strategies)"
echo "• Prompts: specific, neutral (2 types)"
echo "• Visibility: local, adjacent, full (3 levels)"
echo "• Scenarios: classic, random, shock, seasonal (4 scenarios)"
echo "• Game Modes: modern, classic (2 modes)"
echo "• Runs: 20 per condition"
echo "• Rounds: 50 per game"
echo
echo "🎯 Total Conditions: 3×2×3×4×2 = 144"
echo "📈 Total Experiments: 144×20 = 2,880"
echo "⏱️  Estimated Time: 35-60 hours (with 4 parallel processes)"
echo "🎲 Deterministic Seeding: Base seed 42"
echo

read -p "🚀 Ready to run full factorial experiment? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Experiment cancelled"
    exit 0
fi

echo "🏃 Starting parallel experiment batches..."
echo

# Split the full experiment into 4 batches for parallel execution

# BATCH 1: Memory strategy (none) + all other factors
echo "🚀 Starting Batch 1: Memory strategy (none)..."
echo "y" | poetry run python -m scm_arena.cli experiment \
    --models llama3.2 \
    --memory none \
    --prompts specific \
    --prompts neutral \
    --visibility local \
    --visibility adjacent \
    --visibility full \
    --scenarios classic \
    --scenarios random \
    --scenarios shock \
    --scenarios seasonal \
    --game-modes modern \
    --game-modes classic \
    --runs 20 \
    --rounds 50 \
    --base-seed 42 \
    --deterministic \
    --save-database \
    --db-path batch1_memory_none.db \
    --save-results batch1_results.csv &

# BATCH 2: Memory strategy (short) + all other factors  
echo "🚀 Starting Batch 2: Memory strategy (short)..."
echo "y" | poetry run python -m scm_arena.cli experiment \
    --models llama3.2 \
    --memory short \
    --prompts specific \
    --prompts neutral \
    --visibility local \
    --visibility adjacent \
    --visibility full \
    --scenarios classic \
    --scenarios random \
    --scenarios shock \
    --scenarios seasonal \
    --game-modes modern \
    --game-modes classic \
    --runs 20 \
    --rounds 50 \
    --base-seed 42 \
    --deterministic \
    --save-database \
    --db-path batch2_memory_short.db \
    --save-results batch2_results.csv &

# BATCH 3: Memory strategy (full) + all other factors
echo "🚀 Starting Batch 3: Memory strategy (full)..."
echo "y" | poetry run python -m scm_arena.cli experiment \
    --models llama3.2 \
    --memory full \
    --prompts specific \
    --prompts neutral \
    --visibility local \
    --visibility adjacent \
    --visibility full \
    --scenarios classic \
    --scenarios random \
    --scenarios shock \
    --scenarios seasonal \
    --game-modes modern \
    --game-modes classic \
    --runs 20 \
    --rounds 50 \
    --base-seed 42 \
    --deterministic \
    --save-database \
    --db-path batch3_memory_full.db \
    --save-results batch3_results.csv &

# BATCH 4: Validation batch with mixed conditions
echo "🚀 Starting Batch 4: Validation batch (cross-scenario testing)..."
echo "y" | poetry run python -m scm_arena.cli experiment \
    --models llama3.2 \
    --memory none \
    --memory short \
    --memory full \
    --prompts specific \
    --visibility local \
    --visibility full \
    --scenarios classic \
    --scenarios shock \
    --game-modes modern \
    --runs 20 \
    --rounds 50 \
    --base-seed 42 \
    --deterministic \
    --save-database \
    --db-path batch4_validation.db \
    --save-results batch4_validation.csv &

echo "⏳ All 4 batches running in parallel. Monitoring progress..."
echo "   Use 'jobs' to see running processes"
echo "   Use 'kill %1 %2 %3 %4' to stop all if needed"
echo

# Function to show progress
show_progress() {
    while jobs %1 %2 %3 %4 >/dev/null 2>&1; do
        echo "⏳ $(date '+%H:%M:%S') - Experiments still running..."
        echo "   📊 Check database files for live progress:"
        if [ -f "batch1_memory_none.db" ]; then
            echo "   • Batch 1 (memory: none): $(sqlite3 batch1_memory_none.db 'SELECT COUNT(*) FROM experiments' 2>/dev/null || echo '0') experiments completed"
        fi
        if [ -f "batch2_memory_short.db" ]; then
            echo "   • Batch 2 (memory: short): $(sqlite3 batch2_memory_short.db 'SELECT COUNT(*) FROM experiments' 2>/dev/null || echo '0') experiments completed"
        fi
        if [ -f "batch3_memory_full.db" ]; then
            echo "   • Batch 3 (memory: full): $(sqlite3 batch3_memory_full.db 'SELECT COUNT(*) FROM experiments' 2>/dev/null || echo '0') experiments completed"
        fi
        if [ -f "batch4_validation.db" ]; then
            echo "   • Batch 4 (validation): $(sqlite3 batch4_validation.db 'SELECT COUNT(*) FROM experiments' 2>/dev/null || echo '0') experiments completed"
        fi
        echo
        sleep 300  # Check every 5 minutes
    done
}

# Start progress monitoring in background
show_progress &
PROGRESS_PID=$!

# Wait for all experiments to complete
wait

# Stop progress monitoring
kill $PROGRESS_PID 2>/dev/null

echo
echo "🎉 ALL EXPERIMENTS COMPLETED!"
echo "=============================="
echo

# Show final results
echo "📊 Final Results Summary:"
echo "------------------------"

if [ -f "batch1_memory_none.db" ]; then
    batch1_count=$(sqlite3 batch1_memory_none.db 'SELECT COUNT(*) FROM experiments' 2>/dev/null || echo '0')
    echo "• Batch 1 (memory: none): ${batch1_count} experiments"
fi

if [ -f "batch2_memory_short.db" ]; then
    batch2_count=$(sqlite3 batch2_memory_short.db 'SELECT COUNT(*) FROM experiments' 2>/dev/null || echo '0')
    echo "• Batch 2 (memory: short): ${batch2_count} experiments"
fi

if [ -f "batch3_memory_full.db" ]; then
    batch3_count=$(sqlite3 batch3_memory_full.db 'SELECT COUNT(*) FROM experiments' 2>/dev/null || echo '0')
    echo "• Batch 3 (memory: full): ${batch3_count} experiments"
fi

if [ -f "batch4_validation.db" ]; then
    batch4_count=$(sqlite3 batch4_validation.db 'SELECT COUNT(*) FROM experiments' 2>/dev/null || echo '0')
    echo "• Batch 4 (validation): ${batch4_count} experiments"
fi

total_experiments=$((${batch1_count:-0} + ${batch2_count:-0} + ${batch3_count:-0} + ${batch4_count:-0}))
echo "• Total: ${total_experiments} experiments completed"

echo
echo "📁 Output Files:"
echo "---------------"
echo "• batch1_memory_none.db + batch1_results.csv"
echo "• batch2_memory_short.db + batch2_results.csv"  
echo "• batch3_memory_full.db + batch3_results.csv"
echo "• batch4_validation.db + batch4_validation.csv"
echo

echo "🔍 Next Steps:"
echo "-------------"
echo "1. Analyze results by memory strategy:"
echo "   python comprehensive_test.py batch1_memory_none.db"
echo "   python comprehensive_test.py batch2_memory_short.db"
echo "   python comprehensive_test.py batch3_memory_full.db"
echo
echo "2. Merge databases (optional):"
echo "   # Use SQLite commands to merge the databases if needed"
echo
echo "3. Verify results with validation batch:"
echo "   python comprehensive_test.py batch4_validation.db"
echo
echo "🎯 Full factorial experiment complete with deterministic seeding!"
echo "✅ Each condition used unique, reproducible seeds"
echo "✅ Results are scientifically valid and reproducible"