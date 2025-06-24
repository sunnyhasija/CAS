#!/bin/bash
# Fixed Full Factorial SCM-Arena Experiment - Concurrent Execution
# FIXES: Array handling, error checking, better debugging

echo "🚀 FULL FACTORIAL SCM-ARENA EXPERIMENT - FIXED VERSION"
echo "======================================================"

# Configuration
BASE_SEED=42
TOTAL_RUNS=20
TOTAL_ROUNDS=52
MAX_CONCURRENT_JOBS=4  # Reduced for stability

# Create output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="full_factorial_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "📁 Output directory: $OUTPUT_DIR"

# Quick test first - let's run a single small experiment
echo
echo "🧪 RUNNING QUICK TEST FIRST..."
echo "=============================="

test_cmd="poetry run python -m scm_arena.cli experiment --models llama3.2 --memory none --prompts specific --visibility local --scenarios classic --game-modes modern --runs 1 --rounds 5 --base-seed 42 --deterministic --save-database --db-path ${OUTPUT_DIR}/test.db --save-results ${OUTPUT_DIR}/test.csv"

echo "Test command: $test_cmd"
echo

# Run test
echo "y" | $test_cmd > "${OUTPUT_DIR}/test.log" 2>&1
test_exit_code=$?

if [ $test_exit_code -eq 0 ]; then
    echo "✅ Quick test PASSED - CLI working correctly"
    
    # Check if database was created
    if [ -f "${OUTPUT_DIR}/test.db" ]; then
        test_count=$(sqlite3 "${OUTPUT_DIR}/test.db" 'SELECT COUNT(*) FROM experiments' 2>/dev/null || echo '0')
        echo "✅ Test database created with $test_count experiments"
    else
        echo "⚠️  Test database not found"
    fi
else
    echo "❌ Quick test FAILED (exit code: $test_exit_code)"
    echo "📋 Check the test log:"
    echo "----------------------------------------"
    cat "${OUTPUT_DIR}/test.log"
    echo "----------------------------------------"
    echo
    echo "🛑 STOPPING - Fix the CLI issues before running full experiment"
    exit 1
fi

echo
read -p "✅ Test passed! Continue with full experiment? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Full experiment cancelled"
    exit 0
fi

echo
echo "🚀 STARTING FULL EXPERIMENT..."
echo "============================="

# Fixed function to run experiment batches
run_experiment_batch() {
    local batch_name="$1"
    local memory_strategies="$2"
    local visibility_levels="$3" 
    local scenarios="$4"
    local batch_id="$5"
    
    echo "🚀 Starting Batch $batch_id: $batch_name" >&2
    
    local db_file="${OUTPUT_DIR}/batch_${batch_id}_${batch_name// /_}.db"
    local csv_file="${OUTPUT_DIR}/batch_${batch_id}_${batch_name// /_}.csv"
    local log_file="${OUTPUT_DIR}/batch_${batch_id}_${batch_name// /_}.log"
    
    # Build the command more carefully
    local cmd="poetry run python -m scm_arena.cli experiment"
    cmd="$cmd --models llama3.2"
    
    # Add memory strategies
    for mem in $memory_strategies; do
        cmd="$cmd --memory $mem"
    done
    
    # Add prompts (always both)
    cmd="$cmd --prompts specific --prompts neutral"
    
    # Add visibility levels
    for vis in $visibility_levels; do
        cmd="$cmd --visibility $vis"
    done
    
    # Add scenarios
    for scenario in $scenarios; do
        cmd="$cmd --scenarios $scenario"
    done
    
    # Add game modes (always both)
    cmd="$cmd --game-modes modern --game-modes classic"
    
    # Add other parameters
    cmd="$cmd --runs $TOTAL_RUNS"
    cmd="$cmd --rounds $TOTAL_ROUNDS"
    cmd="$cmd --base-seed $BASE_SEED"
    cmd="$cmd --deterministic"
    cmd="$cmd --save-database"
    cmd="$cmd --db-path $db_file"
    cmd="$cmd --save-results $csv_file"
    
    # Log the command
    {
        echo "Batch: $batch_name"
        echo "Command: $cmd"
        echo "Started: $(date)"
        echo "----------------------------------------"
    } > "$log_file"
    
    # Run the experiment with proper input handling
    echo "y" | $cmd >> "$log_file" 2>&1
    local exit_code=$?
    
    # Log completion
    {
        echo "----------------------------------------"
        echo "Finished: $(date)"
        echo "Exit code: $exit_code"
    } >> "$log_file"
    
    if [ $exit_code -eq 0 ]; then
        # Count experiments in database
        local exp_count=0
        if [ -f "$db_file" ]; then
            exp_count=$(sqlite3 "$db_file" 'SELECT COUNT(*) FROM experiments' 2>/dev/null || echo '0')
        fi
        echo "✅ Batch $batch_id completed: $exp_count experiments" >&2
    else
        echo "❌ Batch $batch_id failed (exit code: $exit_code)" >&2
    fi
    
    return $exit_code
}

# Export function and variables
export -f run_experiment_batch
export OUTPUT_DIR TOTAL_RUNS TOTAL_ROUNDS BASE_SEED

# Simplified batch definitions (fewer batches for testing)
declare -a batches=(
    "Memory_None|none|local adjacent full|classic random|1"
    "Memory_Short|short|local adjacent full|classic random|2"
    "Memory_Full|full|local adjacent full|classic random|3"
    "Validation|none short full|local full|shock seasonal|4"
)

# Track running jobs with fixed array handling
declare -a job_pids=()
declare -a job_names=()

echo "Starting ${#batches[@]} experiment batches..."
echo

# Start batches with improved job management
for batch_spec in "${batches[@]}"; do
    # Parse batch specification
    IFS='|' read -r batch_name memory_strategies visibility_levels scenarios batch_id <<< "$batch_spec"
    
    # Wait if we've hit the concurrency limit
    while [ ${#job_pids[@]} -ge $MAX_CONCURRENT_JOBS ]; do
        echo "⏳ $(date '+%H:%M:%S') - Waiting for job slot (${#job_pids[@]}/$MAX_CONCURRENT_JOBS running)..."
        
        # Check for completed jobs
        new_pids=()
        new_names=()
        
        for i in "${!job_pids[@]}"; do
            if kill -0 "${job_pids[$i]}" 2>/dev/null; then
                # Job still running
                new_pids+=("${job_pids[$i]}")
                new_names+=("${job_names[$i]}")
            else
                # Job completed
                echo "✅ Job ${job_pids[$i]} (${job_names[$i]}) completed"
            fi
        done
        
        # Update arrays
        job_pids=("${new_pids[@]}")
        job_names=("${new_names[@]}")
        
        sleep 10
    done
    
    # Start the batch
    echo "▶️  Starting Batch $batch_id: $batch_name"
    run_experiment_batch "$batch_name" "$memory_strategies" "$visibility_levels" "$scenarios" "$batch_id" &
    
    # Add to tracking arrays
    job_pids+=($!)
    job_names+=("$batch_name")
    
    echo "   Job PID: $! (${#job_pids[@]}/$MAX_CONCURRENT_JOBS slots used)"
    
    # Brief delay to stagger starts
    sleep 5
done

echo
echo "🔄 ALL BATCHES STARTED - WAITING FOR COMPLETION"
echo "==============================================="

# Wait for all jobs to complete
echo "Waiting for jobs: ${job_pids[*]}"

for i in "${!job_pids[@]}"; do
    echo "⏳ Waiting for ${job_names[$i]} (PID: ${job_pids[$i]})..."
    wait "${job_pids[$i]}"
    echo "✅ ${job_names[$i]} finished"
done

echo
echo "🎉 ALL BATCHES COMPLETED!"
echo "========================"

# Results summary
echo "📊 RESULTS SUMMARY:"
echo "------------------"

total_experiments=0
successful_batches=0
failed_batches=0

for db_file in "$OUTPUT_DIR"/batch_*.db; do
    if [ -f "$db_file" ]; then
        exp_count=$(sqlite3 "$db_file" 'SELECT COUNT(*) FROM experiments' 2>/dev/null || echo '0')
        batch_name=$(basename "$db_file" .db)
        
        if [ "$exp_count" -gt 0 ]; then
            echo "✅ $batch_name: $exp_count experiments"
            total_experiments=$((total_experiments + exp_count))
            successful_batches=$((successful_batches + 1))
        else
            echo "❌ $batch_name: 0 experiments (failed)"
            failed_batches=$((failed_batches + 1))
        fi
    fi
done

echo
echo "📈 SUMMARY:"
echo "• Total experiments: $total_experiments"
echo "• Successful batches: $successful_batches"
echo "• Failed batches: $failed_batches"

if [ $failed_batches -gt 0 ]; then
    echo
    echo "🔍 CHECK FAILED BATCH LOGS:"
    for log_file in "$OUTPUT_DIR"/batch_*.log; do
        if [ -f "$log_file" ]; then
            batch_name=$(basename "$log_file" .log)
            db_file="${log_file%.log}.db"
            if [ ! -f "$db_file" ] || [ "$(sqlite3 "$db_file" 'SELECT COUNT(*) FROM experiments' 2>/dev/null || echo '0')" -eq 0 ]; then
                echo "❌ $batch_name - check $log_file"
            fi
        fi
    done
fi

echo
echo "📁 All results in: $OUTPUT_DIR/"
echo "🎯 Next: Analyze logs and fix any issues before running full-scale experiment"