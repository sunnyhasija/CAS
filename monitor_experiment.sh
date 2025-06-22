#!/bin/bash
# Fixed Real-time SCM-Arena Experiment Monitor
# Shows running speed and progress updates for your actual directory structure

echo "🔍 SCM-Arena Experiment Monitor (FIXED)"
echo "======================================="

# Function to get current stats
get_stats() {
    local timestamp=$(date '+%H:%M:%S')
    local total_experiments=0
    local total_rounds=0
    local total_interactions=0
    local completed_experiments=0
    
    echo "[$timestamp] Current Status:"
    
    # Check for the actual directory structure
    for dir in full_factorial_*/; do
        if [ -d "$dir" ]; then
            echo "  Checking directory: $dir"
            
            for db in "$dir"batch_*.db; do
                if [ -f "$db" ]; then
                    local batch_name=$(basename "$db" .db)
                    local experiments=$(sqlite3 "$db" 'SELECT COUNT(*) FROM experiments' 2>/dev/null || echo '0')
                    local rounds=$(sqlite3 "$db" 'SELECT COUNT(*) FROM rounds' 2>/dev/null || echo '0')
                    local interactions=$(sqlite3 "$db" 'SELECT COUNT(*) FROM agent_rounds' 2>/dev/null || echo '0')
                    
                    # Get completed experiments (those with rounds)
                    local completed=$(sqlite3 "$db" 'SELECT COUNT(DISTINCT experiment_id) FROM rounds' 2>/dev/null || echo '0')
                    
                    local size=$(ls -l "$db" | awk '{print $5}')
                    local size_mb=$(echo "scale=1; $size / 1024 / 1024" | bc -l 2>/dev/null || echo "0")
                    
                    echo "  $batch_name: $experiments total ($completed completed), $rounds rounds, ${size_mb}MB"
                    
                    total_experiments=$((total_experiments + experiments))
                    total_rounds=$((total_rounds + rounds))
                    total_interactions=$((total_interactions + interactions))
                    completed_experiments=$((completed_experiments + completed))
                fi
            done
        fi
    done
    
    # Also check current directory for any batch files
    for db in batch_*.db; do
        if [ -f "$db" ]; then
            local batch_name=$(basename "$db" .db)
            local experiments=$(sqlite3 "$db" 'SELECT COUNT(*) FROM experiments' 2>/dev/null || echo '0')
            local rounds=$(sqlite3 "$db" 'SELECT COUNT(*) FROM rounds' 2>/dev/null || echo '0')
            local interactions=$(sqlite3 "$db" 'SELECT COUNT(*) FROM agent_rounds' 2>/dev/null || echo '0')
            local completed=$(sqlite3 "$db" 'SELECT COUNT(DISTINCT experiment_id) FROM rounds' 2>/dev/null || echo '0')
            local size=$(ls -l "$db" | awk '{print $5}')
            local size_mb=$(echo "scale=1; $size / 1024 / 1024" | bc -l 2>/dev/null || echo "0")
            
            echo "  $batch_name (current dir): $experiments total ($completed completed), $rounds rounds, ${size_mb}MB"
            
            total_experiments=$((total_experiments + experiments))
            total_rounds=$((total_rounds + rounds))
            total_interactions=$((total_interactions + interactions))
            completed_experiments=$((completed_experiments + completed))
        fi
    done
    
    echo "  TOTAL: $total_experiments experiments ($completed_experiments completed), $total_rounds rounds, $total_interactions interactions"
    
    # Calculate progress based on completed experiments
    local progress=$(echo "scale=2; $completed_experiments * 100 / 2880" | bc -l 2>/dev/null || echo "0")
    local queued=$((total_experiments - completed_experiments))
    
    echo "  Progress: $progress% complete ($queued experiments queued/in-progress)"
    echo
}

# Function to calculate and display rates
monitor_with_rates() {
    echo "🚀 Starting real-time monitoring (Ctrl+C to stop)..."
    echo "Updates every 2 minutes with rate calculations"
    echo
    
    # Get initial state
    local prev_time=$(date +%s)
    local prev_total_exp=0
    local prev_total_rounds=0
    local prev_completed=0
    
    # Calculate initial totals
    for dir in full_factorial_*/; do
        if [ -d "$dir" ]; then
            for db in "$dir"batch_*.db; do
                if [ -f "$db" ]; then
                    local exp=$(sqlite3 "$db" 'SELECT COUNT(*) FROM experiments' 2>/dev/null || echo '0')
                    local rounds=$(sqlite3 "$db" 'SELECT COUNT(*) FROM rounds' 2>/dev/null || echo '0')
                    local completed=$(sqlite3 "$db" 'SELECT COUNT(DISTINCT experiment_id) FROM rounds' 2>/dev/null || echo '0')
                    prev_total_exp=$((prev_total_exp + exp))
                    prev_total_rounds=$((prev_total_rounds + rounds))
                    prev_completed=$((prev_completed + completed))
                fi
            done
        fi
    done
    
    # Also check current directory
    for db in batch_*.db; do
        if [ -f "$db" ]; then
            local exp=$(sqlite3 "$db" 'SELECT COUNT(*) FROM experiments' 2>/dev/null || echo '0')
            local rounds=$(sqlite3 "$db" 'SELECT COUNT(*) FROM rounds' 2>/dev/null || echo '0')
            local completed=$(sqlite3 "$db" 'SELECT COUNT(DISTINCT experiment_id) FROM rounds' 2>/dev/null || echo '0')
            prev_total_exp=$((prev_total_exp + exp))
            prev_total_rounds=$((prev_total_rounds + rounds))
            prev_completed=$((prev_completed + completed))
        fi
    done
    
    while true; do
        get_stats
        
        # Calculate current totals
        local curr_time=$(date +%s)
        local curr_total_exp=0
        local curr_total_rounds=0
        local curr_completed=0
        
        for dir in full_factorial_*/; do
            if [ -d "$dir" ]; then
                for db in "$dir"batch_*.db; do
                    if [ -f "$db" ]; then
                        local exp=$(sqlite3 "$db" 'SELECT COUNT(*) FROM experiments' 2>/dev/null || echo '0')
                        local rounds=$(sqlite3 "$db" 'SELECT COUNT(*) FROM rounds' 2>/dev/null || echo '0')
                        local completed=$(sqlite3 "$db" 'SELECT COUNT(DISTINCT experiment_id) FROM rounds' 2>/dev/null || echo '0')
                        curr_total_exp=$((curr_total_exp + exp))
                        curr_total_rounds=$((curr_total_rounds + rounds))
                        curr_completed=$((curr_completed + completed))
                    fi
                done
            fi
        done
        
        for db in batch_*.db; do
            if [ -f "$db" ]; then
                local exp=$(sqlite3 "$db" 'SELECT COUNT(*) FROM experiments' 2>/dev/null || echo '0')
                local rounds=$(sqlite3 "$db" 'SELECT COUNT(*) FROM rounds' 2>/dev/null || echo '0')
                local completed=$(sqlite3 "$db" 'SELECT COUNT(DISTINCT experiment_id) FROM rounds' 2>/dev/null || echo '0')
                curr_total_exp=$((curr_total_exp + exp))
                curr_total_rounds=$((curr_total_rounds + rounds))
                curr_completed=$((curr_completed + completed))
            fi
        done
        
        # Calculate rates
        local time_diff=$((curr_time - prev_time))
        if [ $time_diff -gt 0 ]; then
            local exp_diff=$((curr_completed - prev_completed))
            local rounds_diff=$((curr_total_rounds - prev_total_rounds))
            
            local exp_rate=$(echo "scale=2; $exp_diff * 3600 / $time_diff" | bc -l 2>/dev/null || echo "0")
            local rounds_rate=$(echo "scale=1; $rounds_diff * 3600 / $time_diff" | bc -l 2>/dev/null || echo "0")
            
            echo "📈 RATES (last $(echo "scale=1; $time_diff / 60" | bc -l) min):"
            echo "  Completed Experiments: $exp_rate/hour"
            echo "  Rounds: $rounds_rate/hour"
            
            # ETA calculation based on completed experiments
            local remaining_exp=$((2880 - curr_completed))
            if (( $(echo "$exp_rate > 0" | bc -l) )); then
                local eta_hours=$(echo "scale=1; $remaining_exp / $exp_rate" | bc -l)
                local eta_days=$(echo "scale=1; $eta_hours / 24" | bc -l)
                echo "  ETA: ${eta_hours}h (${eta_days} days)"
            fi
            
            echo "  Token processing: ~$(echo "scale=0; $rounds_rate * 4 * 783 / 1000" | bc -l)k tokens/hour"
        fi
        
        echo "----------------------------------------"
        
        # Update for next iteration
        prev_time=$curr_time
        prev_total_exp=$curr_total_exp
        prev_total_rounds=$curr_total_rounds
        prev_completed=$curr_completed
        
        sleep 120  # Wait 2 minutes
    done
}

# Function for one-time snapshot with detailed breakdown
detailed_snapshot() {
    echo "📊 DETAILED EXPERIMENT SNAPSHOT"
    echo "==============================="
    
    get_stats
    
    echo "📋 BATCH BREAKDOWN:"
    for dir in full_factorial_*/; do
        if [ -d "$dir" ]; then
            echo "  Directory: $dir"
            for db in "$dir"batch_*.db; do
                if [ -f "$db" ]; then
                    local batch_name=$(basename "$db" .db)
                    
                    echo "    $batch_name:"
                    echo "      Total Experiments: $(sqlite3 "$db" 'SELECT COUNT(*) FROM experiments' 2>/dev/null || echo '0')"
                    echo "      Completed Experiments: $(sqlite3 "$db" 'SELECT COUNT(DISTINCT experiment_id) FROM rounds' 2>/dev/null || echo '0')"
                    echo "      Rounds: $(sqlite3 "$db" 'SELECT COUNT(*) FROM rounds' 2>/dev/null || echo '0')"
                    echo "      Agent interactions: $(sqlite3 "$db" 'SELECT COUNT(*) FROM agent_rounds' 2>/dev/null || echo '0')"
                    echo "      Game states: $(sqlite3 "$db" 'SELECT COUNT(*) FROM game_states' 2>/dev/null || echo '0')"
                    
                    # Get latest experiment info
                    local latest=$(sqlite3 "$db" "SELECT model_name, memory_strategy, prompt_type, run_number FROM experiments ORDER BY rowid DESC LIMIT 1" 2>/dev/null || echo "none|none|none|0")
                    IFS='|' read -r model memory prompt run <<< "$latest"
                    if [ "$model" != "none" ]; then
                        echo "      Latest: $model-$memory-$prompt run $run"
                    fi
                    
                    # Check for incomplete experiments
                    local incomplete=$(sqlite3 "$db" "SELECT COUNT(*) FROM experiments e WHERE e.experiment_id NOT IN (SELECT DISTINCT experiment_id FROM rounds)" 2>/dev/null || echo '0')
                    echo "      Queued/In-progress: $incomplete"
                    echo
                fi
            done
        fi
    done
    
    # Check current directory too
    echo "  Current Directory:"
    for db in batch_*.db; do
        if [ -f "$db" ]; then
            local batch_name=$(basename "$db" .db)
            echo "    $batch_name:"
            echo "      Total Experiments: $(sqlite3 "$db" 'SELECT COUNT(*) FROM experiments' 2>/dev/null || echo '0')"
            echo "      Completed Experiments: $(sqlite3 "$db" 'SELECT COUNT(DISTINCT experiment_id) FROM rounds' 2>/dev/null || echo '0')"
            echo "      Rounds: $(sqlite3 "$db" 'SELECT COUNT(*) FROM rounds' 2>/dev/null || echo '0')"
        fi
    done
    
    # Check if processes are still running
    echo "🔄 PROCESS STATUS:"
    local running_procs=$(ps aux | grep "scm_arena.cli experiment" | grep -v grep | wc -l)
    echo "  Active experiment processes: $running_procs"
    
    if [ $running_procs -gt 0 ]; then
        echo "  ✅ Experiment is running"
        echo "  Running processes:"
        ps aux | grep "scm_arena.cli experiment" | grep -v grep | while read line; do
            echo "    $line"
        done
    else
        echo "  ⚠️  No active experiment processes found"
    fi
}

# Function for quick check
quick_check() {
    local total_exp=0
    local total_rounds=0
    local completed_exp=0
    
    for dir in full_factorial_*/; do
        if [ -d "$dir" ]; then
            for db in "$dir"batch_*.db; do
                if [ -f "$db" ]; then
                    local exp=$(sqlite3 "$db" 'SELECT COUNT(*) FROM experiments' 2>/dev/null || echo '0')
                    local rounds=$(sqlite3 "$db" 'SELECT COUNT(*) FROM rounds' 2>/dev/null || echo '0')
                    local completed=$(sqlite3 "$db" 'SELECT COUNT(DISTINCT experiment_id) FROM rounds' 2>/dev/null || echo '0')
                    total_exp=$((total_exp + exp))
                    total_rounds=$((total_rounds + rounds))
                    completed_exp=$((completed_exp + completed))
                fi
            done
        fi
    done
    
    for db in batch_*.db; do
        if [ -f "$db" ]; then
            local exp=$(sqlite3 "$db" 'SELECT COUNT(*) FROM experiments' 2>/dev/null || echo '0')
            local rounds=$(sqlite3 "$db" 'SELECT COUNT(*) FROM rounds' 2>/dev/null || echo '0')
            local completed=$(sqlite3 "$db" 'SELECT COUNT(DISTINCT experiment_id) FROM rounds' 2>/dev/null || echo '0')
            total_exp=$((total_exp + exp))
            total_rounds=$((total_rounds + rounds))
            completed_exp=$((completed_exp + completed))
        fi
    done
    
    local progress=$(echo "scale=2; $completed_exp * 100 / 2880" | bc -l 2>/dev/null || echo "0")
    local queued=$((total_exp - completed_exp))
    echo "$(date '+%H:%M:%S') - $total_exp total experiments ($completed_exp completed, $queued queued), $total_rounds rounds ($progress% complete)"
}

# Main menu
echo "Choose monitoring option:"
echo "1. Real-time monitor with rates (updates every 2 min)"
echo "2. Detailed snapshot (one-time)"
echo "3. Quick check (one-time)"
echo "4. Continuous quick checks (every 30 sec)"
echo
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        monitor_with_rates
        ;;
    2)
        detailed_snapshot
        ;;
    3)
        quick_check
        ;;
    4)
        echo "🔄 Quick monitoring (Ctrl+C to stop)..."
        while true; do
            quick_check
            sleep 30
        done
        ;;
    *)
        echo "Invalid choice. Running quick check..."
        quick_check
        ;;
esac