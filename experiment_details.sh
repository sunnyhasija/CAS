#!/bin/bash
# Show latest experiment details with settings and costs

echo "🔍 LATEST EXPERIMENT DETAILS"
echo "============================="

show_latest_experiment() {
    local db_file="$1"
    local batch_name=$(basename "$db_file" .db)
    
    echo "📊 BATCH: $batch_name"
    echo "------------------------"
    
    # Get the most recent completed experiment
    local latest_completed=$(sqlite3 "$db_file" "
    SELECT e.experiment_id, e.model_name, e.memory_strategy, e.prompt_type, 
           e.visibility_level, e.scenario, e.game_mode, e.run_number,
           e.total_cost, e.service_level, e.bullwhip_ratio, e.timestamp,
           e.temperature, e.top_p, e.seed, e.deterministic_seeding,
           COUNT(r.round_number) as rounds_completed
    FROM experiments e
    LEFT JOIN rounds r ON e.experiment_id = r.experiment_id
    WHERE e.experiment_id IN (SELECT DISTINCT experiment_id FROM rounds)
    GROUP BY e.experiment_id
    ORDER BY e.timestamp DESC
    LIMIT 1;" 2>/dev/null)
    
    if [ -n "$latest_completed" ]; then
        echo "$latest_completed" | while IFS='|' read -r exp_id model memory prompt visibility scenario game_mode run total_cost service bullwhip timestamp temp top_p seed deterministic rounds; do
            echo "🎯 Latest COMPLETED Experiment:"
            echo "  ID: ${exp_id:0:8}..."
            echo "  Settings:"
            echo "    Model: $model"
            echo "    Memory: $memory"
            echo "    Prompt: $prompt" 
            echo "    Visibility: $visibility"
            echo "    Scenario: $scenario"
            echo "    Game Mode: $game_mode"
            echo "    Run: #$run"
            echo "  LLM Config:"
            echo "    Temperature: $temp"
            echo "    Top-P: $top_p"
            echo "    Seed: $seed"
            echo "    Deterministic: $deterministic"
            echo "  Results:"
            echo "    Total Cost: \$$total_cost"
            echo "    Service Level: $(echo "scale=1; $service * 100" | bc)%"
            echo "    Bullwhip Ratio: $bullwhip"
            echo "    Rounds: $rounds/50"
            echo "  Timestamp: $timestamp"
        done
    else
        echo "  ❌ No completed experiments found"
    fi
    
    echo ""
    
    # Get the most recent started experiment (may be in progress)
    local latest_started=$(sqlite3 "$db_file" "
    SELECT experiment_id, model_name, memory_strategy, prompt_type, 
           visibility_level, scenario, game_mode, run_number,
           total_cost, timestamp, temperature, seed
    FROM experiments 
    ORDER BY timestamp DESC 
    LIMIT 1;" 2>/dev/null)
    
    if [ -n "$latest_started" ]; then
        echo "$latest_started" | while IFS='|' read -r exp_id model memory prompt visibility scenario game_mode run total_cost timestamp temp seed; do
            echo "🚀 Latest STARTED Experiment:"
            echo "  ID: ${exp_id:0:8}..."
            echo "  Settings: $model-$memory-$prompt-$visibility-$scenario-$game_mode Run#$run"
            
            # Check if this experiment has any rounds (is it progressing?)
            local round_count=$(sqlite3 "$db_file" "SELECT COUNT(*) FROM rounds WHERE experiment_id = '$exp_id';" 2>/dev/null || echo "0")
            
            if [ "$round_count" -gt 0 ]; then
                echo "  Status: ✅ In Progress ($round_count rounds completed)"
                
                # Get latest round data
                local latest_round=$(sqlite3 "$db_file" "
                SELECT round_number, total_system_cost, customer_demand 
                FROM rounds 
                WHERE experiment_id = '$exp_id' 
                ORDER BY round_number DESC 
                LIMIT 1;" 2>/dev/null)
                
                if [ -n "$latest_round" ]; then
                    echo "$latest_round" | while IFS='|' read -r round_num cost demand; do
                        echo "  Latest Round: #$round_num, Cost: \$$cost, Demand: $demand"
                    done
                fi
            else
                echo "  Status: ⏳ Queued/Starting (no rounds yet)"
            fi
            
            echo "  Started: $timestamp"
        done
    fi
    
    echo ""
}

# Function to show latest from all batches
show_all_latest() {
    echo "🔍 LATEST EXPERIMENTS FROM ALL BATCHES"
    echo "======================================"
    echo ""
    
    # Check timestamped directories
    for dir in full_factorial_*/; do
        if [ -d "$dir" ]; then
            echo "📁 Directory: $dir"
            for db in "$dir"batch_*.db; do
                if [ -f "$db" ]; then
                    show_latest_experiment "$db"
                fi
            done
        fi
    done
    
    # Check current directory
    for db in batch_*.db; do
        if [ -f "$db" ]; then
            echo "📁 Current Directory:"
            show_latest_experiment "$db"
        fi
    done
}

# Function to show top performers
show_top_performers() {
    echo "🏆 TOP PERFORMING EXPERIMENTS"
    echo "=============================="
    echo ""
    
    for dir in full_factorial_*/; do
        if [ -d "$dir" ]; then
            for db in "$dir"batch_*.db; do
                if [ -f "$db" ]; then
                    local batch_name=$(basename "$db" .db)
                    echo "📊 $batch_name - Best Performers:"
                    
                    # Best service level
                    echo "  🎯 Highest Service Level:"
                    sqlite3 "$db" "
                    SELECT '    ' || memory_strategy || '-' || visibility_level || '-' || scenario || 
                           ' Run#' || run_number || ': ' || ROUND(service_level * 100, 1) || '% service, $' || 
                           ROUND(total_cost, 0) || ' cost'
                    FROM experiments 
                    WHERE service_level > 0
                    ORDER BY service_level DESC 
                    LIMIT 1;" 2>/dev/null
                    
                    # Lowest cost
                    echo "  💰 Lowest Cost:"
                    sqlite3 "$db" "
                    SELECT '    ' || memory_strategy || '-' || visibility_level || '-' || scenario || 
                           ' Run#' || run_number || ': $' || ROUND(total_cost, 0) || ' cost, ' || 
                           ROUND(service_level * 100, 1) || '% service'
                    FROM experiments 
                    WHERE total_cost > 0
                    ORDER BY total_cost ASC 
                    LIMIT 1;" 2>/dev/null
                    
                    # Best bullwhip ratio
                    echo "  📈 Best Bullwhip Control:"
                    sqlite3 "$db" "
                    SELECT '    ' || memory_strategy || '-' || visibility_level || '-' || scenario || 
                           ' Run#' || run_number || ': ' || ROUND(bullwhip_ratio, 2) || ' ratio, $' || 
                           ROUND(total_cost, 0) || ' cost'
                    FROM experiments 
                    WHERE bullwhip_ratio > 0
                    ORDER BY bullwhip_ratio ASC 
                    LIMIT 1;" 2>/dev/null
                    
                    echo ""
                fi
            done
        fi
    done
}

# Function to show specific experiment details
show_experiment_details() {
    local search_term="$1"
    echo "🔍 SEARCHING FOR EXPERIMENT: $search_term"
    echo "========================================="
    
    for dir in full_factorial_*/; do
        if [ -d "$dir" ]; then
            for db in "$dir"batch_*.db; do
                if [ -f "$db" ]; then
                    local batch_name=$(basename "$db" .db)
                    
                    # Search by experiment ID prefix or settings
                    local results=$(sqlite3 "$db" "
                    SELECT experiment_id, model_name, memory_strategy, prompt_type, visibility_level, 
                           scenario, game_mode, run_number, total_cost, service_level, bullwhip_ratio
                    FROM experiments 
                    WHERE experiment_id LIKE '$search_term%' 
                       OR (memory_strategy || '-' || visibility_level || '-' || scenario) LIKE '%$search_term%'
                    ORDER BY timestamp DESC;" 2>/dev/null)
                    
                    if [ -n "$results" ]; then
                        echo "📊 Found in $batch_name:"
                        echo "$results" | while IFS='|' read -r exp_id model memory prompt visibility scenario game_mode run cost service bullwhip; do
                            echo "  ${exp_id:0:8}... | $memory-$visibility-$scenario Run#$run | Cost:\$$cost Service:$(echo "scale=1; $service * 100" | bc)% Bullwhip:$bullwhip"
                        done
                        echo ""
                    fi
                fi
            done
        fi
    done
}

# Main menu
echo "Choose option:"
echo "1. Show latest experiments from all batches"
echo "2. Show top performers across all batches" 
echo "3. Search for specific experiment"
echo "4. Show latest from specific batch"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        show_all_latest
        ;;
    2)
        show_top_performers
        ;;
    3)
        read -p "Enter search term (experiment ID prefix or setting like 'full-local'): " search_term
        show_experiment_details "$search_term"
        ;;
    4)
        echo "Available batches:"
        ls -1 full_factorial_*/batch_*.db batch_*.db 2>/dev/null | head -10
        read -p "Enter database filename: " db_file
        if [ -f "$db_file" ]; then
            show_latest_experiment "$db_file"
        elif [ -f "full_factorial_*/$db_file" ]; then
            show_latest_experiment "full_factorial_*/$db_file"
        else
            echo "Database file not found"
        fi
        ;;
    *)
        echo "Invalid choice. Showing latest from all batches..."
        show_all_latest
        ;;
esac