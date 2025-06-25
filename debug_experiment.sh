#!/bin/bash
# Debug script to identify experiment issues

echo "🔍 DEBUGGING EXPERIMENT ISSUES"
echo "=============================="

# Check the failed batch logs
LOG_DIR="full_factorial_20250622_022845"

if [ ! -d "$LOG_DIR" ]; then
    echo "❌ Log directory not found: $LOG_DIR"
    exit 1
fi

echo "📋 Checking batch logs in: $LOG_DIR"
echo

# Check each batch log
for log_file in "$LOG_DIR"/batch_*.log; do
    if [ -f "$log_file" ]; then
        batch_name=$(basename "$log_file" .log)
        echo "🔍 $batch_name:"
        echo "----------------------------------------"
        
        # Show last 20 lines of log
        tail -20 "$log_file"
        
        echo "----------------------------------------"
        echo
    fi
done

# Test a simple CLI command manually
echo "🧪 TESTING CLI COMMAND MANUALLY:"
echo "--------------------------------"

test_cmd="poetry run python -m scm_arena.cli experiment --models llama3.2 --memory none --prompts specific --visibility local --scenarios classic --game-modes modern --runs 1 --rounds 5 --base-seed 42 --deterministic --save-database --db-path debug_test.db --save-results debug_test.csv"

echo "Command: $test_cmd"
echo

# Run test command interactively
echo "Running test command..."
echo "y" | $test_cmd

exit_code=$?
echo
echo "Exit code: $exit_code"

if [ $exit_code -eq 0 ]; then
    echo "✅ CLI command works!"
    
    if [ -f "debug_test.db" ]; then
        exp_count=$(sqlite3 "debug_test.db" 'SELECT COUNT(*) FROM experiments' 2>/dev/null || echo '0')
        echo "✅ Database created with $exp_count experiments"
        
        # Clean up
        rm -f debug_test.db debug_test.csv
    fi
else
    echo "❌ CLI command failed"
fi

echo
echo "🔍 COMMON ISSUES TO CHECK:"
echo "1. Is Ollama server running? (ollama serve)"
echo "2. Is llama3.2 model available? (ollama list)"
echo "3. Are there any import errors in the CLI?"
echo "4. Is the seeding system properly integrated?"
echo

echo "🛠️  SUGGESTED FIXES:"
echo "1. Check Ollama status: curl -s http://localhost:11434/api/tags"
echo "2. Test CLI help: poetry run python -m scm_arena.cli experiment --help"
echo "3. Test seeding: poetry run python test_seeding_fix.py"
echo "4. Check for Python errors in the logs above"