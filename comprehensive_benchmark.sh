#!/bin/bash
# Comprehensive SCM-Arena Benchmark Test
# Tests all fixes with complete experimental coverage

echo "🧪 COMPREHENSIVE SCM-ARENA BENCHMARK TEST"
echo "=========================================="
echo
echo "This will run a complete test of all fixes:"
echo "✅ Service level calculation fix"
echo "✅ Bullwhip ratio calculation fix (Lee, Padmanabhan & Whang methodology)"
echo "✅ Memory window consistency fix"
echo "✅ Demand scenario consistency fix"
echo "✅ Complete experimental factor coverage"
echo
echo "📊 EXPERIMENTAL DESIGN:"
echo "  Models: llama3.2"
echo "  Memory: none, short, full (3 strategies)"
echo "  Prompts: specific, neutral (2 types)"
echo "  Visibility: local, adjacent, full (3 levels)"
echo "  Scenarios: classic (1 scenario)"
echo "  Game Modes: modern, classic (2 modes)"
echo "  Runs: 3 per condition"
echo "  Rounds: 20 per game"
echo
echo "  Total conditions: 3×2×3×1×2 = 36"
echo "  Total experiments: 36×3 = 108"
echo "  Estimated time: 3-5 hours"
echo

# Check if Ollama is running
echo "🔍 CHECKING PREREQUISITES:"
echo "-------------------------"

if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "❌ Ollama server not running"
    echo "Please start Ollama first:"
    echo "  ollama serve"
    echo
    exit 1
else
    echo "✅ Ollama server is running"
fi

# Check if llama3.2 model is available
if ! ollama list | grep -q "llama3.2"; then
    echo "❌ llama3.2 model not found"
    echo "Please pull the model first:"
    echo "  ollama pull llama3.2"
    echo
    exit 1
else
    echo "✅ llama3.2 model is available"
fi

echo "✅ Poetry environment ready"
echo

# Confirm before running
read -p "🚀 Ready to run comprehensive benchmark? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Benchmark cancelled"
    exit 0
fi

echo
echo "🏃 STARTING COMPREHENSIVE BENCHMARK..."
echo "======================================"
echo

# Run the comprehensive experiment
poetry run python -m scm_arena.cli experiment \
    --models llama3.2 \
    --memory none --memory short --memory full \
    --prompts specific --prompts neutral \
    --visibility local --visibility adjacent --visibility full \
    --scenarios classic \
    --game-modes modern --game-modes classic \
    --runs 3 \
    --rounds 20 \
    --save-database \
    --save-results comprehensive_benchmark_results.csv \
    --db-path comprehensive_benchmark.db

# Check if experiment completed successfully
if [ $? -eq 0 ]; then
    echo
    echo "🎉 COMPREHENSIVE BENCHMARK COMPLETED!"
    echo "===================================="
    echo
    echo "📊 Results saved to:"
    echo "  📁 comprehensive_benchmark.db (complete audit trail)"
    echo "  📄 comprehensive_benchmark_results.csv (summary statistics)"
    echo
    echo "🔍 NEXT STEPS:"
    echo "1. Run validation tests:"
    echo "   python test_db.py comprehensive_benchmark.db"
    echo
    echo "2. Analyze results:"
    echo "   - Service level range and variation"
    echo "   - Bullwhip ratio patterns by condition"
    echo "   - Memory strategy effectiveness"
    echo "   - Game mode differences (modern vs classic)"
    echo "   - Visibility level impacts"
    echo
    echo "3. Compare to previous results:"
    echo "   - Verify fixes eliminated previous issues"
    echo "   - Confirm all experimental factors represented"
    echo "   - Check for reasonable performance variation"
    echo
else
    echo
    echo "❌ BENCHMARK FAILED"
    echo "==================="
    echo
    echo "Check the output above for errors."
    echo "Common issues:"
    echo "  - Ollama connection problems"
    echo "  - Model not responding"
    echo "  - Disk space insufficient"
    echo "  - Permission issues"
    echo
fi