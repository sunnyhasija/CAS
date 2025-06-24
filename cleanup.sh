#!/bin/bash
# SCM-Arena Project Cleanup Script
# Removes temporary files, test databases, and other artifacts

echo "🧹 SCM-Arena Project Cleanup"
echo "============================"
echo

# Function to safely remove files/directories
safe_remove() {
    if [ -e "$1" ]; then
        echo "🗑️  Removing: $1"
        rm -rf "$1"
    else
        echo "ℹ️  Not found: $1"
    fi
}

echo "📂 CLEANING TEST DATABASES:"
echo "----------------------------"
safe_remove "test_fixed_memory.db"
safe_remove "test_both_modes.db"
safe_remove "test_complete_coverage.db"
safe_remove "scm_arena_experiments.db"
safe_remove "llama32_canonical_baseline.db"

echo
echo "📂 CLEANING TEMPORARY FILES:"
echo "----------------------------"
safe_remove "__pycache__"
safe_remove "src/__pycache__"
safe_remove "src/scm_arena/__pycache__"
safe_remove "src/scm_arena/beer_game/__pycache__"
safe_remove "src/scm_arena/models/__pycache__"
safe_remove "src/scm_arena/evaluation/__pycache__"
safe_remove "src/scm_arena/visualization/__pycache__"
safe_remove ".pytest_cache"
safe_remove "tests/__pycache__"

echo
echo "📂 CLEANING RESULT FILES:"
echo "-------------------------"
safe_remove "*.csv"
safe_remove "*.json"
safe_remove "results_*.csv"
safe_remove "experimental_results_*.csv"
safe_remove "llama32_canonical_baseline.csv"

echo
echo "📂 CLEANING ANALYSIS OUTPUTS:"
echo "-----------------------------"
safe_remove "game_analysis/"
safe_remove "analysis_output/"
safe_remove "plots/"
safe_remove "*.png"
safe_remove "*.pdf"

echo
echo "📂 CLEANING LOG FILES:"
echo "----------------------"
safe_remove "*.log"
safe_remove "debug.log"
safe_remove "experiment.log"

echo
echo "📂 CLEANING DEVELOPMENT FILES:"
echo "------------------------------"
safe_remove ".coverage"
safe_remove "coverage.xml"
safe_remove "htmlcov/"
safe_remove ".mypy_cache"
safe_remove ".tox"

echo
echo "📂 OPTIONAL: REMOVE TESTING SCRIPTS"
echo "-----------------------------------"
echo "The following files can be removed if no longer needed:"
echo "  - test_db.py (our testing script)"
echo "  - comprehensive_test.py (comprehensive testing script)"
echo "  - Any custom test scripts you created"
echo
read -p "Remove testing scripts? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    safe_remove "test_db.py"
    safe_remove "comprehensive_test.py"
    echo "✅ Testing scripts removed"
else
    echo "ℹ️  Testing scripts kept"
fi

echo
echo "📂 FILES TO KEEP:"
echo "-----------------"
echo "✅ src/ - Core source code"
echo "✅ tests/ - Unit tests"
echo "✅ pyproject.toml - Project configuration"
echo "✅ README.md - Documentation"
echo "✅ documentation.txt - Technical docs"
echo "✅ LICENSE - License file"
echo "✅ .gitignore - Git ignore rules"

echo
echo "🎉 CLEANUP COMPLETE!"
echo "==================="
echo
echo "📊 Project is now clean and ready for:"
echo "  1. Full comprehensive testing"
echo "  2. Production benchmark runs"
echo "  3. Research experiments"
echo
echo "🚀 Next steps:"
echo "  poetry run python -m scm_arena.cli experiment --help"
echo "  # Run comprehensive benchmark"