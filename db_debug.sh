#!/bin/bash

# Database Debug Script
# Find out why experiments aren't being saved

DB_FILE="scm_arena_benchmark_phi4_latest.db"

echo "🔍 Database Debug Analysis"
echo "=========================="

# Check all database files
echo "📁 DATABASE FILES:"
echo "------------------"
ls -la *benchmark*.db 2>/dev/null || echo "No benchmark database files found"
ls -la *scm*.db 2>/dev/null || echo "No SCM database files found"
echo ""

# Check progress file
echo "📋 PROGRESS FILE:"
echo "----------------"
if [ -f "benchmark_progress_phi4_latest.json" ]; then
    echo "✅ Progress file exists"
    echo "Size: $(du -h benchmark_progress_phi4_latest.json | cut -f1)"
    echo "Last modified: $(stat -c %y benchmark_progress_phi4_latest.json)"
    echo ""
    echo "Sample progress data:"
    head -20 benchmark_progress_phi4_latest.json
else
    echo "❌ Progress file not found"
fi

echo ""

# Check database permissions
echo "🔒 DATABASE PERMISSIONS:"
echo "-----------------------"
if [ -f "$DB_FILE" ]; then
    ls -la "$DB_FILE"
    echo "Writable: $([ -w "$DB_FILE" ] && echo "✅ Yes" || echo "❌ No")"
else
    echo "❌ Database file doesn't exist"
fi

echo ""

# Check table structure
echo "🏗️ DATABASE STRUCTURE:"
echo "----------------------"
if [ -f "$DB_FILE" ]; then
    echo "Tables:"
    sqlite3 "$DB_FILE" ".tables"
    echo ""
    
    echo "Experiments table structure:"
    sqlite3 "$DB_FILE" ".schema experiments"
    echo ""
    
    echo "Row count:"
    sqlite3 "$DB_FILE" "SELECT COUNT(*) FROM experiments;"
    echo ""
    
    echo "Sample of any data:"
    sqlite3 "$DB_FILE" "SELECT * FROM experiments LIMIT 3;"
else
    echo "❌ No database file to inspect"
fi

echo ""

# Test database write
echo "🧪 DATABASE WRITE TEST:"
echo "----------------------"
TEST_DB="test_write.db"
if sqlite3 "$TEST_DB" "CREATE TABLE test (id INTEGER); INSERT INTO test VALUES (1); SELECT * FROM test;" 2>/dev/null; then
    echo "✅ Can create and write to database"
    rm -f "$TEST_DB"
else
    echo "❌ Cannot write to database"
fi

echo ""

# Check if experiments are using different database path
echo "🔍 CHECKING FOR OTHER DATABASE FILES:"
echo "------------------------------------"
find . -name "*.db" -type f 2>/dev/null | while read db; do
    count=$(sqlite3 "$db" "SELECT COUNT(*) FROM experiments;" 2>/dev/null || echo "0")
    echo "$db: $count experiments"
done

echo ""

# Check recent benchmark logs/output
echo "📝 RECENT ACTIVITY:"
echo "------------------"
echo "Files modified in last hour:"
find . -type f -newermt "1 hour ago" 2>/dev/null | head -10

echo ""
echo "🎯 DIAGNOSIS COMPLETE"
echo "===================="
echo ""
echo "🔧 Next steps:"
echo "1. Check if experiments are writing to a different database file"
echo "2. Verify database path in benchmark script"
echo "3. Test single experiment with verbose output"
echo ""
echo "🧪 Test single experiment:"
echo "poetry run python -m scm_arena.cli experiment --models phi4:latest --memory none --prompts neutral --visibility local --scenarios classic --game-modes modern --runs 1 --rounds 5 --save-database --db-path test_single.db"