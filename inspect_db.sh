#!/bin/bash

# SCM-Arena Database Inspector
# Quick script to check your benchmark database

DB_FILE="scm_arena_benchmark_phi4_latest.db"

echo "🔍 SCM-Arena Database Inspector"
echo "======================================"

# Check if database exists
if [ ! -f "$DB_FILE" ]; then
    echo "❌ Database file not found: $DB_FILE"
    exit 1
fi

echo "✅ Database found: $DB_FILE"
echo "📊 File size: $(du -h "$DB_FILE" | cut -f1)"
echo ""

# Basic database info
echo "📋 BASIC DATABASE INFO:"
echo "------------------------"
sqlite3 "$DB_FILE" "SELECT 
    COUNT(*) as total_experiments,
    COUNT(DISTINCT model_name) as models,
    COUNT(DISTINCT memory_strategy) as memory_strategies,
    COUNT(DISTINCT prompt_type) as prompt_types,
    COUNT(DISTINCT visibility_level) as visibility_levels,
    COUNT(DISTINCT scenario) as scenarios,
    COUNT(DISTINCT game_mode) as game_modes
FROM experiments;" | while IFS='|' read -r total models memory prompts visibility scenarios game_modes; do
    echo "  Total experiments: $total"
    echo "  Models: $models"
    echo "  Memory strategies: $memory"
    echo "  Prompt types: $prompts"
    echo "  Visibility levels: $visibility"
    echo "  Scenarios: $scenarios"
    echo "  Game modes: $game_modes"
done

echo ""

# Progress by condition
echo "📊 PROGRESS BY CONDITION:"
echo "--------------------------"
sqlite3 "$DB_FILE" "SELECT 
    memory_strategy || '-' || prompt_type || '-' || visibility_level || '-' || scenario || '-' || game_mode as condition,
    COUNT(*) as completed_runs
FROM experiments 
GROUP BY memory_strategy, prompt_type, visibility_level, scenario, game_mode
ORDER BY completed_runs DESC
LIMIT 10;" | while IFS='|' read -r condition count; do
    printf "  %-50s %2d/20 runs\n" "$condition" "$count"
done

echo ""

# Recent experiments
echo "⏰ RECENT EXPERIMENTS (Last 10):"
echo "--------------------------------"
sqlite3 "$DB_FILE" "SELECT 
    created_at,
    model_name,
    memory_strategy || '-' || prompt_type || '-' || visibility_level as config,
    scenario || '-' || game_mode as setup
FROM experiments 
ORDER BY created_at DESC 
LIMIT 10;" | while IFS='|' read -r timestamp model config setup; do
    echo "  $timestamp | $model | $config | $setup"
done

echo ""

# Sample data structure
echo "🔬 SAMPLE EXPERIMENT DATA:"
echo "-------------------------"
sqlite3 "$DB_FILE" "SELECT 
    experiment_id,
    total_rounds,
    final_total_cost,
    avg_service_level
FROM experiments 
WHERE total_rounds IS NOT NULL 
LIMIT 5;" | while IFS='|' read -r exp_id rounds cost service; do
    echo "  ID: $exp_id"
    echo "    Rounds: $rounds | Cost: $cost | Service Level: $service"
done

echo ""

# Quick validation
echo "🔍 DATA VALIDATION:"
echo "-------------------"

# Check for nulls in key fields
null_count=$(sqlite3 "$DB_FILE" "SELECT COUNT(*) FROM experiments WHERE total_rounds IS NULL OR final_total_cost IS NULL;")
echo "  Experiments with missing key data: $null_count"

# Check round counts
round_stats=$(sqlite3 "$DB_FILE" "SELECT MIN(total_rounds), MAX(total_rounds), AVG(total_rounds) FROM experiments WHERE total_rounds IS NOT NULL;")
echo "  Round counts - Min: $(echo $round_stats | cut -d'|' -f1), Max: $(echo $round_stats | cut -d'|' -f2), Avg: $(echo $round_stats | cut -d'|' -f3 | cut -d'.' -f1)"

# Check for reasonable cost values
cost_stats=$(sqlite3 "$DB_FILE" "SELECT MIN(final_total_cost), MAX(final_total_cost), AVG(final_total_cost) FROM experiments WHERE final_total_cost IS NOT NULL;")
echo "  Cost range - Min: $(echo $cost_stats | cut -d'|' -f1 | cut -d'.' -f1), Max: $(echo $cost_stats | cut -d'|' -f2 | cut -d'.' -f1), Avg: $(echo $cost_stats | cut -d'|' -f3 | cut -d'.' -f1)"

echo ""

# Table structure
echo "🏗️ DATABASE SCHEMA:"
echo "-------------------"
sqlite3 "$DB_FILE" ".schema experiments" | head -20

echo ""
echo "✅ Database inspection complete!"
echo ""
echo "💡 Quick commands:"
echo "  Full table count: sqlite3 $DB_FILE 'SELECT COUNT(*) FROM experiments;'"
echo "  Latest experiment: sqlite3 $DB_FILE 'SELECT * FROM experiments ORDER BY created_at DESC LIMIT 1;'"
echo "  All tables: sqlite3 $DB_FILE '.tables'"