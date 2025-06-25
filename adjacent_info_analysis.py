#!/usr/bin/env python3
"""
Fixed Nuanced Analysis of Adjacent Visibility
Exploring when adjacent visibility helps vs hurts performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv("full_factorial_merged.csv")

print("🔍 NUANCED ANALYSIS: When Does Adjacent Visibility Help?")
print("=" * 60)

# Create comprehensive visualization
fig = plt.figure(figsize=(18, 12))

# 1. Visibility performance by other factors
ax1 = plt.subplot(2, 3, 1)
vis_by_memory = df.groupby(['visibility_level', 'memory_strategy'])['total_cost'].mean().unstack()
vis_by_memory.plot(kind='bar', ax=ax1)
ax1.set_title('Visibility × Memory Strategy', fontweight='bold', fontsize=12)
ax1.set_ylabel('Total Cost ($)')
ax1.legend(title='Memory', bbox_to_anchor=(1.05, 1))
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# 2. Visibility by game mode
ax2 = plt.subplot(2, 3, 2)
vis_by_mode = df.groupby(['visibility_level', 'game_mode'])['total_cost'].mean().unstack()
vis_by_mode.plot(kind='bar', ax=ax2)
ax2.set_title('Visibility × Game Mode', fontweight='bold', fontsize=12)
ax2.set_ylabel('Total Cost ($)')
ax2.legend(title='Mode')
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# 3. Visibility by prompt type
ax3 = plt.subplot(2, 3, 3)
vis_by_prompt = df.groupby(['visibility_level', 'prompt_type'])['total_cost'].mean().unstack()
vis_by_prompt.plot(kind='bar', ax=ax3)
ax3.set_title('Visibility × Prompt Type', fontweight='bold', fontsize=12)
ax3.set_ylabel('Total Cost ($)')
ax3.legend(title='Prompt')
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45)

# 4. Detailed performance comparison
ax4 = plt.subplot(2, 3, 4)

# Calculate when adjacent beats local
comparison_data = []
for memory in df['memory_strategy'].unique():
    for mode in df['game_mode'].unique():
        subset = df[(df['memory_strategy'] == memory) & (df['game_mode'] == mode)]
        
        costs = {}
        for vis in ['local', 'adjacent', 'full']:
            vis_data = subset[subset['visibility_level'] == vis]
            if len(vis_data) > 0:
                costs[vis] = vis_data['total_cost'].mean()
        
        if 'local' in costs and 'adjacent' in costs:
            diff = costs['adjacent'] - costs['local']
            comparison_data.append({
                'config': f"{memory[:4]}-{mode[:4]}",
                'difference': diff,
                'adjacent_better': diff < 0
            })

# Plot the differences
if comparison_data:
    comp_df = pd.DataFrame(comparison_data)
    colors = ['green' if x else 'red' for x in comp_df['adjacent_better']]
    bars = ax4.bar(comp_df['config'], comp_df['difference'], color=colors, alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_ylabel('Adjacent - Local Cost ($)')
    ax4.set_title('When Adjacent Beats Local Visibility', fontweight='bold', fontsize=12)
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add annotations
    for bar, val in zip(bars, comp_df['difference']):
        if val < 0:
            ax4.text(bar.get_x() + bar.get_width()/2, val - 50, 
                    f'${abs(val):.0f}', ha='center', va='top', fontsize=9)

# 5. Heatmap of visibility performance
ax5 = plt.subplot(2, 3, 5)

# Create pivot table for heatmap
pivot_data = df.pivot_table(
    values='total_cost',
    index=['memory_strategy', 'game_mode'],
    columns='visibility_level',
    aggfunc='mean'
)

# Calculate relative performance (vs local)
if 'local' in pivot_data.columns:
    relative_perf = pivot_data.copy()
    for col in relative_perf.columns:
        relative_perf[col] = ((pivot_data[col] - pivot_data['local']) / pivot_data['local'] * 100)
    
    # Create heatmap
    sns.heatmap(relative_perf, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                center=0, cbar_kws={'label': '% Difference from Local'},
                ax=ax5)
    ax5.set_title('Visibility Performance Relative to Local (%)', fontweight='bold', fontsize=12)

# 6. Summary statistics
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

# Calculate key statistics
stats_text = "📊 VISIBILITY PERFORMANCE SUMMARY\n" + "="*35 + "\n\n"

# Overall averages
vis_avg = df.groupby('visibility_level')['total_cost'].agg(['mean', 'std', 'count'])
stats_text += "Overall Averages:\n"
for vis in ['local', 'adjacent', 'full']:
    if vis in vis_avg.index:
        stats_text += f"  {vis:8}: ${vis_avg.loc[vis, 'mean']:,.0f} "
        stats_text += f"(±${vis_avg.loc[vis, 'std']:,.0f})\n"

# Count where adjacent wins
adjacent_wins = 0
total_comparisons = 0

for memory in df['memory_strategy'].unique():
    for mode in df['game_mode'].unique():
        for prompt in df['prompt_type'].unique():
            subset = df[(df['memory_strategy'] == memory) & 
                       (df['game_mode'] == mode) & 
                       (df['prompt_type'] == prompt)]
            
            if len(subset) > 0:
                vis_costs = subset.groupby('visibility_level')['total_cost'].mean()
                if 'adjacent' in vis_costs and 'local' in vis_costs:
                    total_comparisons += 1
                    if vis_costs['adjacent'] < vis_costs['local']:
                        adjacent_wins += 1

stats_text += f"\n\nAdjacent vs Local Performance:\n"
stats_text += f"  Adjacent better: {adjacent_wins}/{total_comparisons} "
stats_text += f"({adjacent_wins/total_comparisons*100:.1f}%)\n"

# Best config for adjacent
best_adjacent = df[df['visibility_level'] == 'adjacent'].groupby(
    ['memory_strategy', 'game_mode', 'prompt_type']
)['total_cost'].mean().idxmin()

stats_text += f"\n\nBest config for Adjacent:\n"
stats_text += f"  Memory: {best_adjacent[0]}\n"
stats_text += f"  Mode: {best_adjacent[1]}\n"
stats_text += f"  Prompt: {best_adjacent[2]}\n"

ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, 
         va='top', fontsize=10, fontfamily='monospace')

plt.suptitle('The Nuanced Truth About Adjacent Visibility', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('visibility_nuance_analysis_fixed.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Analysis complete!")
print("💾 Saved: visibility_nuance_analysis_fixed.png")

# Print detailed findings
print("\n📊 DETAILED FINDINGS:")
print("-" * 40)

# Find specific conditions where adjacent performs well
print("\nConditions where Adjacent performs BETTER than Local:")
found_better = False

for memory in df['memory_strategy'].unique():
    for mode in df['game_mode'].unique():
        for prompt in df['prompt_type'].unique():
            for scenario in df['scenario'].unique():
                subset = df[(df['memory_strategy'] == memory) & 
                           (df['game_mode'] == mode) & 
                           (df['prompt_type'] == prompt) &
                           (df['scenario'] == scenario)]
                
                if len(subset) > 0:
                    vis_costs = subset.groupby('visibility_level')['total_cost'].mean()
                    
                    if 'adjacent' in vis_costs and 'local' in vis_costs:
                        if vis_costs['adjacent'] < vis_costs['local']:
                            saving = vis_costs['local'] - vis_costs['adjacent']
                            print(f"  • {memory}-{mode}-{prompt}-{scenario}: "
                                  f"Adjacent saves ${saving:.0f}")
                            found_better = True

if not found_better:
    print("  No conditions found where adjacent beats local")

# Interaction analysis
print("\n📈 KEY INSIGHTS:")
print("-" * 40)

# Memory interaction
mem_interaction = df.groupby(['memory_strategy', 'visibility_level'])['total_cost'].mean().unstack()
print("\n1. Memory × Visibility Interaction:")
for memory in mem_interaction.index:
    if 'adjacent' in mem_interaction.columns and 'local' in mem_interaction.columns:
        diff = mem_interaction.loc[memory, 'adjacent'] - mem_interaction.loc[memory, 'local']
        print(f"   {memory:5} memory: Adjacent is ${abs(diff):.0f} {'better' if diff < 0 else 'worse'} than local")

# Game mode interaction
mode_interaction = df.groupby(['game_mode', 'visibility_level'])['total_cost'].mean().unstack()
print("\n2. Game Mode × Visibility Interaction:")
for mode in mode_interaction.index:
    if 'adjacent' in mode_interaction.columns and 'local' in mode_interaction.columns:
        diff = mode_interaction.loc[mode, 'adjacent'] - mode_interaction.loc[mode, 'local']
        print(f"   {mode:7} mode: Adjacent is ${abs(diff):.0f} {'better' if diff < 0 else 'worse'} than local")

print("\n✨ The nuance: Adjacent visibility's performance is highly context-dependent!")