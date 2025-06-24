#!/usr/bin/env python3
"""
Prompt Length vs Performance Analysis using full_factorial_merged.db
Analyzes how prompt complexity affects llama3.2 performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sqlite3
import warnings
warnings.filterwarnings('ignore')

# Use the main merged database
DB_PATH = "full_factorial_merged.db"

print("📏 ANALYZING PROMPT LENGTH VS PERFORMANCE")
print("=" * 60)
print(f"Using database: {DB_PATH}")

# Connect to database
conn = sqlite3.connect(DB_PATH)

# First check the structure
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print(f"\nTables found: {[t[0] for t in tables]}")

# Get agent_rounds columns
cursor.execute("PRAGMA table_info(agent_rounds)")
columns = cursor.fetchall()
agent_columns = [col[1] for col in columns]
print(f"\nagent_rounds columns: {agent_columns[:15]}...")

# Check if prompt_sent exists
if 'prompt_sent' not in agent_columns:
    print("\n❌ No prompt_sent column found!")
    conn.close()
    exit(1)

# Load the data - join experiments with agent_rounds
query = """
SELECT 
    e.experiment_id,
    e.memory_strategy,
    e.visibility_level,
    e.prompt_type,
    e.game_mode,
    e.scenario,
    e.total_cost as experiment_total_cost,
    e.service_level,
    e.bullwhip_ratio,
    ar.round_number,
    ar.position,
    LENGTH(ar.prompt_sent) as prompt_length,
    ar.prompt_sent,
    ar.round_cost,
    ar.decision,
    ar.outgoing_order,
    ar.backlog,
    ar.inventory
FROM experiments e
JOIN agent_rounds ar ON e.experiment_id = ar.experiment_id
WHERE ar.prompt_sent IS NOT NULL AND ar.prompt_sent != ''
LIMIT 500000
"""

print("\nLoading data from database...")
df = pd.read_sql_query(query, conn)
print(f"✅ Loaded {len(df):,} agent decision records")

# Basic statistics
print(f"\nPrompt length statistics:")
print(f"  Mean: {df['prompt_length'].mean():.0f} characters")
print(f"  Median: {df['prompt_length'].median():.0f} characters")
print(f"  Min: {df['prompt_length'].min()} characters")
print(f"  Max: {df['prompt_length'].max():,} characters")

# Create comprehensive visualizations
fig = plt.figure(figsize=(20, 16))

# 1. Average prompt length by configuration
ax1 = plt.subplot(3, 4, 1)
config_prompt_length = df.groupby(['memory_strategy', 'visibility_level'])['prompt_length'].mean().unstack()
sns.heatmap(config_prompt_length, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax1)
ax1.set_title('Average Prompt Length by Configuration', fontweight='bold')

# 2. Prompt length distribution by memory strategy
ax2 = plt.subplot(3, 4, 2)
for memory in ['none', 'short', 'full']:
    data = df[df['memory_strategy'] == memory]['prompt_length']
    ax2.hist(data, bins=50, alpha=0.5, label=f'{memory} (μ={data.mean():.0f})', density=True)
ax2.set_xlabel('Prompt Length (characters)')
ax2.set_ylabel('Density')
ax2.set_title('Prompt Length Distribution by Memory', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Prompt length evolution over rounds
ax3 = plt.subplot(3, 4, 3)
round_evolution = df.groupby(['round_number', 'memory_strategy'])['prompt_length'].mean().unstack()
for memory in ['none', 'short', 'full']:
    if memory in round_evolution.columns:
        ax3.plot(round_evolution.index, round_evolution[memory], 
                marker='o', label=memory, linewidth=2, markersize=3)
ax3.set_xlabel('Round Number')
ax3.set_ylabel('Average Prompt Length')
ax3.set_title('Prompt Length Growth Over Game', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Cost vs prompt length scatter
ax4 = plt.subplot(3, 4, 4)
# Sample for visualization
sample_df = df.sample(min(10000, len(df)))
colors = {'none': 'green', 'short': 'orange', 'full': 'red'}
for memory in ['none', 'short', 'full']:
    data = sample_df[sample_df['memory_strategy'] == memory]
    ax4.scatter(data['prompt_length'], data['round_cost'], 
               alpha=0.3, s=10, label=memory, color=colors[memory])

# Add trend line
z = np.polyfit(sample_df['prompt_length'], sample_df['round_cost'], 1)
p = np.poly1d(z)
x_trend = sorted(sample_df['prompt_length'])
ax4.plot(x_trend, p(x_trend), "k--", alpha=0.8, linewidth=2, label='Trend')

ax4.set_xlabel('Prompt Length (characters)')
ax4.set_ylabel('Round Cost ($)')
ax4.set_title('Round Cost vs Prompt Length', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Calculate and show correlation
corr, p_value = stats.pearsonr(sample_df['prompt_length'], sample_df['round_cost'])
ax4.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_value:.2e}', 
        transform=ax4.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 5. Performance quintiles by prompt length
ax5 = plt.subplot(3, 4, 5)
df['prompt_quintile'] = pd.qcut(df['prompt_length'], q=5, 
                                labels=['Q1\n(Shortest)', 'Q2', 'Q3', 'Q4', 'Q5\n(Longest)'])
quintile_performance = df.groupby('prompt_quintile')['round_cost'].agg(['mean', 'std', 'count'])

x = range(len(quintile_performance))
bars = ax5.bar(x, quintile_performance['mean'], 
               yerr=quintile_performance['std']/np.sqrt(quintile_performance['count']),
               capsize=5, alpha=0.7, 
               color=['darkgreen', 'green', 'yellow', 'orange', 'red'])
ax5.set_xticks(x)
ax5.set_xticklabels(quintile_performance.index)
ax5.set_ylabel('Average Round Cost ($)')
ax5.set_title('Performance by Prompt Length Quintile', fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# Add values on bars
for i, (idx, row) in enumerate(quintile_performance.iterrows()):
    ax5.text(i, row['mean'] + 1, f"${row['mean']:.0f}", ha='center', fontsize=9)

# 6. Prompt length by visibility level
ax6 = plt.subplot(3, 4, 6)
vis_prompt = df.groupby(['visibility_level', 'memory_strategy'])['prompt_length'].mean().unstack()
vis_prompt.plot(kind='bar', ax=ax6)
ax6.set_xlabel('Visibility Level')
ax6.set_ylabel('Average Prompt Length')
ax6.set_title('Prompt Length: Visibility × Memory', fontweight='bold')
ax6.legend(title='Memory', bbox_to_anchor=(1.05, 1))
ax6.grid(True, alpha=0.3, axis='y')

# 7. Decision variance vs prompt length
ax7 = plt.subplot(3, 4, 7)
# Group by prompt quintiles and calculate decision variance
decision_stats = df.groupby('prompt_quintile').agg({
    'decision': ['mean', 'std'],
    'backlog': 'mean',
    'prompt_length': 'mean'
})

ax7.bar(range(len(decision_stats)), decision_stats[('decision', 'std')], 
        alpha=0.7, color='steelblue')
ax7.set_xticks(range(len(decision_stats)))
ax7.set_xticklabels(decision_stats.index)
ax7.set_ylabel('Decision Std Dev')
ax7.set_title('Decision Variability by Prompt Length', fontweight='bold')
ax7.grid(True, alpha=0.3, axis='y')

# 8. Prompt type comparison
ax8 = plt.subplot(3, 4, 8)
prompt_type_analysis = df.groupby(['prompt_type', 'memory_strategy'])['prompt_length'].mean().unstack()
prompt_type_analysis.plot(kind='bar', ax=ax8)
ax8.set_xlabel('Prompt Type')
ax8.set_ylabel('Average Prompt Length')
ax8.set_title('Prompt Length by Type and Memory', fontweight='bold')
ax8.legend(title='Memory')
ax8.grid(True, alpha=0.3, axis='y')

# 9. Best vs worst configurations
ax9 = plt.subplot(3, 4, 9)
config_analysis = df.groupby(['memory_strategy', 'visibility_level', 'prompt_type']).agg({
    'prompt_length': 'mean',
    'round_cost': 'mean',
    'experiment_total_cost': lambda x: x.iloc[0]  # Get the experiment total
}).reset_index()

# Sort by total cost
config_analysis = config_analysis.sort_values('experiment_total_cost')

# Plot best 5 and worst 5
best5 = config_analysis.head(5)
worst5 = config_analysis.tail(5)

ax9.scatter(best5['prompt_length'], best5['round_cost'], 
           color='green', s=150, label='Best 5', marker='^', edgecolor='black')
ax9.scatter(worst5['prompt_length'], worst5['round_cost'], 
           color='red', s=150, label='Worst 5', marker='v', edgecolor='black')

# Add labels
for _, row in best5.iterrows():
    label = f"{row['memory_strategy'][:1]}-{row['visibility_level'][:1]}"
    ax9.annotate(label, (row['prompt_length'], row['round_cost']), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)

for _, row in worst5.iterrows():
    label = f"{row['memory_strategy'][:1]}-{row['visibility_level'][:1]}"
    ax9.annotate(label, (row['prompt_length'], row['round_cost']), 
                xytext=(5, -10), textcoords='offset points', fontsize=8)

ax9.set_xlabel('Average Prompt Length')
ax9.set_ylabel('Average Round Cost ($)')
ax9.set_title('Best vs Worst Configurations', fontweight='bold')
ax9.legend()
ax9.grid(True, alpha=0.3)

# 10. Correlation by factor
ax10 = plt.subplot(3, 4, 10)
correlations = []
factors = ['memory_strategy', 'visibility_level', 'prompt_type', 'game_mode']

for factor in factors:
    for level in df[factor].unique():
        subset = df[df[factor] == level]
        if len(subset) > 100:
            corr, _ = stats.pearsonr(subset['prompt_length'], subset['round_cost'])
            correlations.append({'factor': factor, 'level': level, 'correlation': corr})

corr_df = pd.DataFrame(correlations)
corr_pivot = corr_df.pivot(index='level', columns='factor', values='correlation')
sns.heatmap(corr_pivot, annot=True, fmt='.3f', cmap='RdBu_r', center=0, ax=ax10)
ax10.set_title('Prompt Length-Cost Correlation by Factor', fontweight='bold')

# 11. Prompt length ranges
ax11 = plt.subplot(3, 4, 11)
# Show prompt length ranges for each configuration
config_ranges = df.groupby(['memory_strategy', 'visibility_level']).agg({
    'prompt_length': ['min', 'max', 'mean']
}).reset_index()

configs = []
for _, row in config_ranges.iterrows():
    configs.append(f"{row['memory_strategy']}-{row['visibility_level']}")

y_pos = range(len(configs))
mins = config_ranges[('prompt_length', 'min')]
maxs = config_ranges[('prompt_length', 'max')]
means = config_ranges[('prompt_length', 'mean')]

ax11.barh(y_pos, maxs - mins, left=mins, alpha=0.3)
ax11.scatter(means, y_pos, color='red', s=50, zorder=5)
ax11.set_yticks(y_pos)
ax11.set_yticklabels(configs)
ax11.set_xlabel('Prompt Length Range')
ax11.set_title('Prompt Length Ranges by Config', fontweight='bold')
ax11.grid(True, alpha=0.3, axis='x')

# 12. Statistical summary
ax12 = plt.subplot(3, 4, 12)
ax12.axis('off')

# Calculate key statistics
stats_text = "📊 KEY FINDINGS\n" + "="*25 + "\n\n"

# Overall correlation
overall_corr, overall_p = stats.pearsonr(df['prompt_length'], df['round_cost'])
stats_text += f"Overall correlation:\nr = {overall_corr:.3f}, p = {overall_p:.2e}\n\n"

# By memory strategy
stats_text += "Avg length by memory:\n"
for memory in ['none', 'short', 'full']:
    avg_len = df[df['memory_strategy'] == memory]['prompt_length'].mean()
    avg_cost = df[df['memory_strategy'] == memory]['round_cost'].mean()
    stats_text += f"{memory:5}: {avg_len:4.0f} chars → ${avg_cost:.0f}\n"

# Prompt growth rate
stats_text += "\nPrompt growth (char/round):\n"
for memory in ['none', 'short', 'full']:
    subset = df[df['memory_strategy'] == memory]
    if len(subset) > 0:
        growth = np.polyfit(subset['round_number'], subset['prompt_length'], 1)[0]
        stats_text += f"{memory:5}: +{growth:.1f} chars/round\n"

# Best configuration
best_config = config_analysis.iloc[0]
stats_text += f"\nBest config:\n{best_config['memory_strategy']}-{best_config['visibility_level']}-{best_config['prompt_type']}\n"
stats_text += f"Avg length: {best_config['prompt_length']:.0f} chars"

ax12.text(0.05, 0.95, stats_text, transform=ax12.transAxes, 
         va='top', fontsize=10, fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.suptitle('Prompt Length Impact on LLM Performance - SCM Arena', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('prompt_length_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ Analysis complete!")
print("💾 Saved: prompt_length_comprehensive_analysis.png")

# Print summary findings
print("\n🔍 SUMMARY OF FINDINGS:")
print("-" * 40)
print(f"Overall correlation (prompt length vs cost): r = {overall_corr:.3f}")
if overall_p < 0.001:
    print("✅ Highly statistically significant (p < 0.001)")
else:
    print(f"Statistical significance: p = {overall_p:.4f}")

print("\nPrompt length by memory strategy:")
memory_summary = df.groupby('memory_strategy')['prompt_length'].agg(['mean', 'min', 'max'])
print(memory_summary.round(0))

print("\nPerformance by prompt length quintile:")
print(quintile_performance[['mean']].round(2))

if overall_corr > 0.2:
    print("\n🎯 CONCLUSION: Strong evidence that longer prompts lead to worse performance!")
    print("   This supports the hypothesis that llama3.2 gets confused by complex contexts.")
else:
    print("\n🤔 CONCLUSION: Weak correlation between prompt length and performance.")
    print("   Other factors may be more important than raw prompt length.")

# Close connection
conn.close()