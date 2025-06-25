#!/usr/bin/env python3
"""
Research-grade analysis dashboard for SCM-Arena experiments
Analyzes factorial design: Models × Memory × Visibility × Prompts × Scenarios × Game Modes
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import argparse
import sys
from pathlib import Path
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

class SCMArenaResearchDashboard:
    def __init__(self, db_path):
        """Initialize dashboard with database connection"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.experiments = None
        self.llama_data = None
        self.or_data = None
        
    def load_data(self):
        """Load all experiments from database"""
        # Load all experiments
        self.experiments = pd.read_sql_query("SELECT * FROM experiments", self.conn)
        
        # Separate Llama and OR experiments
        self.llama_data = self.experiments[self.experiments['model_name'] == 'llama3.2'].copy()
        self.or_data = self.experiments[self.experiments['model_name'] != 'llama3.2'].copy()
        
        # Print experimental design summary
        print("📊 EXPERIMENTAL DESIGN SUMMARY")
        print("=" * 50)
        print(f"Total experiments: {len(self.experiments)}")
        print(f"Llama 3.2: {len(self.llama_data)} experiments")
        print(f"OR Baselines: {len(self.or_data)} experiments")
        print("\nFACTORS:")
        
        # Analyze factors for Llama
        print("\nLlama 3.2 Factorial Design:")
        factors = ['memory_strategy', 'visibility_level', 'prompt_type', 'scenario', 'game_mode']
        for factor in factors:
            if factor in self.llama_data.columns:
                levels = self.llama_data[factor].unique()
                counts = self.llama_data[factor].value_counts()
                print(f"  {factor}: {len(levels)} levels - {list(levels)}")
                
        # Calculate number of unique conditions
        llama_conditions = self.llama_data.groupby(factors).size().reset_index(name='count')
        print(f"\nUnique experimental conditions: {len(llama_conditions)}")
        print(f"Replications per condition: {llama_conditions['count'].value_counts().to_dict()}")
        
    def generate_research_dashboard(self):
        """Generate comprehensive research analysis dashboard"""
        # Create multi-page figure
        self._create_main_effects_analysis()
        self._create_interaction_effects_analysis()
        self._create_conditional_comparison_analysis()
        self._create_factor_importance_analysis()
        
    def _create_main_effects_analysis(self):
        """Analyze main effects of each factor"""
        fig = plt.figure(figsize=(20, 24))
        fig.suptitle('SCM-Arena: Main Effects Analysis - Llama 3.2 vs OR Baselines', fontsize=24, fontweight='bold')
        
        # Create 6x3 grid for comprehensive analysis
        gs = fig.add_gridspec(6, 3, hspace=0.4, wspace=0.3)
        
        # 1. Memory Strategy Effects
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_memory_effects(ax1)
        
        # 2. Visibility Level Effects
        ax2 = fig.add_subplot(gs[1, :])
        self._plot_visibility_effects(ax2)
        
        # 3. Prompt Type Effects
        ax3 = fig.add_subplot(gs[2, :])
        self._plot_prompt_effects(ax3)
        
        # 4. Scenario Effects
        ax4 = fig.add_subplot(gs[3, :])
        self._plot_scenario_effects(ax4)
        
        # 5. Game Mode Effects
        ax5 = fig.add_subplot(gs[4, :])
        self._plot_game_mode_effects(ax5)
        
        # 6. Best OR Baseline Comparison
        ax6 = fig.add_subplot(gs[5, :])
        self._plot_best_conditions_comparison(ax6)
        
        plt.tight_layout()
        
        # Save
        output_path = Path(self.db_path).parent / 'scm_arena_main_effects.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Main effects analysis saved to: {output_path}")
        
    def _create_interaction_effects_analysis(self):
        """Analyze interaction effects between factors"""
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('SCM-Arena: Interaction Effects Analysis', fontsize=24, fontweight='bold')
        
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Key interactions to analyze
        interactions = [
            ('memory_strategy', 'visibility_level', gs[0, :]),
            ('memory_strategy', 'game_mode', gs[1, 0]),
            ('visibility_level', 'scenario', gs[1, 1]),
            ('prompt_type', 'memory_strategy', gs[1, 2]),
            ('game_mode', 'scenario', gs[2, :2]),
            ('visibility_level', 'game_mode', gs[2, 2])
        ]
        
        for factor1, factor2, subplot_spec in interactions:
            ax = fig.add_subplot(subplot_spec)
            self._plot_interaction_heatmap(ax, factor1, factor2)
        
        plt.tight_layout()
        
        # Save
        output_path = Path(self.db_path).parent / 'scm_arena_interaction_effects.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Interaction effects analysis saved to: {output_path}")
        
    def _create_conditional_comparison_analysis(self):
        """Create detailed conditional comparisons"""
        fig = plt.figure(figsize=(20, 20))
        fig.suptitle('SCM-Arena: Conditional Performance Analysis', fontsize=24, fontweight='bold')
        
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
        
        # 1. Best vs Worst Conditions
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_best_worst_conditions(ax1)
        
        # 2. Performance by Memory-Visibility Combination
        ax2 = fig.add_subplot(gs[1, :])
        self._plot_memory_visibility_combinations(ax2)
        
        # 3. Scenario-Game Mode Performance Matrix
        ax3 = fig.add_subplot(gs[2, 0])
        self._plot_scenario_gamemode_matrix(ax3)
        
        # 4. OR Method Performance Across Conditions
        ax4 = fig.add_subplot(gs[2, 1])
        self._plot_or_method_breakdown(ax4)
        
        # 5. Statistical Significance Table
        ax5 = fig.add_subplot(gs[3, :])
        self._plot_significance_table(ax5)
        
        plt.tight_layout()
        
        # Save
        output_path = Path(self.db_path).parent / 'scm_arena_conditional_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Conditional analysis saved to: {output_path}")
        
    def _create_factor_importance_analysis(self):
        """Analyze relative importance of each factor"""
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('SCM-Arena: Factor Importance & Effect Sizes', fontsize=24, fontweight='bold')
        
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Effect Size Analysis
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_effect_sizes(ax1)
        
        # 2. Variance Decomposition
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_variance_decomposition(ax2)
        
        # 3. Performance Stability
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_performance_stability(ax3)
        
        plt.tight_layout()
        
        # Save
        output_path = Path(self.db_path).parent / 'scm_arena_factor_importance.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Factor importance analysis saved to: {output_path}")
        
    def _plot_memory_effects(self, ax):
        """Plot memory strategy effects with confidence intervals"""
        # Get OR baseline average
        or_baseline = self.or_data.groupby('model_name')['total_cost'].mean().min()
        
        # Calculate statistics for each memory strategy
        memory_stats = self.llama_data.groupby('memory_strategy').agg({
            'total_cost': ['mean', 'std', 'count', 'sem']
        }).round(2)
        
        # Plot with error bars
        x = range(len(memory_stats))
        means = memory_stats['total_cost']['mean'].values
        sems = memory_stats['total_cost']['sem'].values
        labels = memory_stats.index
        
        bars = ax.bar(x, means, yerr=sems*1.96, capsize=5, 
                      color='steelblue', edgecolor='black', linewidth=1, alpha=0.8)
        
        # Add OR baseline line
        ax.axhline(y=or_baseline, color='red', linestyle='--', linewidth=2, 
                  label=f'Best OR Baseline: ${or_baseline:.0f}')
        
        # Add value labels
        for i, (bar, mean, sem) in enumerate(zip(bars, means, sems)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                   f'${mean:.0f}\n±{sem*1.96:.0f}', ha='center', va='bottom', fontsize=10)
        
        # Calculate percentage difference from baseline
        for i, (bar, mean) in enumerate(zip(bars, means)):
            diff = ((mean - or_baseline) / or_baseline) * 100
            color = 'red' if diff > 0 else 'green'
            ax.text(bar.get_x() + bar.get_width()/2, or_baseline - 500,
                   f'{diff:+.1f}%', ha='center', va='top', fontsize=9, 
                   color=color, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel('Memory Strategy', fontsize=12)
        ax.set_ylabel('Average Total Cost ($)', fontsize=12)
        ax.set_title('Effect of Memory Strategy on Performance (95% CI)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
    def _plot_visibility_effects(self, ax):
        """Plot visibility level effects"""
        # Similar structure but for visibility levels
        or_baseline = self.or_data.groupby('model_name')['total_cost'].mean().min()
        
        visibility_stats = self.llama_data.groupby('visibility_level').agg({
            'total_cost': ['mean', 'std', 'count', 'sem']
        }).round(2)
        
        x = range(len(visibility_stats))
        means = visibility_stats['total_cost']['mean'].values
        sems = visibility_stats['total_cost']['sem'].values
        labels = visibility_stats.index
        
        bars = ax.bar(x, means, yerr=sems*1.96, capsize=5,
                      color='darkgreen', edgecolor='black', linewidth=1, alpha=0.8)
        
        ax.axhline(y=or_baseline, color='red', linestyle='--', linewidth=2,
                  label=f'Best OR Baseline: ${or_baseline:.0f}')
        
        for i, (bar, mean, sem) in enumerate(zip(bars, means, sems)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                   f'${mean:.0f}\n±{sem*1.96:.0f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel('Visibility Level', fontsize=12)
        ax.set_ylabel('Average Total Cost ($)', fontsize=12)
        ax.set_title('Effect of Visibility Level on Performance (95% CI)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
    def _plot_prompt_effects(self, ax):
        """Plot prompt type effects"""
        or_baseline = self.or_data.groupby('model_name')['total_cost'].mean().min()
        
        prompt_stats = self.llama_data.groupby('prompt_type').agg({
            'total_cost': ['mean', 'std', 'count', 'sem']
        }).round(2)
        
        x = range(len(prompt_stats))
        means = prompt_stats['total_cost']['mean'].values
        sems = prompt_stats['total_cost']['sem'].values
        labels = prompt_stats.index
        
        bars = ax.bar(x, means, yerr=sems*1.96, capsize=5,
                      color='coral', edgecolor='black', linewidth=1, alpha=0.8)
        
        ax.axhline(y=or_baseline, color='red', linestyle='--', linewidth=2,
                  label=f'Best OR Baseline: ${or_baseline:.0f}')
        
        for i, (bar, mean, sem) in enumerate(zip(bars, means, sems)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                   f'${mean:.0f}\n±{sem*1.96:.0f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel('Prompt Type', fontsize=12)
        ax.set_ylabel('Average Total Cost ($)', fontsize=12)
        ax.set_title('Effect of Prompt Type on Performance (95% CI)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
    def _plot_scenario_effects(self, ax):
        """Plot scenario effects comparing Llama vs OR methods"""
        # Group by scenario for both Llama and OR
        llama_scenario = self.llama_data.groupby('scenario')['total_cost'].agg(['mean', 'sem']).reset_index()
        or_scenario = self.or_data.groupby('scenario')['total_cost'].agg(['mean', 'sem']).reset_index()
        
        # Find common scenarios
        common_scenarios = set(llama_scenario['scenario']) & set(or_scenario['scenario'])
        
        # Filter to common scenarios only
        llama_scenario = llama_scenario[llama_scenario['scenario'].isin(common_scenarios)].sort_values('scenario')
        or_scenario = or_scenario[or_scenario['scenario'].isin(common_scenarios)].sort_values('scenario')
        
        x = np.arange(len(llama_scenario))
        width = 0.35
        
        # Plot bars
        bars1 = ax.bar(x - width/2, llama_scenario['mean'], width,
                       yerr=llama_scenario['sem']*1.96, capsize=5,
                       label='Llama 3.2', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, or_scenario['mean'], width,
                       yerr=or_scenario['sem']*1.96, capsize=5,
                       label='Best OR', color='darkgreen', alpha=0.8)
        
        # Add percentage differences
        for i, (idx, row) in enumerate(llama_scenario.iterrows()):
            scenario = row['scenario']
            llama_cost = row['mean']
            or_row = or_scenario[or_scenario['scenario'] == scenario]
            if not or_row.empty:
                or_cost = or_row.iloc[0]['mean']
                diff = ((llama_cost - or_cost) / or_cost) * 100
                color = 'red' if diff > 0 else 'green'
                ax.text(i, max(llama_cost, or_cost) + 500,
                       f'{diff:+.1f}%', ha='center', fontsize=10,
                       color=color, fontweight='bold')
        
        # Add note about missing scenarios
        all_llama_scenarios = set(self.llama_data['scenario'].unique())
        missing_scenarios = all_llama_scenarios - common_scenarios
        if missing_scenarios:
            ax.text(0.02, 0.98, f"Note: OR baselines not available for: {', '.join(missing_scenarios)}",
                   transform=ax.transAxes, fontsize=9, va='top', style='italic', color='gray')
        
        ax.set_xlabel('Scenario', fontsize=12)
        ax.set_ylabel('Average Total Cost ($)', fontsize=12)
        ax.set_title('Performance by Scenario: Llama 3.2 vs OR Methods', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(llama_scenario['scenario'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
    def _plot_game_mode_effects(self, ax):
        """Plot game mode effects"""
        # Group by game mode for both
        llama_mode = self.llama_data.groupby('game_mode')['total_cost'].agg(['mean', 'sem']).reset_index()
        or_mode = self.or_data.groupby('game_mode')['total_cost'].agg(['mean', 'sem']).reset_index()
        
        x = np.arange(len(llama_mode))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, llama_mode['mean'], width,
                       yerr=llama_mode['sem']*1.96, capsize=5,
                       label='Llama 3.2', color='purple', alpha=0.8)
        bars2 = ax.bar(x + width/2, or_mode['mean'], width,
                       yerr=or_mode['sem']*1.96, capsize=5,
                       label='Best OR', color='orange', alpha=0.8)
        
        ax.set_xlabel('Game Mode', fontsize=12)
        ax.set_ylabel('Average Total Cost ($)', fontsize=12)
        ax.set_title('Performance by Game Mode: Classic vs Modern', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(llama_mode['game_mode'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
    def _plot_best_conditions_comparison(self, ax):
        """Find and compare best performing conditions"""
        # Find best Llama conditions
        factors = ['memory_strategy', 'visibility_level', 'prompt_type', 'scenario', 'game_mode']
        llama_conditions = self.llama_data.groupby(factors)['total_cost'].agg(['mean', 'count']).reset_index()
        llama_conditions = llama_conditions[llama_conditions['count'] >= 5]  # At least 5 runs
        llama_conditions = llama_conditions.sort_values('mean').head(10)
        
        # Create condition labels
        llama_conditions['condition'] = llama_conditions.apply(
            lambda x: f"{x['memory_strategy'][:4]}-{x['visibility_level'][:3]}-{x['prompt_type'][:4]}-{x['scenario'][:4]}-{x['game_mode'][:3]}",
            axis=1
        )
        
        # Get best OR performance
        or_best = self.or_data['total_cost'].min()
        or_avg = self.or_data['total_cost'].mean()
        
        # Plot
        y_pos = np.arange(len(llama_conditions))
        bars = ax.barh(y_pos, llama_conditions['mean'], 
                       color='steelblue', edgecolor='black', alpha=0.8)
        
        # Add OR reference lines
        ax.axvline(x=or_best, color='green', linestyle='--', linewidth=2,
                  label=f'Best OR: ${or_best:.0f}')
        ax.axvline(x=or_avg, color='red', linestyle='--', linewidth=2,
                  label=f'Avg OR: ${or_avg:.0f}')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(llama_conditions['condition'], fontsize=9)
        ax.set_xlabel('Average Total Cost ($)', fontsize=12)
        ax.set_title('Top 10 Best Performing Llama 3.2 Conditions', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
    def _plot_interaction_heatmap(self, ax, factor1, factor2):
        """Plot interaction effects between two factors"""
        # Calculate mean costs for each combination
        interaction_data = self.llama_data.groupby([factor1, factor2])['total_cost'].mean().reset_index()
        heatmap_data = interaction_data.pivot(index=factor1, columns=factor2, values='total_cost')
        
        # Create heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='RdYlGn_r',
                   cbar_kws={'label': 'Average Total Cost ($)'}, ax=ax)
        
        ax.set_title(f'Interaction: {factor1} × {factor2}', fontsize=12, fontweight='bold')
        ax.set_xlabel(factor2.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel(factor1.replace('_', ' ').title(), fontsize=10)
        
    def _plot_best_worst_conditions(self, ax):
        """Compare best and worst performing conditions in detail"""
        factors = ['memory_strategy', 'visibility_level', 'prompt_type', 'scenario', 'game_mode']
        
        # Get all condition performances
        condition_perf = self.llama_data.groupby(factors).agg({
            'total_cost': ['mean', 'std', 'count'],
            'service_level': 'mean',
            'bullwhip_ratio': 'mean'
        }).reset_index()
        
        # Flatten column names
        condition_perf.columns = factors + ['cost_mean', 'cost_std', 'count', 'service_level', 'bullwhip_ratio']
        
        # Filter for sufficient data
        condition_perf = condition_perf[condition_perf['count'] >= 5]
        
        # Get best and worst 5
        best_5 = condition_perf.nsmallest(5, 'cost_mean')
        worst_5 = condition_perf.nlargest(5, 'cost_mean')
        
        # Combine and plot
        combined = pd.concat([best_5, worst_5])
        combined['label'] = combined.apply(
            lambda x: f"{x['memory_strategy'][:2]}-{x['visibility_level'][:2]}-{x['scenario'][:3]}",
            axis=1
        )
        
        x = range(len(combined))
        colors = ['green']*5 + ['red']*5
        
        bars = ax.bar(x, combined['cost_mean'], yerr=combined['cost_std'],
                      color=colors, alpha=0.7, edgecolor='black', capsize=5)
        
        # Add OR baseline
        or_baseline = self.or_data['total_cost'].mean()
        ax.axhline(y=or_baseline, color='blue', linestyle='--', linewidth=2,
                  label=f'OR Average: ${or_baseline:.0f}')
        
        ax.set_xticks(x)
        ax.set_xticklabels(combined['label'], rotation=45, ha='right')
        ax.set_ylabel('Average Total Cost ($)', fontsize=12)
        ax.set_title('Best vs Worst Performing Conditions', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
    def _plot_memory_visibility_combinations(self, ax):
        """Analyze memory-visibility interaction in detail"""
        # Create grouped bar plot for each combination
        combo_stats = self.llama_data.groupby(['memory_strategy', 'visibility_level']).agg({
            'total_cost': ['mean', 'sem'],
            'service_level': 'mean'
        }).reset_index()
        
        # Flatten columns
        combo_stats.columns = ['memory', 'visibility', 'cost_mean', 'cost_sem', 'service_level']
        
        # Create x-axis positions
        memory_levels = combo_stats['memory'].unique()
        visibility_levels = combo_stats['visibility'].unique()
        
        x = np.arange(len(memory_levels))
        width = 0.25
        multiplier = 0
        
        # Plot each visibility level
        for visibility in visibility_levels:
            subset = combo_stats[combo_stats['visibility'] == visibility]
            offset = width * multiplier
            bars = ax.bar(x + offset, subset['cost_mean'], width,
                          yerr=subset['cost_sem']*1.96, capsize=3,
                          label=visibility, alpha=0.8)
            multiplier += 1
        
        # Add OR baseline
        or_baseline = self.or_data['total_cost'].mean()
        ax.axhline(y=or_baseline, color='red', linestyle='--', linewidth=2,
                  label=f'OR Average: ${or_baseline:.0f}', alpha=0.7)
        
        ax.set_xlabel('Memory Strategy', fontsize=12)
        ax.set_ylabel('Average Total Cost ($)', fontsize=12)
        ax.set_title('Performance by Memory Strategy and Visibility Level', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(memory_levels)
        ax.legend(title='Visibility Level')
        ax.grid(True, alpha=0.3, axis='y')
        
    def _plot_scenario_gamemode_matrix(self, ax):
        """Create matrix view of scenario × game mode performance"""
        matrix_data = self.llama_data.groupby(['scenario', 'game_mode'])['total_cost'].mean().unstack()
        
        # Add OR baseline for comparison
        or_matrix = self.or_data.groupby(['scenario', 'game_mode'])['total_cost'].mean().unstack()
        
        # Calculate relative performance (% difference from OR)
        relative_perf = ((matrix_data - or_matrix) / or_matrix * 100).round(1)
        
        # Create annotated heatmap
        sns.heatmap(relative_perf, annot=True, fmt='.1f', cmap='RdYlGn_r',
                   center=0, cbar_kws={'label': '% Difference from OR'},
                   ax=ax, vmin=-20, vmax=20)
        
        ax.set_title('Relative Performance: Llama vs OR (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Game Mode', fontsize=10)
        ax.set_ylabel('Scenario', fontsize=10)
        
    def _plot_or_method_breakdown(self, ax):
        """Show how each OR method performs"""
        or_performance = self.or_data.groupby('model_name').agg({
            'total_cost': ['mean', 'std', 'min', 'max', 'count']
        }).round(0)
        
        # Flatten columns
        or_performance.columns = ['mean', 'std', 'min', 'max', 'count']
        or_performance = or_performance.sort_values('mean')
        
        # Create bar plot with error bars
        x = range(len(or_performance))
        bars = ax.bar(x, or_performance['mean'], yerr=or_performance['std'],
                      capsize=5, color='lightcoral', edgecolor='black', alpha=0.8)
        
        # Add value labels
        for i, (idx, row) in enumerate(or_performance.iterrows()):
            ax.text(i, row['mean'] + row['std'] + 100,
                   f"${row['mean']:.0f}\nn={row['count']}", 
                   ha='center', va='bottom', fontsize=9)
        
        # Add Llama average for comparison
        llama_avg = self.llama_data['total_cost'].mean()
        ax.axhline(y=llama_avg, color='blue', linestyle='--', linewidth=2,
                  label=f'Llama 3.2 Avg: ${llama_avg:.0f}')
        
        ax.set_xticks(x)
        ax.set_xticklabels(or_performance.index, rotation=45, ha='right')
        ax.set_ylabel('Average Total Cost ($)', fontsize=12)
        ax.set_title('OR Method Performance Breakdown', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
    def _plot_significance_table(self, ax):
        """Create statistical significance table"""
        ax.axis('tight')
        ax.axis('off')
        
        # Perform t-tests for key comparisons
        comparisons = []
        
        # 1. Llama vs Best OR
        best_or = self.or_data[self.or_data['model_name'] == 'basestock']['total_cost']
        llama_all = self.llama_data['total_cost']
        t_stat, p_val = stats.ttest_ind(llama_all, best_or)
        comparisons.append(['Llama 3.2 vs Best OR (Basestock)', 
                          f'{llama_all.mean():.0f} vs {best_or.mean():.0f}',
                          f'{t_stat:.3f}', f'{p_val:.4f}',
                          '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'])
        
        # 2. Memory strategies
        for memory in ['none', 'short', 'full']:
            if memory in self.llama_data['memory_strategy'].values:
                memory_data = self.llama_data[self.llama_data['memory_strategy'] == memory]['total_cost']
                t_stat, p_val = stats.ttest_ind(memory_data, best_or)
                comparisons.append([f'Memory: {memory} vs OR',
                                  f'{memory_data.mean():.0f} vs {best_or.mean():.0f}',
                                  f'{t_stat:.3f}', f'{p_val:.4f}',
                                  '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'])
        
        # 3. Best Llama condition vs OR
        best_llama_condition = self.llama_data.groupby(['memory_strategy', 'visibility_level'])['total_cost'].mean().idxmin()
        best_llama_data = self.llama_data[(self.llama_data['memory_strategy'] == best_llama_condition[0]) & 
                                         (self.llama_data['visibility_level'] == best_llama_condition[1])]['total_cost']
        t_stat, p_val = stats.ttest_ind(best_llama_data, best_or)
        comparisons.append([f'Best Llama ({best_llama_condition[0]}-{best_llama_condition[1]}) vs OR',
                          f'{best_llama_data.mean():.0f} vs {best_or.mean():.0f}',
                          f'{t_stat:.3f}', f'{p_val:.4f}',
                          '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'])
        
        # Create table
        table = ax.table(cellText=comparisons,
                        colLabels=['Comparison', 'Mean Costs', 't-statistic', 'p-value', 'Significance'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.35, 0.2, 0.15, 0.15, 0.15])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        # Style the table
        for i in range(len(comparisons) + 1):
            for j in range(5):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#34495e')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    if j == 4:  # Significance column
                        sig = comparisons[i-1][4]
                        if sig == '***':
                            cell.set_facecolor('#90EE90')
                        elif sig == '**':
                            cell.set_facecolor('#FFFFE0')
                        elif sig == '*':
                            cell.set_facecolor('#FFE4B5')
                    else:
                        cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax.set_title('Statistical Significance Tests (*** p<0.001, ** p<0.01, * p<0.05)',
                    fontsize=12, fontweight='bold', pad=20)
        
    def _plot_effect_sizes(self, ax):
        """Calculate and visualize effect sizes for each factor"""
        # Calculate Cohen's d for each factor
        effect_sizes = []
        
        # Baseline: best OR method performance
        or_baseline = self.or_data[self.or_data['model_name'] == 'basestock']['total_cost']
        baseline_mean = or_baseline.mean()
        baseline_std = or_baseline.std()
        
        # Memory strategy effects
        for memory in self.llama_data['memory_strategy'].unique():
            data = self.llama_data[self.llama_data['memory_strategy'] == memory]['total_cost']
            cohen_d = (data.mean() - baseline_mean) / np.sqrt((data.std()**2 + baseline_std**2) / 2)
            effect_sizes.append({'Factor': f'Memory: {memory}', 'Cohen_d': cohen_d, 
                               'Category': 'Memory Strategy'})
        
        # Visibility effects
        for vis in self.llama_data['visibility_level'].unique():
            data = self.llama_data[self.llama_data['visibility_level'] == vis]['total_cost']
            cohen_d = (data.mean() - baseline_mean) / np.sqrt((data.std()**2 + baseline_std**2) / 2)
            effect_sizes.append({'Factor': f'Visibility: {vis}', 'Cohen_d': cohen_d,
                               'Category': 'Visibility Level'})
        
        # Other factors...
        effect_df = pd.DataFrame(effect_sizes)
        effect_df = effect_df.sort_values('Cohen_d')
        
        # Create horizontal bar plot
        colors = {'Memory Strategy': 'steelblue', 'Visibility Level': 'darkgreen'}
        bar_colors = [colors.get(cat, 'gray') for cat in effect_df['Category']]
        
        bars = ax.barh(range(len(effect_df)), effect_df['Cohen_d'], 
                       color=bar_colors, edgecolor='black', alpha=0.8)
        
        # Add reference lines for effect size interpretation
        ax.axvline(x=0, color='black', linewidth=1)
        ax.axvline(x=-0.2, color='gray', linestyle=':', alpha=0.5, label='Small')
        ax.axvline(x=-0.5, color='gray', linestyle='--', alpha=0.5, label='Medium')
        ax.axvline(x=-0.8, color='gray', linestyle='-', alpha=0.5, label='Large')
        
        ax.set_yticks(range(len(effect_df)))
        ax.set_yticklabels(effect_df['Factor'])
        ax.set_xlabel("Cohen's d (Effect Size)", fontsize=12)
        ax.set_title('Effect Sizes: Factor Impact on Performance vs OR Baseline', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='x')
        
    def _plot_variance_decomposition(self, ax):
        """Analyze variance contribution of each factor"""
        # Simple ANOVA-style variance decomposition
        factors = ['memory_strategy', 'visibility_level', 'prompt_type', 'scenario', 'game_mode']
        
        variance_contrib = []
        total_variance = self.llama_data['total_cost'].var()
        
        for factor in factors:
            # Calculate between-group variance
            group_means = self.llama_data.groupby(factor)['total_cost'].mean()
            group_counts = self.llama_data.groupby(factor).size()
            grand_mean = self.llama_data['total_cost'].mean()
            
            between_var = sum(group_counts * (group_means - grand_mean)**2) / len(self.llama_data)
            var_explained = (between_var / total_variance) * 100
            
            variance_contrib.append({'Factor': factor.replace('_', ' ').title(), 
                                   'Variance Explained (%)': var_explained})
        
        var_df = pd.DataFrame(variance_contrib)
        var_df = var_df.sort_values('Variance Explained (%)', ascending=False)
        
        # Create pie chart
        colors = plt.cm.Set3(range(len(var_df)))
        wedges, texts, autotexts = ax.pie(var_df['Variance Explained (%)'], 
                                          labels=var_df['Factor'],
                                          autopct='%1.1f%%',
                                          colors=colors,
                                          explode=[0.05]*len(var_df))
        
        ax.set_title('Variance Decomposition by Factor', fontsize=12, fontweight='bold')
        
    def _plot_performance_stability(self, ax):
        """Analyze performance stability across runs"""
        # Calculate coefficient of variation for each condition
        factors = ['memory_strategy', 'visibility_level']
        
        stability_data = self.llama_data.groupby(factors).agg({
            'total_cost': ['mean', 'std', 'count']
        }).reset_index()
        
        stability_data.columns = factors + ['mean', 'std', 'count']
        stability_data = stability_data[stability_data['count'] >= 5]
        stability_data['cv'] = stability_data['std'] / stability_data['mean']
        stability_data['label'] = stability_data['memory_strategy'] + '-' + stability_data['visibility_level']
        
        # Sort by CV
        stability_data = stability_data.sort_values('cv')
        
        # Create bar plot
        bars = ax.bar(range(len(stability_data)), stability_data['cv'],
                      color='purple', edgecolor='black', alpha=0.8)
        
        # Color bars by performance level
        for i, (idx, row) in enumerate(stability_data.iterrows()):
            if row['cv'] < 0.1:
                bars[i].set_facecolor('green')
            elif row['cv'] < 0.2:
                bars[i].set_facecolor('yellow')
            else:
                bars[i].set_facecolor('red')
        
        ax.set_xticks(range(len(stability_data)))
        ax.set_xticklabels(stability_data['label'], rotation=45, ha='right')
        ax.set_ylabel('Coefficient of Variation', fontsize=12)
        ax.set_title('Performance Stability by Condition (Lower is Better)', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add legend
        green_patch = mpatches.Patch(color='green', label='High Stability (CV < 0.1)')
        yellow_patch = mpatches.Patch(color='yellow', label='Medium Stability (0.1 < CV < 0.2)')
        red_patch = mpatches.Patch(color='red', label='Low Stability (CV > 0.2)')
        ax.legend(handles=[green_patch, yellow_patch, red_patch], loc='upper left')

def main():
    parser = argparse.ArgumentParser(description='Generate SCM-Arena research analysis dashboard')
    parser.add_argument('database', help='Path to the SQLite database')
    args = parser.parse_args()
    
    # Check if database exists
    if not Path(args.database).exists():
        print(f"❌ Error: Database not found at {args.database}")
        sys.exit(1)
    
    print("🔬 SCM-Arena Research Analysis Dashboard")
    print("=" * 50)
    
    # Create and run dashboard
    dashboard = SCMArenaResearchDashboard(args.database)
    dashboard.load_data()
    dashboard.generate_research_dashboard()
    
    print("\n✅ All dashboards generated successfully!")
    print("📊 Check your directory for:")
    print("   - scm_arena_main_effects.png")
    print("   - scm_arena_interaction_effects.png")
    print("   - scm_arena_conditional_analysis.png")
    print("   - scm_arena_factor_importance.png")

if __name__ == "__main__":
    main()