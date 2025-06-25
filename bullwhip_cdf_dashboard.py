#!/usr/bin/env python3
"""
Bullwhip Effect and Cost CDF Analysis Dashboard for SCM-Arena
Shows bullwhip ratios and cumulative distribution functions for each experimental cell
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
from matplotlib.gridspec import GridSpec

class BullwhipCDFDashboard:
    def __init__(self, db_path):
        """Initialize dashboard with database connection"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.experiments = None
        self.llama_data = None
        self.or_data = None
        
    def load_data(self):
        """Load all experiments from database"""
        # Load experiments
        self.experiments = pd.read_sql_query("SELECT * FROM experiments", self.conn)
        
        # Separate Llama and OR experiments
        self.llama_data = self.experiments[self.experiments['model_name'] == 'llama3.2'].copy()
        self.or_data = self.experiments[self.experiments['model_name'] != 'llama3.2'].copy()
        
        print("📊 DATA LOADED")
        print("=" * 50)
        print(f"Llama 3.2: {len(self.llama_data)} experiments")
        print(f"OR Baselines: {len(self.or_data)} experiments")
        
    def generate_bullwhip_dashboard(self):
        """Generate comprehensive bullwhip and CDF analysis"""
        # Create main dashboard
        self._create_bullwhip_overview()
        self._create_memory_strategy_analysis()
        self._create_visibility_level_analysis()
        self._create_scenario_analysis()
        self._create_combined_factor_analysis()
        
    def _create_bullwhip_overview(self):
        """Create overview of bullwhip effects across all conditions"""
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('SCM-Arena: Bullwhip Effect & Cost Distribution Overview', fontsize=24, fontweight='bold')
        
        gs = GridSpec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. Overall Bullwhip Comparison
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_overall_bullwhip_comparison(ax1)
        
        # 2. Cost CDF Comparison
        ax2 = fig.add_subplot(gs[1, :2])
        self._plot_overall_cost_cdf(ax2)
        
        # 3. Bullwhip vs Cost Scatter
        ax3 = fig.add_subplot(gs[1, 2])
        self._plot_bullwhip_cost_scatter(ax3)
        
        # 4. Bullwhip Distribution by Model
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_bullwhip_distribution(ax4)
        
        # 5. Cost Distribution by Model
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_cost_distribution_comparison(ax5)
        
        # 6. Performance Quadrant Analysis
        ax6 = fig.add_subplot(gs[2, 2])
        self._plot_performance_quadrants(ax6)
        
        # 7. Summary Statistics
        ax7 = fig.add_subplot(gs[3, :])
        self._plot_summary_statistics(ax7)
        
        plt.tight_layout()
        
        # Save
        output_path = Path(self.db_path).parent / 'bullwhip_overview.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Bullwhip overview saved to: {output_path}")
        
    def _create_memory_strategy_analysis(self):
        """Analyze bullwhip and cost by memory strategy"""
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('Bullwhip Effect & Cost CDF by Memory Strategy', fontsize=22, fontweight='bold')
        
        memory_strategies = sorted(self.llama_data['memory_strategy'].unique())
        n_strategies = len(memory_strategies)
        
        gs = GridSpec(2, n_strategies, hspace=0.3, wspace=0.3)
        
        for i, memory in enumerate(memory_strategies):
            # Bullwhip effect subplot
            ax_bull = fig.add_subplot(gs[0, i])
            self._plot_bullwhip_by_condition(ax_bull, 'memory_strategy', memory)
            
            # Cost CDF subplot
            ax_cdf = fig.add_subplot(gs[1, i])
            self._plot_cost_cdf_by_condition(ax_cdf, 'memory_strategy', memory)
        
        plt.tight_layout()
        
        # Save
        output_path = Path(self.db_path).parent / 'bullwhip_memory_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Memory strategy analysis saved to: {output_path}")
        
    def _create_visibility_level_analysis(self):
        """Analyze bullwhip and cost by visibility level"""
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('Bullwhip Effect & Cost CDF by Visibility Level', fontsize=22, fontweight='bold')
        
        visibility_levels = sorted(self.llama_data['visibility_level'].unique())
        n_levels = len(visibility_levels)
        
        gs = GridSpec(2, n_levels, hspace=0.3, wspace=0.3)
        
        for i, visibility in enumerate(visibility_levels):
            # Bullwhip effect subplot
            ax_bull = fig.add_subplot(gs[0, i])
            self._plot_bullwhip_by_condition(ax_bull, 'visibility_level', visibility)
            
            # Cost CDF subplot
            ax_cdf = fig.add_subplot(gs[1, i])
            self._plot_cost_cdf_by_condition(ax_cdf, 'visibility_level', visibility)
        
        plt.tight_layout()
        
        # Save
        output_path = Path(self.db_path).parent / 'bullwhip_visibility_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visibility level analysis saved to: {output_path}")
        
    def _create_scenario_analysis(self):
        """Analyze bullwhip and cost by scenario"""
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('Bullwhip Effect & Cost CDF by Scenario', fontsize=22, fontweight='bold')
        
        scenarios = sorted(self.llama_data['scenario'].unique())
        n_scenarios = len(scenarios)
        
        # Create 2 rows for each scenario
        gs = GridSpec(2, n_scenarios, hspace=0.3, wspace=0.3)
        
        for i, scenario in enumerate(scenarios):
            # Bullwhip effect subplot
            ax_bull = fig.add_subplot(gs[0, i])
            self._plot_bullwhip_by_condition(ax_bull, 'scenario', scenario)
            
            # Cost CDF subplot
            ax_cdf = fig.add_subplot(gs[1, i])
            self._plot_cost_cdf_by_condition(ax_cdf, 'scenario', scenario)
        
        plt.tight_layout()
        
        # Save
        output_path = Path(self.db_path).parent / 'bullwhip_scenario_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Scenario analysis saved to: {output_path}")
        
    def _create_combined_factor_analysis(self):
        """Create matrix view of bullwhip and cost for factor combinations"""
        fig = plt.figure(figsize=(24, 20))
        fig.suptitle('Bullwhip & Cost Analysis: Memory × Visibility Combinations', fontsize=24, fontweight='bold')
        
        memory_strategies = sorted(self.llama_data['memory_strategy'].unique())
        visibility_levels = sorted(self.llama_data['visibility_level'].unique())
        
        n_rows = len(memory_strategies)
        n_cols = len(visibility_levels)
        
        gs = GridSpec(n_rows * 2, n_cols, hspace=0.4, wspace=0.3)
        
        for i, memory in enumerate(memory_strategies):
            for j, visibility in enumerate(visibility_levels):
                # Get data for this combination
                mask = (self.llama_data['memory_strategy'] == memory) & \
                       (self.llama_data['visibility_level'] == visibility)
                cell_data = self.llama_data[mask]
                
                if len(cell_data) > 0:
                    # Bullwhip subplot
                    ax_bull = fig.add_subplot(gs[i*2, j])
                    self._plot_cell_bullwhip(ax_bull, cell_data, f"{memory}-{visibility}")
                    
                    # Cost CDF subplot
                    ax_cdf = fig.add_subplot(gs[i*2+1, j])
                    self._plot_cell_cost_cdf(ax_cdf, cell_data, f"{memory}-{visibility}")
        
        plt.tight_layout()
        
        # Save
        output_path = Path(self.db_path).parent / 'bullwhip_combined_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Combined factor analysis saved to: {output_path}")
        
    def _plot_overall_bullwhip_comparison(self, ax):
        """Plot overall bullwhip comparison between Llama and OR methods"""
        # Prepare data
        llama_bull = self.llama_data['bullwhip_ratio'].dropna()
        or_bulls = {}
        
        for method in self.or_data['model_name'].unique():
            or_bulls[method] = self.or_data[self.or_data['model_name'] == method]['bullwhip_ratio'].dropna()
        
        # Create violin plot
        all_data = [llama_bull] + list(or_bulls.values())
        all_labels = ['Llama 3.2'] + list(or_bulls.keys())
        
        parts = ax.violinplot(all_data, positions=range(len(all_labels)), 
                             showmeans=True, showextrema=True, showmedians=True)
        
        # Color the violins
        colors = ['#FF6B6B'] + ['#4ECDC4'] * (len(all_labels) - 1)
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        # Add reference line at 1
        ax.axhline(y=1, color='green', linestyle='--', linewidth=2, 
                  label='No Amplification', alpha=0.7)
        
        # Add mean values
        for i, data in enumerate(all_data):
            mean_val = np.mean(data)
            ax.text(i, mean_val + 0.1, f'{mean_val:.2f}', 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xticks(range(len(all_labels)))
        ax.set_xticklabels(all_labels, rotation=45, ha='right')
        ax.set_ylabel('Bullwhip Ratio', fontsize=12)
        ax.set_title('Bullwhip Effect Comparison: All Methods', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
    def _plot_overall_cost_cdf(self, ax):
        """Plot cost CDFs for Llama vs OR methods"""
        # Llama CDF
        llama_costs = np.sort(self.llama_data['total_cost'].values)
        llama_cdf = np.arange(1, len(llama_costs) + 1) / len(llama_costs)
        ax.plot(llama_costs, llama_cdf, label='Llama 3.2', linewidth=3, color='#FF6B6B')
        
        # OR method CDFs
        colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#F7DC6F', '#BB8FCE']
        for i, method in enumerate(self.or_data['model_name'].unique()):
            method_costs = np.sort(self.or_data[self.or_data['model_name'] == method]['total_cost'].values)
            method_cdf = np.arange(1, len(method_costs) + 1) / len(method_costs)
            ax.plot(method_costs, method_cdf, label=method, linewidth=2, 
                   color=colors[i % len(colors)], alpha=0.8)
        
        # Add percentile markers
        percentiles = [25, 50, 75, 90]
        for p in percentiles:
            llama_percentile = np.percentile(self.llama_data['total_cost'], p)
            ax.axvline(x=llama_percentile, color='red', linestyle=':', alpha=0.3)
            ax.text(llama_percentile, p/100, f'P{p}', rotation=90, 
                   va='bottom', ha='right', fontsize=8, color='red')
        
        ax.set_xlabel('Total Cost ($)', fontsize=12)
        ax.set_ylabel('Cumulative Probability', fontsize=12)
        ax.set_title('Cost Cumulative Distribution Functions', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, np.percentile(self.experiments['total_cost'], 95))
        
    def _plot_bullwhip_cost_scatter(self, ax):
        """Scatter plot of bullwhip ratio vs cost"""
        # Plot each method
        for method in self.experiments['model_name'].unique():
            method_data = self.experiments[self.experiments['model_name'] == method]
            color = '#FF6B6B' if method == 'llama3.2' else '#4ECDC4'
            size = 50 if method == 'llama3.2' else 30
            alpha = 0.6 if method == 'llama3.2' else 0.4
            
            ax.scatter(method_data['total_cost'], method_data['bullwhip_ratio'],
                      label=method, color=color, s=size, alpha=alpha, edgecolors='black', linewidth=0.5)
        
        # Add reference lines
        ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='No Amplification')
        
        # Add trend line for Llama
        llama_data = self.experiments[self.experiments['model_name'] == 'llama3.2']
        z = np.polyfit(llama_data['total_cost'], llama_data['bullwhip_ratio'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(llama_data['total_cost'].min(), llama_data['total_cost'].max(), 100)
        ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Llama Trend')
        
        ax.set_xlabel('Total Cost ($)', fontsize=12)
        ax.set_ylabel('Bullwhip Ratio', fontsize=12)
        ax.set_title('Bullwhip Effect vs Cost Trade-off', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
    def _plot_bullwhip_distribution(self, ax):
        """Plot bullwhip ratio distributions"""
        # Prepare data for box plot
        data_dict = {}
        for method in self.experiments['model_name'].unique():
            data_dict[method] = self.experiments[self.experiments['model_name'] == method]['bullwhip_ratio'].dropna()
        
        # Sort by median
        sorted_methods = sorted(data_dict.keys(), 
                               key=lambda x: data_dict[x].median())
        
        # Create box plot
        bp = ax.boxplot([data_dict[m] for m in sorted_methods],
                       labels=sorted_methods, patch_artist=True,
                       showmeans=True, meanline=True)
        
        # Color boxes
        for i, (patch, method) in enumerate(zip(bp['boxes'], sorted_methods)):
            color = '#FF6B6B' if method == 'llama3.2' else '#4ECDC4'
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add reference line
        ax.axhline(y=1, color='green', linestyle='--', linewidth=2, alpha=0.7)
        
        ax.set_xticklabels(sorted_methods, rotation=45, ha='right')
        ax.set_ylabel('Bullwhip Ratio', fontsize=12)
        ax.set_title('Bullwhip Distribution by Method', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
    def _plot_cost_distribution_comparison(self, ax):
        """Plot cost distributions using KDE"""
        # Plot KDE for each method
        for method in self.experiments['model_name'].unique():
            method_data = self.experiments[self.experiments['model_name'] == method]['total_cost']
            color = '#FF6B6B' if method == 'llama3.2' else '#4ECDC4'
            linewidth = 3 if method == 'llama3.2' else 2
            
            method_data.plot.kde(ax=ax, label=method, color=color, 
                               linewidth=linewidth, alpha=0.8)
        
        # Add mean lines
        for method in self.experiments['model_name'].unique():
            mean_cost = self.experiments[self.experiments['model_name'] == method]['total_cost'].mean()
            color = '#FF6B6B' if method == 'llama3.2' else '#4ECDC4'
            ax.axvline(x=mean_cost, color=color, linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Total Cost ($)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Cost Distribution Comparison', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, np.percentile(self.experiments['total_cost'], 95))
        
    def _plot_performance_quadrants(self, ax):
        """Create quadrant analysis of cost vs bullwhip"""
        # Calculate medians for quadrant boundaries
        median_cost = self.experiments['total_cost'].median()
        median_bullwhip = self.experiments['bullwhip_ratio'].median()
        
        # Plot each experiment
        for method in self.experiments['model_name'].unique():
            method_data = self.experiments[self.experiments['model_name'] == method]
            color = '#FF6B6B' if method == 'llama3.2' else '#4ECDC4'
            
            ax.scatter(method_data['total_cost'], method_data['bullwhip_ratio'],
                      label=method, color=color, alpha=0.6, s=30)
        
        # Add quadrant lines
        ax.axvline(x=median_cost, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=median_bullwhip, color='gray', linestyle='--', alpha=0.5)
        
        # Label quadrants
        ax.text(median_cost * 0.5, median_bullwhip * 1.5, 'Low Cost\nHigh Amplification',
               ha='center', va='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
               facecolor='red', alpha=0.2))
        ax.text(median_cost * 1.5, median_bullwhip * 1.5, 'High Cost\nHigh Amplification',
               ha='center', va='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3",
               facecolor='orange', alpha=0.2))
        ax.text(median_cost * 0.5, median_bullwhip * 0.5, 'Low Cost\nLow Amplification',
               ha='center', va='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3",
               facecolor='green', alpha=0.2))
        ax.text(median_cost * 1.5, median_bullwhip * 0.5, 'High Cost\nLow Amplification',
               ha='center', va='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3",
               facecolor='yellow', alpha=0.2))
        
        ax.set_xlabel('Total Cost ($)', fontsize=12)
        ax.set_ylabel('Bullwhip Ratio', fontsize=12)
        ax.set_title('Performance Quadrants', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
    def _plot_summary_statistics(self, ax):
        """Create summary statistics table"""
        ax.axis('tight')
        ax.axis('off')
        
        # Calculate statistics
        stats_data = []
        
        for method in sorted(self.experiments['model_name'].unique()):
            method_data = self.experiments[self.experiments['model_name'] == method]
            
            stats_data.append([
                method,
                f"${method_data['total_cost'].mean():.0f} ± {method_data['total_cost'].std():.0f}",
                f"{method_data['bullwhip_ratio'].mean():.2f} ± {method_data['bullwhip_ratio'].std():.2f}",
                f"{method_data['service_level'].mean()*100:.1f}%",
                f"${method_data['total_cost'].quantile(0.5):.0f}",
                f"{method_data['bullwhip_ratio'].quantile(0.5):.2f}",
                len(method_data)
            ])
        
        # Create table
        table = ax.table(cellText=stats_data,
                        colLabels=['Method', 'Avg Cost ± SD', 'Avg Bullwhip ± SD', 
                                  'Avg Service', 'Median Cost', 'Median Bullwhip', 'N'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.15, 0.18, 0.18, 0.12, 0.12, 0.13, 0.08])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        # Style the table
        for i in range(len(stats_data) + 1):
            for j in range(7):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#2C3E50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    # Highlight Llama row
                    if stats_data[i-1][0] == 'llama3.2':
                        cell.set_facecolor('#FFE5E5')
                    else:
                        cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax.set_title('Summary Statistics: Bullwhip Effect and Cost Performance',
                    fontsize=14, fontweight='bold', pad=20)
        
    def _plot_bullwhip_by_condition(self, ax, factor, level):
        """Plot bullwhip distribution for a specific condition"""
        # Get data for this condition
        llama_condition = self.llama_data[self.llama_data[factor] == level]
        
        # Also get OR baseline for comparison
        or_baseline = self.or_data['bullwhip_ratio'].dropna()
        
        # Create violin plot
        data_to_plot = [llama_condition['bullwhip_ratio'].dropna(), or_baseline]
        parts = ax.violinplot(data_to_plot, positions=[0, 1], 
                             showmeans=True, showextrema=True)
        
        # Color violins
        colors = ['#FF6B6B', '#4ECDC4']
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        # Add reference line
        ax.axhline(y=1, color='green', linestyle='--', linewidth=2, alpha=0.7)
        
        # Add statistics
        ax.text(0, 0.5, f"n={len(llama_condition)}", transform=ax.transAxes,
               fontsize=9, va='center')
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Llama 3.2', 'OR Baseline'])
        ax.set_ylabel('Bullwhip Ratio', fontsize=10)
        ax.set_title(f'{factor.replace("_", " ").title()}: {level}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
    def _plot_cost_cdf_by_condition(self, ax, factor, level):
        """Plot cost CDF for a specific condition"""
        # Get data for this condition
        llama_condition = self.llama_data[self.llama_data[factor] == level]
        
        # Plot Llama CDF
        llama_costs = np.sort(llama_condition['total_cost'].values)
        llama_cdf = np.arange(1, len(llama_costs) + 1) / len(llama_costs)
        ax.plot(llama_costs, llama_cdf, label=f'Llama ({level})', 
               linewidth=3, color='#FF6B6B')
        
        # Plot OR baseline CDF for comparison
        or_costs = np.sort(self.or_data['total_cost'].values)
        or_cdf = np.arange(1, len(or_costs) + 1) / len(or_costs)
        ax.plot(or_costs, or_cdf, label='OR Baseline', 
               linewidth=2, color='#4ECDC4', alpha=0.7, linestyle='--')
        
        # Add percentile markers
        percentiles = [25, 50, 75]
        for p in percentiles:
            if len(llama_costs) > 0:
                percentile_val = np.percentile(llama_costs, p)
                ax.axvline(x=percentile_val, color='red', linestyle=':', alpha=0.3)
        
        ax.set_xlabel('Total Cost ($)', fontsize=10)
        ax.set_ylabel('Cumulative Probability', fontsize=10)
        ax.set_title(f'Cost CDF: {level}', fontsize=11, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
    def _plot_cell_bullwhip(self, ax, cell_data, label):
        """Plot bullwhip distribution for a specific cell"""
        if len(cell_data) < 2:
            ax.text(0.5, 0.5, f'Insufficient data\n(n={len(cell_data)})', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{label}', fontsize=10)
            return
            
        # Create histogram with KDE
        cell_data['bullwhip_ratio'].hist(ax=ax, bins=15, alpha=0.7, 
                                         color='steelblue', edgecolor='black')
        
        # Add KDE if enough data
        if len(cell_data) > 5:
            cell_data['bullwhip_ratio'].plot.kde(ax=ax, secondary_y=True, 
                                                 color='red', linewidth=2)
        
        # Add statistics
        mean_val = cell_data['bullwhip_ratio'].mean()
        median_val = cell_data['bullwhip_ratio'].median()
        ax.axvline(x=mean_val, color='red', linestyle='--', alpha=0.7, 
                  label=f'Mean: {mean_val:.2f}')
        ax.axvline(x=median_val, color='green', linestyle='--', alpha=0.7,
                  label=f'Median: {median_val:.2f}')
        
        # Add reference line at 1
        ax.axvline(x=1, color='black', linestyle=':', linewidth=2, alpha=0.5)
        
        ax.set_xlabel('Bullwhip Ratio', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.set_title(f'{label} (n={len(cell_data)})', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        
    def _plot_cell_cost_cdf(self, ax, cell_data, label):
        """Plot cost CDF for a specific cell"""
        if len(cell_data) < 2:
            ax.text(0.5, 0.5, f'Insufficient data\n(n={len(cell_data)})', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{label}', fontsize=10)
            return
            
        # Sort costs and create CDF
        costs = np.sort(cell_data['total_cost'].values)
        cdf = np.arange(1, len(costs) + 1) / len(costs)
        
        # Plot CDF
        ax.plot(costs, cdf, linewidth=2, color='darkgreen')
        
        # Add percentile markers
        percentiles = [25, 50, 75, 90]
        for p in percentiles:
            percentile_val = np.percentile(costs, p)
            ax.axvline(x=percentile_val, color='red', linestyle=':', alpha=0.3)
            ax.text(percentile_val, p/100, f'P{p}', rotation=90,
                   va='bottom', ha='right', fontsize=7)
        
        # Add mean
        mean_cost = cell_data['total_cost'].mean()
        ax.axvline(x=mean_cost, color='blue', linestyle='--', alpha=0.7,
                  label=f'Mean: ${mean_cost:.0f}')
        
        ax.set_xlabel('Total Cost ($)', fontsize=9)
        ax.set_ylabel('Cumulative Probability', fontsize=9)
        ax.set_title(f'{label} Cost CDF', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # Set reasonable x-limits
        ax.set_xlim(0, np.percentile(costs, 95) * 1.1)

def main():
    parser = argparse.ArgumentParser(description='Generate Bullwhip and CDF analysis dashboard')
    parser.add_argument('database', help='Path to the SQLite database')
    args = parser.parse_args()
    
    # Check if database exists
    if not Path(args.database).exists():
        print(f"❌ Error: Database not found at {args.database}")
        sys.exit(1)
    
    print("📊 SCM-Arena: Bullwhip Effect & Cost CDF Analysis")
    print("=" * 50)
    
    # Create and run dashboard
    dashboard = BullwhipCDFDashboard(args.database)
    dashboard.load_data()
    dashboard.generate_bullwhip_dashboard()
    
    print("\n✅ All dashboards generated successfully!")
    print("📊 Generated files:")
    print("   - bullwhip_overview.png")
    print("   - bullwhip_memory_analysis.png")
    print("   - bullwhip_visibility_analysis.png")
    print("   - bullwhip_scenario_analysis.png")
    print("   - bullwhip_combined_analysis.png")

if __name__ == "__main__":
    main()