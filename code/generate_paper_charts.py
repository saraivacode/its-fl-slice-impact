import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import json
import glob
import os

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================

# Configure style for academic publication (IEEE/Scientific)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

# Define the directory containing the result files
# Adjust this path if necessary to point to your data folder
results_dir = '../results/v1/paper_artifacts'
summary_table_file = os.path.join(results_dir, 'summary_table.csv')

# Load the dataset directly from the CSV file
df = pd.read_csv(summary_table_file)

# Robust data cleaning function
def clean_currency(x):
    """Removes 's' suffix and handles '-' by converting to NaN."""
    if isinstance(x, str):
        return pd.to_numeric(x.replace('s', '').replace('-', ''), errors='coerce')
    return x

# Apply cleaning to relevant columns
df['Total'] = df['Total'].apply(clean_currency)
df['Avg Train'] = df['Avg Train'].apply(clean_currency)
df['Acc'] = pd.to_numeric(df['Acc'], errors='coerce')

# ==========================================
# CHART 1: PERFORMANCE GAP (BAR CHART)
# Visualizes the impact of Non-IID distribution
# ==========================================
def plot_performance_gap():
    plt.figure(figsize=(10, 6))
    
    # Filter data: Exclude Centralized rows (they will be reference lines) 
    # and Redundant IID FedProx (FedAvg is sufficient baseline)
    plot_df = df[
        (df['Dist'] != 'CENTRAL') & 
        ~((df['Dist'] == 'IID') & (df['Strategy'] == 'FEDPROX'))
    ].copy()
    
    # Create descriptive legend labels
    plot_df['Scenario'] = plot_df.apply(lambda x: f"{x['Dist']} ({x['Strategy']})", axis=1)
    
    # Define order of bars for consistency
    scenario_order = ['IID (FEDAVG)', 'NONIID (FEDAVG)', 'NONIID (FEDPROX)']
    
    # Plot bars
    ax = sns.barplot(
        data=plot_df, 
        x='Model', 
        y='Acc', 
        hue='Scenario', 
        hue_order=scenario_order, 
        palette=['#bdc3c7', '#e74c3c', '#2980b9'], # Grey (Baseline), Red (Drop), Blue (Attempt)
        edgecolor="black",
        linewidth=1
    )
    
    # Add Centralized Baselines as dashed lines (Theoretical Upper Bound)
    centralized_data = df[df['Dist'] == 'CENTRAL'].set_index('Model')['Acc']
    
    for i, model in enumerate(['DNN', 'LSTM', 'GRU']):
        if model in centralized_data.index:
            val = centralized_data[model]
            x_min, x_max = i - 0.4, i + 0.4
            plt.hlines(y=val, xmin=x_min, xmax=x_max, colors='black', linestyles='--', linewidth=2)
            plt.text(i, val + 0.005, f"Central: {val:.3f}", ha='center', va='bottom', 
                     fontsize=10, fontweight='bold', color='black')

    plt.ylim(0.70, 1.05)
    plt.title('Performance Gap: IID vs. Non-IID Scenarios', fontweight='bold', pad=15)
    plt.ylabel('Accuracy')
    plt.xlabel('')
    plt.legend(title='', loc='lower right', framealpha=0.95, fancybox=True)

    # Annotation
    plt.annotate('~20% Drop', xy=(0.1, 0.77), xytext=(0.5, 0.85),
                 arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=11)

    plt.tight_layout()
    fig_perf = os.path.join(results_dir, 'paper_fig1_performance_gap_v3.png')
    plt.savefig(fig_perf, dpi=300)
    plt.show()
    
    print(f"Chart 1 saved as {fig_perf}")

# ==========================================
# CHART 2: EFFICIENCY TRADE-OFF (SCATTER)
# Visualizes Cost vs Benefit
# ==========================================
def plot_efficiency_tradeoff():
    plt.figure(figsize=(9, 7))
    
    # Focus only on Non-IID scenarios (The real-world challenge)
    fed_df = df[df['Dist'] == 'NONIID'].copy()
    
    sns.scatterplot(
        data=fed_df, 
        x='Total', 
        y='Acc', 
        hue='Model', 
        style='Strategy', 
        s=250, # Marker size
        palette='viridis', 
        markers={'FEDAVG': 'o', 'FEDPROX': 'X'},
        edgecolor='black'
    )
    
    # Annotate points with model names
    for i in range(fed_df.shape[0]):
        row = fed_df.iloc[i]
        plt.text(row['Total']+0.8, row['Acc'], f"{row['Model']}", fontsize=10, weight='semibold')

    plt.title('Efficiency Trade-off (Non-IID)', fontweight='bold')
    plt.xlabel('Total Training Time (s)')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
               fancybox=True, shadow=False, ncol=3)

    plt.subplots_adjust(bottom=0.2)
    fig_eff = os.path.join(results_dir, 'paper_fig2_efficiency_v3.png')
    plt.savefig(fig_eff, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Chart 2 saved as {fig_eff}")

# ==========================================
# CHART 3: CONVERGENCE CURVES (FROM JSONs)
# Shows learning stability over rounds
# ==========================================
def plot_convergence_from_json():
    # List of files to compare (Non-IID FedAvg vs FedProx for GRU/LSTM)
    target_files = [
        'gru_noniid_fedavg_e5.json', 
        'gru_noniid_fedprox_e5.json',
        'lstm_noniid_fedavg_e5.json',
        'lstm_noniid_fedprox_e5.json'
    ]
    
    plt.figure(figsize=(10, 6))
    colors = {'gru': 'green', 'lstm': 'blue'}
    styles = {'fedavg': '-', 'fedprox': '--'}
    
    file_found = False
    for fname in target_files:
        full_path = os.path.join(results_dir, fname)
        try:
            with open(full_path, 'r') as f:
                data = json.load(f)
                
            model = data['config']['model_type']
            strategy = data['config']['strategy']
            rounds = data['rounds']
            acc = data['accuracy']
            
            label = f"{model.upper()} ({strategy})"
            plt.plot(rounds, acc, label=label, color=colors.get(model, 'gray'), 
                     linestyle=styles.get(strategy, '-'), linewidth=3, marker='o')
            file_found = True
            
        except FileNotFoundError:
            print(f"Warning: File {fname} not found in {results_dir}. Skipping line.")

    if file_found:
        plt.title('Convergence Analysis: Non-IID Scenarios', fontweight='bold', pad=15)
        plt.xlabel('Communication Round')
        plt.ylabel('Global Accuracy')
        plt.xticks(range(1, 11)) # rounds 1 to 10
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()

        fig_conv = os.path.join(results_dir, 'paper_fig3_convergence_v3.png')
        plt.savefig(fig_conv, dpi=300)
        plt.show()
        print(f"Chart 3 saved as {fig_conv}")
    else:
        print("No JSON files found. Skipping convergence plot.")

# ==========================================
# EXECUTE ALL
# ==========================================
if __name__ == "__main__":
    plot_performance_gap()
    plot_efficiency_tradeoff()
    plot_convergence_from_json()