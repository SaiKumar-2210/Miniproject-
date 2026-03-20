"""
Evaluation Metrics Visualization
Generates publication-quality charts from evaluation_results.csv
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import sys

# Use non-interactive backend
matplotlib.use('Agg')

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# --- Config ---
RESULTS_PATH = os.path.join("models", "evaluation_results.csv")
OUTPUT_DIR = os.path.join("models", "metrics_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color palette
COLORS = {
    'Baseline': '#3b82f6',   # blue
    'LSTM': '#f59e0b',       # amber
    'Hybrid': '#ef4444',     # red
}

def load_results():
    df = pd.read_csv(RESULTS_PATH)
    df['Label'] = df['Commodity'] + '\n' + df['District']
    return df

# ─────────────────────────────────────────────
# 1. RMSE Comparison - Grouped Bar Chart
# ─────────────────────────────────────────────
def plot_rmse_comparison(df):
    fig, ax = plt.subplots(figsize=(16, 7))
    
    labels = df['Label'].values
    x = np.arange(len(labels))
    width = 0.25
    
    # Only plot models that have data
    bars_plotted = 0
    if df['Baseline_RMSE'].notna().any():
        ax.bar(x - width, df['Baseline_RMSE'], width, label='Baseline (ARIMA)',
               color=COLORS['Baseline'], edgecolor='white', linewidth=0.5)
        bars_plotted += 1
    if df['LSTM_RMSE'].notna().any():
        ax.bar(x, df['LSTM_RMSE'], width, label='LSTM',
               color=COLORS['LSTM'], edgecolor='white', linewidth=0.5)
        bars_plotted += 1
    if df['Hybrid_RMSE'].notna().any():
        offset = 0 if bars_plotted < 2 else width
        ax.bar(x + offset, df['Hybrid_RMSE'], width, label='Hybrid (ARIMA+LSTM)',
               color=COLORS['Hybrid'], edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Commodity / District', fontsize=11, fontweight='bold')
    ax.set_ylabel('RMSE (₹)', fontsize=11, fontweight='bold')
    ax.set_title('Model Comparison — Root Mean Squared Error (RMSE)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, ha='center')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'rmse_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {path}")

# ─────────────────────────────────────────────
# 2. MAPE Comparison - Grouped Bar Chart
# ─────────────────────────────────────────────
def plot_mape_comparison(df):
    fig, ax = plt.subplots(figsize=(16, 7))
    
    labels = df['Label'].values
    x = np.arange(len(labels))
    width = 0.25
    
    bars_plotted = 0
    if df['Baseline_MAPE'].notna().any():
        ax.bar(x - width, df['Baseline_MAPE'] * 100, width, label='Baseline (ARIMA)',
               color=COLORS['Baseline'], edgecolor='white', linewidth=0.5)
        bars_plotted += 1
    if df['LSTM_MAPE'].notna().any():
        ax.bar(x, df['LSTM_MAPE'] * 100, width, label='LSTM',
               color=COLORS['LSTM'], edgecolor='white', linewidth=0.5)
        bars_plotted += 1
    if df['Hybrid_MAPE'].notna().any():
        offset = 0 if bars_plotted < 2 else width
        ax.bar(x + offset, df['Hybrid_MAPE'] * 100, width, label='Hybrid (ARIMA+LSTM)',
               color=COLORS['Hybrid'], edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Commodity / District', fontsize=11, fontweight='bold')
    ax.set_ylabel('MAPE (%)', fontsize=11, fontweight='bold')
    ax.set_title('Model Comparison — Mean Absolute Percentage Error (MAPE)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, ha='center')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'mape_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {path}")

# ─────────────────────────────────────────────
# 3. Average RMSE by Commodity - Horizontal Bar
# ─────────────────────────────────────────────
def plot_avg_rmse_by_commodity(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    grouped = df.groupby('Commodity')[['Baseline_RMSE', 'LSTM_RMSE', 'Hybrid_RMSE']].mean()
    commodities = grouped.index.tolist()
    y = np.arange(len(commodities))
    height = 0.25
    
    if grouped['Baseline_RMSE'].notna().any():
        ax.barh(y - height, grouped['Baseline_RMSE'], height, label='Baseline (ARIMA)',
                color=COLORS['Baseline'], edgecolor='white')
    if grouped['LSTM_RMSE'].notna().any():
        ax.barh(y, grouped['LSTM_RMSE'], height, label='LSTM',
                color=COLORS['LSTM'], edgecolor='white')
    if grouped['Hybrid_RMSE'].notna().any():
        ax.barh(y + height, grouped['Hybrid_RMSE'], height, label='Hybrid (ARIMA+LSTM)',
                color=COLORS['Hybrid'], edgecolor='white')
    
    ax.set_xlabel('Average RMSE (₹)', fontsize=11, fontweight='bold')
    ax.set_title('Average RMSE by Commodity', fontsize=14, fontweight='bold', pad=15)
    ax.set_yticks(y)
    ax.set_yticklabels(commodities, fontsize=11)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'avg_rmse_by_commodity.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {path}")

# ─────────────────────────────────────────────
# 4. Average MAPE by Commodity - Horizontal Bar
# ─────────────────────────────────────────────
def plot_avg_mape_by_commodity(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    grouped = df.groupby('Commodity')[['Baseline_MAPE', 'LSTM_MAPE', 'Hybrid_MAPE']].mean()
    commodities = grouped.index.tolist()
    y = np.arange(len(commodities))
    height = 0.25
    
    if grouped['Baseline_MAPE'].notna().any():
        ax.barh(y - height, grouped['Baseline_MAPE'] * 100, height, label='Baseline (ARIMA)',
                color=COLORS['Baseline'], edgecolor='white')
    if grouped['LSTM_MAPE'].notna().any():
        ax.barh(y, grouped['LSTM_MAPE'] * 100, height, label='LSTM',
                color=COLORS['LSTM'], edgecolor='white')
    if grouped['Hybrid_MAPE'].notna().any():
        ax.barh(y + height, grouped['Hybrid_MAPE'] * 100, height, label='Hybrid (ARIMA+LSTM)',
                color=COLORS['Hybrid'], edgecolor='white')
    
    ax.set_xlabel('Average MAPE (%)', fontsize=11, fontweight='bold')
    ax.set_title('Average MAPE by Commodity', fontsize=14, fontweight='bold', pad=15)
    ax.set_yticks(y)
    ax.set_yticklabels(commodities, fontsize=11)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'avg_mape_by_commodity.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {path}")

# ─────────────────────────────────────────────
# 5. RMSE Heatmap per District × Commodity
# ─────────────────────────────────────────────
def plot_rmse_heatmap(df):
    # Use Baseline RMSE (most complete) for the heatmap
    pivot = df.pivot_table(index='Commodity', columns='District', values='Baseline_RMSE')
    
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticklabels(pivot.index, fontsize=10)
    
    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text_color = 'white' if val > pivot.values[~np.isnan(pivot.values)].mean() else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                        fontsize=9, fontweight='bold', color=text_color)
    
    ax.set_title('Baseline (ARIMA) RMSE — District × Commodity Heatmap', fontsize=13, fontweight='bold', pad=15)
    plt.colorbar(im, ax=ax, label='RMSE (₹)', shrink=0.8)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'rmse_heatmap_baseline.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {path}")

    # Hybrid RMSE heatmap
    pivot_h = df.pivot_table(index='Commodity', columns='District', values='Hybrid_RMSE')
    
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot_h.values, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(np.arange(len(pivot_h.columns)))
    ax.set_yticks(np.arange(len(pivot_h.index)))
    ax.set_xticklabels(pivot_h.columns, fontsize=10)
    ax.set_yticklabels(pivot_h.index, fontsize=10)
    
    for i in range(len(pivot_h.index)):
        for j in range(len(pivot_h.columns)):
            val = pivot_h.values[i, j]
            if not np.isnan(val):
                text_color = 'white' if val > pivot_h.values[~np.isnan(pivot_h.values)].mean() else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                        fontsize=9, fontweight='bold', color=text_color)
    
    ax.set_title('Hybrid (ARIMA+LSTM) RMSE — District × Commodity Heatmap', fontsize=13, fontweight='bold', pad=15)
    plt.colorbar(im, ax=ax, label='RMSE (₹)', shrink=0.8)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'rmse_heatmap_hybrid.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {path}")

# ─────────────────────────────────────────────
# 6. Model Win Rate - Pie Chart
# ─────────────────────────────────────────────
def plot_model_win_rate(df):
    """Which model has lower RMSE per commodity-district?"""
    valid = df.dropna(subset=['Baseline_RMSE', 'Hybrid_RMSE'])
    
    baseline_wins = (valid['Baseline_RMSE'] < valid['Hybrid_RMSE']).sum()
    hybrid_wins = (valid['Hybrid_RMSE'] < valid['Baseline_RMSE']).sum()
    ties = (valid['Baseline_RMSE'] == valid['Hybrid_RMSE']).sum()
    
    labels = []
    sizes = []
    colors = []
    if baseline_wins > 0:
        labels.append(f'Baseline Wins ({baseline_wins})')
        sizes.append(baseline_wins)
        colors.append(COLORS['Baseline'])
    if hybrid_wins > 0:
        labels.append(f'Hybrid Wins ({hybrid_wins})')
        sizes.append(hybrid_wins)
        colors.append(COLORS['Hybrid'])
    if ties > 0:
        labels.append(f'Tie ({ties})')
        sizes.append(ties)
        colors.append('#94a3b8')
    
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
                                        startangle=90, textprops={'fontsize': 12})
    for at in autotexts:
        at.set_fontweight('bold')
        at.set_color('white')
    
    ax.set_title('Model Win Rate (RMSE-based)', fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'model_win_rate.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {path}")

# ─────────────────────────────────────────────
# 7. Summary Table Image
# ─────────────────────────────────────────────
def plot_summary_table(df):
    summary = df.groupby('Commodity').agg({
        'Baseline_RMSE': 'mean',
        'Baseline_MAPE': lambda x: f"{x.mean()*100:.2f}%",
        'Hybrid_RMSE': 'mean',
        'Hybrid_MAPE': lambda x: f"{x.mean()*100:.2f}%",
    }).reset_index()
    
    summary.columns = ['Commodity', 'Baseline\nRMSE', 'Baseline\nMAPE', 'Hybrid\nRMSE', 'Hybrid\nMAPE']
    summary['Baseline\nRMSE'] = summary['Baseline\nRMSE'].apply(lambda x: f'{x:.2f}')
    summary['Hybrid\nRMSE'] = summary['Hybrid\nRMSE'].apply(lambda x: f'{x:.2f}')
    
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')
    
    table = ax.table(
        cellText=summary.values,
        colLabels=summary.columns,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Style header
    for j, col in enumerate(summary.columns):
        cell = table[0, j]
        cell.set_facecolor('#1e293b')
        cell.set_text_props(color='white', fontweight='bold')
    
    # Alternate row colors
    for i in range(1, len(summary) + 1):
        for j in range(len(summary.columns)):
            cell = table[i, j]
            cell.set_facecolor('#f1f5f9' if i % 2 == 0 else 'white')
    
    ax.set_title('Average Evaluation Metrics Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'summary_table.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {path}")

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("GENERATING EVALUATION METRIC GRAPHS")
    print("=" * 50)
    
    df = load_results()
    print(f"Loaded {len(df)} evaluation records.\n")
    
    plot_rmse_comparison(df)
    plot_mape_comparison(df)
    plot_avg_rmse_by_commodity(df)
    plot_avg_mape_by_commodity(df)
    plot_rmse_heatmap(df)
    plot_model_win_rate(df)
    plot_summary_table(df)
    
    print(f"\n✓ All graphs saved to: {OUTPUT_DIR}/")
    print("=" * 50)
