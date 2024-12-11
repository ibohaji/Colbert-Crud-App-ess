from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots 
import numpy as np
import altair as alt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np




def test_plot(data1_set1: dict, data2_set1: dict,
              data1_set2: dict, data2_set2: dict,
              model1_name: str, model2_name: str,
              dataset1_name: str, dataset2_name: str) -> plt.Figure:
    """
    Creates publication-quality plots comparing two models across two datasets.
    """
    # Set clean, professional style
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'DejaVu Serif',  # More widely available font
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.2,
        'axes.axisbelow': True,
    })
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Colors for the models
    color1 = '#2ecc71'  # Green
    color2 = '#3498db'  # Blue
    
    def plot_comparison(ax, data1, data2, dataset_name):
        k_values = list(data1['NDCG'].keys())
        x_positions = range(len(k_values))
        
        # Calculate metrics for both models
        means1 = [np.mean(data1['NDCG'][k]) for k in k_values]
        stderr1 = [np.std(data1['NDCG'][k]) / np.sqrt(len(data1['NDCG'][k])) * 5 for k in k_values]
        
        means2 = [np.mean(data2['NDCG'][k]) for k in k_values]
        stderr2 = [np.std(data2['NDCG'][k]) / np.sqrt(len(data2['NDCG'][k])) * 5 for k in k_values]
        
        offset = 0.15
        
        # First model
        ax.errorbar([x - offset for x in x_positions], means1, yerr=stderr1, 
                   fmt='o', color=color1,
                   capsize=4, capthick=1.5, markersize=7, elinewidth=1.5,
                   label=model1_name, zorder=3)
        
        # Second model
        ax.errorbar([x + offset for x in x_positions], means2, yerr=stderr2, 
                   fmt='o', color=color2,
                   capsize=4, capthick=1.5, markersize=7, elinewidth=1.5,
                   label=model2_name, zorder=3)
        
        # Add value annotations
        for i, (mean1, mean2) in enumerate(zip(means1, means2)):
            ax.annotate(f'{mean1:.3f}', 
                       (x_positions[i] - offset, mean1),
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', va='bottom', fontsize=9,
                       alpha=0.8)
            ax.annotate(f'{mean2:.3f}', 
                       (x_positions[i] + offset, mean2),
                       xytext=(0, -20), textcoords='offset points',
                       ha='center', va='top', fontsize=9,
                       alpha=0.8)
        
        # Configure axis
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f'k@{k}' for k in k_values])
        ax.set_xlabel('Cutoff Value', fontweight='normal', labelpad=10)
        ax.set_ylabel('NDCG Score', fontweight='normal', labelpad=10)
        
        # Dataset name as title
        ax.set_title(f'{dataset_name}', fontweight='bold', pad=15)
        
        # Set limits
        all_means = means1 + means2
        all_stderr = stderr1 + stderr2
        ax.set_ylim(max(0, min(all_means) - max(all_stderr) - 0.1),
                    min(1, max(all_means) + max(all_stderr) + 0.1))
        ax.set_xlim(-0.5, len(k_values) - 0.5)
        
        # Refine grid and spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        return ax
    
    # Plot both datasets
    plot_comparison(ax1, data1_set1, data2_set1, dataset1_name)
    plot_comparison(ax2, data1_set2, data2_set2, dataset2_name)
    
    # Add single legend for both plots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels,
              loc='upper center',
              bbox_to_anchor=(0.5, 1.05),
              ncol=2,
              frameon=True,
              facecolor='white',
              framealpha=1,
              edgecolor='none')
    
    plt.tight_layout()
    return fig


def test_plotly(data: dict, model_name: str) -> go.Figure:
    """
    Creates a sophisticated, publication-quality plot using Plotly.
    """
    k_values = list(data['NDCG'].keys())
    means = [np.mean(data['NDCG'][k]) for k in k_values]
    stderr = [np.std(data['NDCG'][k]) / np.sqrt(len(data['NDCG'][k])) * 5 for k in k_values]
    
    fig = go.Figure()
    
    # Add error bars and points
    fig.add_trace(go.Scatter(
        x=[f'k={k}' for k in k_values],  # Categorical x-values
        y=means,
        error_y=dict(
            type='data',
            array=stderr,
            color='rgba(53, 97, 235, 0.3)',
            thickness=1.5,
            width=6,
        ),
        mode='markers+text',
        marker=dict(
            symbol='circle',
            size=14,
            color='rgba(53, 97, 235, 0.9)',
            line=dict(color='white', width=2)
        ),
        text=[f'{m:.3f}' for m in means],
        textposition='top center',
        textfont=dict(
            family='Arial',
            size=13,
            color='rgba(50, 50, 50, 0.8)'
        ),
        name=model_name
    ))

    fig.update_layout(
        template='none',
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(
            family='Arial',
            size=14,
            color='#2c3e50'
        ),
        title=dict(
            text=f'NDCG Scores for {model_name}',
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=20)
        ),
        xaxis=dict(
            title='Cutoff Value',
            titlefont=dict(size=16),
            type='category',  # Ensure categorical x-axis
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor='rgba(0,0,0,0.2)'
        ),
        yaxis=dict(
            title='NDCG Score',
            titlefont=dict(size=16),
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True,
            zeroline=False,
            range=[0, 1],
            showline=True,
            linewidth=1,
            linecolor='rgba(0,0,0,0.2)'
        ),
        showlegend=False,
        margin=dict(t=80, b=60, l=60, r=40),
        width=800,
        height=500
    )
    
    return fig

def test_plot_altair(data: dict, model_name: str) -> alt.Chart:
    """
    Creates an elegant, minimalist plot using Altair.
    """
    # Prepare data
    df = pd.DataFrame([
        {
            'k': f'k={k}',
            'mean': np.mean(data['NDCG'][k]),
            'stderr': np.std(data['NDCG'][k]) / np.sqrt(len(data['NDCG'][k])) * 5
        }
        for k in data['NDCG'].keys()
    ])
    
    df['upper'] = df['mean'] + df['stderr']
    df['lower'] = df['mean'] - df['stderr']
    
    # Base chart
    base = alt.Chart(df).encode(
        x=alt.X('k:N',
                title='Cutoff Value',
                axis=alt.Axis(labelAngle=0))
    )
    
    # Error bars
    error_bars = base.mark_rule(
        opacity=0.4
    ).encode(
        y='lower:Q',
        y2='upper:Q'
    )
    
    # Points
    points = base.mark_circle(
        size=150,
        opacity=0.9,
        color='#3561eb'  # Using hex color directly
    ).encode(
        y=alt.Y('mean:Q',
                title='NDCG Score',
                scale=alt.Scale(domain=[0, 1])),
        tooltip=['k', alt.Tooltip('mean:Q', format='.3f')]
    )
    
    # Text labels
    text = base.mark_text(
        align='center',
        baseline='bottom',
        dy=-15,
        fontSize=12,
        color='#333333'
    ).encode(
        y='mean:Q',
        text=alt.Text('mean:Q', format='.3f')
    )
    
    # Combine all elements
    chart = (error_bars + points + text).properties(
        width=600,
        height=400,
        title=alt.TitleParams(
            text=f'NDCG Scores for {model_name}',
            fontSize=20,
            color='#333333'
        )
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14,
        grid=True,
        gridColor='#EEEEEE',
        gridOpacity=0.5
    ).configure_view(
        strokeWidth=0
    )
    
    return chart