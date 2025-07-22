import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

def create_cp_plot(cp_results):
    """
    Create Critical Power plot with corrected hyperbolic model visualization.
    Shows the relationship P = CP + W'/t as described in the paper.
    
    Parameters:
    -----------
    cp_results : dict
        Dictionary containing CP analysis results with corrected calculation
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the CP plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    durations = cp_results['durations']
    max_powers = cp_results['max_powers']
    cp = cp_results['cp']
    w_prime = cp_results['w_prime']
    r_squared = cp_results['r_squared']
    
    # Plot data points
    duration_labels = ['1 min', '5 min', '12 min']
    valid_durations = [d for d, p in zip(durations, max_powers) if not np.isnan(p)]
    valid_powers = [p for p in max_powers if not np.isnan(p)]
    
    ax.scatter(valid_durations, valid_powers, s=150, color='red', zorder=5, 
               label='Maximum Average Power', edgecolors='darkred', linewidth=2)
    
    # Plot fitted hyperbolic curve: P = CP + W'/t
    t_extended = np.linspace(30, 1200, 1000)
    p_fitted = cp + (w_prime / t_extended)
    ax.plot(t_extended, p_fitted, 'b--', linewidth=3, alpha=0.8,
            label=f'Hyperbolic Model: P = {cp:.1f} + {w_prime:.0f}/t')
    
    # Add CP asymptote line
    ax.axhline(y=cp, color='green', linestyle=':', linewidth=3, alpha=0.8,
               label=f'Critical Power = {cp:.1f} W')
    
    # Add annotations for data points
    for i, (dur, pwr, label) in enumerate(zip(valid_durations, valid_powers, duration_labels[:len(valid_powers)])):
        percent_cp = (pwr / cp) * 100 if cp > 0 else 0
        ax.annotate(f'{label}\n{pwr:.0f} W\n({percent_cp:.0f}% CP)', 
                   xy=(dur, pwr), 
                   xytext=(20, 20), 
                   textcoords='offset points',
                   fontsize=11,
                   ha='left',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8, edgecolor='orange'),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', color='black'))
    
    # Add model quality information
    ax.text(0.02, 0.98, f'Model Quality:\nR² = {r_squared:.3f}\nW\' = {w_prime:.0f} J', 
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    ax.set_xlabel('Duration (seconds)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Power (W)', fontsize=14, fontweight='bold')
    ax.set_title('Critical Power Analysis\n(Monod-Scherrer Hyperbolic Model)', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    # Set reasonable axis limits
    ax.set_xlim(0, max(1300, max(valid_durations) * 1.1))
    ax.set_ylim(cp * 0.8, max(valid_powers) * 1.1)
    
    plt.tight_layout()
    return fig

def create_quartile_plots(df_15min):
    """
    Create temporal quartile analysis plots showing rHRI, heart rate derivative, and power.
    Uses TEMPORAL quartiles (Q1-Q4 based on time) as per paper methodology.
    
    Parameters:
    -----------
    df_15min : pandas.DataFrame
        DataFrame containing 15-minute interval data with temporal quartiles
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the quartile plots
    """
    quartiles = ['Q1', 'Q2', 'Q3', 'Q4']
    quartile_labels = ['Q1\n(0-25% time)', 'Q2\n(25-50% time)', 'Q3\n(50-75% time)', 'Q4\n(75-100% time)']
    
    # Calculate metrics by temporal quartile
    rHRI_by_quartile = []
    fc_deriv_by_quartile = []
    power_percent_by_quartile = []
    sample_counts = []
    
    for q in quartiles:
        if 'quartile' in df_15min.columns:
            q_data = df_15min[df_15min['quartile'] == q]
        else:
            # Fallback to power_quartile if quartile column doesn't exist
            q_data = df_15min[df_15min.get('power_quartile', pd.Series()) == q]
        
        if len(q_data) > 0:
            rHRI_by_quartile.append(q_data['rHRI'].mean())
            fc_deriv_by_quartile.append(q_data['fc_deriv'].mean())
            power_percent_by_quartile.append(q_data['power_percent_cp'].mean())
            sample_counts.append(len(q_data))
        else:
            rHRI_by_quartile.append(0)
            fc_deriv_by_quartile.append(0)
            power_percent_by_quartile.append(0)
            sample_counts.append(0)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # rHRI plot (main metric from paper)
    colors_rHRI = ['#E3F2FD', '#BBDEFB', '#64B5F6', '#1976D2']
    bars1 = axes[0].bar(quartile_labels, rHRI_by_quartile, 
                        color=colors_rHRI, edgecolor='navy', linewidth=1.5)
    axes[0].set_xlabel('Temporal Quartile', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('rHRI (bpm/s)', fontsize=12, fontweight='bold')
    axes[0].set_title('Relative Heart Rate Increase (rHRI)\nby Temporal Quartiles', fontsize=14, fontweight='bold')
    
    # Add value labels and sample counts
    for bar, value, count in zip(bars1, rHRI_by_quartile, sample_counts):
        height = bar.get_height()
        if not np.isnan(value) and height != 0:
            axes[0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.4f}\n(n={count})', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Heart rate derivative plot
    colors_hr = ['#E8F5E8', '#A5D6A7', '#66BB6A', '#2E7D32']
    bars2 = axes[1].bar(quartile_labels, fc_deriv_by_quartile, 
                        color=colors_hr, edgecolor='darkgreen', linewidth=1.5)
    axes[1].set_xlabel('Temporal Quartile', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Heart Rate Derivative (bpm/s)', fontsize=12, fontweight='bold')
    axes[1].set_title('Heart Rate Change Rate\nby Temporal Quartiles', fontsize=14, fontweight='bold')
    
    for bar, value, count in zip(bars2, fc_deriv_by_quartile, sample_counts):
        height = bar.get_height()
        if not np.isnan(value) and abs(height) > 1e-6:
            axes[1].text(bar.get_x() + bar.get_width()/2., height + (abs(height)*0.05 if height >= 0 else -abs(height)*0.05),
                        f'{value:.4f}\n(n={count})', ha='center', va='bottom' if height >= 0 else 'top', 
                        fontsize=10, fontweight='bold')
    
    # Power percentage plot
    colors_power = ['#FFF3E0', '#FFE0B2', '#FFCC02', '#F57C00']
    bars3 = axes[2].bar(quartile_labels, power_percent_by_quartile, 
                        color=colors_power, edgecolor='darkorange', linewidth=1.5)
    axes[2].set_xlabel('Temporal Quartile', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Power (% of CP)', fontsize=12, fontweight='bold')
    axes[2].set_title('Average Power as % of CP\nby Temporal Quartiles', fontsize=14, fontweight='bold')
    
    for bar, value, count in zip(bars3, power_percent_by_quartile, sample_counts):
        height = bar.get_height()
        if not np.isnan(value) and height != 0:
            axes[2].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.1f}%\n(n={count})', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add grid to all subplots
    for ax in axes:
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
    
    # Add main title
    fig.suptitle('Cardiovascular Dynamics Analysis by Temporal Quartiles\n(Based on Race Time Progression)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    return fig

def create_time_series_plot(df, df_15min, cp):
    """
    Create time series plot with 15-minute intervals showing temporal progression.
    Emphasizes the temporal quartile methodology from the paper.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing full cycling data
    df_15min : pandas.DataFrame
        DataFrame containing 15-minute interval data
    cp : float
        Critical Power value
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the time series plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    time_labels = df_15min['time_bin'] / 60  # Convert to minutes
    
    # Plot 1: Power analysis with CP reference
    bars_power = axes[0].bar(time_labels, df_15min['power_smooth'], width=14, alpha=0.7, 
                            color='steelblue', edgecolor='navy', linewidth=1, label='Average Power')
    axes[0].axhline(y=cp, color='red', linestyle='--', linewidth=3, alpha=0.8,
                    label=f'Critical Power = {cp:.0f} W')
    
    # Add power as % of CP on secondary axis
    ax0_twin = axes[0].twinx()
    power_percent_line = ax0_twin.plot(time_labels + 7.5, df_15min['power_percent_cp'], 
                                      'go-', linewidth=3, markersize=8, alpha=0.8, label='Power (% CP)')
    ax0_twin.set_ylabel('Power (% of CP)', color='green', fontsize=12, fontweight='bold')
    ax0_twin.tick_params(axis='y', labelcolor='green')
    
    axes[0].set_ylabel('Power (W)', fontsize=12, fontweight='bold')
    axes[0].set_title('Power Output and Intensity Progression', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper left', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Heart Rate progression
    hr_line = axes[1].plot(time_labels + 7.5, df_15min['heart_rate_smooth'], 
                          'ro-', linewidth=3, markersize=8, alpha=0.8, label='Average HR')
    
    # Add HR derivative on secondary axis
    ax1_twin = axes[1].twinx()
    hr_deriv_bars = ax1_twin.bar(time_labels, df_15min['fc_deriv'], width=14, alpha=0.5, 
                                color='orange', edgecolor='darkorange', linewidth=1, label='HR Derivative')
    ax1_twin.set_ylabel('HR Derivative (bpm/s)', color='orange', fontsize=12, fontweight='bold')
    ax1_twin.tick_params(axis='y', labelcolor='orange')
    
    axes[1].set_ylabel('Heart Rate (bpm)', color='red', fontsize=12, fontweight='bold')
    axes[1].set_title('Heart Rate and Cardiovascular Response', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper left', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='y', labelcolor='red')
    
    # Plot 3: rHRI colored by temporal quartiles
    quartile_column = 'quartile' if 'quartile' in df_15min.columns else 'power_quartile'
    colors = {'Q1': '#E3F2FD', 'Q2': '#BBDEFB', 'Q3': '#64B5F6', 'Q4': '#1976D2'}
    bar_colors = [colors.get(str(q), 'gray') for q in df_15min[quartile_column]]
    
    rHRI_bars = axes[2].bar(time_labels, df_15min['rHRI'], width=14, alpha=0.8, 
                           color=bar_colors, edgecolor='navy', linewidth=1)
    
    axes[2].set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('rHRI (bpm/s)', fontsize=12, fontweight='bold')
    axes[2].set_title('Relative Heart Rate Increase (rHRI) by Temporal Quartiles', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    # Add temporal quartile legend
    legend_elements = [Patch(facecolor=colors[q], edgecolor='navy', label=f'{q} (Time)') 
                      for q in ['Q1', 'Q2', 'Q3', 'Q4']]
    axes[2].legend(handles=legend_elements, title='Temporal Quartile', loc='upper right', fontsize=11)
    
    # Add quartile dividers
    total_time = time_labels.max()
    for i, frac in enumerate([0.25, 0.5, 0.75]):
        quartile_time = total_time * frac
        for ax in axes:
            ax.axvline(x=quartile_time, color='black', linestyle=':', alpha=0.5, linewidth=1)
            if ax == axes[0]:  # Only label on top plot
                ax.text(quartile_time, ax.get_ylim()[1]*0.95, f'Q{i+1}|Q{i+2}', 
                       ha='center', va='top', fontsize=10, 
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def create_overview_plot(df):
    """
    Create overview plot of entire ride with improved temporal quartile visualization.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing cycling data
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the overview plot
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    time_minutes = df['time'] / 60
    
    # Power plot with quartile backgrounds
    ax1.plot(time_minutes, df['power_smooth'], 'b-', linewidth=2, alpha=0.8, label='Power (Smoothed)')
    ax1.fill_between(time_minutes, 0, df['power_smooth'], alpha=0.3, color='blue')
    ax1.set_ylabel('Power (W)', fontsize=12, fontweight='bold')
    ax1.set_title('Complete Ride Overview with Temporal Quartiles', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=11)
    
    # Heart rate plot
    ax2.plot(time_minutes, df['heart_rate_smooth'], 'r-', linewidth=2, alpha=0.8, label='Heart Rate (Smoothed)')
    ax2.fill_between(time_minutes, df['heart_rate_smooth'].min()*0.9, df['heart_rate_smooth'], alpha=0.3, color='red')
    ax2.set_ylabel('Heart Rate (bpm)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=11)
    
    # rHRI plot (if available)
    if 'rHRI' in df.columns:
        rHRI_clean = df['rHRI'].dropna()
        time_clean = time_minutes[df['rHRI'].notna()]
        if len(rHRI_clean) > 0:
            ax3.scatter(time_clean, rHRI_clean, c='green', alpha=0.6, s=20, label='rHRI')
            ax3.plot(time_clean, rHRI_clean.rolling(window=50, center=True, min_periods=1).mean(), 
                    'darkgreen', linewidth=2, label='rHRI (Moving Average)')
    
    ax3.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('rHRI (bpm/s)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left', fontsize=11)
    
    # Add temporal quartile dividers
    total_time = time_minutes.max()
    quartile_colors = ['#E3F2FD', '#BBDEFB', '#64B5F6', '#1976D2']
    
    for i, (start_frac, end_frac, color, label) in enumerate([(0, 0.25, quartile_colors[0], 'Q1'),
                                                             (0.25, 0.5, quartile_colors[1], 'Q2'),
                                                             (0.5, 0.75, quartile_colors[2], 'Q3'),
                                                             (0.75, 1.0, quartile_colors[3], 'Q4')]):
        start_time = total_time * start_frac
        end_time = total_time * end_frac
        
        for ax in [ax1, ax2, ax3]:
            ax.axvspan(start_time, end_time, alpha=0.1, color=color)
            if ax == ax1:  # Only label on top plot
                mid_time = (start_time + end_time) / 2
                ax.text(mid_time, ax.get_ylim()[1]*0.95, label, ha='center', va='top', 
                       fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))
    
    plt.tight_layout()
    return fig