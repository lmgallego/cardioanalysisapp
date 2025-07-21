import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

def create_cp_plot(cp_results):
    """
    Create Critical Power plot with fitted curve and data points.
    
    Parameters:
    -----------
    cp_results : dict
        Dictionary containing CP analysis results
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the CP plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    durations = cp_results['durations']
    max_powers = cp_results['max_powers']
    cp = cp_results['cp']
    w_prime = cp_results['w_prime']
    
    # Plot data points
    duration_labels = ['1 min', '5 min', '12 min']
    ax.scatter(durations, max_powers, s=100, color='red', zorder=5, label='Max Average Power')
    
    # Plot fitted curve
    x_extended = np.linspace(30, 1200, 100)
    x_inv_extended = 1 / x_extended
    y_fitted = cp + (w_prime * x_inv_extended)
    ax.plot(x_extended, y_fitted, 'b--', linewidth=2, 
            label=f'CP Model: CP={cp:.1f}W, W\'={w_prime:.0f}J')
    
    # Add CP line
    ax.axhline(y=cp, color='green', linestyle=':', linewidth=2, 
               label=f'Critical Power = {cp:.1f}W')
    
    # Annotations
    for i, (dur, pwr, label) in enumerate(zip(durations, max_powers, duration_labels)):
        if not np.isnan(pwr):
            ax.annotate(f'{label}\n{pwr:.0f}W', 
                       xy=(dur, pwr), 
                       xytext=(10, 10), 
                       textcoords='offset points',
                       fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel('Duration (seconds)', fontsize=12)
    ax.set_ylabel('Power (W)', fontsize=12)
    ax.set_title('Critical Power Analysis', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    return fig

def create_quartile_plots(df_15min):
    """
    Create quartile analysis plots for rHRI, heart rate derivative, and power percentage.
    
    Parameters:
    -----------
    df_15min : pandas.DataFrame
        DataFrame containing 15-minute interval data with quartiles
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the quartile plots
    """
    quartiles = ['Q1', 'Q2', 'Q3', 'Q4']
    
    # Calculate metrics by quartile
    rHRI_by_quartile = []
    fc_deriv_by_quartile = []
    power_percent_by_quartile = []
    
    for q in quartiles:
        q_data = df_15min[df_15min['power_quartile'] == q]
        rHRI_by_quartile.append(q_data['rHRI'].mean() if len(q_data) > 0 else 0)
        fc_deriv_by_quartile.append(q_data['fc_deriv'].mean() if len(q_data) > 0 else 0)
        power_percent_by_quartile.append(q_data['power_percent_cp'].mean() if len(q_data) > 0 else 0)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # rHRI plot
    bars1 = axes[0].bar(quartiles, rHRI_by_quartile, 
                        color=['lightblue', 'skyblue', 'cornflowerblue', 'darkblue'])
    axes[0].set_xlabel('Power Quartile')
    axes[0].set_ylabel('rHRI (bpm/s/W)')
    axes[0].set_title('Relative Heart Rate Increase\n(15-minute intervals)')
    
    for bar, value in zip(bars1, rHRI_by_quartile):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.6f}', ha='center', va='bottom')
    
    # Heart rate derivative plot
    bars2 = axes[1].bar(quartiles, fc_deriv_by_quartile, 
                        color=['lightgreen', 'mediumseagreen', 'seagreen', 'darkgreen'])
    axes[1].set_xlabel('Power Quartile')
    axes[1].set_ylabel('Heart Rate Derivative (bpm/s)')
    axes[1].set_title('Heart Rate Change Rate\n(15-minute intervals)')
    
    for bar, value in zip(bars2, fc_deriv_by_quartile):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}', ha='center', va='bottom')
    
    # Power percentage plot
    bars3 = axes[2].bar(quartiles, power_percent_by_quartile, 
                        color=['lightyellow', 'gold', 'orange', 'darkorange'])
    axes[2].set_xlabel('Power Quartile')
    axes[2].set_ylabel('Power (% of CP)')
    axes[2].set_title('Average Power as % of CP\n(15-minute intervals)')
    
    for bar, value in zip(bars3, power_percent_by_quartile):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def create_time_series_plot(df, df_15min, cp):
    """
    Create time series plot with 15-minute intervals.
    
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
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    time_labels = df_15min['time_bin'] / 60
    
    # Power and heart rate plot
    axes[0].bar(time_labels, df_15min['power_smooth'], width=14, alpha=0.7, 
                color='blue', label='Average Power')
    axes[0].axhline(y=cp, color='red', linestyle='--', linewidth=2, 
                    label=f'CP = {cp:.0f}W')
    axes[0].set_ylabel('Power (W)', fontsize=12)
    axes[0].set_title('Power and Heart Rate by 15-minute Intervals', fontsize=14)
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # Heart rate on secondary axis
    ax1_twin = axes[0].twinx()
    ax1_twin.plot(time_labels + 7.5, df_15min['heart_rate_smooth'], 
                  'ro-', linewidth=2, markersize=8, label='Average HR')
    ax1_twin.set_ylabel('Heart Rate (bpm)', color='red', fontsize=12)
    ax1_twin.tick_params(axis='y', labelcolor='red')
    
    # rHRI and Power Quartiles plot
    colors = {'Q1': 'lightblue', 'Q2': 'skyblue', 'Q3': 'cornflowerblue', 'Q4': 'darkblue'}
    bar_colors = [colors.get(q, 'gray') for q in df_15min['power_quartile']]
    
    bars = axes[1].bar(time_labels, df_15min['rHRI'], width=14, alpha=0.8, color=bar_colors)
    axes[1].set_xlabel('Time (minutes)', fontsize=12)
    axes[1].set_ylabel('rHRI (bpm/s/W)', fontsize=12)
    axes[1].set_title('rHRI by 15-minute Intervals (colored by power quartile)', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    # Add legend for quartiles
    legend_elements = [Patch(facecolor=colors[q], label=f'{q}') for q in ['Q1', 'Q2', 'Q3', 'Q4']]
    axes[1].legend(handles=legend_elements, title='Power Quartile', loc='upper right')
    
    plt.tight_layout()
    return fig

def create_overview_plot(df):
    """
    Create overview plot of entire ride.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing cycling data
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the overview plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    time_minutes = df['time'] / 60
    
    # Power plot
    ax1.plot(time_minutes, df['power_smooth'], 'b-', linewidth=1, alpha=0.8, label='Power')
    ax1.set_ylabel('Power (W)', fontsize=12)
    ax1.set_title('Complete Ride Overview', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Heart rate plot
    ax2.plot(time_minutes, df['heart_rate_smooth'], 'r-', linewidth=1, alpha=0.8, label='Heart Rate')
    ax2.set_xlabel('Time (minutes)', fontsize=12)
    ax2.set_ylabel('Heart Rate (bpm)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    return fig