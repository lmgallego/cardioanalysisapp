import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

def create_cp_plot(cp_results):
    """
    Create interactive Critical Power plot with Plotly
    """
    durations = cp_results['durations']
    max_powers = cp_results['max_powers']
    cp = cp_results['cp']
    w_prime = cp_results['w_prime']
    r_squared = cp_results['r_squared']
    
    # Filter valid data
    valid_durations = [d for d, p in zip(durations, max_powers) if not np.isnan(p)]
    valid_powers = [p for p in max_powers if not np.isnan(p)]
    duration_labels = ['1 min', '5 min', '12 min']
    
    fig = go.Figure()
    
    # Add data points
    fig.add_trace(go.Scatter(
        x=valid_durations,
        y=valid_powers,
        mode='markers',
        marker=dict(size=15, color='red', line=dict(width=2, color='darkred')),
        name='Maximum Average Power',
        hovertemplate='<b>%{text}</b><br>' +
                     'Duration: %{x}s<br>' +
                     'Power: %{y:.0f}W<br>' +
                     'CP Percentage: %{customdata:.0f}%<br>' +
                     '<extra></extra>',
        text=[duration_labels[i] for i in range(len(valid_powers))],
        customdata=[(p/cp)*100 if cp > 0 else 0 for p in valid_powers]
    ))
    
    # Add hyperbolic curve
    t_extended = np.linspace(30, 1200, 1000)
    p_fitted = cp + (w_prime / t_extended)
    
    fig.add_trace(go.Scatter(
        x=t_extended,
        y=p_fitted,
        mode='lines',
        line=dict(color='blue', width=3, dash='dash'),
        name=f'Hyperbolic Model',
        hovertemplate='P = %{y:.0f}W<br>t = %{x:.0f}s<br>' +
                     f'Model: P = {cp:.0f} + {w_prime:.0f}/t<br>' +
                     '<extra></extra>'
    ))
    
    # Add CP line
    fig.add_hline(
        y=cp,
        line=dict(color='green', width=3, dash='dot'),
        annotation_text=f"Critical Power = {cp:.0f}W",
        annotation_position="top right"
    )
    
    # Layout
    fig.update_layout(
        title=dict(
            text='Critical Power Analysis<br><sub>(Monod-Scherrer Hyperbolic Model)</sub>',
            font=dict(size=18),
            x=0.5
        ),
        xaxis=dict(
            title='Duration (seconds)',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        yaxis=dict(
            title='Power (W)',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        hovermode='closest',
        width=900,
        height=500,
        showlegend=True,
        legend=dict(x=0.7, y=0.95),
        template='plotly_white'
    )
    
    # Add model quality annotation
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"Model Quality:<br>R² = {r_squared:.3f}<br>W' = {w_prime:.0f} J",
        showarrow=False,
        font=dict(size=11),
        bgcolor="rgba(173,216,230,0.8)",
        bordercolor="rgba(0,0,139,0.8)",
        borderwidth=1
    )
    
    return fig

def create_quartile_plots(df_15min):
    """
    Create interactive temporal quartile analysis plots
    """
    quartiles = ['Q1', 'Q2', 'Q3', 'Q4']
    quartile_labels = ['Q1<br>(0-25% time)', 'Q2<br>(25-50% time)', 
                      'Q3<br>(50-75% time)', 'Q4<br>(75-100% time)']
    
    # Calculate metrics by temporal quartile
    rHRI_by_quartile = []
    fc_deriv_by_quartile = []
    power_percent_by_quartile = []
    sample_counts = []
    
    quartile_column = 'quartile' if 'quartile' in df_15min.columns else 'power_quartile'
    
    for q in quartiles:
        q_data = df_15min[df_15min[quartile_column] == q]
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
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Relative Heart Rate Increase (rHRI)', 
                       'Heart Rate Change Rate', 
                       'Average Power as % of CP'),
        horizontal_spacing=0.08
    )
    
    # Color schemes
    colors_rHRI = ['#E3F2FD', '#BBDEFB', '#64B5F6', '#1976D2']
    colors_hr = ['#E8F5E8', '#A5D6A7', '#66BB6A', '#2E7D32']  
    colors_power = ['#FFF3E0', '#FFE0B2', '#FFCC02', '#F57C00']
    
    # rHRI plot
    fig.add_trace(
        go.Bar(
            x=quartile_labels,
            y=rHRI_by_quartile,
            marker_color=colors_rHRI,
            marker_line=dict(width=2, color='navy'),
            name='rHRI',
            hovertemplate='<b>%{x}</b><br>' +
                         'rHRI: %{y:.6f} bpm/s<br>' +
                         'Sample size: %{customdata}<br>' +
                         '<extra></extra>',
            customdata=sample_counts,
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Heart rate derivative plot
    fig.add_trace(
        go.Bar(
            x=quartile_labels,
            y=fc_deriv_by_quartile,
            marker_color=colors_hr,
            marker_line=dict(width=2, color='darkgreen'),
            name='HR Derivative',
            hovertemplate='<b>%{x}</b><br>' +
                         'HR Derivative: %{y:.4f} bpm/s<br>' +
                         'Sample size: %{customdata}<br>' +
                         '<extra></extra>',
            customdata=sample_counts,
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Power percentage plot
    fig.add_trace(
        go.Bar(
            x=quartile_labels,
            y=power_percent_by_quartile,
            marker_color=colors_power,
            marker_line=dict(width=2, color='darkorange'),
            name='Power % CP',
            hovertemplate='<b>%{x}</b><br>' +
                         'Power: %{y:.1f}% of CP<br>' +
                         'Sample size: %{customdata}<br>' +
                         '<extra></extra>',
            customdata=sample_counts,
            showlegend=False
        ),
        row=1, col=3
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Cardiovascular Dynamics Analysis by Temporal Quartiles<br>' +
                 '<sub>(Based on Race Time Progression)</sub>',
            font=dict(size=16),
            x=0.5
        ),
        width=1200,
        height=500,
        template='plotly_white',
        showlegend=False
    )
    
    # Update axes
    fig.update_yaxes(title_text="rHRI (bpm/s)", row=1, col=1)
    fig.update_yaxes(title_text="HR Derivative (bpm/s)", row=1, col=2)
    fig.update_yaxes(title_text="Power (% of CP)", row=1, col=3)
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig

def create_time_series_plot(df, df_15min, cp):
    """
    Create interactive time series plot with multiple metrics
    """
    time_labels = df_15min['time_bin'] / 60  # Convert to minutes
    quartile_column = 'quartile' if 'quartile' in df_15min.columns else 'power_quartile'
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Power Output and Intensity', 
                       'Heart Rate Response', 
                       'Relative Heart Rate Increase (rHRI)'),
        vertical_spacing=0.08,
        specs=[[{"secondary_y": True}],
               [{"secondary_y": True}], 
               [{"secondary_y": False}]]
    )
    
    # Plot 1: Power with % CP
    fig.add_trace(
        go.Bar(
            x=time_labels,
            y=df_15min['power_smooth'],
            width=14,
            marker_color='steelblue',
            marker_line=dict(width=1, color='navy'),
            name='Average Power',
            hovertemplate='Time: %{x:.0f}-%{customdata:.0f} min<br>' +
                         'Power: %{y:.0f}W<br>' +
                         '<extra></extra>',
            customdata=time_labels + 15
        ),
        row=1, col=1
    )
    
    # Add CP line
    fig.add_hline(
        y=cp,
        line=dict(color='red', width=3, dash='dash'),
        annotation_text=f"CP = {cp:.0f}W",
        annotation_position="top right",
        row=1
    )
    
    # Add power % CP on secondary axis
    fig.add_trace(
        go.Scatter(
            x=time_labels + 7.5,
            y=df_15min['power_percent_cp'],
            mode='markers+lines',
            marker=dict(size=8, color='green'),
            line=dict(color='green', width=3),
            name='Power (% CP)',
            hovertemplate='Time: %{x:.0f} min<br>' +
                         'Power: %{y:.1f}% CP<br>' +
                         '<extra></extra>',
            yaxis='y2'
        ),
        row=1, col=1, secondary_y=True
    )
    
    # Plot 2: Heart Rate with derivative
    fig.add_trace(
        go.Scatter(
            x=time_labels + 7.5,
            y=df_15min['heart_rate_smooth'],
            mode='markers+lines',
            marker=dict(size=8, color='red'),
            line=dict(color='red', width=3),
            name='Average HR',
            hovertemplate='Time: %{x:.0f} min<br>' +
                         'HR: %{y:.0f} bpm<br>' +
                         '<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Add HR derivative bars on secondary axis
    fig.add_trace(
        go.Bar(
            x=time_labels,
            y=df_15min['fc_deriv'],
            width=14,
            marker_color='orange',
            marker_line=dict(width=1, color='darkorange'),
            opacity=0.6,
            name='HR Derivative',
            hovertemplate='Time: %{x:.0f}-%{customdata:.0f} min<br>' +
                         'HR Derivative: %{y:.4f} bpm/s<br>' +
                         '<extra></extra>',
            customdata=time_labels + 15,
            yaxis='y4'
        ),
        row=2, col=1, secondary_y=True
    )
    
    # Plot 3: rHRI colored by quartiles
    colors = {'Q1': '#E3F2FD', 'Q2': '#BBDEFB', 'Q3': '#64B5F6', 'Q4': '#1976D2'}
    bar_colors = [colors.get(str(q), 'gray') for q in df_15min[quartile_column]]
    
    fig.add_trace(
        go.Bar(
            x=time_labels,
            y=df_15min['rHRI'],
            width=14,
            marker_color=bar_colors,
            marker_line=dict(width=1, color='navy'),
            name='rHRI',
            hovertemplate='Time: %{x:.0f}-%{customdata[0]:.0f} min<br>' +
                         'rHRI: %{y:.6f} bpm/s<br>' +
                         'Quartile: %{customdata[1]}<br>' +
                         '<extra></extra>',
            customdata=list(zip(time_labels + 15, df_15min[quartile_column]))
        ),
        row=3, col=1
    )
    
    # Add quartile dividers
    total_time = time_labels.max()
    for i, frac in enumerate([0.25, 0.5, 0.75]):
        quartile_time = total_time * frac
        fig.add_vline(
            x=quartile_time,
            line=dict(color='black', width=1, dash='dot'),
            opacity=0.5,
            annotation_text=f'Q{i+1}|Q{i+2}',
            annotation_position="top"
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Power, Heart Rate, and Cardiovascular Response Over Time',
            font=dict(size=16),
            x=0.5
        ),
        width=1000,
        height=800,
        template='plotly_white',
        showlegend=True,
        legend=dict(x=1.02, y=1)
    )
    
    # Update axes labels
    fig.update_yaxes(title_text="Power (W)", row=1, col=1)
    fig.update_yaxes(title_text="Power (% CP)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Heart Rate (bpm)", row=2, col=1)  
    fig.update_yaxes(title_text="HR Derivative (bpm/s)", row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="rHRI (bpm/s)", row=3, col=1)
    fig.update_xaxes(title_text="Time (minutes)", row=3, col=1)
    
    return fig

def create_overview_plot(df):
    """
    Create interactive overview plot of entire ride
    """
    time_minutes = df['time'] / 60
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Power Output', 'Heart Rate', 'Relative Heart Rate Increase'),
        vertical_spacing=0.08,
        shared_xaxes=True
    )
    
    # Power plot
    fig.add_trace(
        go.Scatter(
            x=time_minutes,
            y=df['power_smooth'],
            mode='lines',
            line=dict(color='blue', width=1),
            fill='tonexty',
            fillcolor='rgba(0,0,255,0.2)',
            name='Power',
            hovertemplate='Time: %{x:.1f} min<br>' +
                         'Power: %{y:.0f}W<br>' +
                         '<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Heart rate plot
    fig.add_trace(
        go.Scatter(
            x=time_minutes,
            y=df['heart_rate_smooth'],
            mode='lines',
            line=dict(color='red', width=1),
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.2)',
            name='Heart Rate',
            hovertemplate='Time: %{x:.1f} min<br>' +
                         'HR: %{y:.0f} bpm<br>' +
                         '<extra></extra>'
        ),
        row=2, col=1
    )
    
    # rHRI plot (if available)
    if 'rHRI' in df.columns:
        rHRI_clean = df['rHRI'].dropna()
        time_clean = time_minutes[df['rHRI'].notna()]
        
        if len(rHRI_clean) > 0:
            # Scatter plot for individual points
            fig.add_trace(
                go.Scatter(
                    x=time_clean,
                    y=rHRI_clean,
                    mode='markers',
                    marker=dict(size=3, color='green', opacity=0.6),
                    name='rHRI (raw)',
                    hovertemplate='Time: %{x:.1f} min<br>' +
                                 'rHRI: %{y:.6f} bpm/s<br>' +
                                 '<extra></extra>'
                ),
                row=3, col=1
            )
            
            # Moving average line
            if len(rHRI_clean) > 50:
                rHRI_smooth = rHRI_clean.rolling(window=50, center=True, min_periods=1).mean()
                fig.add_trace(
                    go.Scatter(
                        x=time_clean,
                        y=rHRI_smooth,
                        mode='lines',
                        line=dict(color='darkgreen', width=2),
                        name='rHRI (smoothed)',
                        hovertemplate='Time: %{x:.1f} min<br>' +
                                     'rHRI (smoothed): %{y:.6f} bpm/s<br>' +
                                     '<extra></extra>'
                    ),
                    row=3, col=1
                )
    
    # Add temporal quartile backgrounds
    total_time = time_minutes.max()
    quartile_colors = ['rgba(227,242,253,0.3)', 'rgba(187,222,251,0.3)', 
                      'rgba(100,181,246,0.3)', 'rgba(25,118,210,0.3)']
    
    for i, (start_frac, end_frac, color) in enumerate([(0, 0.25, quartile_colors[0]),
                                                       (0.25, 0.5, quartile_colors[1]),
                                                       (0.5, 0.75, quartile_colors[2]),
                                                       (0.75, 1.0, quartile_colors[3])]):
        start_time = total_time * start_frac
        end_time = total_time * end_frac
        
        # Add background rectangles for each subplot
        for row in [1, 2, 3]:
            fig.add_vrect(
                x0=start_time, x1=end_time,
                fillcolor=color,
                opacity=0.3,
                layer="below",
                line_width=0,
                row=row, col=1
            )
            
        # Add quartile labels (only on top plot)
        mid_time = (start_time + end_time) / 2
        fig.add_annotation(
            x=mid_time,
            y=1,
            yref=f"y domain",
            text=f"Q{i+1}",
            showarrow=False,
            font=dict(size=12, color="black"),
            bgcolor="rgba(255,255,255,0.8)",
            row=1, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Complete Ride Overview with Temporal Quartiles',
            font=dict(size=16),
            x=0.5
        ),
        width=1000,
        height=700,
        template='plotly_white',
        showlegend=True,
        hovermode='x unified'
    )
    
    # Update axes
    fig.update_yaxes(title_text="Power (W)", row=1, col=1)
    fig.update_yaxes(title_text="Heart Rate (bpm)", row=2, col=1)
    fig.update_yaxes(title_text="rHRI (bpm/s)", row=3, col=1)
    fig.update_xaxes(title_text="Time (minutes)", row=3, col=1)
    
    return fig