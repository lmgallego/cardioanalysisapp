import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import io
import base64

# Page configuration
st.set_page_config(
    page_title="Cycling Performance Analyzer",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🚴 Cycling Performance Analyzer</h1>', unsafe_allow_html=True)
st.markdown("### Advanced analysis of cycling power and heart rate data")

# Sidebar for file upload
with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with columns: time, watts/power, heartrate/heart_rate"
    )
    
    st.markdown("---")
    st.header("ℹ️ About")
    st.info("""
    This app analyzes cycling performance data including:
    - Critical Power (CP) calculation
    - W' (anaerobic capacity)
    - Cardiovascular efficiency (rHRI)
    - Performance metrics by power quartiles
    - 15-minute interval analysis
    """)

# Analysis functions
@st.cache_data
def load_and_clean_data(file):
    """Load and clean cycling data"""
    df = pd.read_csv(file, encoding='utf-8-sig')
    
    # Rename columns if necessary
    if 'watts' in df.columns:
        df = df.rename(columns={'watts': 'power'})
    if 'heartrate' in df.columns:
        df = df.rename(columns={'heartrate': 'heart_rate'})
    
    # Clean outliers
    df.loc[(df['power'] > 1800) | (df['power'] < 0), 'power'] = np.nan
    df.loc[(df['heart_rate'] < 30) | (df['heart_rate'] > 220), 'heart_rate'] = np.nan
    
    # Interpolate missing values
    df['power'] = df['power'].interpolate(method='linear', limit_direction='both')
    df['heart_rate'] = df['heart_rate'].interpolate(method='linear', limit_direction='both')
    
    # Apply smoothing
    df['power_smooth'] = df['power'].rolling(window=30, center=True, min_periods=1).mean()
    df['heart_rate_smooth'] = df['heart_rate'].rolling(window=30, center=True, min_periods=1).mean()
    
    # Calculate derivatives
    df['fc_deriv'] = df['heart_rate_smooth'].diff() / df['time'].diff()
    df['power_deriv'] = df['power_smooth'].diff() / df['time'].diff()
    
    return df

@st.cache_data
def calculate_critical_power(df, durations=[60, 300, 720]):
    """Calculate Critical Power and W'"""
    max_powers = []
    
    for duration in durations:
        if len(df) >= duration:
            rolling_power = df['power'].rolling(window=duration, min_periods=1).mean()
            max_power = rolling_power.max()
            max_powers.append(max_power)
        else:
            max_powers.append(np.nan)
    
    valid_data = [(1/d, p) for d, p in zip(durations, max_powers) if not np.isnan(p)]
    
    if len(valid_data) >= 2:
        x_vals = np.array([x[0] for x in valid_data])
        y_vals = np.array([x[1] for x in valid_data])
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
        
        return {
            'cp': intercept,
            'w_prime': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'max_powers': max_powers,
            'durations': durations
        }
    else:
        return None

@st.cache_data
def calculate_rhri_and_quartiles(df, cp):
    """Calculate rHRI and power quartiles"""
    # Calculate rHRI
    df['rHRI'] = np.nan
    mask = (df['power_smooth'] > 0) & (df['heart_rate_smooth'] > 0)
    df.loc[mask, 'rHRI'] = df.loc[mask, 'fc_deriv'] / df.loc[mask, 'power_smooth']
    
    # Calculate power as percentage of CP
    if not np.isnan(cp) and cp > 0:
        df['power_percent_cp'] = (df['power_smooth'] / cp) * 100
        
        # Define quartiles
        if df['power_percent_cp'].notna().sum() > 4:
            df['power_quartile'] = pd.qcut(df['power_percent_cp'].dropna(), 
                                          q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], 
                                          duplicates='drop')
    
    return df

@st.cache_data
def analyze_15min_intervals(df):
    """Analyze data in 15-minute intervals"""
    df['time_bin'] = (df['time'] // 900) * 900
    
    df_15min = df.groupby('time_bin').agg({
        'power_smooth': 'mean',
        'heart_rate_smooth': 'mean',
        'fc_deriv': 'mean',
        'rHRI': 'mean',
        'power_percent_cp': 'mean'
    }).reset_index()
    
    if 'power_percent_cp' in df_15min.columns and df_15min['power_percent_cp'].notna().sum() > 4:
        df_15min['power_quartile'] = pd.qcut(df_15min['power_percent_cp'].dropna(), 
                                            q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], 
                                            duplicates='drop')
    
    return df_15min

def create_cp_plot(cp_results):
    """Create Critical Power plot"""
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
    """Create quartile analysis plots"""
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
    """Create time series plot with 15-minute intervals"""
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
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[q], label=f'{q}') for q in ['Q1', 'Q2', 'Q3', 'Q4']]
    axes[1].legend(handles=legend_elements, title='Power Quartile', loc='upper right')
    
    plt.tight_layout()
    return fig

def create_overview_plot(df):
    """Create overview plot of entire ride"""
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

# Main application
if uploaded_file is not None:
    # Load data
    with st.spinner('Loading and processing data...'):
        df = load_and_clean_data(uploaded_file)
        st.success(f'Data loaded successfully! {len(df)} data points found.')
    
    # Calculate Critical Power
    with st.spinner('Calculating Critical Power...'):
        cp_results = calculate_critical_power(df)
        
        if cp_results:
            cp = cp_results['cp']
            w_prime = cp_results['w_prime']
            
            # Display CP metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Critical Power", f"{cp:.0f} W")
            with col2:
                st.metric("W' (Anaerobic Capacity)", f"{w_prime:.0f} J")
            with col3:
                st.metric("R²", f"{cp_results['r_squared']:.3f}")
            with col4:
                duration_min = len(df) / 60
                st.metric("Ride Duration", f"{duration_min:.1f} min")
        else:
            cp = 200  # Default value if calculation fails
            st.warning("Could not calculate Critical Power. Using default value of 200W.")
    
    # Calculate rHRI and quartiles
    df = calculate_rhri_and_quartiles(df, cp)
    
    # Analyze 15-minute intervals
    df_15min = analyze_15min_intervals(df)
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "⚡ Critical Power", "📈 Quartile Analysis", 
                                             "⏱️ 15-min Intervals", "📋 Data Export"])
    
    with tab1:
        st.header("Ride Overview")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Power", f"{df['power_smooth'].mean():.0f} W")
        with col2:
            st.metric("Max Power", f"{df['power_smooth'].max():.0f} W")
        with col3:
            st.metric("Average HR", f"{df['heart_rate_smooth'].mean():.0f} bpm")
        with col4:
            st.metric("Max HR", f"{df['heart_rate_smooth'].max():.0f} bpm")
        
        # Overview plot
        fig_overview = create_overview_plot(df)
        st.pyplot(fig_overview)
    
    with tab2:
        st.header("Critical Power Analysis")
        
        if cp_results:
            # Display CP equation
            st.markdown(f"### Power-Duration Model")
            st.latex(f"P = {cp:.1f} + \\frac{{{w_prime:.0f}}}{{t}}")
            
            # CP plot
            fig_cp = create_cp_plot(cp_results)
            st.pyplot(fig_cp)
            
            # Max power table
            st.subheader("Maximum Average Powers")
            power_df = pd.DataFrame({
                'Duration': ['1 min', '5 min', '12 min'],
                'Max Power (W)': [f"{p:.0f}" if not np.isnan(p) else "N/A" 
                                for p in cp_results['max_powers']],
                '% of CP': [f"{(p/cp*100):.0f}%" if not np.isnan(p) else "N/A" 
                          for p in cp_results['max_powers']]
            })
            st.dataframe(power_df, use_container_width=True)
    
    with tab3:
        st.header("Quartile Analysis (15-minute intervals)")
        
        # Quartile plots
        fig_quartiles = create_quartile_plots(df_15min)
        st.pyplot(fig_quartiles)
        
        # Quartile summary
        st.subheader("Quartile Summary Statistics")
        quartile_summary = df.groupby('power_quartile').agg({
            'power_smooth': ['mean', 'std'],
            'heart_rate_smooth': ['mean', 'std'],
            'rHRI': ['mean', 'std']
        }).round(2)
        st.dataframe(quartile_summary, use_container_width=True)
    
    with tab4:
        st.header("15-Minute Interval Analysis")
        
        # Time series plot
        fig_ts = create_time_series_plot(df, df_15min, cp)
        st.pyplot(fig_ts)
        
        # Interval table
        st.subheader("Interval Summary")
        interval_table = pd.DataFrame({
            'Interval': [f'{int(t/60)}-{int(t/60+15)} min' for t in df_15min['time_bin']],
            'Avg Power (W)': df_15min['power_smooth'].round(1),
            'Avg Power (% CP)': df_15min['power_percent_cp'].round(1),
            'Avg HR (bpm)': df_15min['heart_rate_smooth'].round(1),
            'rHRI': df_15min['rHRI'].round(6),
            'Quartile': df_15min['power_quartile']
        })
        st.dataframe(interval_table, use_container_width=True)
    
    with tab5:
        st.header("Data Export")
        
        # Create downloadable Excel file
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Write different sheets
            df[['time', 'power', 'heart_rate', 'power_smooth', 'heart_rate_smooth', 
                'rHRI', 'power_percent_cp', 'power_quartile']].to_excel(
                writer, sheet_name='Raw Data', index=False)
            
            interval_table.to_excel(writer, sheet_name='15min Intervals', index=False)
            
            if cp_results:
                cp_df = pd.DataFrame({
                    'Metric': ['Critical Power (W)', 'W\' (J)', 'R²'],
                    'Value': [cp, w_prime, cp_results['r_squared']]
                })
                cp_df.to_excel(writer, sheet_name='CP Analysis', index=False)
        
        output.seek(0)
        
        # Download button
        st.download_button(
            label="📥 Download Complete Analysis (Excel)",
            data=output,
            file_name=f"cycling_analysis_{uploaded_file.name.split('.')[0]}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # CSV export option
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Processed Data (CSV)",
            data=csv,
            file_name=f"processed_{uploaded_file.name}",
            mime="text/csv"
        )

else:
    # Instructions when no file is uploaded
    st.info("👆 Please upload a CSV file to begin analysis")
    
    st.markdown("### Expected CSV Format")
    st.markdown("""
    Your CSV file should contain at least these columns:
    - `time`: Time in seconds
    - `watts` or `power`: Power output in watts
    - `heartrate` or `heart_rate`: Heart rate in bpm
    
    Additional columns (optional):
    - `cadence`: Pedaling cadence in rpm
    - `distance`: Distance in meters
    - `altitude`: Altitude in meters
    """)
    
    # Sample data structure
    st.markdown("### Sample Data Structure")
    sample_df = pd.DataFrame({
        'time': [0, 1, 2, 3, 4],
        'watts': [150, 155, 160, 158, 162],
        'heartrate': [120, 121, 123, 125, 127],
        'cadence': [85, 86, 87, 86, 88]
    })
    st.dataframe(sample_df, use_container_width=True)
