import streamlit as st
import pandas as pd
import numpy as np
import base64
import logging

# Import modules from src
from src.data.loader import load_and_clean_data, generate_sample_data
from src.analysis.critical_power import calculate_critical_power
from src.analysis.rhri import calculate_rhri_and_quartiles, analyze_15min_intervals
from src.visualization.plots import (
    create_cp_plot, create_quartile_plots, 
    create_time_series_plot, create_overview_plot
)
from src.utils.helpers import create_excel_download, create_csv_download, validate_input_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('cardio_analysis')

# Page configuration
st.set_page_config(
    page_title="Cycling Performance Analyzer",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #1E88E5;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F0F2F6;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5 !important;
        color: white !important;
    }
    .stDataFrame {
        max-height: 400px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.title("🚴 Cycling Performance Analyzer")
st.markdown("""
**Analyze your cycling performance with advanced metrics:**
- Critical Power (CP) and W' calculation
- Relative Heart Rate Increase (rHRI) analysis
- Power quartile distribution
- 15-minute interval breakdown
""")

# Sidebar for file upload
st.sidebar.header("Data Input")
upload_file = st.sidebar.file_uploader("Upload your cycling data (CSV)", type=["csv"])

# Sample data option
use_sample_data = st.sidebar.checkbox("Use sample data instead")

# Main application logic
def main():
    if upload_file is not None or use_sample_data:
        try:
            # Load data
            if use_sample_data:
                logger.info("Using sample data")
                df = generate_sample_data()
                st.sidebar.success("Sample data loaded successfully!")
            else:
                logger.info(f"Loading data from uploaded file: {upload_file.name}")
                df = pd.read_csv(upload_file)
                
                # Validate input data
                valid, error_msg = validate_input_data(df)
                if not valid:
                    st.error(f"Invalid data format: {error_msg}")
                    st.stop()
                    
                st.sidebar.success(f"File {upload_file.name} loaded successfully!")
            
            # Process data
            df = load_and_clean_data(df)
            
            # Calculate Critical Power
            cp_results = calculate_critical_power(df)
            cp = cp_results['cp']
            
            # Calculate rHRI and quartiles
            df = calculate_rhri_and_quartiles(df, cp)
            
            # Analyze 15-minute intervals
            df_15min = analyze_15min_intervals(df)
            
            # Create tabs for different analyses
            tabs = st.tabs(["Overview", "Critical Power", "Quartile Analysis", "15-min Intervals", "Data Export"])
            
            # Overview tab
            with tabs[0]:
                st.header("Ride Overview")
                overview_fig = create_overview_plot(df)
                st.pyplot(overview_fig)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Critical Power", f"{cp_results['cp']:.1f} W")
                    st.metric("W'", f"{cp_results['w_prime']:.0f} J")
                with col2:
                    st.metric("Average Power", f"{df['power_smooth'].mean():.1f} W")
                    st.metric("Average Heart Rate", f"{df['heart_rate_smooth'].mean():.1f} bpm")
            
            # Critical Power tab
            with tabs[1]:
                st.header("Critical Power Analysis")
                st.markdown(f"""
                **Power-Duration Model Results:**
                - Critical Power (CP): **{cp_results['cp']:.1f} W**
                - W' (Anaerobic Work Capacity): **{cp_results['w_prime']:.0f} J**
                - R²: **{cp_results['r_squared']:.4f}**
                
                *CP represents your sustainable power output, while W' represents your anaerobic energy reserves.*
                """)
                
                cp_fig = create_cp_plot(cp_results)
                st.pyplot(cp_fig)
                
                # Max power table
                st.subheader("Maximum Average Power")
                max_power_df = pd.DataFrame({
                    'Duration': ['1 minute', '5 minutes', '12 minutes'],
                    'Max Power (W)': cp_results['max_powers'],
                    '% of CP': [p/cp_results['cp']*100 for p in cp_results['max_powers']]
                })
                st.dataframe(max_power_df.style.format({'Max Power (W)': '{:.1f}', '% of CP': '{:.1f}%'}))
            
            # Quartile Analysis tab
            with tabs[2]:
                st.header("Quartile Analysis")
                st.markdown("""
                **Power Quartiles:**
                - **Q1**: < 75% of CP (Recovery)
                - **Q2**: 75-90% of CP (Endurance)
                - **Q3**: 90-105% of CP (Tempo/Threshold)
                - **Q4**: > 105% of CP (VO2max/Anaerobic)
                
                *rHRI (Relative Heart Rate Increase) measures cardiac strain relative to power output.*
                """)
                
                quartile_fig = create_quartile_plots(df_15min)
                st.pyplot(quartile_fig)
                
                # Quartile summary statistics
                st.subheader("Quartile Summary Statistics")
                quartile_stats = df.groupby('power_quartile').agg({
                    'power_smooth': 'mean',
                    'heart_rate_smooth': 'mean',
                    'power_percent_cp': 'mean',
                    'rHRI': 'mean',
                    'time': lambda x: len(x) / len(df) * 100  # Percentage of time
                }).reset_index()
                
                quartile_stats = quartile_stats.rename(columns={
                    'power_smooth': 'Avg Power (W)',
                    'heart_rate_smooth': 'Avg HR (bpm)',
                    'power_percent_cp': 'Power (% of CP)',
                    'rHRI': 'rHRI (bpm/s/W)',
                    'time': 'Time (%)'
                })
                
                st.dataframe(quartile_stats.style.format({
                    'Avg Power (W)': '{:.1f}',
                    'Avg HR (bpm)': '{:.1f}',
                    'Power (% of CP)': '{:.1f}%',
                    'rHRI (bpm/s/W)': '{:.6f}',
                    'Time (%)': '{:.1f}%'
                }))
            
            # 15-min Intervals tab
            with tabs[3]:
                st.header("15-Minute Interval Analysis")
                st.markdown("""
                This analysis breaks down your ride into 15-minute intervals to track changes in performance metrics over time.
                """)
                
                # Time series plot
                time_series_fig = create_time_series_plot(df, df_15min, cp)
                st.pyplot(time_series_fig)
                
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
                st.dataframe(interval_table)
            
            # Data Export tab
            with tabs[4]:
                st.header("Data Export")
                st.markdown("""
                Download your analysis results in Excel or CSV format.
                """)
                
                # Excel download
                excel_data = create_excel_download(df, df_15min, cp_results)
                excel_b64 = base64.b64encode(excel_data.getvalue()).decode()
                href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{excel_b64}" download="cycling_analysis.xlsx">Download Excel Analysis</a>'
                st.markdown(href, unsafe_allow_html=True)
                
                # CSV download
                csv = create_csv_download(df)
                csv_b64 = base64.b64encode(csv.encode()).decode()
                href_csv = f'<a href="data:text/csv;base64,{csv_b64}" download="cycling_data.csv">Download Processed CSV Data</a>'
                st.markdown(href_csv, unsafe_allow_html=True)
                
                st.info("""
                **Excel file contains:**
                - Raw data with calculated metrics
                - 15-minute interval summary
                - Critical Power analysis results
                
                **CSV file contains:**
                - Full processed dataset with all calculated metrics
                """)
                
        except Exception as e:
            logger.error(f"Error in application: {str(e)}")
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)
    else:
        # Instructions when no file is uploaded
        st.info("""
        ### 📊 Upload your cycling data to get started
        
        **Expected CSV format:**
        - Must include columns for time, power, and heart rate
        - Column names can be: 'time', 'power'/'watts', 'heart_rate'/'heartrate'
        - Data should be in seconds and standard units (watts, bpm)
        
        **Sample data structure:**
        ```
        time,power,heart_rate
        0,150,80
        1,155,82
        2,160,85
        ...
        ```
        
        Alternatively, use the sample data option in the sidebar.
        """)

# Run the app
if __name__ == "__main__":
    main()