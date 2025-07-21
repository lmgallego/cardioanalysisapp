import numpy as np
import pandas as pd
import streamlit as st

@st.cache_data
def calculate_rhri_and_quartiles(df, cp):
    """
    Calculate relative Heart Rate Increase (rHRI) and power quartiles.
    
    rHRI is defined as the heart rate derivative divided by power,
    representing cardiovascular efficiency.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing cycling data with required columns
    cp : float
        Critical Power value used to calculate power as percentage of CP
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added rHRI and power quartile columns
    """
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
    """
    Analyze cycling data in 15-minute intervals.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing cycling data with required columns
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with aggregated metrics for each 15-minute interval
    """
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