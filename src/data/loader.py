import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def load_and_clean_data(data):
    """
    Clean and process cycling data.
    
    Parameters:
    -----------
    data : pandas.DataFrame or file object
        The DataFrame containing cycling data or the uploaded CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned and processed dataframe with cycling metrics
    """
    # Check if input is a DataFrame or a file object
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        # If it's a file object, read it
        df = pd.read_csv(data, encoding='utf-8-sig')
    
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
def generate_sample_data():
    """
    Generate sample cycling data for demonstration purposes.
    
    Returns:
    --------
    pandas.DataFrame
        Sample dataframe with cycling metrics
    """
    # Create time array (30 minutes with 1-second intervals)
    time = np.arange(0, 30*60, 1)
    
    # Generate power data with some variability
    base_power = 150
    power_trend = np.concatenate([
        np.linspace(base_power, base_power+50, 5*60),  # 5 min warmup
        np.linspace(base_power+50, base_power+100, 5*60),  # 5 min increase
        np.linspace(base_power+100, base_power+150, 5*60),  # 5 min hard effort
        np.linspace(base_power+150, base_power+50, 5*60),  # 5 min decrease
        np.linspace(base_power+50, base_power, 10*60)   # 10 min cooldown
    ])
    
    # Add some random noise to power
    np.random.seed(42)  # For reproducibility
    power_noise = np.random.normal(0, 15, len(time))
    power = power_trend + power_noise
    
    # Generate heart rate data (lagging behind power changes)
    base_hr = 120
    hr_trend = np.concatenate([
        np.linspace(base_hr, base_hr+20, 5*60),  # 5 min warmup
        np.linspace(base_hr+20, base_hr+40, 5*60),  # 5 min increase
        np.linspace(base_hr+40, base_hr+60, 5*60),  # 5 min hard effort
        np.linspace(base_hr+60, base_hr+30, 5*60),  # 5 min decrease
        np.linspace(base_hr+30, base_hr, 10*60)   # 10 min cooldown
    ])
    
    # Add some random noise to heart rate
    hr_noise = np.random.normal(0, 3, len(time))
    heart_rate = hr_trend + hr_noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': time,
        'power': power,
        'heart_rate': heart_rate
    })
    
    # Apply the same processing as real data
    df['power_smooth'] = df['power'].rolling(window=30, center=True, min_periods=1).mean()
    df['heart_rate_smooth'] = df['heart_rate'].rolling(window=30, center=True, min_periods=1).mean()
    df['fc_deriv'] = df['heart_rate_smooth'].diff() / df['time'].diff()
    df['power_deriv'] = df['power_smooth'].diff() / df['time'].diff()
    
    return df