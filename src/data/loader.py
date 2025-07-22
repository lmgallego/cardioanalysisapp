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
    
    # Apply 2-second smoothing as per paper (corrected from 30-second)
    df['power_smooth'] = df['power'].rolling(window=2, center=True, min_periods=1).mean()
    df['heart_rate_smooth'] = df['heart_rate'].rolling(window=2, center=True, min_periods=1).mean()
    
    # Calculate derivatives
    df['fc_deriv'] = df['heart_rate_smooth'].diff() / df['time'].diff()
    df['power_deriv'] = df['power_smooth'].diff() / df['time'].diff()
    
    return df

@st.cache_data
def generate_sample_data():
    """
    Generate more realistic sample cycling data with proper power-HR relationships
    
    Returns:
    --------
    pandas.DataFrame
        Sample dataframe with cycling metrics
    """
    # Create time array (2 hours with 1-second intervals for more realistic data)
    time = np.arange(0, 120*60, 1)  # 2 hours
    
    # Create more realistic power profile with intervals
    np.random.seed(42)
    
    # Base power levels
    base_power = 180  # More realistic base for trained cyclist
    
    # Create structured workout with intervals
    power_profile = []
    
    # 10 min warm-up (gradually increasing)
    warmup = np.linspace(100, base_power, 10*60)
    power_profile.extend(warmup)
    
    # Main workout with 4x15min intervals
    for interval in range(4):
        # 15 min moderate (base power)
        moderate = np.full(15*60, base_power) + np.random.normal(0, 10, 15*60)
        power_profile.extend(moderate)
        
        # 5 min hard effort (higher power)
        if interval < 3:  # Skip last hard effort
            hard_power = base_power + 80 + interval * 10  # Progressive intensity
            hard = np.full(5*60, hard_power) + np.random.normal(0, 15, 5*60)
            power_profile.extend(hard)
    
    # Cool down (20 minutes, decreasing)
    remaining_time = len(time) - len(power_profile)
    if remaining_time > 0:
        cooldown = np.linspace(base_power, 120, remaining_time)
        power_profile.extend(cooldown)
    
    # Trim to exact length
    power_profile = power_profile[:len(time)]
    power = np.array(power_profile)
    
    # Add realistic noise
    power += np.random.normal(0, 8, len(power))
    power = np.clip(power, 50, 400)  # Realistic bounds
    
    # Generate heart rate with realistic lag and relationship to power
    base_hr = 125
    max_hr = 185
    
    # HR responds to power changes with delay
    hr_response = []
    current_hr = base_hr
    
    for i, p in enumerate(power):
        # Target HR based on power (non-linear relationship)
        power_ratio = (p - 100) / (300 - 100)  # Normalize power
        target_hr = base_hr + (max_hr - base_hr) * (power_ratio ** 0.7)
        target_hr = np.clip(target_hr, base_hr - 10, max_hr)
        
        # HR changes gradually towards target (lag effect)
        hr_change_rate = 0.02  # 2% adjustment per second
        if target_hr > current_hr:
            current_hr += min((target_hr - current_hr) * hr_change_rate, 1.0)
        else:
            current_hr += max((target_hr - current_hr) * hr_change_rate * 0.5, -0.5)
        
        hr_response.append(current_hr)
    
    heart_rate = np.array(hr_response)
    
    # Add realistic HR noise and drift
    hr_noise = np.random.normal(0, 2, len(heart_rate))
    heart_rate += hr_noise
    heart_rate = np.clip(heart_rate, 80, 200)
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': time,
        'power': power,
        'heart_rate': heart_rate
    })
    
    print(f"Generated sample data: {len(df)} points")
    print(f"Power: {power.min():.0f}-{power.max():.0f}W (avg: {power.mean():.0f}W)")
    print(f"HR: {heart_rate.min():.0f}-{heart_rate.max():.0f}bpm (avg: {heart_rate.mean():.0f}bpm)")
    
    return df