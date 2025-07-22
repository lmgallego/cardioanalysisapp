import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import streamlit as st

def sigmoid(t, A, B, C, D):
    """
    Sigmoid function for heart rate modeling as defined in the paper:
    fc(t) = A + B/(1 + e^(-C(t-D)))
    """
    return A + (B / (1 + np.exp(-C * (t - D))))

def find_sequences(df, column, threshold, direction="increase", min_points=6):
    """
    Find consecutive sequences with more flexible criteria
    """
    sequences = []
    start = None
    
    for i in range(len(df)):
        value = df[column].iloc[i]
        if pd.isna(value):
            if start is not None and i - start >= min_points:
                sequences.append((start, i-1))
            start = None
            continue
            
        if direction == "increase" and value > threshold:
            if start is None:
                start = i
        elif direction == "decrease" and value < -threshold:
            if start is None:
                start = i
        else:
            if start is not None and i - start >= min_points:
                sequences.append((start, i-1))
            start = None
    
    if start is not None and len(df) - start >= min_points:
        sequences.append((start, len(df)-1))
    
    return sequences

def calculate_simple_rhri(df):
    """
    Fallback: Calculate simplified rHRI when sigmoid fitting fails
    rHRI = mean(dHR/dt) / mean(power) for periods of increasing power
    """
    # Find periods where power is increasing
    power_increase = df['power_deriv'] > 0
    
    if power_increase.sum() > 0:
        # Calculate rHRI as HR derivative divided by power during increases
        mask = power_increase & (df['power_smooth'] > 0) & (df['fc_deriv'] > 0)
        if mask.sum() > 0:
            rhri_simple = df.loc[mask, 'fc_deriv'] / df.loc[mask, 'power_smooth']
            return rhri_simple.mean()
    
    return np.nan

@st.cache_data
def calculate_rhri_and_quartiles(df, cp):
    """
    Calculate rHRI with enhanced robustness and fallback methods
    """
    df = df.copy()
    
    # Apply 2-second moving average
    window_size = 2
    if len(df) >= window_size:
        df['power_smooth'] = df['power'].rolling(window=window_size, center=True, min_periods=1).mean()
        df['heart_rate_smooth'] = df['heart_rate'].rolling(window=window_size, center=True, min_periods=1).mean()
    else:
        df['power_smooth'] = df['power']
        df['heart_rate_smooth'] = df['heart_rate']
    
    # Calculate power as percentage of CP
    if not np.isnan(cp) and cp > 0:
        df['power_percent_cp'] = (df['power_smooth'] / cp) * 100
    else:
        df['power_percent_cp'] = df['power_smooth'] / 200 * 100  # Use reasonable fallback
    
    # Calculate derivatives
    time_diff = df['time'].diff()
    time_diff = time_diff.fillna(1.0)
    time_diff = np.where(time_diff == 0, 1e-5, time_diff)
    
    df['power_deriv'] = df['power_smooth'].diff() / time_diff
    df['power_percent_deriv'] = df['power_percent_cp'].diff() / time_diff
    df['fc_deriv'] = df['heart_rate_smooth'].diff() / time_diff
    
    # More flexible threshold (reduced from 0.3 to 0.1)
    power_percent_deriv_std = df['power_percent_deriv'].std()
    if pd.isna(power_percent_deriv_std) or power_percent_deriv_std == 0:
        power_threshold = 0.1  # Lower fixed threshold
    else:
        power_threshold = max(0.1 * power_percent_deriv_std, 0.01)  # Minimum threshold
    
    # Find sequences with relaxed criteria (minimum 3 points instead of 6)
    increase_sequences = find_sequences(df, 'power_percent_deriv', power_threshold, "increase", min_points=3)
    
    # Initialize rHRI column
    df['rHRI'] = np.nan
    df['sigmoid_A'] = np.nan
    df['sigmoid_B'] = np.nan
    df['sigmoid_C'] = np.nan
    df['sigmoid_D'] = np.nan
    
    # Track successful fits
    successful_fits = 0
    total_sequences = len(increase_sequences)
    
    # Try sigmoid fitting for each sequence
    for start_idx, end_idx in increase_sequences:
        if end_idx - start_idx < 3:  # Reduced minimum points
            continue
            
        seq_data = df.iloc[start_idx:end_idx+1].copy()
        time_seq = seq_data['time'].values
        fc_seq = seq_data['heart_rate_smooth'].values
        
        # Remove NaN values
        valid_mask = ~(pd.isna(time_seq) | pd.isna(fc_seq))
        if valid_mask.sum() < 3:
            continue
            
        time_seq = time_seq[valid_mask]
        fc_seq = fc_seq[valid_mask]
        
        # Check if we have meaningful HR variation
        if np.max(fc_seq) - np.min(fc_seq) < 1:  # Less than 1 bpm variation
            continue
        
        try:
            # More flexible initial parameters
            A_init = np.min(fc_seq)
            B_init = max(np.max(fc_seq) - np.min(fc_seq), 1.0)
            C_init = 0.01  # Slower rate parameter
            D_init = np.mean(time_seq)
            
            # More lenient bounds
            lower_bounds = [A_init - 10, 0.1, -0.5, np.min(time_seq) - 60]
            upper_bounds = [np.max(fc_seq) + 10, B_init * 3, 0.5, np.max(time_seq) + 60]
            
            # Fit sigmoid with multiple attempts
            popt, _ = curve_fit(sigmoid, time_seq, fc_seq,
                              p0=[A_init, B_init, C_init, D_init],
                              bounds=(lower_bounds, upper_bounds),
                              maxfev=2000)
            
            A, B, C, D = popt
            
            # Calculate rHRI more robustly
            if abs(C) > 1e-6:  # Avoid division by very small numbers
                t_extended = np.linspace(np.min(time_seq), np.max(time_seq), 500)
                exp_term = np.exp(-C * (t_extended - D))
                # Avoid overflow
                exp_term = np.clip(exp_term, 1e-10, 1e10)
                sigmoid_derivative = (B * C * exp_term) / ((1 + exp_term) ** 2)
                
                if len(sigmoid_derivative) > 0 and not np.all(np.isnan(sigmoid_derivative)):
                    rHRI_value = np.max(sigmoid_derivative)
                    
                    # Sanity check: rHRI should be positive and reasonable
                    if rHRI_value > 0 and rHRI_value < 1.0:  # Max 1 bpm/s is reasonable
                        df.loc[start_idx:end_idx, 'rHRI'] = rHRI_value
                        df.loc[start_idx:end_idx, 'sigmoid_A'] = A
                        df.loc[start_idx:end_idx, 'sigmoid_B'] = B
                        df.loc[start_idx:end_idx, 'sigmoid_C'] = C
                        df.loc[start_idx:end_idx, 'sigmoid_D'] = D
                        successful_fits += 1
            
        except (RuntimeError, ValueError, TypeError, OverflowError, OptimizeWarning):
            continue
    
    # If sigmoid fitting mostly failed, use simplified rHRI calculation
    if successful_fits < max(1, total_sequences * 0.1):  # Less than 10% success rate
        print(f"Warning: Sigmoid fitting had low success rate ({successful_fits}/{total_sequences}). Using simplified rHRI calculation.")
        
        # Apply simplified rHRI calculation
        simple_rHRI = calculate_simple_rhri(df)
        if not np.isnan(simple_rHRI):
            # Fill NaN rHRI values with simplified calculation
            mask = df['rHRI'].isna() & (df['power_deriv'] > 0)
            if mask.sum() > 0:
                df.loc[mask, 'rHRI'] = simple_rHRI
    
    # Calculate TEMPORAL quartiles
    total_duration = df['time'].max() - df['time'].min()
    if total_duration > 0:
        quartile_breaks = [
            df['time'].min(),
            df['time'].min() + 0.25 * total_duration,
            df['time'].min() + 0.50 * total_duration,
            df['time'].min() + 0.75 * total_duration,
            df['time'].max()
        ]
        
        df['quartile'] = pd.cut(df['time'], bins=quartile_breaks,
                               labels=['Q1', 'Q2', 'Q3', 'Q4'],
                               include_lowest=True)
    else:
        df['quartile'] = 'Q1'
    
    # Power quartiles for comparison
    if df['power_percent_cp'].notna().sum() > 4:
        try:
            df['power_quartile'] = pd.qcut(df['power_percent_cp'].dropna(),
                                          q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'],
                                          duplicates='drop')
        except ValueError:
            df['power_quartile'] = pd.cut(df['power_percent_cp'],
                                         bins=4, labels=['Q1', 'Q2', 'Q3', 'Q4'],
                                         include_lowest=True)
    else:
        df['power_quartile'] = 'Q1'
    
    return df

@st.cache_data
def analyze_15min_intervals(df):
    """
    Enhanced 15-minute interval analysis with better rHRI handling
    """
    # Create 15-minute time bins
    df['time_bin'] = (df['time'] // 900) * 900
    
    # Custom aggregation function for rHRI (take mean of non-NaN values)
    def safe_mean(x):
        clean_x = x.dropna()
        return clean_x.mean() if len(clean_x) > 0 else np.nan
    
    # Aggregate with robust functions
    agg_dict = {
        'power_smooth': safe_mean,
        'heart_rate_smooth': safe_mean,
        'fc_deriv': safe_mean,
        'rHRI': safe_mean,  # Use safe_mean for rHRI
        'power_percent_cp': safe_mean
    }
    
    if 'quartile' in df.columns:
        agg_dict['quartile'] = lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Q1'
    
    df_15min = df.groupby('time_bin').agg(agg_dict).reset_index()
    
    # Ensure temporal quartiles
    if len(df_15min) > 0:
        total_duration = df_15min['time_bin'].max() - df_15min['time_bin'].min()
        if total_duration > 0:
            quartile_breaks = [
                df_15min['time_bin'].min(),
                df_15min['time_bin'].min() + 0.25 * total_duration,
                df_15min['time_bin'].min() + 0.50 * total_duration,
                df_15min['time_bin'].min() + 0.75 * total_duration,
                df_15min['time_bin'].max()
            ]
            df_15min['quartile'] = pd.cut(df_15min['time_bin'], bins=quartile_breaks,
                                         labels=['Q1', 'Q2', 'Q3', 'Q4'],
                                         include_lowest=True)
        else:
            df_15min['quartile'] = 'Q1'
    
    # Power quartiles
    if 'power_percent_cp' in df_15min.columns and df_15min['power_percent_cp'].notna().sum() > 1:
        try:
            df_15min['power_quartile'] = pd.qcut(df_15min['power_percent_cp'].dropna(),
                                                q=min(4, df_15min['power_percent_cp'].notna().sum()),
                                                labels=['Q1', 'Q2', 'Q3', 'Q4'][:min(4, df_15min['power_percent_cp'].notna().sum())],
                                                duplicates='drop')
        except (ValueError, TypeError):
            df_15min['power_quartile'] = 'Q1'
    else:
        df_15min['power_quartile'] = 'Q1'
    
    return df_15min