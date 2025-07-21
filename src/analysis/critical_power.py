import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st

@st.cache_data
def calculate_critical_power(df, durations=[60, 300, 720]):
    """
    Calculate Critical Power (CP) and W' (anaerobic work capacity) using the hyperbolic model.
    
    The critical power model is based on the relationship: P = CP + W'/t
    where P is power output, t is time, CP is critical power, and W' is anaerobic work capacity.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing cycling data with 'power' column
    durations : list
        List of durations (in seconds) to calculate maximum average power
        
    Returns:
    --------
    dict or None
        Dictionary containing CP, W', R², p-value, max powers, and durations
        Returns None if insufficient data for calculation
    """
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