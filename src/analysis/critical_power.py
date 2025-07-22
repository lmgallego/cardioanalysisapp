import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st

@st.cache_data
def calculate_critical_power(df, durations=[60, 300, 720]):
    """
    Calculate Critical Power with enhanced debugging and validation
    """
    max_powers = []
    
    print(f"Data length: {len(df)} points")
    print(f"Power range: {df['power'].min():.1f} - {df['power'].max():.1f} W")
    
    # Calculate maximum mean power for each duration
    for duration in durations:
        if len(df) >= duration:
            rolling_power = df['power'].rolling(window=duration, min_periods=1).mean()
            max_power = rolling_power.max()
            max_powers.append(max_power)
            print(f"Max {duration}s power: {max_power:.1f} W")
        else:
            max_powers.append(np.nan)
            print(f"Insufficient data for {duration}s duration")
    
    # Filter valid data points
    valid_pairs = []
    for i, (duration, power) in enumerate(zip(durations, max_powers)):
        if not np.isnan(power) and power > 0:
            valid_pairs.append((duration, power))
    
    print(f"Valid data points for regression: {len(valid_pairs)}")
    
    if len(valid_pairs) >= 2:
        # Method 1: Direct hyperbolic model fitting P = CP + W'/t
        durations_valid = [p[0] for p in valid_pairs]
        powers_valid = [p[1] for p in valid_pairs]
        
        # Try nonlinear regression first: P = CP + W'/t
        try:
            from scipy.optimize import curve_fit
            
            def hyperbolic_model(t, CP, W_prime):
                return CP + W_prime / t
            
            # Initial guesses
            cp_guess = min(powers_valid) * 0.8  # CP should be lower than shortest duration
            w_prime_guess = (max(powers_valid) - cp_guess) * max(durations_valid)
            
            popt, pcov = curve_fit(hyperbolic_model, durations_valid, powers_valid,
                                  p0=[cp_guess, w_prime_guess],
                                  bounds=([100, 0], [400, 50000]))
            
            cp_nonlinear, w_prime_nonlinear = popt
            
            # Calculate R²
            y_pred = [hyperbolic_model(t, cp_nonlinear, w_prime_nonlinear) for t in durations_valid]
            ss_res = sum((y - y_pred)**2 for y, y_pred in zip(powers_valid, y_pred))
            ss_tot = sum((y - np.mean(powers_valid))**2 for y in powers_valid)
            r_squared_nonlinear = 1 - (ss_res / ss_tot)
            
            print(f"Nonlinear fit: CP={cp_nonlinear:.1f}, W'={w_prime_nonlinear:.0f}, R²={r_squared_nonlinear:.3f}")
            
        except Exception as e:
            print(f"Nonlinear fitting failed: {e}")
            cp_nonlinear, w_prime_nonlinear, r_squared_nonlinear = None, None, 0
        
        # Method 2: Linear regression on 1/P vs 1/t (original paper method)
        try:
            x_vals = np.array([1/d for d in durations_valid])  # 1/t
            y_vals = np.array([1/p for p in powers_valid])     # 1/P
            
            slope, intercept, r_value_linear, p_value, std_err = stats.linregress(x_vals, y_vals)
            
            if intercept > 0:
                cp_linear = 1 / intercept
                w_prime_linear = slope / intercept
                r_squared_linear = r_value_linear**2
            else:
                cp_linear, w_prime_linear, r_squared_linear = None, None, 0
                
            print(f"Linear fit: CP={cp_linear:.1f}, W'={w_prime_linear:.0f}, R²={r_squared_linear:.3f}")
            
        except Exception as e:
            print(f"Linear fitting failed: {e}")
            cp_linear, w_prime_linear, r_squared_linear = None, None, 0
        
        # Choose best method
        if (cp_nonlinear is not None and r_squared_nonlinear > r_squared_linear and 
            100 < cp_nonlinear < 400 and 0 < w_prime_nonlinear < 50000):
            cp, w_prime, r_squared = cp_nonlinear, w_prime_nonlinear, r_squared_nonlinear
            method_used = "Nonlinear"
        elif (cp_linear is not None and 100 < cp_linear < 400 and w_prime_linear > 0):
            cp, w_prime, r_squared = cp_linear, w_prime_linear, r_squared_linear
            method_used = "Linear"
        else:
            # Fallback: estimate CP as 85% of 12-min power
            cp = max(powers_valid) * 0.85 if powers_valid else 200
            w_prime = (max(powers_valid) - cp) * max(durations_valid) if powers_valid else 10000
            r_squared = 0.5
            method_used = "Fallback"
        
        print(f"Selected method: {method_used}")
        print(f"Final values: CP={cp:.1f}W, W'={w_prime:.0f}J, R²={r_squared:.3f}")
        
        return {
            'cp': max(100, min(400, cp)),  # Clamp to reasonable range
            'w_prime': max(0, w_prime),
            'r_squared': r_squared,
            'p_value': p_value if 'p_value' in locals() else 0.05,
            'max_powers': max_powers,
            'durations': durations,
            'method_used': method_used
        }
    
    else:
        print("Insufficient valid data points for regression")
        # Use simple estimates
        if valid_pairs:
            # Estimate CP as 90% of longest duration available
            longest_power = max(p[1] for p in valid_pairs)
            cp = longest_power * 0.9
            w_prime = longest_power * 300  # Rough estimate
        else:
            cp, w_prime = 200, 10000  # Default values
        
        return {
            'cp': cp,
            'w_prime': w_prime,
            'r_squared': 0.0,
            'p_value': 1.0,
            'max_powers': max_powers,
            'durations': durations,
            'method_used': 'Default'
        }