import pandas as pd
import io
import base64
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('cardio_analysis')

def create_excel_download(df, df_15min, cp_results=None, filename="cycling_analysis"):
    """
    Create a downloadable Excel file with analysis results.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing full cycling data
    df_15min : pandas.DataFrame
        DataFrame containing 15-minute interval data
    cp_results : dict, optional
        Dictionary containing CP analysis results
    filename : str, optional
        Base filename for the Excel file
        
    Returns:
    --------
    bytes
        Excel file as bytes object for download
    """
    logger.info(f"Creating Excel download for {filename}")
    
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Write different sheets
            df[['time', 'power', 'heart_rate', 'power_smooth', 'heart_rate_smooth', 
                'rHRI', 'power_percent_cp', 'power_quartile']].to_excel(
                writer, sheet_name='Raw Data', index=False)
            
            # Create interval table for Excel
            interval_table = pd.DataFrame({
                'Interval': [f'{int(t/60)}-{int(t/60+15)} min' for t in df_15min['time_bin']],
                'Avg Power (W)': df_15min['power_smooth'].round(1),
                'Avg Power (% CP)': df_15min['power_percent_cp'].round(1),
                'Avg HR (bpm)': df_15min['heart_rate_smooth'].round(1),
                'rHRI': df_15min['rHRI'].round(6),
                'Quartile': df_15min['power_quartile']
            })
            interval_table.to_excel(writer, sheet_name='15min Intervals', index=False)
            
            if cp_results:
                cp_df = pd.DataFrame({
                    'Metric': ['Critical Power (W)', 'W\' (J)', 'R²'],
                    'Value': [cp_results['cp'], cp_results['w_prime'], cp_results['r_squared']]
                })
                cp_df.to_excel(writer, sheet_name='CP Analysis', index=False)
        
        output.seek(0)
        logger.info("Excel file created successfully")
        return output
    except Exception as e:
        logger.error(f"Error creating Excel file: {str(e)}")
        raise

def create_csv_download(df):
    """
    Create a downloadable CSV file with processed data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing cycling data
        
    Returns:
    --------
    str
        CSV data as string for download
    """
    logger.info("Creating CSV download")
    try:
        csv = df.to_csv(index=False)
        logger.info("CSV file created successfully")
        return csv
    except Exception as e:
        logger.error(f"Error creating CSV file: {str(e)}")
        raise

def validate_input_data(df):
    """
    Validate that the input data contains the required columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to validate
        
    Returns:
    --------
    bool
        True if data is valid, False otherwise
    str
        Error message if data is invalid, None otherwise
    """
    required_columns = ['time']
    power_columns = ['watts', 'power']
    hr_columns = ['heartrate', 'heart_rate']
    
    # Check for time column
    if 'time' not in df.columns:
        return False, "Missing required column: 'time'"
    
    # Check for power column
    if not any(col in df.columns for col in power_columns):
        return False, "Missing required column: 'watts' or 'power'"
    
    # Check for heart rate column
    if not any(col in df.columns for col in hr_columns):
        return False, "Missing required column: 'heartrate' or 'heart_rate'"
    
    # Check data types
    try:
        # Ensure time is numeric
        df['time'] = pd.to_numeric(df['time'])
        
        # Ensure power is numeric
        for col in power_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col])
        
        # Ensure heart rate is numeric
        for col in hr_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col])
    except ValueError:
        return False, "Non-numeric values found in data columns"
    
    return True, None