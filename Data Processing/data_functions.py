import os
import pandas as pd
import numpy as np

def load_us_spot_rates(directory):
    # Define the ordered list of files
    ordered_files = [
        "DGS3m.csv", "DGS6m.csv", "DGS1.csv", "DGS3.csv",
        "DGS5.csv", "DGS10.csv"
    ]

    # Define the maturity mapping
    maturity_map = {
        "DGS3m.csv": "US_3m",
        "DGS6m.csv": "US_6m",
        "DGS1.csv": "US_1y",
        "DGS3.csv": "US_3y",
        "DGS5.csv": "US_5y",
        "DGS10.csv": "US_10y",
    }

    # Read index from DGS1.csv for consistency (assuming it's always present)
    index_file = os.path.join(directory, "DGS1.csv")
    if not os.path.exists(index_file):
        raise FileNotFoundError("DGS1.csv is missing, cannot establish index.")

    index_df = pd.read_csv(index_file)
    index_col = index_df.columns[0]  # Assume first column is the date
    index_df[index_col] = pd.to_datetime(index_df[index_col])  # Convert to datetime
    min_date = index_df[index_col].min()
    max_date = index_df[index_col].max()

    # Initialize list to store dataframes
    data_frames = []

    for file in ordered_files:
        file_path = os.path.join(directory, file)
        if not os.path.exists(file_path):
            print(f"Warning: {file} is missing, skipping.")
            continue

        # Read CSV
        df = pd.read_csv(file_path)

        # Convert first column to datetime
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], errors='coerce')

        # Filter to only include dates within the valid range
        df = df[(df[df.columns[0]] >= min_date) & (df[df.columns[0]] <= max_date)]

        # Rename the rate column
        if len(df.columns) < 2:
            print(f"Warning: {file} has unexpected format, skipping.")
            continue

        df = df.iloc[:, [0, 1]].rename(columns={df.columns[1]: maturity_map[file]})
        data_frames.append(df)

    # Merge all dataframes on date column
    final_df = index_df[[index_col]]
    for df in data_frames:
        final_df = final_df.merge(df, on=index_col, how="left")

    # Set index to date column
    final_df.set_index(index_col, inplace=True)
    final_df.index.names = ['date']

    return final_df

def load_input(base_path, start_date):
    folders = [
        "Economic Indicator and Surprises",
        "Market Factors",
        "Sentiment Factors"
    ]

    end_date = "2025-03-05"

    all_data = []

    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                file_path = os.path.join(folder_path, file)
                try:
                    df = pd.read_csv(file_path, parse_dates=True)

                    # Use observation_date if available
                    if 'observation_date' in df.columns:
                        # Handle Excel-style serial numbers if needed
                        if pd.api.types.is_numeric_dtype(df['observation_date']):
                            df['observation_date'] = pd.to_datetime(
                                '1899-12-30') + pd.to_timedelta(df['observation_date'], unit='D')
                        else:
                            df['observation_date'] = pd.to_datetime(
                                df['observation_date'], dayfirst=True, errors='coerce')

                        df = df.dropna(subset=['observation_date'])
                        df = df[~df['observation_date'].duplicated(keep='first')]
                        df.set_index('observation_date', inplace=True)
                    else:
                        df.index = pd.to_datetime(df.index, errors='coerce')
                        df = df.dropna(subset=[df.index.name])
                        df = df[~df.index.duplicated(keep='first')]

                    df.sort_index(inplace=True)
                    all_data.append(df)
                except Exception as e:
                    print(f"Failed to load {file_path}: {e}")

    # Merge all into one daily DataFrame
    merged_df = pd.concat(all_data, axis=1)
    merged_df = merged_df.resample("D").asfreq()

    # Custom 45-day forward fill
    def limited_ffill(series, limit_days=90):
        return series.ffill(limit=limit_days)

    # List of monthly variables for which release indicators should be created
    monthly_vars = ['Actual_INF', 'Actual_UN', 'Actual_CU', 'Actual_NFP', 'PMI']

    for col in merged_df.columns:
        # Skip if the column has no valid values
        if merged_df[col].notna().sum() == 0:
            continue

        # Create a "_release" version only for known monthly variables
        if col in monthly_vars:
            base_name = col.replace("Actual_", "")
            release_col = f"{base_name}_release"

            # Create a zero-initialized series with the same dtype
            release_series = pd.Series(0, index=merged_df.index, dtype=merged_df[col].dtype)

            # Set values only on actual data release dates (non-NaN)
            valid_mask = merged_df[col].notna()
            release_series.loc[valid_mask] = merged_df[col].loc[valid_mask]
            merged_df[release_col] = release_series

    # Handle forecast error creation and cleanup
    macro_vars = ['INF', 'NFP', 'UN', 'CU']
    for var in macro_vars:
        actual_col = f"Actual_{var}"
        forecast_col = f"Forecast_{var}"

        if actual_col in merged_df.columns and forecast_col in merged_df.columns:
            merged_df[actual_col] = pd.to_numeric(merged_df[actual_col], errors='coerce')
            #merged_df[forecast_col] = pd.to_numeric(merged_df[forecast_col], errors='coerce')

            error = merged_df[actual_col] - merged_df[forecast_col]
            std_error = (error - error.mean()) / error.std()
            std_col = f"{var}S"
            #merged_df[std_col] = std_error

            merged_df.drop(columns=[forecast_col], inplace=True)
            merged_df.rename(columns={actual_col: var}, inplace=True)

            # Create release column for standardized error
            release_col = f"{std_col}_release"
            release_series = pd.Series(0, index=merged_df.index, dtype=std_error.dtype)

            # Use original Actual_* col to determine release dates
            release_mask = merged_df[var].notna()
            #release_series.loc[release_mask] = merged_df[std_col].loc[release_mask]
            #merged_df[release_col] = release_series


    for col in merged_df.columns:
        # Skip if the column has no valid values
        if merged_df[col].notna().sum() == 0:
            continue

        # Apply forward fill (limited to 50 days) to the original column
        first_valid = merged_df[col].first_valid_index()
        if pd.notna(first_valid):  # use pd.notna() to avoid NaT comparison issues
            temp_series = merged_df[col].copy()
            mask = temp_series.index >= first_valid
            filled = limited_ffill(temp_series[mask])
            merged_df.loc[mask, col] = filled

    # Final sort and filter
    merged_df.sort_index(inplace=True)
    merged_df = merged_df[(merged_df.index >= start_date) & (merged_df.index <= end_date)]

    return merged_df


def prepare_tables_for_merge(df1, df2, date_col='date'):

    # Ensure the index is in datetime format
    df1.index = pd.to_datetime(df1.index, errors='coerce')
    df2.index = pd.to_datetime(df2.index, errors='coerce')

    # Drop rows where index conversion failed (NaT values)
    df1 = df1[~df1.index.isna()]
    df2 = df2[~df2.index.isna()]

    # Find the common date range based on index values
    min_date = max(df1.index.min(), df2.index.min())
    max_date = min(df1.index.max(), df2.index.max())

    # Filter both DataFrames to the common index date range
    df1 = df1.loc[min_date:max_date]
    df2 = df2.loc[min_date:max_date]

    return df1, df2


def compute_y_df_changes(Y_df: pd.DataFrame) -> dict:
    """
    Compute percentage changes over different horizons: d/d, m/m, 3m/3m, y/y.
    Returns a dictionary with keys: 'd', 'm', '3m', 'y'.
    """
    trading_days = {
        'd': 1,
        'm': 21,     # Approx 1 month
        'w': 5,
        '3m': 63,    # Approx 3 months
        'y': 252     # Approx 1 year
    }

    result = {}
    for label, lag in trading_days.items():
        df_change = Y_df.pct_change(periods=lag).dropna()
        result[f'Y_df_change_{lag}'] = df_change

    return result


def compute_change_directions(Y_df: pd.DataFrame) -> dict:
    """
    Computes directional binary labels for Y_df over different horizons:
    Returns 1 if change > 0, else 0, for:
    - 1 trading day (d/d)
    - 21 trading days (m/m)
    - 63 trading days (3m/3m)
    - 252 trading days (y/y)

    Returns:
        dict: Keys are 'Y_df_change_{lag}', values are DataFrames of 0/1.
    """
    trading_days = {
        'd': 1,
        'w': 5,
        'm': 21,
        '3m': 63,
        'y': 252
    }

    result = {}
    for label, lag in trading_days.items():
        pct_change = Y_df.pct_change(periods=lag)
        direction = (pct_change > 0).astype(int).dropna()
        result[f'Y_df_change_{lag}'] = direction

    return result

def load_economic_data(directory):
    # Define an ordered list of files to process
    ordered_files = sorted([f for f in os.listdir(directory) if f.endswith('.csv')])

    # Use the second file (if available) as the reference index
    if len(ordered_files) < 2:
        raise FileNotFoundError("Not enough files to establish a reference index.")

    index_df = pd.read_csv(directory + '\CS.csv')
    index_df['observation_date'] = pd.to_datetime(index_df['observation_date'], errors='coerce')
    index_df.set_index('observation_date', inplace=True)
    index_df = index_df.asfreq('D', method='ffill').reset_index()  # Keep daily frequency and forward-fill missing values
    index_col = index_df.columns[0]

    min_date = index_df[index_col].min()
    max_date = index_df[index_col].max()

    # Initialize list to store dataframes
    data_frames = []

    for file in ordered_files:
        file_path = os.path.join(directory, file)

        # Read CSV
        df = pd.read_csv(file_path)

        # Attempt to find the date column
        date_col = None
        for col in df.columns:
            if 'date' in col.lower() or 'observation_date' in col.lower():
                date_col = col
                break

        # Convert first column to datetime
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df.set_index(date_col, inplace=True)

        # Apply CPI-specific processing if needed
        if 'CPILFESL' in df.columns:
            df = process_cpi_changes(df)

        # Ensure all datasets are daily by forward-filling missing values
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        df = df.asfreq('D', method='ffill')

        # Filter dates within range
        df = df[(df.index >= min_date) & (df.index <= max_date)]

        # Ensure there are at least two columns
        if df.shape[1] < 1:
            print(f"Warning: {file} has unexpected format, skipping.")
            continue

        # Rename columns based on the file name
        file_id = file.split("_")[-1].replace(".csv", "")
        new_col_name = f"{file_id}"
        df.rename(columns={df.columns[0]: new_col_name}, inplace=True)

        data_frames.append(df)

    # Merge all dataframes on date column
    final_df = index_df.set_index(index_col).asfreq('D', method='ffill')
    final_df.drop(columns='CS', inplace=True)

    for df in data_frames:
        final_df = final_df.merge(df, left_index=True, right_index=True, how="left")

    # Forward-fill any missing values in the final dataset
    final_df = final_df.ffill()

    # Set index names
    final_df.index.names = ['date']

    return final_df