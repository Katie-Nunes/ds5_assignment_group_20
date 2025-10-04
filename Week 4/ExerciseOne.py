# Imports
import pandas as pd
import numpy as np



# Aaron 
def copy(df):
    """Return a shallow copy to avoid in-place mutation."""
    return df.copy()

# Cleaning and fixing errors
# MEAL: strip spaces + constrain to {BB, HB, FB}
def clean_meal(df, col="meal"):
    """
    Normalize meal codes:
      - strip whitespace
      - uppercase
      - map varianto a strict set {BB, HB, FB}; others -> 'UNK'
    """
    out = _copy(df)
    if col not in out.columns:
        return out

    s = (
        out[col]
        .astype("string")
        .str.strip()
        .str.upper()
        .replace({"": pd.NA})
    )
    # simple canonical map 
    canonical = {"BB": "BB", "HB": "HB", "FB": "FB"}
    out[col] = s.map(canonical).fillna("UNK")
    return out

# COUNTRY: strip/upper + validate length, else 'UNK'
def standardize_country(df, col="country", unknown="UNK"):
    """
    Clean country codes:
      - strip + uppercase
      - if missing or length not in {2,3} -> 'UNK'
    """
    out = _copy(df)
    if col not in out.columns:
        return out

    s = (
        out[col]
        .astype("string")
        .str.strip()
        .str.upper()
        .replace({"": pd.NA, "N/A": pd.NA, "NA": pd.NA, "NONE": pd.NA, "NULL": pd.NA})
    )

    def _fix(val):
        if pd.isna(val):
            return unknown
        n = len(val)
        return val if n in (2, 3) and val.isalpha() else unknown

    out[col] = s.map(_fix)
    return out
# DATES reservation_status_date parses cleanly with dayfirst=True
def parse_reservation_date(df, col="reservation_status_date", dayfirst=True):
    """
    Parse reservation_status_date to datetime (NaT on failure).
    """
    out = _copy(df)
    if col in out.columns:
        out[col] = pd.to_datetime(out[col], dayfirst=dayfirst, errors="coerce")
    return out

# NUMERIC: replace non-numeric with NaN, cast counts to Int64
def fix_numerics(
    df,
    non_negative=("lead_time", "adr", "adults", "children", "babies",
                  "stays_in_weekend_nights", "stays_in_week_nights",
                  "previous_cancellations", "previous_bookings_not_canceled",
                  "booking_changes", "days_in_waiting_list",
                  "required_car_parking_spaces", "total_of_special_requests"),
    integer_cols=("adults", "children", "babies", "stays_in_weekend_nights", "stays_in_week_nights",
                  "previous_cancellations", "previous_bookings_not_canceled",
                  "booking_changes", "days_in_waiting_list",
                  "required_car_parking_spaces", "total_of_special_requests"),
):
    """
    - For listed columns: coerce to numeric, NaN->0, negatives->0.
    - For count-like columns: round and cast to nullable Int64.
    """
    out = _copy(df)

    for col in non_negative:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
            out[col] = out[col].mask(out[col] < 0, 0)

    for col in integer_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).round().astype("Int64")

    return out

# MISSING: fill agent/company/country; children/babies -> 0
def fill_missing_common(df, mapping=None):
    """
    Conservative defaults (overridable via `mapping`):
        children/babies -> 0
        agent/company   -> 'Unknown'
        country         -> 'UNK'
    """
    out = _copy(df)
    defaults = {"children": 0, "babies": 0, "agent": "Unknown", "company": "Unknown", "country": "UNK"}
    to_fill = {**defaults, **(mapping or {})}
    for col, val in to_fill.items():
        if col in out.columns:
            out[col] = out[col].fillna(val)
    return out

# YEAR OUTLIERS: handle unrealistic arrival_date_year
def handle_arrival_year_outliers(df, col="arrival_date_year", max_year=2050):
    """
    Replace implausible future years ( > max_year ) with NaN.
    """
    out = _copy(df)
    if col in out.columns:
        s = pd.to_numeric(out[col], errors="coerce")
        out[col] = s.mask(s > max_year)
    return out

# DROPS EXACT DUPLICATES
def drop_exact_duplicates(df, subset=None):
    """
    Drop exact duplicates (keep first).
    """
    out = _copy(df)
    return out.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)

# hala
def clean_hotel_data(file_path, verbose=True):
    """
    Main data cleaning pipeline that integrates all cleaning functions.
    
    Args:
        file_path (str): Path to the Excel file
        verbose (bool): Whether to print cleaning steps and results
        
    Returns:
        pd.DataFrame: Cleaned hotel bookings data
        
    hala
    """
    if verbose:
        print("=== HOTEL BOOKINGS DATA CLEANING PIPELINE ===")
        print("Loading data...")
    
    # Load data (Person A)
    df = load_data(file_path)
    
    if verbose:
        print(f"Initial data shape: {df.shape}")
        print("\nInitial inspection:")
        inspection = inspect_data(df)
        print(f"Missing values: {sum(inspection['missing_values'].values())}")
        print(f"Duplicates: {inspection['duplicates']}")
    
    # Data cleaning pipeline (Person B's functions)
    cleaning_steps = [
        ("Dropping exact duplicates", lambda x: drop_exact_duplicates(x)),
        ("Cleaning meal codes", clean_meal),
        ("Standardizing country codes", standardize_country),
        ("Parsing reservation dates", parse_reservation_date),
        ("Fixing numeric values", fix_numerics),
        ("Filling missing values", fill_missing_common),
        ("Handling year outliers", handle_arrival_year_outliers)
    ]
    
    # Apply all cleaning steps
    cleaned_df = df.copy()
    for step_name, step_func in cleaning_steps:
        if verbose:
            print(f"Applying: {step_name}")
        cleaned_df = step_func(cleaned_df)
    
    if verbose:
        print(f"\nFinal data shape: {cleaned_df.shape}")
        final_inspection = inspect_data(cleaned_df)
        print(f"Remaining missing values: {sum(final_inspection['missing_values'].values())}")
        print(f"Remaining duplicates: {final_inspection['duplicates']}")
        print("\nCleaning completed successfully!")
    
    return cleaned_df


## Decisions we took:
# 1. For meal codes, we standardized to uppercase and mapped unknowns to 'UNK'.
# 2. Country codes were validated for 2-3 character codes.
# 3. Reservation dates were parsed with dayfirst=True to handle European date formats.  
# 4. Numeric fields were constrained to numeric types, negatives were set to zero and count fields to Int64.
# 5. Missing values for children and babies were filled with 0, agent and company were set to 'Unknown'.
# 6. Arrival years greater than 2050 were set to NaN.
# 7. Exact duplicate rows were dropped.

# Main execution block
if __name__ == "__main__":
    try:
        # Clean the data
        cleaned_data = clean_hotel_data('hotelBookings.xlsx', verbose=True)
        print("\nData cleaning completed successfully!")
        
    except FileNotFoundError:
        print("Error: hotelBookings.xlsx file not found.")
    except Exception as e:
        print(f"Error during cleaning: {e}")