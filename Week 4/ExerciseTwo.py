# Imports 
import pandas as pd

# Fixed calendar order for sorting
_MONTH_ORDER = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]
_MONTH_DTYPE = pd.CategoricalDtype(categories=_MONTH_ORDER, ordered=True)

# Normalize month names
def _normalize_month_names(df, month_col="Month"):
    """
    Normalize Month names so grouping works.
    - strips spaces, title-cases (e.g., '  march ' -> 'March')
    - maps common variants (e.g., 'Sept' -> 'September')
    - invalid values become NaN (so they won't pollute totals)
    """
    out = df.copy()
    if month_col not in out.columns:
        raise ValueError(f"'{month_col}' not found in DataFrame.")

    s = out[month_col].astype("string").str.strip().str.title()

    alias = {
        "Jan": "January", "Feb": "February", "Mar": "March", "Apr": "April",
        "Jun": "June", "Jul": "July", "Aug": "August",
        "Sep": "September", "Sept": "September",
        "Oct": "October", "Nov": "November", "Dec": "December",
    }
    s = s.replace(alias)

    out[month_col] = s.where(s.isin(_MONTH_ORDER))
    return out
# Sales per month
def sales_per_month(df, month_col="Month", sales_col="Sales"):
    """
    compute total Sales per calendar month (Jan to Dec)

    Returns a DataFrame with columns:
        [month_col, 'Sales']
    """
    out = _normalize_month_names(df, month_col=month_col).copy()

    if sales_col not in out.columns:
        raise ValueError(f"'{sales_col}' not found in DataFrame.")
    out[sales_col] = pd.to_numeric(out[sales_col], errors="coerce").fillna(0)

    monthly = (
        out.groupby(month_col, as_index=False)[sales_col]
        .sum()
        .rename(columns={sales_col: "Sales"})
    )

    monthly[month_col] = monthly[month_col].astype(_MONTH_DTYPE)
    monthly = monthly.sort_values(month_col).reset_index(drop=True)
    return monthly

# Sales percentage 
def add_sales_percentages(df_monthly, sales_col="Sales", pct_col="Sales %"):
    """
    append percentage of total column in [0, 1].
    """
    out = df_monthly.copy()
    total = out[sales_col].sum()
    out[pct_col] = 0.0 if total == 0 else out[sales_col] / total
    return out