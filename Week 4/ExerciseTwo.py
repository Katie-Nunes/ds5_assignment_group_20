# Imports 
import pandas as pd

# Katie
def load_retail_data(file_path):
    df = pd.read_excel(file_path)
    return df

def sales_per_category(df):
    return df.groupby('Category')['Sales'].sum().reset_index()

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
    total = out["Sales"].sum()
    out[pct_col] = 0.0 if total == 0 else out[sales_col] / total
    return out

# hala

def sales_per_manager(df, manager_col="Sales Manager", sales_col="Sales"):
    """
    Compute total sales for each sales manager.
    
    Args:
        df (pd.DataFrame): Input DataFrame with sales data
        manager_col (str): Name of the sales manager column
        sales_col (str): Name of the sales amount column
        
    Returns:
        pd.DataFrame: DataFrame with managers and their total sales
        
     hala
    """
    out = df.copy()
    
    # Validate columns exist
    if manager_col not in out.columns:
        raise ValueError(f"'{manager_col}' not found in DataFrame.")
    if sales_col not in out.columns:
        raise ValueError(f"'{sales_col}' not found in DataFrame.")
    
    # Ensure sales column is numeric
    out[sales_col] = pd.to_numeric(out[sales_col], errors="coerce").fillna(0)
    
    # Group by sales manager and sum sales
    manager_sales = (
        out.groupby(manager_col, as_index=False)[sales_col]
        .sum()
        .rename(columns={sales_col: "Sales"})
        .sort_values("Sales", ascending=False)
        .reset_index(drop=True)
    )
    
    return manager_sales

def add_percentage_column(df, sales_col="Sales", pct_col="Percentage"):
    """
    Add percentage column to any sales DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with sales data
        sales_col (str): Name of the sales column
        pct_col (str): Name for the new percentage column
        
    Returns:
        pd.DataFrame: DataFrame with added percentage column
        
    hala
    """
    out = df.copy()
    total_sales = out[sales_col].sum()
    
    if total_sales == 0:
        out[pct_col] = 0.0
    else:
        out[pct_col] = out[sales_col] / total_sales
    
    return out

def assemble_final_report(category_df, monthly_df, manager_df):
    """
    Assemble all metrics into final report DataFrame.
    
    Args:
        category_df (pd.DataFrame): Category sales data
        monthly_df (pd.DataFrame): Monthly sales data  
        manager_df (pd.DataFrame): Manager sales data
        
    Returns:
        pd.DataFrame: Final assembled report
        
    hala
    """
    report_data = []
    
    # Add category section
    report_data.append({"Metric": "=== SALES BY CATEGORY ===", "Sales": "", "Percentage": ""})
    for _, row in category_df.iterrows():
        report_data.append({
            "Metric": row.get("Category", "Unknown"),
            "Sales": row["Sales"],
            "Percentage": row["Percentage"]
        })
    
    # Add monthly section
    report_data.append({"Metric": "", "Sales": "", "Percentage": ""})
    report_data.append({"Metric": "=== SALES BY MONTH ===", "Sales": "", "Percentage": ""})
    for _, row in monthly_df.iterrows():
        report_data.append({
            "Metric": row.get("Month", "Unknown"),
            "Sales": row["Sales"],
            "Percentage": row["Percentage"]
        })
    
    # Add manager section
    report_data.append({"Metric": "", "Sales": "", "Percentage": ""})
    report_data.append({"Metric": "=== SALES BY MANAGER ===", "Sales": "", "Percentage": ""})
    for _, row in manager_df.iterrows():
        report_data.append({
            "Metric": row.get("Sales Manager", "Unknown"),
            "Sales": row["Sales"],
            "Percentage": row["Percentage"]
        })
    
    return pd.DataFrame(report_data)

def save_report_to_excel(report_df, output_file="reportRetail.xlsx"):
    """
    Save the final report to an Excel file.
    
    Args:
        report_df (pd.DataFrame): The final report DataFrame
        output_file (str): Output Excel file name

    hala    
    
    """
    report_df.to_excel(output_file, index=False, sheet_name="Sales Report")


def generate_sales_report(input_file="detailedRetail-1.xlsx", output_file="reportRetail.xlsx"):
    """
    Main function to generate complete sales report.
    
    Args:
        input_file (str): Input Excel file with detailed sales data
        output_file (str): Output Excel file for the report
        
    Returns:
        pd.DataFrame: Final assembled report
        
    hala
    """
    #try:
    print("=== SALES REPORT GENERATION ===")
    print("Loading data...")

    # Load data and compute category sales
    df = load_retail_data(input_file)
    category_sales = sales_per_category(df)
    category_sales = add_percentage_column(category_sales)

    print("✓ Category sales computed")

    # Compute monthly sales
    monthly_sales = sales_per_month(df)
    monthly_sales = add_percentage_column(monthly_sales)

    print("✓ Monthly sales computed")

    # Compute manager sales
    manager_sales = sales_per_manager(df)
    manager_sales = add_percentage_column(manager_sales)

    print("✓ Manager sales computed")

    # Assemble final report
    final_report = assemble_final_report(category_sales, monthly_sales, manager_sales)

    # Save to Excel
    save_report_to_excel(final_report, output_file)

    print(f"✓ Report saved to '{output_file}'")
    print("=== REPORT GENERATION COMPLETED ===")

    return final_report

    #except FileNotFoundError:
    #    print(f"Error: File '{input_file}' not found.")
    #    raise
    #except Exception as e:
    #    print(f"Error during report generation: {e}")
    #    raise

# Main execution block
if __name__ == "__main__":
    """
    Main execution - runs when script is executed directly.
    """
    #try:
        # Generate the complete sales report
    report = generate_sales_report()
    print("\nFinal Report Preview:")
    print(report.head(10))
        
    #except Exception as e:
    #    print(f"Error: {e}")