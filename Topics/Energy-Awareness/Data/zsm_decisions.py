import os
import psycopg2
import pandas as pd

# Function to retrieve and display ZSM decisions
# This script executes the query from shw_decisions.sql and prints the results as a formatted table.

def main():
    # Database connection parameters - set these as environment variables or replace directly
    db_params = {
        'host': os.getenv('DB_HOST', '143.129.82.101'),
        'port': os.getenv('DB_PORT', '5432'),
        'dbname': os.getenv('DB_NAME', 'zsmmetrics'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'postgres')
    }
    # Establish connection
    conn = psycopg2.connect(**db_params)
    # Query for ZSM decisions
    query = '''
    SELECT
    n.hostname AS node,
    power,
    latency,
    to_timestamp(timestamp) AT TIME ZONE 'Europe/Brussels' AS datetime_belgium
FROM decisions
    JOIN nodes n ON decisions.node_id = n.id
WHERE
    to_timestamp(timestamp) AT TIME ZONE 'Europe/Brussels' BETWEEN '2025-10-18 00:37:50.660664' AND '2025-10-18 00:37:52.414687'
ORDER BY datetime_belgium ASC;
    '''

    # Load into pandas DataFrame
    df = pd.read_sql(query, conn)

    # Close connection and print formatted table
    conn.close()
    
    # Add columns to match the table format
    df['E2E latency (ms)'] = df['latency'].round(0).astype(int)
    df['Time (hh:mm:ss)'] = df['datetime_belgium'].dt.strftime('%H:%M:%S')  # Format to hh:mm:ss
    df['Power (W)'] = df['power'].round(0).astype(int)  # Add rounded power column
    
    # Generate LaTeX table
    latex_table = df[['node', 'E2E latency (ms)', 'Power (W)', 'Time (hh:mm:ss)']].to_latex(index=False, header=['Node', 'E2E latency (ms)', 'Power (W)', 'Time (hh:mm:ss)'])
    
    # Remove any \toprule, \midrule, \bottomrule
    latex_table = latex_table.replace('\\toprule', '').replace('\\midrule', '').replace('\\bottomrule', '')
    
    latex_code = f"""\\begin{{table}}[!t]
    \\centering
    \\caption{{Latency and power measurements for nodes from the \\gls{{zsm}} orchestrator.}}
    \\label{{tab:latency-decisions}}
    {latex_table}
\\end{{table}}"""
    
    print(latex_code)


if __name__ == '__main__':
    main()