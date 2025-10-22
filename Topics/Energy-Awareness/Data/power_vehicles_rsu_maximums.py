import os
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

# Function to retrieve and display vehicle and power metrics
# This script executes a UNION ALL query to fetch the latest 100 entries from both tables
# and prints them as a formatted table.


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
    # Query: generate vehicle counts per node (0..max) and pick representative metrics rows + closest power reading
    query = '''
-- Restrict to 2025-10-18 (00:00:00) through midnight (exclusive)
WITH RECURSIVE max_v AS (
    SELECT COALESCE(MAX(value), 0) AS max_value
    FROM metrics_vehicles
    WHERE timestamp >= extract(epoch from timestamp '2025-10-18 00:00:00')
      AND timestamp < extract(epoch from timestamp '2025-10-19 00:00:00')
),
nums(n) AS (
    SELECT 0
    UNION ALL
    SELECT n + 1 FROM nums, max_v WHERE n < max_v.max_value
)
,
mv_min AS (
        SELECT DISTINCT ON (node, value) id, node, value, timestamp
        FROM metrics_vehicles
        WHERE timestamp >= extract(epoch from timestamp '2025-10-18 00:00:00')
            AND timestamp < extract(epoch from timestamp '2025-10-19 00:00:00')
        ORDER BY node, value, id
)
SELECT n.hostname AS node_name,
       nums.n AS vehicles,
       mv_min.id AS record,
       mv_min.timestamp AS recorded_at,
       mp.value AS power,
       mp.id AS power_id
FROM (
    SELECT id, hostname FROM nodes WHERE hostname <> 'obu2'
) n
CROSS JOIN nums
LEFT JOIN mv_min ON mv_min.node = n.id AND mv_min.value = nums.n
-- pick the single closest metrics_power row (by timestamp difference) within +/-5 seconds
LEFT JOIN LATERAL (
     SELECT id, value, timestamp
     FROM metrics_power
     WHERE node = mv_min.node
        AND mv_min.timestamp IS NOT NULL
        AND ABS(timestamp - mv_min.timestamp) <= 5
        -- pick rows closest in time first, then highest power among equally-close, then smallest id
        ORDER BY ABS(timestamp - mv_min.timestamp) ASC, value DESC, id ASC
     LIMIT 1
) mp ON true
    '''

    # Load into pandas DataFrame
    df = pd.read_sql(query, conn)

    # Close DB connection early
    conn.close()

    # Ensure numeric types
    if 'vehicles' in df.columns:
        df['vehicles'] = pd.to_numeric(df['vehicles'], errors='coerce')
    if 'power' in df.columns:
        df['power'] = pd.to_numeric(df['power'], errors='coerce')

    # Convert recorded_at epoch to datetime if present
    if 'recorded_at' in df.columns:
        df['recorded_at_dt'] = pd.to_datetime(df['recorded_at'], unit='s', errors='coerce')

    # Filter rows that have power measurements (we only plot points with power)
    df_plot = df[df['power'].notna()].copy()
    if df_plot.empty:
        print('No power measurements found for the generated vehicle counts.')
        return

    # Prepare consistent color palette per node
    unique_nodes = sorted(df_plot['node_name'].dropna().unique())
    palette = sns.color_palette('tab10', max(3, len(unique_nodes)))
    color_mapping = dict(zip(unique_nodes, palette))

    # Plot: x = vehicles, y = power, series per node_name
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(12, 6))

    # Sort data so line plots connect in vehicle order
    df_plot.sort_values(['node_name', 'vehicles'], inplace=True)

    sns.lineplot(data=df_plot, x='vehicles', y='power', hue='node_name',
                 marker='o', dashes=False, palette=color_mapping, estimator=None)

    plt.title('Power vs Vehicles per RSU')
    plt.xlabel('Number of vehicles')
    plt.ylabel('Power (W)')
    plt.legend(title='RSU', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
