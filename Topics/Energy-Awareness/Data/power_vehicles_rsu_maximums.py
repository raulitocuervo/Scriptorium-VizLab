import numpy as np  # Add at the top with other imports
import math
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
        df['recorded_at_dt'] = pd.to_datetime(
            df['recorded_at'], unit='s', errors='coerce')

    # Filter rows that have power measurements (we only plot points with power)
    df_plot = df[df['power'].notna()].copy()
    if df_plot.empty:
        print('No power measurements found for the generated vehicle counts.')
        return

    # Prepare consistent color palette per node
    unique_nodes = sorted(df_plot['node_name'].dropna().unique())
    palette = sns.color_palette('tab10', max(3, len(unique_nodes)))
    color_mapping = dict(zip(unique_nodes, palette))

    # Faceted scatter plots with trend lines for each node
    sns.set_theme(style='whitegrid')
    n_nodes = len(unique_nodes)
    ncols = 2
    nrows = math.ceil(n_nodes / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(
        7 * ncols, 4 * nrows), sharex=True, sharey=True)
    if n_nodes == 1:
        axes = [[axes]]
    elif n_nodes == 2:
        axes = [axes]
    axes = np.array(axes).reshape(-1)

    for idx, node in enumerate(unique_nodes):
        ax = axes[idx]
        node_data = df_plot[df_plot['node_name'] == node]
        if node_data.empty:
            continue
        # Scatter plot
        ax.scatter(node_data['vehicles'], node_data['power'],
                   color=color_mapping[node], alpha=0.7, label='Data')
        # Linear regression (trend line)
        if len(node_data) > 1:
            try:
                z = np.polyfit(node_data['vehicles'], node_data['power'], 1)
                p = np.poly1d(z)
                ax.plot(node_data['vehicles'], p(
                    node_data['vehicles']), linestyle='--', color='black', label='Linear Trend')
            except Exception:
                pass
            # Moving average
            window = min(3, len(node_data))
            if window > 1:
                ma = node_data['power'].rolling(
                    window=window, min_periods=1, center=True).mean()
                ax.plot(node_data['vehicles'], ma, linestyle=':',
                        color='gray', label='Moving Avg')
        ax.set_title(f'Node: {node}')
        ax.set_xlabel('Number of vehicles')
        ax.set_ylabel('Power (W)')
        ax.legend()

    # Hide unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle('Power vs Vehicles per RSU (Faceted with Trend Lines)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

    # Combined chart: all RSUs together, logarithmic y-axis, log-space trend lines
    plt.figure(figsize=(10, 6))
    for node in unique_nodes:
        node_data = df_plot[df_plot['node_name'] == node]
        if node_data.empty:
            continue
        # Scatter plot
        plt.scatter(node_data['vehicles'], node_data['power'],
                    color=color_mapping[node], alpha=0.6, label=f'{node} Data')
        # Linear regression in log-space (if all power > 0)
        valid = (node_data['power'] > 0) & (node_data['vehicles'] > 0)
        if valid.sum() > 1:
            x = node_data.loc[valid, 'vehicles']
            y = node_data.loc[valid, 'power']
            try:
                z_log = np.polyfit(x, np.log(y), 1)
                p_log = np.poly1d(z_log)
                y_fit = np.exp(p_log(x))
                plt.plot(x, y_fit, linestyle='--',
                         color=color_mapping[node], label=f'{node} Log Trend')
            except Exception:
                pass

    plt.yscale('log')
    plt.title('Power vs Vehicles (All RSUs, Logarithmic Trend)')
    plt.xlabel('Number of vehicles')
    plt.ylabel('Power (W, log scale)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

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

    # --- Export analysis CSV: vehicles per watt and trend ---
    export_df = df_plot.copy()
    export_df['vehicles_per_watt'] = export_df['vehicles'] / export_df['power']
    export_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    export_df['trend'] = np.nan

    # Compute linear trend (per node) and assign to each row
    for node in unique_nodes:
        node_mask = export_df['node_name'] == node
        node_data = export_df.loc[node_mask]
        
        if len(node_data.dropna(subset=['vehicles', 'power'])) > 1:
            try:
                fit_data = node_data.dropna(subset=['vehicles', 'power'])
                z = np.polyfit(fit_data['vehicles'], fit_data['power'], 1)
                p = np.poly1d(z)
                export_df.loc[node_mask, 'trend'] = p(node_data['vehicles'])
            except Exception as e:
                print(f"Could not calculate trend for node {node}: {e}")
                pass

    analysis_export_path = os.path.join(os.path.dirname(__file__), 'power_vehicles_rsu_analysis.csv')
    export_df[['node_name', 'vehicles', 'power', 'vehicles_per_watt', 'trend']].to_csv(analysis_export_path, index=False)
    print(f"Exported analysis data to {analysis_export_path}")


if __name__ == '__main__':
    main()
