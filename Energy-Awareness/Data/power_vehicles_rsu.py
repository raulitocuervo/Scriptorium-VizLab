import os
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

# Function to retrieve and display metrics
# This script executes a UNION ALL query to fetch the latest entries from multiple tables
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
    # Combined query for vehicle, power, CPU, and latency metrics
    query = '''
    SELECT *, to_timestamp(combined.timestamp) AT TIME ZONE 'Europe/Brussels' AS datetime_belgium
    FROM (
        SELECT 'vehicle' AS source, n.hostname AS node, m.value, m.timestamp
        FROM metrics_vehicles m
            JOIN nodes n ON m.node = n.id
        UNION ALL
        SELECT 'power' AS source, n.hostname AS node, m.value, m.timestamp
        FROM metrics_power m
            JOIN nodes n ON m.node = n.id
        UNION ALL
        SELECT 'cpu' AS source, n.hostname AS node, m.value, m.timestamp
        FROM metrics_cpu m
            JOIN nodes n ON m.node = n.id
        UNION ALL
        SELECT 'latency' AS source, n.hostname AS node, m.value, m.timestamp
        FROM metrics_latency m
            JOIN nodes n ON m.node = n.id
    ) AS combined
    WHERE
        to_timestamp(combined.timestamp) AT TIME ZONE 'Europe/Brussels' >= '2025-10-18 00:00:00'
        AND to_timestamp(combined.timestamp) AT TIME ZONE 'Europe/Brussels' < '2025-10-19 00:00:00'
    ORDER BY datetime_belgium DESC;
    '''
    # WHERE combined.timestamp >= extract(epoch from timestamp '2025-10-16 17:53:00') AND combined.timestamp <= extract(epoch from timestamp '2025-10-17 05:51:00')

    # Load into pandas DataFrame
    df = pd.read_sql(query, conn)

    # Convert epoch timestamp to naive datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    # Limit milliseconds to two digits for readability (centiseconds)
    df['timestamp'] = df['timestamp'].dt.strftime(
        '%Y-%m-%d %H:%M:%S.%f').str[:-4]

    # Close connection and print formatted table
    conn.close()
    print(df.to_string(index=False))

    # Prepare timestamp for plotting
    df['datetime_belgium'] = pd.to_datetime(df['datetime_belgium'])

    # Define a shared color palette for consistent RSU colors across charts
    unique_nodes = df['node'].unique()
    palette = sns.color_palette('tab10', len(unique_nodes))
    color_mapping = dict(zip(unique_nodes, palette))

    # Create a vertically split chart for vehicles, power, CPU, and latency
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 15),
                                             sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 1]})

    # Filter data for each metric
    df_vehicles = df[df['source'] == 'vehicle']
    df_power = df[df['source'] == 'power']
    df_cpu = df[df['source'] == 'cpu']
    df_latency = df[df['source'] == 'latency']

    # Plot vehicles on the first axis
    sns.lineplot(data=df_vehicles, x='datetime_belgium', y='value', hue='node',
                 ax=ax1, palette=color_mapping, markers=True, dashes=False)
    ax1.set_title('Vehicle Counts Over Time by Node')
    ax1.set_ylabel('Vehicle Count')
    # Sort legend items alphabetically
    handles, labels = ax1.get_legend_handles_labels()
    sorted_legend = sorted(zip(labels, handles), key=lambda x: x[0])
    labels, handles = zip(*sorted_legend)
    ax1.legend(handles, labels, title='Node',
               bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plot power estimation on the second axis
    sns.lineplot(data=df_power, x='datetime_belgium', y='value',
                 hue='node', ax=ax2, palette=color_mapping, markers=True, dashes=False)
    ax2.set_title('Power Consumption Over Time by Node')
    ax2.set_ylabel('Power Consumption (W)')
    # Sort legend items alphabetically
    handles, labels = ax2.get_legend_handles_labels()
    sorted_legend = sorted(zip(labels, handles), key=lambda x: x[0])
    labels, handles = zip(*sorted_legend)
    ax2.legend(handles, labels, title='Node',
               bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plot CPU usage on the third axis
    sns.lineplot(data=df_cpu, x='datetime_belgium', y='value',
                 hue='node', ax=ax3, palette=color_mapping, markers=True, dashes=False)
    ax3.set_title('CPU Usage Over Time by Node')
    ax3.set_ylabel('CPU Usage (%)')
    # Sort legend items alphabetically
    handles, labels = ax3.get_legend_handles_labels()
    sorted_legend = sorted(zip(labels, handles), key=lambda x: x[0])
    labels, handles = zip(*sorted_legend)
    ax3.legend(handles, labels, title='Node',
               bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plot E2E latency on the fourth axis
    sns.lineplot(data=df_latency, x='datetime_belgium', y='value',
                 hue='node', ax=ax4, palette=color_mapping, markers=True, dashes=False)
    ax4.set_title('E2E Latency Over Time by Node')
    ax4.set_ylabel('E2E Latency (ms)')
    # Start from minimum value, max at 200 ms
    ax4.set_ylim(bottom=df_latency['value'].min(), top=200)
    # Sort legend items alphabetically
    handles, labels = ax4.get_legend_handles_labels()
    sorted_legend = sorted(zip(labels, handles), key=lambda x: x[0])
    labels, handles = zip(*sorted_legend)
    ax4.legend(handles, labels, title='Node',
               bbox_to_anchor=(1.05, 1), loc='upper left')

    # Format x-axis to show hours and minutes only
    ax4.set_xlabel('Time (HH:MM)')
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
