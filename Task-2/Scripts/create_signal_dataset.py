import os
import pandas as pd
from datetime import timedelta
from scipy.signal import savgol_filter
from data_cleaning import savitzky_golay_filter
from load_files import get_flow_signal, get_sleep_profile


def get_label(win_start, win_end, sleep_df):
    # Construct intervals from sleep stage annotations
    events = []
    timestamps = sleep_df.index.to_list()
    for i in range(len(timestamps) - 1):
        events.append((timestamps[i], timestamps[i+1], sleep_df.iloc[i]['Event']))

    # Match window to sleep stage based on overlap ≥ 15s
    for ev_start, ev_end, ev_label in events:
        overlap = (min(win_end, ev_end) - max(win_start, ev_start)).total_seconds()
        if overlap >= 15:
            return ev_label
    return "Unknown"


def create_dataset(signal_df, sleep_df, output_path):
    windows = []
    start = signal_df.index[0]
    end = signal_df.index[-1]
    win_size = timedelta(seconds=30)
    step = timedelta(seconds=15)

    current = start
    while current + win_size <= end:
        win_end = current + win_size
        segment = signal_df[(signal_df.index >= current) & (signal_df.index < win_end)]

        if len(segment) == 0:
            current += step
            continue

        label = get_label(current, win_end, sleep_df)
        stats = {
            "start_time": current,
            "end_time": win_end,
            "label": label,
            "mean": segment["Nasal Flow"].mean(),
            "std": segment["Nasal Flow"].std(),
            "min": segment["Nasal Flow"].min(),
            "max": segment["Nasal Flow"].max()
        }
        windows.append(stats)
        current += step

    df = pd.DataFrame(windows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Sleep stage dataset saved: {output_path}")


def process_all_folders(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    folders = [f for f in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, f))]

    for folder in folders:
        folder_path = os.path.join(in_dir, folder)
        sleep_df = None
        signals = None

        for file in os.listdir(folder_path):
            filepath = os.path.join(folder_path, file)
            base_name = os.path.splitext(file)[0].lower().replace(" ", "")

            if 'sleepprofile' in base_name or 'sleep_profile' in base_name:
                sleep_df = get_sleep_profile(filepath)
            elif 'flow' in base_name or 'flowsignal' in base_name:
                signals = get_flow_signal(filepath)

        if sleep_df is None or signals is None:
            print(f"Skipping folder {folder} due to missing data.")
            continue

        filtered_signal = savitzky_golay_filter(signals['Nasal Flow'])
        filtered_df = pd.DataFrame({'Nasal Flow': filtered_signal}, index=signals.index)

        output_path = os.path.join(out_dir, f"{folder}_sleep_stage_dataset.parquet")
        create_dataset(filtered_df, sleep_df, output_path)


process_all_folders('../Data', '../Dataset')