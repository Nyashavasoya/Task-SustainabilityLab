from data_cleaning import savitzky_golay_filter
from load_files import get_flow_events, get_flow_signal
import os
import argparse
import pandas as pd
from datetime import timedelta

def get_label(win_start, win_end, events):
    for ev_start, ev_end, ev_label in events:
        delta = min(win_end, ev_end) - max(win_start, ev_start)
        overlap = max(delta.total_seconds(), 0) # converting timedelat back to seconds to make it usable for max()
        if overlap >= 15: # as the 50% of 30 sec is 15
            return ev_label
    return "Normal"


def create_dataset(signal_df, events, output_path):
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

        label = get_label(current, win_end, events)
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
    print(f"Dataset saved: {output_path}")


def process_all_folders(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    folders = [f for f in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, f))]

    for folder in folders:
        folder_path = os.path.join(in_dir, folder)
        events = None
        signals = None

        for file in os.listdir(folder_path):
            filepath = os.path.join(folder_path, file)
            base_name = os.path.splitext(file)[0].lower().replace(" ", "")

            if 'flowevent' in base_name or 'flow_event' in base_name:
                events_df = get_flow_events(filepath)
                events = list(events_df[['start_time', 'end_time', 'event_type']].itertuples(index=False, name=None))
            elif 'flow' in base_name or 'flowsignal' in base_name:
                signals = get_flow_signal(filepath)

        filtered_signal = savitzky_golay_filter(signals['Nasal Flow'])
        filtered_df = pd.DataFrame({'Nasal Flow': filtered_signal}, index=signals.index)
        output_path = os.path.join(out_dir, f"{folder}_flow_dataset.parquet")
        create_dataset(filtered_df, events, output_path)


parser = argparse.ArgumentParser(description='Data creation from the filtered signals')
parser.add_argument('-in_dir', type=str, required=True, help='Path to participant data folder')
parser.add_argument('-out_dir', type=str, required=True, help='Path to output filtered data')
args = parser.parse_args()


process_all_folders(args.in_dir, args.out_dir)