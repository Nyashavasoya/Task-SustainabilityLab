import pandas as pd
import os

import pandas as pd

def get_flow_signal(path, signal_name='Nasal Flow', sampling_rate=32):
    with open(path, 'r', encoding='utf-8') as f:
        rows = f.readlines()

    start_idx = next(i for i, row in enumerate(rows) if 'data:' in row.strip().lower()) + 1
    data = rows[start_idx:]

    timestamps = []
    values = []

    for row in data:
        if ";" not in row:
            continue

        timestamp_str, value_str = row.strip().split(";")
        timestamp = pd.to_datetime(timestamp_str.strip(), format="%d.%m.%Y %H:%M:%S,%f")
        value = float(value_str.strip())
        timestamps.append(timestamp)
        values.append(value)

    df = pd.DataFrame({signal_name: values}, index=pd.to_datetime(timestamps))
    df.index.name = 'Timestamp'
    print(len(df))
    return df



def get_flow_events(path):
    with open(path, 'r', encoding='utf-8') as f:
        rows = f.readlines()

    start_idx = next(i for i, row in enumerate(rows) if row.strip() == 'Signal Type: Impuls') + 2
    data = rows[start_idx: ]

    start_times = []
    end_times = []
    values = []
    event_types = []
    sleep_stages = []

    for row in data:
        time_range, value_str, event_type, sleep_stage = [x.strip() for x in row.split(";")]
        start_str, end_str = time_range.split("-")
        if len(end_str.split(" ")[0].split(".")) < 3:
            end_str = start_str.split(" ")[0] + " " + end_str

        start_dt = pd.to_datetime(start_str, format="%d.%m.%Y %H:%M:%S,%f")
        end_dt = pd.to_datetime(end_str, format="%d.%m.%Y %H:%M:%S,%f")
        value = float(value_str)
        start_times.append(start_dt)
        end_times.append(end_dt)
        values.append(value)
        event_types.append(event_type)
        sleep_stages.append(sleep_stage)

    df = pd.DataFrame({
        "start_time": start_times,
        "end_time": end_times,
        "values": values,
        "event_type": event_types,
        "sleep_stage": sleep_stages
    })
    print(len(df))
    return df


def get_sleep_profile(path, signal_name='Event'):
    with open(path, 'r', encoding='utf-8') as f:
        rows = f.readlines()

    start_idx = next(i for i, row in enumerate(rows) if "Rate: 30 s" in row) + 2
    data = rows[start_idx:]


    timestamps = []
    events = []

    for row in data:
        timestamp_str, events_str = row.strip().split(";")
        timestamp = pd.to_datetime(timestamp_str.strip(), format="%d.%m.%Y %H:%M:%S,%f")
        timestamps.append(timestamp)
        events.append(events_str)

    df = pd.DataFrame({signal_name: events}, index=pd.to_datetime(timestamps))
    df.index.name = 'Timestamp'
    print(len(df))
    return df


def get_spo2(path, signal_name='SPO2 Type', sampling_rate=4):
    with open(path, 'r', encoding='utf-8') as f:
        rows = f.readlines()

    start_idx = next(i for i, row in enumerate(rows) if row.strip() == 'Data:') + 1
    data = rows[start_idx: ]

    timestamps = []
    values = []

    for row in data:
        timestamp_str, value_str = row.strip().split(";")
        timestamp = pd.to_datetime(timestamp_str.strip(), format="%d.%m.%Y %H:%M:%S,%f")
        value = float(value_str.strip())
        timestamps.append(timestamp)
        values.append(value)

    df = pd.DataFrame({signal_name: values}, index=pd.to_datetime(timestamps))
    df.index.name = 'Timestamp'
    print(len(df))
    return df


def get_thorac(path, signal_name='Thorac Value', sampling_rate=32):
    with open(path, 'r', encoding='utf-8') as f:
        rows = f.readlines()

    start_idx = next(i for i, row in enumerate(rows) if row.strip() == 'Data:') + 1
    data = rows[start_idx: ]

    timestamps = []
    values = []

    for row in data:
        timestamp_str, value_str = row.strip().split(";")
        timestamp = pd.to_datetime(timestamp_str.strip(), format="%d.%m.%Y %H:%M:%S,%f")
        value = float(value_str.strip())
        timestamps.append(timestamp)
        values.append(value)

    df = pd.DataFrame({signal_name: values}, index=pd.to_datetime(timestamps))
    df.index.name = 'Timestamp'
    print(len(df))
    return df


def load_all_signals(folder_path):
    signals = {}
    events = pd.DataFrame()

    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        base_name = os.path.splitext(filename)[0].lower().replace(" ", "")

        if 'flowevents' in base_name:
            events = get_flow_events(filepath)
        elif 'flow' in base_name:
            signals['nasal_flow'] = get_flow_signal(filepath)
        elif 'sleepprofile' in base_name:
            signals['sleep_profile'] = get_sleep_profile(filepath)
        elif 'spo2' in base_name:
            signals['spo2'] = get_spo2(filepath)
        elif 'thorac' in base_name:
            signals['thorac'] = get_thorac(filepath)
        else:
            print(f"Unrecognized file: {filename}")

    return signals, events