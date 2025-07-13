from scipy.signal import butter, filtfilt, savgol_filter
import pandas as pd
from load_files import get_flow_signal
import matplotlib.pyplot as plt

def butter_bandpass_filter(data, lowcut=0.17, highcut=0.4, fs=32, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, data)
    return pd.Series(filtered, index=data.index)

def savitzky_golay_filter(data, window_length=101, polyorder=3):
    filtered = savgol_filter(data, window_length, polyorder)
    return pd.Series(filtered, index=data.index)

# df = get_flow_signal('D:\SelectionTask\AP01\Flow - 30-05-2024.txt')
# signal_data = df['Nasal Flow']

# butter_filtered  = butter_bandpass_filter(signal_data)
# savgol_filtered  = savitzky_golay_filter(signal_data)

# plt.figure(figsize=(15, 6))
# plt.plot(signal_data.index, signal_data, label='Raw', alpha=0.3)
# plt.plot(butter_filtered.index, butter_filtered, label='Butterworth', linewidth=1)
# plt.plot(savgol_filtered .index, savgol_filtered , label='Savitzky-Golay', linewidth=1)
# plt.legend()
# plt.title("Filter Comparison - Nasal Flow")
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("filtered_comparison_nasal_flow.png", dpi=300, bbox_inches='tight')
# plt.show()
