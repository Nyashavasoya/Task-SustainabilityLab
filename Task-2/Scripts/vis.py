import matplotlib.pyplot as plt
import pandas as pd
import argparse
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import tempfile
import os
from load_files import load_all_signals

def get_time_range(signals):
    start_times = []
    end_times = []

    for name, data in signals.items():
        start_times.append(data.index.min())
        end_times.append(data.index.max())

    return min(start_times), max(end_times)


def single_plot(signals, events, start_time, end_time, participant_name, segment_num):
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    time_str = f"{start_time.strftime('%H:%M:%S')} to {end_time.strftime('%H:%M:%S')}"
    fig.suptitle(f'{participant_name} - Segment {segment_num} ({time_str})', fontsize=14, fontweight='bold')
    event_colors = {
        'Hypopnea': 'red',
        'Obstructive Apnea': 'orange'
    }

    # plot nasal flow
    nasal_data = signals['nasal_flow']
    mask = (nasal_data.index >= start_time) & (nasal_data.index <= end_time)
    filtered_data = nasal_data[mask]
    
    axes[0].plot(filtered_data.index, filtered_data.iloc[:, 0], 'b-', linewidth=0.8)
    axes[0].set_ylabel('Nasal Flow\n(L/min)', fontsize=10)
    axes[0].set_title('Nasal Airflow Signal', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(start_time, end_time)
    
    # overlay events on nasal flow
    events_in_window = events[(events['start_time'] < end_time) & (events['end_time'] > start_time)]
    for _, event in events_in_window.iterrows():
        color = event_colors.get(event['event_type'], 'gray')
        axes[0].axvspan(max(event['start_time'], start_time), 
                       min(event['end_time'], end_time), 
                       alpha=0.3, color=color)

    # plot thoracic
    thorac_data = signals['thorac']
    mask = (thorac_data.index >= start_time) & (thorac_data.index <= end_time)
    filtered_data = thorac_data[mask]
    
    axes[1].plot(filtered_data.index, filtered_data.iloc[:, 0], 'orange', linewidth=0.8)
    axes[1].set_ylabel('Thoracic Resp.\n(Amplitude)', fontsize=10)
    axes[1].set_title('Thoracic/Abdominal Respiratory Movement', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(start_time, end_time)

    # plot spo2
    spo2_data = signals['spo2']
    mask = (spo2_data.index >= start_time) & (spo2_data.index <= end_time)
    filtered_data = spo2_data[mask]
    
    axes[2].plot(filtered_data.index, filtered_data.iloc[:, 0], 'g-', linewidth=1)
    axes[2].set_ylabel('SpO2 (%)', fontsize=10)
    axes[2].set_xlabel('Time', fontsize=10)
    axes[2].set_title('Blood Oxygen Saturation (SpO2)', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(start_time, end_time)
    axes[2].set_ylim(90, 100)

    # set 5-second intervals on x-axis
    time_ticks = pd.date_range(start=start_time, end=end_time, freq='5s')
    for ax in axes:
        ax.set_xticks(time_ticks)
        ax.set_xticklabels([t.strftime('%H:%M:%S') for t in time_ticks], rotation=45, fontsize=8)

    plt.tight_layout()
    return fig


def create_all_plots(signals, events, participant_name):
    start_time, end_time = get_time_range(signals)
    
    figures = []
    current_time = start_time
    segment_num = 1
    
    while current_time < end_time:
        segment_end = min(current_time + pd.Timedelta(minutes=5), end_time)
        
        fig = single_plot(signals, events, current_time, segment_end, participant_name, segment_num)
        figures.append(fig)
        
        current_time = segment_end
        segment_num += 1
    
    return figures


def save_pdf(figures, output_path, participant_name, events):
    # create temp images
    temp_images = []
    for i, fig in enumerate(figures):
        temp_img = tempfile.NamedTemporaryFile(suffix=f'_{i}.png', delete=False)
        fig.savefig(temp_img.name, dpi=300, bbox_inches='tight')
        temp_images.append(temp_img.name)
        plt.close(fig)
    
    # create PDF
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    
    # title page
    title = Paragraph(f"Sleep Study Report - {participant_name}", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # events summary
    event_summary = events['event_type'].value_counts()
    summary_text = "Event Summary:<br/>"
    for event_type, count in event_summary.items():
        summary_text += f"• {event_type}: {count} events<br/>"
    
    summary_para = Paragraph(summary_text, styles['Normal'])
    story.append(summary_para)
    story.append(Spacer(1, 24))
    
    # add all plots
    for i, temp_img in enumerate(temp_images):
        segment_header = Paragraph(f"<b>Segment {i+1} (5 minutes)</b>", styles['Heading2'])
        story.append(segment_header)
        story.append(Spacer(1, 6))
        
        img = Image(temp_img, width=7.5*inch, height=6*inch)
        story.append(img)
        story.append(Spacer(1, 12))
    
    doc.build(story)
    
    # cleanup
    for temp_img in temp_images:
        os.unlink(temp_img)
    
    print(f"PDF saved: {output_path}")



parser = argparse.ArgumentParser(description='Generate sleep study visualizations')
parser.add_argument('-name', type=str, required=True, help='Path to participant data folder')
args = parser.parse_args()

folder_path = args.name
participant_name = os.path.basename(folder_path)

# create output directory
output_dir = 'Visualizations'
os.makedirs(output_dir, exist_ok=True)

print(f"Processing: {participant_name}")

# load data
signals, events = load_all_signals(folder_path)

# create plots
figures = create_all_plots(signals, events, participant_name)

# save PDF
output_path = os.path.join(output_dir, f'{participant_name}_visualization.pdf')
save_pdf(figures, output_path, participant_name, events)