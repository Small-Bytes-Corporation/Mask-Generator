import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Configuration
csv_file = "metrics.csv"
output_dir = "MetricsLines/"
figure_size = (12, 8)
dpi = 300

Path(output_dir).mkdir(exist_ok=True)
plt.style.use('seaborn-v0_8')
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

def load_data(csv_path):
    """Load and prepare CSV data"""
    print(f"Loading data from {csv_path}...")

    data = {'epoch': [], 'step': [], 'train_loss': [], 'train_acc': [], 'train_iou': [], 'train_dice': []}

    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data['epoch'].append(int(row['epoch']))
            data['step'].append(int(row['step']))
            data['train_loss'].append(float(row['train_loss']))
            data['train_acc'].append(float(row['train_acc']))
            data['train_iou'].append(float(row['train_iou']))
            data['train_dice'].append(float(row['train_dice']))

    max_step_per_epoch = max(data['step'])
    global_steps = []
    for i, (epoch, step) in enumerate(zip(data['epoch'], data['step'])):
        global_step = (epoch - 1) * max_step_per_epoch + step
        global_steps.append(global_step)

    data['global_step'] = global_steps

    print(f"Loaded {len(data['epoch'])} data points")
    return data

def setup_plot():
    """Create a standardized plot setup"""
    fig, ax = plt.subplots(figsize=figure_size, dpi=dpi)
    return fig, ax

def finalize_plot(fig, ax, save_path):
    """Apply common plot formatting and save"""
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

def create_loss_plot(data, save_path):
    """Create loss over time plot"""
    print("Creating loss plot...")

    fig, ax = setup_plot()

    ax.plot(data['global_step'], data['train_loss'], color=colors[0], linewidth=2)
    ax.set_xlabel('Step', fontweight='bold')
    ax.set_ylabel('Training Loss', fontweight='bold')
    ax.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    ax.set_yscale('symlog', linthresh=1e-4)

    finalize_plot(fig, ax, save_path)

def create_single_metric_plot(data, metric_key, metric_name, color, save_path, use_distance=True):
    """Create a plot for a single metric"""
    print(f"Creating {metric_name} plot...")

    fig, ax = setup_plot()

    if use_distance:
        values = [1.0 - val for val in data[metric_key]]
        ylabel = f'1 - {metric_name}'
        title = f'Training {metric_name} Over Time'
        ax.set_yscale('symlog', linthresh=1e-5)
    else:
        values = data[metric_key]
        ylabel = f'Training {metric_name}'
        title = f'Training {metric_name} Over Time'

    ax.plot(data['global_step'], values, color=color, linewidth=2)
    ax.set_xlabel('Step', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')

    finalize_plot(fig, ax, save_path)

def create_metrics_plot(data, save_path):
    """Create combined metrics plot (accuracy, IoU, Dice)"""
    print("Creating combined metrics plot...")

    fig, ax = setup_plot()

    metrics = ['train_acc', 'train_iou', 'train_dice']
    metric_labels = ['Accuracy', 'IoU', 'Dice']

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        transformed_values = [1.0 - val for val in data[metric]]
        ax.plot(data['global_step'], transformed_values, color=colors[i],
               linewidth=2, label=label)

    ax.set_xlabel('Step', fontweight='bold')
    ax.set_ylabel('1 - Metric', fontweight='bold')
    ax.set_title('Training Metrics Over Time', fontsize=14, fontweight='bold')
    ax.set_yscale('symlog', linthresh=1e-5)
    ax.legend(frameon=True, fancybox=True, shadow=True)

    finalize_plot(fig, ax, save_path)

def create_all_plots(data):
    """Create all plots - both combined and individual"""
    # Combined plots
    loss_path = os.path.join(output_dir, "loss_plot.png")
    metrics_path = os.path.join(output_dir, "metrics_plot.png")

    create_loss_plot(data, loss_path)
    create_metrics_plot(data, metrics_path)

    # Individual metric plots
    metrics_config = [
        ('train_acc', 'Accuracy', colors[0], 'accuracy_plot.png'),
        ('train_iou', 'IoU', colors[1], 'iou_plot.png'),
        ('train_dice', 'Dice', colors[2], 'dice_plot.png')
    ]

    for metric_key, metric_name, color, filename in metrics_config:
        save_path = os.path.join(output_dir, filename)
        create_single_metric_plot(data, metric_key, metric_name, color, save_path)

def main():
    """Main processing pipeline"""
    print("Starting metrics visualization...")

    data = load_data(csv_file)
    create_all_plots(data)

    print(f"All plots saved to '{output_dir}' directory")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted, exiting.")
        exit(1)
