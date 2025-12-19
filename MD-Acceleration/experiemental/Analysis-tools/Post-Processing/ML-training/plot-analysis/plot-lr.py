import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import os
import argparse

def plot_learning_dynamics(log_dir="tb_log", output_dir="analysis_plots"):
    """Plot learning rate schedules and gradient statistics if available"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Try to find learning rate data
    lr_events = []
    for event_file in os.listdir(log_dir):
        if "learning_rate" in event_file.lower() or "lr" in event_file.lower():
            lr_events.append(event_file)
    
    if lr_events:
        fig, axes = plt.subplots(1, len(lr_events), figsize=(5*len(lr_events), 4))
        if len(lr_events) == 1:
            axes = [axes]
        
        for idx, event_file in enumerate(lr_events):
            try:
                event = event_accumulator.EventAccumulator(os.path.join(log_dir, event_file))
                event.Reload()
                
                # Try different possible scalar tags
                possible_tags = ["learning_rate", "lr", "Learning Rate", "LR"]
                data = None
                for tag in possible_tags:
                    if tag in event.Tags()['scalars']:
                        data = event.Scalars(tag)
                        break
                
                if data:
                    epochs = [d.step for d in data]
                    lr_values = [d.value for d in data]
                    
                    axes[idx].plot(epochs, lr_values, 'b-', linewidth=2)
                    axes[idx].set_xlabel('Epoch')
                    axes[idx].set_ylabel('Learning Rate')
                    axes[idx].set_title(f'Learning Rate Schedule\n{event_file}')
                    axes[idx].set_yscale('log')
                    axes[idx].grid(True, alpha=0.3)
                    
                    # Add statistics
                    stats_text = f"Initial: {lr_values[0]:.2e}\nFinal: {lr_values[-1]:.2e}\nChange: {lr_values[-1]/lr_values[0]:.2f}x"
                    axes[idx].text(0.02, 0.98, stats_text, transform=axes[idx].transAxes,
                                 verticalalignment='top', fontsize=9,
                                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            except Exception as e:
                print(f"Error processing {event_file}: {e}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'learning_rate_schedule.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Try to find gradient norm data
    grad_events = []
    for event_file in os.listdir(log_dir):
        if "grad" in event_file.lower() or "norm" in event_file.lower():
            grad_events.append(event_file)
    
    if grad_events:
        fig, axes = plt.subplots(1, min(3, len(grad_events)), figsize=(15, 4))
        if min(3, len(grad_events)) == 1:
            axes = [axes]
        
        for idx, event_file in enumerate(grad_events[:3]):  # Limit to first 3
            try:
                event = event_accumulator.EventAccumulator(os.path.join(log_dir, event_file))
                event.Reload()
                
                # Look for gradient-related tags
                for tag in event.Tags()['scalars']:
                    if "grad" in tag.lower() or "norm" in tag.lower():
                        data = event.Scalars(tag)
                        epochs = [d.step for d in data]
                        values = [d.value for d in data]
                        
                        axes[idx].plot(epochs, values, 'g-', alpha=0.7)
                        axes[idx].set_xlabel('Epoch')
                        axes[idx].set_ylabel('Value')
                        axes[idx].set_title(f'{tag}\n{event_file}')
                        axes[idx].set_yscale('log')
                        axes[idx].grid(True, alpha=0.3)
                        
                        # Add statistics
                        stats_text = f"Mean: {np.mean(values):.2e}\nStd: {np.std(values):.2e}\nMax: {np.max(values):.2e}"
                        axes[idx].text(0.02, 0.98, stats_text, transform=axes[idx].transAxes,
                                     verticalalignment='top', fontsize=9,
                                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
                        break
            except Exception as e:
                print(f"Error processing {event_file}: {e}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gradient_statistics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Learning dynamics plots saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot learning dynamics')
    parser.add_argument('--log-dir', type=str, default="tb_log",
                       help='Directory containing tensorboard logs')
    parser.add_argument('--output-dir', type=str, default="analysis_plots",
                       help='Output directory for plots')
    
    args = parser.parse_args()
    plot_learning_dynamics(args.log_dir, args.output_dir)
