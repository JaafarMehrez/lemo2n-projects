import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import os

def generate_training_report(log_dir="tb_log", output_dir="training_report"):
    """Generate a comprehensive training report"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all available metrics
    metrics_to_load = [
        "loss_training", "loss_validation",
        "MAE update_velocities_training", "MAE update_velocities_validation",
        "MAE displacements_training", "MAE displacements_validation"
    ]
    
    data = {}
    for metric in metrics_to_load:
        try:
            event = event_accumulator.EventAccumulator(os.path.join(log_dir, metric))
            event.Reload()
            # Get the first scalar tag (assuming there's only one per file)
            scalar_tags = event.Tags()['scalars']
            if scalar_tags:
                tag = scalar_tags[0]
                metric_data = event.Scalars(tag)
                data[metric] = {
                    'epochs': [d.step for d in metric_data],
                    'values': [d.value for d in metric_data]
                }
        except Exception as e:
            print(f"Could not load {metric}: {e}")
    
    # Create comprehensive report
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # Plot 1: Loss curves
    if 'loss_training' in data and 'loss_validation' in data:
        ax = axes[0, 0]
        ax.plot(data['loss_training']['epochs'], data['loss_training']['values'], 
                'b-', label='Training', alpha=0.7)
        ax.plot(data['loss_validation']['epochs'], data['loss_validation']['values'], 
                'r-', label='Validation', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 2: MAE update velocities
    if 'MAE update_velocities_training' in data and 'MAE update_velocities_validation' in data:
        ax = axes[0, 1]
        ax.plot(data['MAE update_velocities_training']['epochs'], 
                data['MAE update_velocities_training']['values'], 
                'b-', label='Training', alpha=0.7)
        ax.plot(data['MAE update_velocities_validation']['epochs'], 
                data['MAE update_velocities_validation']['values'], 
                'r-', label='Validation', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE Update Velocities')
        ax.set_title('MAE Update Velocities')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 3: MAE displacements
    if 'MAE displacements_training' in data and 'MAE displacements_validation' in data:
        ax = axes[0, 2]
        ax.plot(data['MAE displacements_training']['epochs'], 
                data['MAE displacements_training']['values'], 
                'b-', label='Training', alpha=0.7)
        ax.plot(data['MAE displacements_validation']['epochs'], 
                data['MAE displacements_validation']['values'], 
                'r-', label='Validation', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE Displacements')
        ax.set_title('MAE Displacements')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Final values comparison (bar chart)
    ax = axes[1, 0]
    final_values = []
    labels = []
    colors = []
    
    for metric in ['loss', 'MAE update_velocities', 'MAE displacements']:
        train_key = f"{metric}_training"
        val_key = f"{metric}_validation"
        
        if train_key in data and val_key in data:
            final_values.extend([data[train_key]['values'][-1], data[val_key]['values'][-1]])
            labels.extend([f'{metric}\n(Train)', f'{metric}\n(Val)'])
            colors.extend(['blue', 'red'])
    
    if final_values:
        bars = ax.bar(range(len(final_values)), final_values, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Final Value')
        ax.set_title('Final Metric Values')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, final_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 5: Improvement ratios (final/initial)
    ax = axes[1, 1]
    improvement_ratios = []
    metric_names = []
    
    for metric in ['loss_training', 'MAE update_velocities_training', 'MAE displacements_training']:
        if metric in data:
            initial = data[metric]['values'][0]
            final = data[metric]['values'][-1]
            improvement = (initial - final) / initial * 100  # percentage improvement
            improvement_ratios.append(improvement)
            metric_names.append(metric.replace('_training', ''))
    
    if improvement_ratios:
        colors_improve = ['green' if x > 0 else 'red' for x in improvement_ratios]
        bars = ax.bar(metric_names, improvement_ratios, color=colors_improve)
        ax.set_xlabel('Metric')
        ax.set_ylabel('Improvement (%)')
        ax.set_title('Training Improvement\n(Final vs Initial)')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, improvement_ratios):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}%', ha='center', va='bottom' if val >= 0 else 'top', 
                   fontsize=8)
    
    # Plot 6: Training vs Validation gap
    ax = axes[1, 2]
    gaps = []
    gap_labels = []
    
    for metric_base in ['loss', 'MAE update_velocities', 'MAE displacements']:
        train_key = f"{metric_base}_training"
        val_key = f"{metric_base}_validation"
        
        if train_key in data and val_key in data:
            train_final = data[train_key]['values'][-1]
            val_final = data[val_key]['values'][-1]
            gap = val_final - train_final
            gaps.append(gap)
            gap_labels.append(metric_base)
    
    if gaps:
        colors_gap = ['red' if x > 0 else 'green' for x in gaps]
        bars = ax.bar(gap_labels, gaps, color=colors_gap)
        ax.set_xlabel('Metric')
        ax.set_ylabel('Validation - Training (Final)')
        ax.set_title('Generalization Gap\n(Positive = Overfitting)')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, gaps):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom' if val >= 0 else 'top', 
                   fontsize=8)
    
    # Plot 7-9: Statistics summary (text)
    for i in range(3):
        ax = axes[2, i]
        ax.axis('off')
        
        # Create summary text
        summary_text = "Training Summary\n"
        summary_text += "=" * 20 + "\n"
        summary_text += f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        if i == 0:
            summary_text += "Epoch Statistics:\n"
            if data:
                sample_metric = list(data.values())[0]
                summary_text += f"Total epochs: {len(sample_metric['epochs'])}\n"
                summary_text += f"First epoch: {sample_metric['epochs'][0]}\n"
                summary_text += f"Last epoch: {sample_metric['epochs'][-1]}\n"
        
        elif i == 1:
            summary_text += "Final Metrics:\n"
            for metric_base in ['loss', 'MAE update_velocities', 'MAE displacements']:
                train_key = f"{metric_base}_training"
                val_key = f"{metric_base}_validation"
                if train_key in data and val_key in data:
                    train_val = data[train_key]['values'][-1]
                    val_val = data[val_key]['values'][-1]
                    summary_text += f"{metric_base}:\n"
                    summary_text += f"  Train: {train_val:.6f}\n"
                    summary_text += f"  Val:   {val_val:.6f}\n"
        
        elif i == 2:
            summary_text += "Improvement Summary:\n"
            for metric_base in ['loss', 'MAE update_velocities', 'MAE displacements']:
                train_key = f"{metric_base}_training"
                if train_key in data:
                    initial = data[train_key]['values'][0]
                    final = data[train_key]['values'][-1]
                    improvement = (initial - final) / initial * 100
                    summary_text += f"{metric_base}:\n"
                    summary_text += f"  {improvement:+.1f}%\n"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Comprehensive Training Analysis Report', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the report
    report_path = os.path.join(output_dir, 'training_report.png')
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅Training report generated: {report_path}")
    
    # Also save data as JSON for further analysis
    json_path = os.path.join(output_dir, 'training_data.json')
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
    
    print(f"Training data saved: {json_path}")

if __name__ == "__main__":
    generate_training_report()
