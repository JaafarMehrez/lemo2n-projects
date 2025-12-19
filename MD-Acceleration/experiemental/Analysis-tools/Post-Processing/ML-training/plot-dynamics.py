import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import os
import argparse
from scipy import signal
import pandas as pd

def analyze_training_dynamics(log_dir="tb_log", output_dir="dynamics_analysis"):
    """Analyze training dynamics using only loss/MAE metrics"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store all metrics
    metrics_data = {}
    
    # Load available metrics
    metric_files = {
        "loss_training": "loss_training",
        "loss_validation": "loss_validation",
        "MAE update_velocities_training": "MAE update_velocities_training",
        "MAE update_velocities_validation": "MAE update_velocities_validation",
        "MAE displacements_training": "MAE displacements_training", 
        "MAE displacements_validation": "MAE displacements_validation"
    }
    
    print("Loading metrics from TensorBoard logs...")
    for name, filename in metric_files.items():
        filepath = os.path.join(log_dir, filename)
        if os.path.exists(filepath):
            try:
                ea = event_accumulator.EventAccumulator(filepath)
                ea.Reload()
                
                # Get the scalar tag (usually same as metric name without suffix)
                scalar_tag = name.split('_')[-1]  # Gets 'training' or 'validation'
                if scalar_tag in ["training", "validation"]:
                    scalar_tag = "_".join(name.split('_')[:-1])
                
                if scalar_tag in ea.Tags()['scalars']:
                    scalars = ea.Scalars(scalar_tag)
                    epochs = [s.step for s in scalars]
                    values = [s.value for s in scalars]
                    metrics_data[name] = {'epochs': epochs, 'values': values}
                    print(f"  ✓ Loaded {name} ({len(values)} points)")
                else:
                    print(f"  ✗ No scalar tag '{scalar_tag}' in {filename}")
            except Exception as e:
                print(f"  ✗ Error loading {filename}: {e}")
        else:
            print(f"  ✗ File not found: {filename}")
    
    if not metrics_data:
        print("No metrics loaded. Exiting.")
        return
    
    # Create comprehensive analysis figure
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Original metrics (2x2 grid)
    ax1 = plt.subplot(3, 3, 1)
    ax2 = plt.subplot(3, 3, 2)
    ax3 = plt.subplot(3, 3, 4)
    ax4 = plt.subplot(3, 3, 5)
    
    # Plot available metrics
    plot_metric(ax1, metrics_data, "loss", "Loss")
    plot_metric(ax2, metrics_data, "MAE update_velocities", "MAE Update Velocities")
    plot_metric(ax3, metrics_data, "MAE displacements", "MAE Displacements")
    
    # 2. Ratio plot (validation/training) - overfitting indicator
    ax5 = plt.subplot(3, 3, 3)
    plot_ratio(ax5, metrics_data, "loss", "Loss Ratio (Val/Train)")
    
    # 3. Moving averages and smoothing
    ax6 = plt.subplot(3, 3, 6)
    plot_smoothed_metrics(ax6, metrics_data, window_size=10)
    
    # 4. Convergence analysis
    ax7 = plt.subplot(3, 3, 7)
    plot_convergence_rate(ax7, metrics_data, metric_name="loss")
    
    # 5. Early stopping analysis
    ax8 = plt.subplot(3, 3, 8)
    plot_early_stopping_analysis(ax8, metrics_data, metric_name="loss")
    
    # 6. Statistics summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    plot_statistics_summary(ax9, metrics_data)
    
    plt.suptitle('Training Dynamics Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'training_dynamics_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Training dynamics analysis saved to: {output_path}")
    
    # Generate CSV report
    generate_csv_report(metrics_data, output_dir)
    
    return metrics_data

def plot_metric(ax, metrics_data, metric_base, title):
    """Plot training and validation metrics"""
    train_key = f"{metric_base}_training"
    val_key = f"{metric_base}_validation"
    
    if train_key in metrics_data:
        ax.plot(metrics_data[train_key]['epochs'], 
                metrics_data[train_key]['values'],
                'b-', alpha=0.7, label='Training', linewidth=1.5)
    
    if val_key in metrics_data:
        ax.plot(metrics_data[val_key]['epochs'],
                metrics_data[val_key]['values'],
                'r-', alpha=0.7, label='Validation', linewidth=1.5)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add best value annotation
    if val_key in metrics_data:
        best_idx = np.argmin(metrics_data[val_key]['values'])
        best_epoch = metrics_data[val_key]['epochs'][best_idx]
        best_value = metrics_data[val_key]['values'][best_idx]
        ax.annotate(f'Best: {best_value:.4f}\n@ epoch {best_epoch}',
                   xy=(best_epoch, best_value),
                   xytext=(10, 10),
                   textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
                   fontsize=8)

def plot_ratio(ax, metrics_data, metric_base, title):
    """Plot ratio of validation to training metric (overfitting indicator)"""
    train_key = f"{metric_base}_training"
    val_key = f"{metric_base}_validation"
    
    if train_key in metrics_data and val_key in metrics_data:
        # Align epochs (assume same epochs for both)
        train_epochs = metrics_data[train_key]['epochs']
        train_values = metrics_data[train_key]['values']
        val_values = metrics_data[val_key]['values']
        
        # Calculate ratio
        ratio = np.array(val_values) / (np.array(train_values) + 1e-10)
        
        ax.plot(train_epochs, ratio, 'g-', alpha=0.7, linewidth=1.5)
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Ratio = 1')
        
        # Mark areas of potential overfitting (ratio > 1.1)
        overfit_mask = ratio > 1.1
        if np.any(overfit_mask):
            ax.fill_between(train_epochs, 1.1, ratio, where=overfit_mask,
                           color='red', alpha=0.2, label='Potential overfitting')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Val/Train Ratio')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_ratio = np.mean(ratio)
        final_ratio = ratio[-1]
        ax.text(0.02, 0.98, f'Mean: {mean_ratio:.3f}\nFinal: {final_ratio:.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

def plot_smoothed_metrics(ax, metrics_data, window_size=10):
    """Plot smoothed versions of metrics"""
    train_key = "loss_training"
    val_key = "loss_validation"
    
    if train_key in metrics_data and val_key in metrics_data:
        train_values = metrics_data[train_key]['values']
        val_values = metrics_data[val_key]['values']
        epochs = metrics_data[train_key]['epochs']
        
        # Apply moving average smoothing
        if len(train_values) >= window_size:
            kernel = np.ones(window_size) / window_size
            train_smooth = np.convolve(train_values, kernel, mode='valid')
            val_smooth = np.convolve(val_values, kernel, mode='valid')
            smooth_epochs = epochs[window_size-1:]
            
            ax.plot(smooth_epochs, train_smooth, 'b-', alpha=0.7, 
                   label=f'Training (smoothed)', linewidth=2)
            ax.plot(smooth_epochs, val_smooth, 'r-', alpha=0.7,
                   label=f'Validation (smoothed)', linewidth=2)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Smoothed Loss')
            ax.set_title(f'Smoothed Metrics (window={window_size})')
            ax.legend()
            ax.grid(True, alpha=0.3)

def plot_convergence_rate(ax, metrics_data, metric_name="loss"):
    """Plot convergence rate (derivative of loss)"""
    train_key = f"{metric_name}_training"
    
    if train_key in metrics_data:
        values = metrics_data[train_key]['values']
        epochs = metrics_data[train_key]['epochs']
        
        if len(values) > 1:
            # Calculate derivative (convergence rate)
            gradient = np.gradient(values)
            
            # Smooth the gradient
            if len(gradient) > 10:
                window = 5
                smooth_grad = np.convolve(gradient, np.ones(window)/window, mode='valid')
                grad_epochs = epochs[window-1:window-1+len(smooth_grad)]
                
                ax.plot(grad_epochs, smooth_grad, 'purple', alpha=0.7, linewidth=1.5)
                ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                
                # Mark convergence (gradient near zero)
                convergence_threshold = np.std(smooth_grad) * 0.1
                converged_mask = np.abs(smooth_grad) < convergence_threshold
                
                if np.any(converged_mask):
                    ax.fill_between(grad_epochs, -convergence_threshold, convergence_threshold,
                                   where=converged_mask, color='green', alpha=0.2,
                                   label='Convergence region')
                
                ax.set_xlabel('Epoch')
                ax.set_ylabel('d(Loss)/dEpoch')
                ax.set_title('Convergence Rate')
                ax.legend()
                ax.grid(True, alpha=0.3)

def plot_early_stopping_analysis(ax, metrics_data, metric_name="loss"):
    """Analyze when early stopping might have been optimal"""
    val_key = f"{metric_name}_validation"
    
    if val_key in metrics_data:
        values = metrics_data[val_key]['values']
        epochs = metrics_data[val_key]['epochs']
        
        # Find minimum validation loss
        min_idx = np.argmin(values)
        min_epoch = epochs[min_idx]
        min_value = values[min_idx]
        
        # Plot validation loss
        ax.plot(epochs, values, 'r-', alpha=0.7, label='Validation', linewidth=1.5)
        ax.axvline(x=min_epoch, color='g', linestyle='--', alpha=0.7,
                  label=f'Best epoch: {min_epoch}')
        
        # Mark the minimum point
        ax.plot(min_epoch, min_value, 'go', markersize=10, label=f'Best: {min_value:.4f}')
        
        # Calculate patience window (common early stopping practice)
        patience = min(20, len(values) // 4)
        if min_idx + patience < len(values):
            ax.axvspan(min_epoch, epochs[min_idx + patience], 
                      alpha=0.2, color='yellow', label=f'Patience window ({patience} epochs)')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Early Stopping Analysis')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

def plot_statistics_summary(ax, metrics_data):
    """Create text summary of statistics"""
    summary_text = "Training Statistics Summary\n"
    summary_text += "=" * 30 + "\n\n"
    
    for metric_base in ["loss", "MAE update_velocities", "MAE displacements"]:
        train_key = f"{metric_base}_training"
        val_key = f"{metric_base}_validation"
        
        if train_key in metrics_data and val_key in metrics_data:
            train_values = metrics_data[train_key]['values']
            val_values = metrics_data[val_key]['values']
            
            summary_text += f"{metric_base.replace('_', ' ').title()}:\n"
            summary_text += f"  Training - Final: {train_values[-1]:.6f}, Best: {min(train_values):.6f}\n"
            summary_text += f"  Validation - Final: {val_values[-1]:.6f}, Best: {min(val_values):.6f}\n"
            
            # Calculate improvement
            initial_train = train_values[0]
            final_train = train_values[-1]
            improvement = (initial_train - final_train) / initial_train * 100
            summary_text += f"  Improvement: {improvement:+.1f}%\n\n"
    
    # Calculate overfitting score
    if "loss_training" in metrics_data and "loss_validation" in metrics_data:
        train_final = metrics_data["loss_training"]['values'][-1]
        val_final = metrics_data["loss_validation"]['values'][-1]
        overfitting_score = (val_final - train_final) / train_final * 100
        summary_text += f"Overfitting Score: {overfitting_score:+.1f}%\n"
        if overfitting_score > 10:
            summary_text += "⚠️  Potential overfitting\n"
        elif overfitting_score < -10:
            summary_text += "⚠️  Potential underfitting\n"
        else:
            summary_text += "✓  Good generalization\n"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

def generate_csv_report(metrics_data, output_dir):
    """Generate CSV report with all metrics"""
    all_data = []
    
    for metric_name, data in metrics_data.items():
        for epoch, value in zip(data['epochs'], data['values']):
            all_data.append({
                'epoch': epoch,
                'metric': metric_name,
                'value': value
            })
    
    df = pd.DataFrame(all_data)
    
    # Pivot to wide format
    df_wide = df.pivot(index='epoch', columns='metric', values='value')
    df_wide = df_wide.sort_index()
    
    # Calculate additional statistics
    for metric_base in ["loss", "MAE update_velocities", "MAE displacements"]:
        train_col = f"{metric_base}_training"
        val_col = f"{metric_base}_validation"
        
        if train_col in df_wide.columns and val_col in df_wide.columns:
            df_wide[f"{metric_base}_ratio"] = df_wide[val_col] / df_wide[train_col]
            df_wide[f"{metric_base}_gap"] = df_wide[val_col] - df_wide[train_col]
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'training_metrics_report.csv')
    df_wide.to_csv(csv_path)
    print(f"✅ CSV report saved to: {csv_path}")
    
    # Also save summary statistics
    summary_stats = {}
    for col in df_wide.columns:
        if not pd.isna(df_wide[col]).all():
            summary_stats[col] = {
                'mean': df_wide[col].mean(),
                'std': df_wide[col].std(),
                'min': df_wide[col].min(),
                'max': df_wide[col].max(),
                'final': df_wide[col].iloc[-1]
            }
    
    summary_df = pd.DataFrame(summary_stats).T
    summary_path = os.path.join(output_dir, 'training_summary_stats.csv')
    summary_df.to_csv(summary_path)
    print(f"✅ Summary statistics saved to: {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze training dynamics from TensorBoard logs')
    parser.add_argument('--log-dir', type=str, default="tb_log",
                       help='Directory containing TensorBoard logs')
    parser.add_argument('--output-dir', type=str, default="dynamics_analysis",
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    analyze_training_dynamics(args.log_dir, args.output_dir)
