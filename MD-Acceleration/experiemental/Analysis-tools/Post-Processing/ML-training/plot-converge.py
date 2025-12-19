import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import os
from scipy import signal

def analyze_convergence(log_dir="tb_log", output_dir="convergence_analysis"):
    """Analyze convergence patterns and stability"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load loss data
    training_loss_path = os.path.join(log_dir, "loss_training")
    validation_loss_path = os.path.join(log_dir, "loss_validation")
    
    try:
        # Load training loss
        train_event = event_accumulator.EventAccumulator(training_loss_path)
        train_event.Reload()
        train_data = train_event.Scalars("loss")
        train_epochs = [d.step for d in train_data]
        train_loss = [d.value for d in train_data]
        
        # Load validation loss
        val_event = event_accumulator.EventAccumulator(validation_loss_path)
        val_event.Reload()
        val_data = val_event.Scalars("loss")
        val_epochs = [d.step for d in val_data]
        val_loss = [d.value for d in val_data]
        
        # Ensure same length
        min_len = min(len(train_epochs), len(val_epochs))
        train_epochs = train_epochs[:min_len]
        train_loss = train_loss[:min_len]
        val_epochs = val_epochs[:min_len]
        val_loss = val_loss[:min_len]
        
        # Create analysis plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Raw losses
        axes[0, 0].plot(train_epochs, train_loss, 'b-', label='Training', alpha=0.7)
        axes[0, 0].plot(val_epochs, val_loss, 'r-', label='Validation', alpha=0.7)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training vs Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Loss ratio (overfitting indicator)
        loss_ratio = np.array(val_loss) / np.array(train_loss)
        axes[0, 1].plot(train_epochs, loss_ratio, 'g-', alpha=0.7)
        axes[0, 1].axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Equal')
        axes[0, 1].axhline(y=np.mean(loss_ratio), color='b', linestyle='--', alpha=0.5, 
                          label=f'Mean: {np.mean(loss_ratio):.2f}')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Validation/Training Loss Ratio')
        axes[0, 1].set_title('Overfitting Indicator\n(Ratio > 1 suggests overfitting)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Loss difference
        loss_diff = np.array(val_loss) - np.array(train_loss)
        axes[0, 2].plot(train_epochs, loss_diff, 'm-', alpha=0.7)
        axes[0, 2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 2].axhline(y=np.mean(loss_diff), color='b', linestyle='--', alpha=0.5,
                          label=f'Mean diff: {np.mean(loss_diff):.4f}')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Validation - Training Loss')
        axes[0, 2].set_title('Generalization Gap')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Moving averages
        window_size = min(50, len(train_loss) // 10)
        if window_size > 1:
            train_smooth = np.convolve(train_loss, np.ones(window_size)/window_size, mode='valid')
            val_smooth = np.convolve(val_loss, np.ones(window_size)/window_size, mode='valid')
            smooth_epochs = train_epochs[window_size-1:]
            
            axes[1, 0].plot(smooth_epochs, train_smooth, 'b-', label='Training (smooth)', linewidth=2)
            axes[1, 0].plot(smooth_epochs, val_smooth, 'r-', label='Validation (smooth)', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss (smoothed)')
            axes[1, 0].set_title(f'Smoothed Losses (window={window_size})')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Convergence rate (derivative of loss)
        if len(train_loss) > 1:
            train_grad = np.gradient(train_loss)
            val_grad = np.gradient(val_loss)
            
            # Smooth gradients
            train_grad_smooth = np.convolve(train_grad, np.ones(10)/10, mode='valid')
            val_grad_smooth = np.convolve(val_grad, np.ones(10)/10, mode='valid')
            grad_epochs = train_epochs[10-1:len(train_grad_smooth)+10-1]
            
            axes[1, 1].plot(grad_epochs, train_grad_smooth, 'b-', alpha=0.7, label='Training')
            axes[1, 1].plot(grad_epochs, val_grad_smooth, 'r-', alpha=0.7, label='Validation')
            axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('d(Loss)/dEpoch')
            axes[1, 1].set_title('Convergence Rate (smoothed gradient)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            # 6. Plateau detection
            # Find where gradient magnitude is small (plateaus)
            grad_magnitude = np.abs(train_grad_smooth)
            plateau_threshold = np.percentile(grad_magnitude, 25)
            plateau_mask = grad_magnitude < plateau_threshold
            
            axes[1, 2].plot(grad_epochs, grad_magnitude, 'g-', alpha=0.7, label='|Gradient|')
            axes[1, 2].axhline(y=plateau_threshold, color='r', linestyle='--', 
                              alpha=0.7, label=f'Plateau threshold: {plateau_threshold:.2e}')
            axes[1, 2].fill_between(grad_epochs, 0, plateau_threshold, 
                                   where=plateau_mask, alpha=0.3, color='yellow', label='Potential plateaus')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('|d(Loss)/dEpoch|')
            axes[1, 2].set_title('Plateau Detection')
            axes[1, 2].set_yscale('log')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle('Convergence Analysis', fontsize=16)
        plt.tight_layout()
        
        # Save convergence statistics
        convergence_stats = {
            'Final Training Loss': train_loss[-1],
            'Final Validation Loss': val_loss[-1],
            'Best Training Loss': min(train_loss),
            'Best Validation Loss': min(val_loss),
            'Mean Loss Ratio': np.mean(loss_ratio),
            'Std Loss Ratio': np.std(loss_ratio),
            'Training Convergence Epoch': np.argmin(train_loss) + 1,
            'Validation Convergence Epoch': np.argmin(val_loss) + 1,
            'Overfitting Score (Final Ratio)': loss_ratio[-1],
        }
        
        # Print statistics
        print("\nConvergence Statistics:")
        for key, value in convergence_stats.items():
            print(f"{key:30}: {value:.6f}")
        
        # Save plots
        plt.savefig(os.path.join(output_dir, 'convergence_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save statistics to file
        with open(os.path.join(output_dir, 'convergence_stats.txt'), 'w') as f:
            for key, value in convergence_stats.items():
                f.write(f"{key:30}: {value:.6f}\n")
        
    except Exception as e:
        print(f"Error analyzing convergence: {e}")

if __name__ == "__main__":
    analyze_convergence()
