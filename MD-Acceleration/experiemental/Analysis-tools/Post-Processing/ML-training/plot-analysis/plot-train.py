from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import os
import argparse

def main(start_epoch=None, end_epoch=None):
    # Create plots directory
    out_dir = "plots"
    os.makedirs(out_dir, exist_ok=True)
    
    # Define plot configurations
    plot_configs = [
        {
            "metrics": ["MAE update_velocities_training", "MAE update_velocities_validation"],
            "scalar_tag": "MAE update_velocities",
            "ylabel": "MAE update_velocities",
            "filename": "mae_update_velocities.png"
        },
        {
            "metrics": ["MAE displacements_training", "MAE displacements_validation"],
            "scalar_tag": "MAE displacements",
            "ylabel": "MAE displacements",
            "filename": "mae_displacement.png"
        },
        {
            "metrics": ["loss_training", "loss_validation"],
            "scalar_tag": "loss",
            "ylabel": "Loss",
            "filename": "loss.png"
        }
    ]
    
    # Process each plot configuration
    for config in plot_configs:
        fig, ax = plt.subplots(1, 1)
        
        for metric in config["metrics"]:
            try:
                event = event_accumulator.EventAccumulator(f"tb_log/{metric}")
                event.Reload()
                data = event.Scalars(config["scalar_tag"])
                
                # Filter data by epoch range if specified
                if start_epoch is not None or end_epoch is not None:
                    filtered_data = []
                    for i in data:
                        if start_epoch is not None and i.step < start_epoch:
                            continue
                        if end_epoch is not None and i.step > end_epoch:
                            continue
                        filtered_data.append(i)
                    data = filtered_data
                
                if not data:
                    print(f"No data in range for {metric}")
                    continue
                
                epochs = [i.step for i in data]
                loss = [i.value for i in data]
                ax.plot(epochs, loss, label=metric)
            except Exception as e:
                print(f"Error loading {metric}: {e}")
        
        ax.set_ylabel(config["ylabel"])
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(False)
        fig.tight_layout()
        
        # Add epoch range to filename if specified
        filename = config["filename"]
        if start_epoch is not None and end_epoch is not None:
            name_part = filename.split('.')[0]
            ext_part = filename.split('.')[1] if '.' in filename else 'png'
            filename = f"{name_part}_epochs_{start_epoch}-{end_epoch}.{ext_part}"
        elif start_epoch is not None:
            name_part = filename.split('.')[0]
            ext_part = filename.split('.')[1] if '.' in filename else 'png'
            filename = f"{name_part}_from_epoch_{start_epoch}.{ext_part}"
        elif end_epoch is not None:
            name_part = filename.split('.')[0]
            ext_part = filename.split('.')[1] if '.' in filename else 'png'
            filename = f"{name_part}_to_epoch_{end_epoch}.{ext_part}"
        
        png_path = os.path.join(out_dir, filename)
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {png_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training metrics with epoch range')
    parser.add_argument('--start-epoch', type=int, default=None, 
                       help='Starting epoch (inclusive)')
    parser.add_argument('--end-epoch', type=int, default=None, 
                       help='Ending epoch (inclusive)')
    
    args = parser.parse_args()
    main(start_epoch=args.start_epoch, end_epoch=args.end_epoch)
