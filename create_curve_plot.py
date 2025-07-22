import os
import glob
import pandas as pd
import toml

import matplotlib.pyplot as plt

def load_epochs_per_round(pyproject_path):
    """Load epochs_per_round from pyproject.toml."""
    try:
        config = toml.load(pyproject_path)
        return config.get('tool', {}).get('flwr', {}).get('app', {}).get('config', {}).get('local-epochs', 1)
    except Exception:
        return 1

def plot_training_curves(csv_folder, pyproject_path, output_folder='plots'):
    # Plots train and test loss curves for federated, centralized, and independent models.
    
    os.makedirs(output_folder, exist_ok=True)
    csv_files = glob.glob(os.path.join(csv_folder, '*.csv'))
    epochs_per_round = load_epochs_per_round(pyproject_path)

    train_losses = {}
    test_losses = {}
    x_axes = {}

    for csv_file in csv_files:
        model_name = os.path.splitext(os.path.basename(csv_file))[0]
        # Remove "loss_curve_" prefix if present
        if model_name.startswith("loss_curve_"): model_name = model_name.replace("loss_curve_", "", 1)
        model_name = model_name.capitalize()
        df = pd.read_csv(csv_file)
        # Determine x-axis: epochs or rounds
        if 'epoch' in df.columns:
            x = df['epoch']
        elif 'round' in df.columns:
            x = df['round'] * epochs_per_round
        else:
            continue  # Skip if neither present

        # Collect train loss if available
        if 'train_loss' in df.columns:
            train_losses[model_name] = df['train_loss']
            x_axes[model_name] = x

        # Collect test loss if available
        if 'test_loss' in df.columns:
            test_losses[model_name] = df['test_loss']
            x_axes[model_name] = x

    # Plot train loss
    if train_losses:
        plt.figure()
        for model, loss in train_losses.items():
            plt.plot(x_axes[model], loss, label=model)
        plt.xlabel('Epochs')
        plt.ylabel('Train Loss')
        plt.title('Train Loss Comparison')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'train_loss_comparison.png'))
        plt.close()

    # Plot test loss
    if test_losses:
        plt.figure()
        for model, loss in test_losses.items():
            plt.plot(x_axes[model], loss, label=model)
        plt.xlabel('Epochs')
        plt.ylabel('Test Loss')
        plt.title('Test Loss Comparison')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'test_loss_comparison.png'))
        plt.close()

if __name__ == "__main__":
    plot_training_curves(
        csv_folder='loss_logs',
        pyproject_path='pyproject.toml',
        output_folder='plots'
    )