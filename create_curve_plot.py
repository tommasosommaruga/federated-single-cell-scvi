import os
import glob
import pandas as pd
import toml
import matplotlib.pyplot as plt
import numpy as np

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
    csv_files = sorted(csv_files)
    epochs_per_round = load_epochs_per_round(pyproject_path)

    train_losses = {}
    test_losses = {}
    x_axes = {}
    model_names = []  # List to store the model names

    # Store color assignments
    color_map = {}

    # Color definitions for better visibility (strong and dark)
    predefined_colors = {'Federated': (0.2, 0.4, 0.6),'Centralised': (0, 0.5, 0)}

    # Assign strong orange for all independent clients
    def assign_color(model_name):
        # Check if the model is predefined (Centralised or Federated)
        if model_name in predefined_colors:
            return predefined_colors[model_name]
        
        # For independent clients, assign a strong orange color
        if model_name.startswith('Client'):
            return (1, 0.4, 0)  # Strong Orange (RGB)

        # Otherwise, default to dark gray for any undefined models
        return (0.2, 0.2, 0.2)  # Dark Gray for unknown models

    for csv_file in csv_files:
        model_name = os.path.splitext(os.path.basename(csv_file))[0]
        # Remove "loss_curve_" prefix if present
        if model_name.startswith("loss_curve_"): 
            model_name = model_name.replace("loss_curve_", "", 1)
        model_name = model_name.capitalize()

        model_names.append(model_name)  # Add model name to the list
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

        # Assign color to each model
        color_map[model_name] = assign_color(model_name)

    # Plot train loss
    if train_losses:
        plt.figure()
        for model in train_losses:
            if model in color_map:  # Only plot models with available train loss
                plt.plot(x_axes[model], train_losses[model], label=model, color=color_map[model])
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
        for model in test_losses:
            if model in color_map:  # Only plot models with available test loss
                plt.plot(x_axes[model], test_losses[model], label=model, color=color_map[model])
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
        output_folder='loss_logs/loss_plots'
    )
