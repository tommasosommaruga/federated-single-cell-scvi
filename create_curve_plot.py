import os
import glob
import pandas as pd
import toml
import matplotlib.pyplot as plt
import numpy as np
import re
import random

def load_epochs_per_round(loss_name):
    match = re.search(r'(\d+)_epochs', loss_name)
    if match:
        return int(match.group(1))
    return None
    
def plot_training_curves(csv_folder, loss_name, output_folder='plots', mode='compare_fed_scores', federated_loss_csv=None, max_models=8):
    os.makedirs(output_folder, exist_ok=True)
    all_csv_files = glob.glob(os.path.join(csv_folder, '*.csv'))
    all_csv_files = sorted(all_csv_files)
    if loss_name != 'check':
        epochs_per_round = load_epochs_per_round(loss_name)

    train_losses = {}
    test_losses = {}
    x_axes = {}
    model_names = []
    color_map = {}

    predefined_colors = {'Federated': (0.2, 0.4, 0.6), 'Centralised': (0, 0.5, 0)}

    def assign_color(model_name):
        if model_name in predefined_colors:
            return predefined_colors[model_name]
        if model_name.startswith('Client'):
            return (1, 0.4, 0)
        return (0.2, 0.4, 0.6)

    csv_files = []

    if mode == 'compare_fed_scores':
        federated_csvs = [f for f in all_csv_files if 'federated' in os.path.basename(f).lower()]
        centralized_loss_csv = next(f for f in all_csv_files if 'centralised' in os.path.basename(f).lower())
        selected_federated = random.sample(federated_csvs, k=min(len(federated_csvs), max_models - 1))
        csv_files = selected_federated + [centralized_loss_csv]

    elif mode == 'compare_models':
        if federated_loss_csv is None or not os.path.exists(federated_loss_csv):
            print("Error: federated_loss_csv is required for compare_models mode and must exist.")
            return
        csv_files = [federated_loss_csv] + [f for f in all_csv_files if 'federated' not in os.path.basename(f).lower() and f != federated_loss_csv]

    else:
        print(f"Error: Unknown mode '{mode}'")
        return

    for csv_file in csv_files:
        model_name = os.path.splitext(os.path.basename(csv_file))[0]
        if model_name.startswith("loss_curve_"): 
            model_name = model_name.replace("loss_curve_", "", 1)
        
        if loss_name == 'check':
            epochs_per_round = load_epochs_per_round(model_name)
        model_name = model_name.capitalize()
        model_names.append(model_name)

        df = pd.read_csv(csv_file)

        if 'epoch' in df.columns:
            group_col = 'epoch'
        elif 'round' in df.columns:
            group_col = 'round'
            df['epoch'] = np.minimum(df['round'] * epochs_per_round, 100)
            group_col = 'epoch'
        else:
            continue

        df_grouped = df.groupby(group_col, as_index=False).mean()
        x = df_grouped['epoch']

        if 'train_loss' in df_grouped.columns:
            train_losses[model_name] = df_grouped['train_loss']
            x_axes[model_name] = x

        if 'test_loss' in df_grouped.columns:
            test_losses[model_name] = df_grouped['test_loss']
            x_axes[model_name] = x

        color_map[model_name] = assign_color(model_name)

    # Plot train loss
    if mode != 'compare_fed_scores' and train_losses:
        plt.figure()
        for model in train_losses:
            if model in color_map:
                plt.plot(x_axes[model], train_losses[model], label=model, color=color_map[model])
        plt.xlabel('Epochs')
        plt.ylabel('Negative ELBO Train Loss')
        plt.title('Negative ELBO Train Loss Comparison')
        if mode != 'compare_fed_scores':
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'train_loss_comparison.png'))
        plt.close()

    # Plot test loss
    if test_losses:
        plt.figure()
        for model in test_losses:
            if model in color_map:
                if mode == 'compare_fed_scores':
                    plt.plot(x_axes[model], test_losses[model], label=model, alpha=0.5) # color=color_map[model],
                else:
                    plt.plot(x_axes[model], test_losses[model], label=model, color=color_map[model])
        plt.xlabel('Epochs')
        plt.ylabel('Negative ELBO Test Loss')
        plt.title('Negative ELBO Test Loss Comparison')
        #if mode != 'compare_fed_scores':
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'test_loss_comparison.png'))
        plt.close()

    print(f"[{mode}] Plots saved to:", output_folder)

if __name__ == "__main__":
    name_curve = 'loss_curve_federated_3_clients_34_rounds_3_epochs'
    plot_training_curves(
        csv_folder='loss_logs',
        loss_name=name_curve,
        output_folder='loss_logs/compare_models_plot',
        mode='compare_models',
        federated_loss_csv=f'loss_logs/{name_curve}.csv',
        max_models=8
    )
    plot_training_curves(
        csv_folder='loss_logs',
        loss_name='check',
        output_folder='loss_logs/compare_fed_plot',
        mode='compare_fed_scores'
    )
