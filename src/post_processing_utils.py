
import torch
from sklearn.metrics import roc_curve
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from src.evaluate import _calculate_metrics




def n_random_post_process(df, n_random_splits, sample_fraction=0.75):
    out_df = pd.DataFrame(columns=df.columns)
    for _, row in df.iterrows():
        y = row['y']
        y_pred = row['y_pred']
        y_pred_label = row['y_pred_label']

        if isinstance(y_pred, str):
            y_pred = np.array(literal_eval(y_pred))
        elif isinstance(y_pred, list):
            y_pred = np.array(y_pred)

        if isinstance(y_pred_label, str):
            y_pred_label = np.array(literal_eval(y_pred_label))
        elif isinstance(y_pred_label, list):
            y_pred_label = np.array(y_pred_label)

        if isinstance(y, str):
            y = np.array(literal_eval(y))
        elif isinstance(y, list):
            y = np.array(y)

        total_samples = len(y)
        sample_size = int(total_samples * sample_fraction)
        
        for i, split_num in enumerate(range(n_random_splits)):
            np.random.seed(3*i)  # Use np.random seed instead of random.seed
            random_indices = np.random.choice(total_samples, sample_size, replace=False)

            y_fold = y[random_indices]
            y_pred_fold = y_pred[random_indices]
            y_pred_label_fold = y_pred_label[random_indices]

            eval_results = _calculate_metrics(
                torch.Tensor(y_fold),
                torch.Tensor(y_pred_fold),
                torch.Tensor(y_pred_label_fold)
            )

            eer, t = calculate_eer(y_fold, y_pred_fold)

            eval_results['eer'] = eer
            eval_results['split_num'] = split_num
            eval_results['model'] = row["model"]
            eval_results['evaluated languages'] = row["evaluated languages"]
            eval_results['fine-tuned languages'] = row["fine-tuned languages"]

            out_df = pd.concat([out_df, pd.DataFrame(eval_results, index=[0])], ignore_index=True)

    return out_df

def kfold_post_process(df, k_fold_splits):
    out_df = pd.DataFrame(columns=df.columns)
    for _, row in df.iterrows():
        print(type(row['y']))
        y = row['y']
        y_pred = row['y_pred']
        y_pred_label = row['y_pred_label']
        if isinstance(y_pred, str):
            y_pred = np.array(literal_eval(y_pred))
        elif isinstance(y_pred, list):
            y_pred = np.array(y_pred)
        
        if isinstance(y_pred_label, str):
            y_pred_label = np.array(literal_eval(y_pred_label))
        elif isinstance(y_pred_label, list):
            y_pred_label = np.array(y_pred_label)
        
        if isinstance(y, str):
            y = np.array(literal_eval(y))
        elif isinstance(y, list):
            y = np.array(y)
            
        kf = KFold(n_splits=k_fold_splits, shuffle=True, random_state=42)
        for fold, (idx, _) in enumerate(kf.split(y)):
            y_fold = y[idx]
            y_pred_fold = y_pred[idx]
            y_pred_label_fold = y_pred_label[idx]
            eval_results = _calculate_metrics(
                torch.Tensor(y_fold),
                torch.Tensor(y_pred_fold),
                torch.Tensor(y_pred_label_fold)
            )

            eer, t = calculate_eer(y_fold, y_pred_fold)
               
            eval_results['eer'] = eer
            eval_results['fold'] = fold
            eval_results['model'] = row["model"]
            eval_results['evaluated languages'] = row["evaluated languages"]
            eval_results['fine-tuned languages'] = row["fine-tuned languages"]
        
            out_df = pd.concat([out_df, pd.DataFrame(eval_results, index=[0])], ignore_index=True)
    return out_df

def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    # Find the point where FPR and FNR are closest
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    return eer, eer_threshold

def create_heatmap(df, model_name, metric='eer', metric_label="EER", maximize=False, output_dir="results/plots", title_prefix=""):
    dataframe = df.copy()
    heatmap_data = dataframe.loc[dataframe['model'] == model_name].pivot(index='fine-tuned languages', columns='evaluated languages', values=metric)
    
    plt.figure(figsize=(10, 10))
    
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        cmap="Blues",
        cbar_kws={'label': metric_label},
        annot_kws={"size": 20},
        fmt='.2f'
    )
    plt.xlabel('Evaluated with', fontsize=20)
    plt.ylabel('Fine-Tuned With', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    if '-' in heatmap_data.index:
        idx = heatmap_data.index.get_loc('-')
        n_rows = len(heatmap_data.index)
        n_cols = len(heatmap_data.columns)
        ax.add_patch(plt.Rectangle((0, idx), n_cols, 1, fill=False, edgecolor='black', lw=4))

        ax.vlines(x=[0, 0], ymin=0, ymax=0, colors='black', linewidth=4)

    for col in heatmap_data.columns:
        if maximize:
            best_value = heatmap_data[col].max()
        else:
            best_value = heatmap_data[col].min()

        # Find all indices of the best value in the column
        best_indices = heatmap_data.index[heatmap_data[col] == best_value]

        for idx in best_indices:
            # Find row and column index to add the rectangle patch
            row_idx = heatmap_data.index.get_loc(idx)
            col_idx = heatmap_data.columns.get_loc(col)

            # Draw a red rectangle around the best cells in the column
            ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor='red', lw=3))

    # model_name = model_name.lower()
    
    model_name = rename(model_name)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(metric_label, fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.title(f'{metric_label} for {model_name}', fontsize=18)
    plt.savefig(f"{output_dir}/{title_prefix}_{model_name.replace('+', '-')}.png")
    plt.show()


def prepare_df(dataframe: pd.DataFrame):
    dataframe['y'] = dataframe['y'].apply(lambda x: literal_eval(x))
    dataframe['y_pred'] = dataframe['y_pred'].apply(lambda x: literal_eval(x))
    dataframe['y_pred_label'] = dataframe['y_pred_label'].apply(lambda x: literal_eval(x))
    dataframe['evaluated languages'] = dataframe['evaluated languages'].apply(literal_eval).apply(lambda x: "+".join(x))
    dataframe['model'] = dataframe['model'].apply(lambda x: rename(x))
    return dataframe


def rename(model_name):
    return model_name.replace("whisper_", "Whisper+").replace("frontend_", "LFCC+").replace("rawgat_st", "RawGAT ST").replace("mesonet", "LFCC+MesoNet").replace("aasist", "AASIST").replace("w2v_", "W2V+").replace("lfcc", "LFCC")



def plot_audio_representations_with_columns(
    data_bonafide,
    data_spoof,
    titles,
    figsize,
    cmap='viridis',
    colorbar_label='Amplitude',
    save_path=None,
    log_freq=False,
    vmin=None,
    vmax=None,
):
    """
    Plots positive and negative audio representations in a two-column grid layout.

    Parameters:
    - data_bonafide: List of numpy arrays representing bonafide samples.
    - data_spoof: List of numpy arrays representing spoof samples.
    - titles: List of titles for each row of plots (shared for positive and negative).
    - cmap: Colormap used for plots.
    - colorbar_label: Label for the colorbar.
    - figsize: Figure size for the plot.
    - save_path: Path to save the plot.
    - log_freq: If True, use logarithmic scale for frequency.
    - vmin, vmax: Min and max values for color scaling.
    """
    num_plots = len(data_spoof)
    assert len(data_bonafide) == num_plots, "Positive and negative data lists must have the same length."
    assert len(titles) == num_plots, "Titles list must have the same length as the data lists."

    # Set up figure and grid spec
    fig, axes = plt.subplots(num_plots, 2, figsize=figsize, gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.3, 'hspace': 0.5})
    ims = []

    # Adjust subtitle positioning
    fig.text(0.25, 0.92, 'Bonafide Samples', ha='center', va='center', fontsize=12)
    fig.text(0.75, 0.92, 'Spoof Samples', ha='center', va='center', fontsize=12)
    
    
    

    # Plot each positive and negative pair
    for idx in range(num_plots):
        # Plot positive sample
        ax_pos = axes[idx, 0]
        im_pos = ax_pos.imshow(
            data_bonafide[idx].T,
            aspect='auto',
            origin='lower',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )
        ax_pos.set_title(f"{titles[idx]}", fontsize=6)
        ax_pos.set_xlabel('Time (s)', fontsize=4)
        ax_pos.set_ylabel('', fontsize=4)
        if log_freq:
            ax_pos.set_yscale('log')
        ims.append(im_pos)

        # Plot negative sample
        ax_neg = axes[idx, 1]
        im_neg = ax_neg.imshow(
            data_spoof[idx].T,
            aspect='auto',
            origin='lower',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )
        ax_neg.set_title(f"{titles[idx]}", fontsize=6)
        ax_neg.set_xlabel('Time (s)')
        if log_freq:
            ax_neg.set_yscale('log')
        ims.append(im_neg)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(ims[0], cax=cbar_ax)
    cbar.set_label(colorbar_label)

    plt.tight_layout(rect=[0, 0, 1 , 1]) 
    # plt.show()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
