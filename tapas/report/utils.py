import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

from sklearn.metrics import roc_curve

# List of all metrics that can be used in a report.
ALL_METRICS = [
    "accuracy",
    "true_positive_rate",
    "false_positive_rate",
    "mia_advantage",
    "privacy_gain",
    "auc",
    "effective_epsilon",
]

DEFAULT_METRICS = [
    "accuracy",
    "privacy_gain",
    "auc",
]


# configurable axis ranges
axis_ranges = {
    "accuracy": (-0.2, 1.2),
    "true_positive_rate": (-0.2, 1.2),
    "false_positive_rate": (-0.2, 1.2),
    "mia_advantage": (-0.2, 1.2),
    "privacy_gain": (-0.2, 1.2),
    "auc": (-0.2, 1.2),
    "effective_epsilon": (0, 10),
}
color_pal = sns.color_palette("colorblind", 10)


def metric_comparison_plots(
    data, comparison_label, fixed_pair_label, metrics, marker_label, output_path,
    include_one_marker_plots = True
):

    """
    For a fixed pair of datasets-attacks-generators-target available in the data make a figure comparing
        performance between metrics. Options configure which dimension to compare against. Figures are saved to disk.

    Parameters
    ----------
    data: dataframe
        Input dataframe from the MIAttackReport class
    comparison_label: str
        Name of column that will be used as X axis.
    fixed_pair_columns: list[str]
        Columns in dataframe to fix (groupby) for a given figure in order to make meaningful comparisons.
        It can be any pair of columns from the report.
    metrics:  list[str]
        List of metrics to be used in the report, these can be any of the following:
        "accuracy", "true_positive_rate", "false_positive_rate", "mia_advantage", "privacy_gain", "auc".
    marker_label: str
        Column in dataframe that be used to as marker in a point plot comparison. It can be either: 'generator',
        'attack' or 'target_id'.
    output_path: str
        Path where the figure is to be saved.
    include_one_marker_plots: boolean
        Create also the plots where there is only one item in the legend.


    Returns
    -------
    None

    """
    set_style()
    metrics = list(set(data.columns).intersection(set(metrics)))
    for pair_name, pair in data.groupby(fixed_pair_label):
        
        if len(pair) <= 1 and not include_one_marker_plots:
            continue

        fig, axs = plt.subplots(len(metrics), sharex=True)

        for i, metric in enumerate(metrics):
            # sns.boxplot
            sns.pointplot(
                data=pair,
                y=metric,
                x=comparison_label,
                hue=marker_label,
                order=np.unique(pair[comparison_label]),
                ax=axs[i],
                dodge=True,
                # Disable lines between points for different x.
                join=False,
                # Plot the 95% confidence interval. In most cases, this will only
                # appear when the corresponding Report uses sample bootstrapping.
                errwidth=4,
                errorbar=('pi', 95),
            )
            axs[i].legend([], [], frameon=False)
            axs[i].set_ylabel(metric, fontsize=20)
            axs[i].set_xlabel("")
            if metric in axis_ranges:
                axs[i].set_ylim(axis_ranges[metric])

        axs[-1].set_xlabel(f"{comparison_label}s".capitalize(), fontsize=20)

        handles, labels = axs[i].get_legend_handles_labels()
        fig.subplots_adjust(right=0.82)
        fig.legend(
            handles,
            labels,
            loc="center right",
            prop={"size": 20},
            bbox_to_anchor=(1.05, 0.5),
            title=marker_label, 
            title_fontsize=18 
        )

        fig.suptitle(
            f"Comparison of {comparison_label}s and different targets"
            "\n"
            f"{fixed_pair_label[0]}: {pair_name[0]}, {fixed_pair_label[1]}: {pair_name[1]}",
            fontweight="bold",
            fontsize=24,
        )
        filename = f"{comparison_label}sComparison_Dataset{pair_name[0]}_Attack{pair_name[1]}.png"
          
        # Add condition in case pair_name[0] or pair_name[1] target ids is long replace.
        if fixed_pair_label[0] == 'target_id':
            filename = f"{comparison_label}sComparison_DatasetTarget_Ids_Attack{pair_name[1]}.png"
        if fixed_pair_label[1] == 'target_id':
            filename = f"{comparison_label}sComparison_DatasetTarget_Ids_AttackTargetIds.png"
          
        filename = os.path.join(output_path, filename)

        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        plt.savefig(filename, bbox_inches='tight')

        plt.close(fig)


def plot_roc_curve(
    data,
    names,
    title,
    output_path,
    suffix="",
    eff_epsilon=None,
    zoom_in=1,
    low_corner=True,
):
    """
    Parameters
    ----------
    data: list of pairs (labels, scores), both np.arrays of same lengths
        The true labels and the scores of each attack.
    names: list of str of the same length
        The label for each curve.
    title: str
        Title to display on the figure.
    output_path: str
        Path to the folder where the ROC curve should be saved.
    eff_epsilon: positive float, or None
        If not None, the value of the effective epsilon for this ROC curve,
        for which the TP/FP and (1-FP)/(1-TP) curves are plotted.
    zoom_in: float, default 1
        Maximum value of TP and FP shown on the plot. The default of 1 shows
        the full ROC curve, but this can be used to "zoom in" to the TPR at
        low FPR, an important quantity for privacy analysis.
    low_corner: bool, default True
        Whether to zoom in near (0,0) (True), or (1,1) (False).

    """
    set_style()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.subplots()

    # Plot the "baseline".
    decorum_color = (0.7, 0.7, 0.7)

    ax.plot([0, 1], [0, 1], "--", color=decorum_color)

    if eff_epsilon is not None:
        assert eff_epsilon > 0, "eff_epsilon must be positive."
        slope = np.exp(eff_epsilon)
        tp_inter = slope / (slope + 1)
        fp_inter = 1 / (slope + 1)
        ax.plot([0, fp_inter], [0, tp_inter], "--", color=decorum_color)
        ax.plot([fp_inter, 1], [tp_inter, 1], "--", color=decorum_color)

    for (labels, scores), name in zip(data, names):
        fpr, tpr, thresholds = roc_curve(labels, scores)
        ax.plot(fpr, tpr, label=name)

    ax.legend(loc="lower right", fontsize=20, title="Attacks", title_fontsize=18 )

    # We add a small margin to correctly display [0,1].
    margin = 0.01
    if low_corner:
        ax.set_xlim([0 - margin * zoom_in, zoom_in])
        ax.set_ylim([0, (1 + margin) * zoom_in])
    else:
        ax.set_xlim([1 - zoom_in, 1])
        ax.set_ylim([1 - zoom_in, 1 + margin * zoom_in])
    ax.set_xlabel("False Positive Rate", fontsize=20)
    ax.set_ylabel("True Positive Rate", fontsize=20)

    if title:
        fig.suptitle(title, fontweight="bold", fontsize=24)

    filename = f"ROC_curve{suffix}.png"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plt.savefig(os.path.join(output_path, filename))

    plt.close(fig)
    
def plot_asr_per_sensitive_attribute(data,output_path):
    """Plot AIA related figures

    Parameters
    ----------
    data: dataframe
        Input dataframe from the MIAttackReport class
    output_path: str
        Path where the figure is to be saved.
    
    """
    set_style()
        
    df = data.copy(deep=True)    
    
    df['n_qis'] = df['quasi_identifiers'].apply(lambda x: len(x))

    has_accuracy = 'accuracy' in df.columns and df['accuracy'].notna().any()

    n_qis_levels = sorted(df['n_qis'].unique())

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for qis_val in n_qis_levels:
        qis_df = df[df['n_qis'] == qis_val].copy()
        sensitive_attributes = qis_df['sensitive_attribute'].unique()
        n_rows = len(sensitive_attributes)

        for i, attr in enumerate(sensitive_attributes):
            fig, ax1 = plt.subplots(figsize=(14, 8))
            attr_df = qis_df[qis_df['sensitive_attribute'] == attr].copy()
            if not has_accuracy:
                print("Error: No accuracy df found for sensitive attribute or does not applies.")
            else:
                attr_df['attack_label'] = attr_df['attack'].apply(lambda x: str(x).split('(')[0])
                
                ax1.set_axisbelow(True) 
                ax1.yaxis.grid(True, linestyle='--', color='#DDDDDD', alpha=0.5, zorder=0)
                
                # Filter by worst case attack depending on number of qis  attributes
                attr_df = attr_df.reset_index()
                idx = attr_df.groupby('attack_label')['accuracy'].idxmax()
                result = attr_df.loc[idx]
                
                x = np.arange(len(result))
                width = 0.3

                if 'accuracy_control' in result:
                    # two bars → symmetric layout
                    b1 = ax1.bar(x - width/2, result['accuracy'], width,
                                label='Member Success', color='#56B4E9', zorder=3)

                    b2 = ax1.bar(x + width/2, result['accuracy_control'], width,
                                label='Non Member Success', color='#E69F00', zorder=3)

                else:
                    # one bar → centered
                    b1 = ax1.bar(x, result['accuracy'], width,
                                label='Member Success', color='#56B4E9', zorder=3)
                
                
                ax1.bar_label(b1, labels=[f'{v.get_height()*100:.0f}%' for v in b1], padding=3, fontsize=12, zorder=5)
                if 'accuracy_control' in result.columns: 
                    ax1.bar_label(b2, labels=[f'{v.get_height()*100:.0f}%' for v in b2], padding=3, fontsize=12, zorder=5)
                
                ax1.set_xticks(x)
                ax1.set_xticklabels(result['attack_label'])

                # --- Formatting ---
                ax1.set_title(f"KNOWN ATTRIBUTES COUNT: {qis_val} | SENSITIVE ATTRIBUTE: {attr.upper()}", 
                            loc='left', fontsize=12, fontweight='bold', pad=20)
                
                ax1.set_ylabel("Attack Success Rate (%)", fontweight='bold')
                ax1.set_xlabel("Attack Name", fontweight='bold')
                                
                for ax in [ax1]:
                    ax.set_ylim(0, 1.1)
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{val:.0%}'))
                    # Hide top/side spines for a cleaner look
                    ax.spines['top'].set_visible(False)
                
                ax1.spines['right'].set_visible(False)
                
                if 'accuracy_baseline' in result.columns:
                    baseline = attr_df['accuracy_baseline'].mean()
                    ax1.axhline(baseline, color='#D55E00', linestyle=':', linewidth=2, label='Naive Baseline', zorder=6)
                
                ax1.legend(
                    loc='upper center',
                    bbox_to_anchor=(0.5, 1.15),
                    ncol=3,
                    frameon=False
                )
                
                filename = f"attribute_disclosure_{attr}_nqis{qis_val}.png"         
                plt.tight_layout(rect=[0, 0, 1, 0.94])
                plt.savefig(os.path.join(output_path, filename),dpi=300, bbox_inches='tight')
                plt.close(fig)    


def set_style():

    sns.set_palette(color_pal)
    sns.set_style(
        "whitegrid",
        {
            "axes.spines.right": True,
            "axes.spines.top": True,
            "axes.edgecolor": "k",
            "xtick.color": "k",
            "xtick.rotation": 45,
            "ytick.color": "k",
            "font.family": "sans-serif",
            "font.sans-serif": ["Tahoma", "DejaVu Sans", "Arial", "Liberation Sans"],
            "text.usetex": True,
        },
    )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Tahoma", "DejaVu Sans", "Arial", "Liberation Sans"],
            "font.size": 10,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "savefig.dpi": 75,
            "figure.autolayout": False,
            "figure.figsize": (12, 10),
            "figure.titlesize": 18,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "legend.fontsize": 14,
        }
    )
