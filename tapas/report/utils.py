import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import roc_curve
import scipy.stats as stats

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
        

def plot_interactive_roc_curve(summaries, curve_label, eff_epsilon, zoom_in, low_corner, output_path, current_suffix):
    """
    Interactive Plotly Dashboard for ROC Curves.
    Generates a HTML file with dynamic sliders and metadata.
    
    Parameters
    ----------
    summaries : list of BinaryLabelInferenceAttackSummary
        A list containing the empirical membership inference attack outcomes. Each 
        summary object must expose `.labels` (true membership binary arrays) and 
        `.scores` (the calculated continuous probability predictions or metrics).
    curve_label : str, default "attack"
        The exact attribute string name to extract from each summary instance to serve 
        as its trace label inside the right-hand legend column (e.g., "attack", "generator").
    eff_epsilon : float, default 0.5
        The initial analytical target value for Differential Privacy ($\epsilon$). Determines 
        the starting gradient position of the dotted mathematical safety upper bound.
    zoom_in : float, default 1.0
        The structural view threshold fraction between 0.0 and 1.0. If less than 1.0, configures 
        the initial plot canvas limits to a restricted sub-window slice.
    low_corner : bool, default True
        Specifies the clipping coordinate focus when zoom_in is less than 1.0. 
        If True, locks the view window to the low corner $[0, \text{zoom\_in}]$. 
        If False, locks the view window to the high corner $[1.0 - \text{zoom\_in}, 1.0]$.
    output_path : str or None, default None
        The targeted root directory where the dashboard should be saved. If a string path is 
        supplied, the function dynamically forces directory creation and writes out an HTML file. 
        If None, the file-writing sequence is bypassed.
    current_suffix : str, default ""
        The unique file naming suffix string appended to the exported file name 
        (`ROC_curve{current_suffix}.html`) to prevent cross-center file overwrites.

    Returns
    -------
    plotly.graph_objects.Figure
        The standalone fully configured interactive layout canvas containing empirical attack 
        traces, the theoretical privacy limit, crosshair coordinate trackers, an epsilon slider, 
        and scale drop-downs.
    """
    
    sample_summary = summaries[0]
    labels_array = np.array(sample_summary.labels).ravel()
    num_members = int(np.sum(labels_array == 1))
    num_non_members = int(np.sum(labels_array == 0))
    meta_text = f"Members: {num_members} | Non-Members: {num_non_members} "
    base_title_html = f"<b>ROC AUC Curve</b>"
    
    fig = go.Figure()

    FPR_GRID = np.linspace(0, 1, 300)
    alpha = 0.05  # Enforces 95% Clopper-Pearson interval coverage boundaries

    
    cached_curves = []

    
    for s in summaries:
        labels = np.array(s.labels).ravel()
        scores = np.array(s.scores).ravel()
        
        attack_name = getattr(s, curve_label).replace("SynthMIA_", "")

        if np.max(labels) > 1.0 or np.max(scores) > 1.0:
            scores = scores / 100.0

        member_scores = scores[labels == 1]
        non_member_scores = scores[labels == 0]
        n_members = len(member_scores)

        fpr, tpr, _ = roc_curve(labels, scores)
        
        mean_tpr = np.interp(FPR_GRID, fpr, tpr)
        mean_tpr[0] = 0.0
        mean_tpr[-1] = 1.0

        tpr_lower = []
        tpr_upper = []

        for f_target in FPR_GRID:
            if f_target <= 0.0:
                tpr_lower.append(0.0)
                tpr_upper.append(0.0)
                continue
            if f_target >= 1.0:
                tpr_lower.append(1.0)
                tpr_upper.append(1.0)
                continue

            thresh = np.percentile(non_member_scores, (1.0 - f_target) * 100.0)
            true_positives = np.sum(member_scores >= thresh)
            k = true_positives

            low_val = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n_members - k + 1)
            high_val = 1.0 if k == n_members else stats.beta.ppf(1 - alpha / 2, k + 1, n_members - k)

            tpr_lower.append(low_val)
            tpr_upper.append(high_val)

        tpr_lower = np.array(tpr_lower)
        tpr_upper = np.array(tpr_upper)

        cached_curves.append({
            'upper': tpr_upper,
            'lower': tpr_lower,
            'mean': mean_tpr
        })

        
        fig.add_trace(go.Scatter(
            x=FPR_GRID, y=tpr_upper,
            mode='lines', line=dict(width=0),
            showlegend=False, hoverinfo='skip'
        ))

       
        fig.add_trace(go.Scatter(
            x=FPR_GRID, y=tpr_lower,
            mode='lines', line=dict(width=0),
            fill='tonexty', 
            fillcolor='rgba(0, 128, 255, 0.08)',
            name=f"95% CP Exact Bound ({attack_name})", 
            hoverinfo='skip'
        ))

        
        fig.add_trace(go.Scatter(
            x=FPR_GRID, y=mean_tpr,
            mode='lines',
            name=f"Empirical Attack: {attack_name}",
            line=dict(width=2.5),
            hovertemplate="<b>" + attack_name + "</b><br>FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>"
        ))

    # Random Guess Baseline 
    fig.add_trace(go.Scatter(
        x=FPR_GRID, y=FPR_GRID, mode='lines', name='Random Guess Baseline',
        line=dict(color='black', width=1.5, dash='dash'), hoverinfo='skip'
    ))

    # Effective Epsilon Safety Bound
    init_upper_bound = np.minimum(1.0, np.exp(eff_epsilon) * FPR_GRID)
    fig.add_trace(go.Scatter(
        x=FPR_GRID, y=init_upper_bound, mode='lines', name='DP Safety Bound',
        line=dict(color='rgba(200, 0, 0, 0.7)', width=2, dash='dot'), visible=True
    ))

    
    init_range = [0, zoom_in] if zoom_in < 1 else [0, 1.0]
    if zoom_in < 1 and not low_corner:
        init_range = [1.0 - zoom_in, 1.0]

    zoom_options = [
        {"label": "Full Scale (100% View)", "range": [0, 1.0]},
        {"label": "Moderate (40% View)", "range": [0, 0.4]},
        {"label": "Strict Low (20% View)", "range": [0, 0.2]},
        {"label": "Ultra Low (10% View)", "range": [0, 0.1]}
    ]
    
    if zoom_in == 0.2:
        default_dropdown_idx = 2
    elif zoom_in == 0.4:
        default_dropdown_idx = 1
    elif zoom_in == 0.1:
        default_dropdown_idx = 3
    else:
        default_dropdown_idx = 0

    dropdown_buttons = []
    for opt in zoom_options:
        dropdown_buttons.append(dict(
            method="relayout", label=opt["label"],
            args=[{"xaxis.range": opt["range"], "yaxis.range": opt["range"]}]
        ))

    
    epsilon_range = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    slider_steps = []

    for eps in epsilon_range:
        updated_bound = np.minimum(1.0, np.exp(eps) * FPR_GRID)
        
        
        y_updates = []
        for curve in cached_curves:
            y_updates.append(curve['upper'])  
            y_updates.append(curve['lower'])  
            y_updates.append(curve['mean'])   
            
        y_updates.append(FPR_GRID)       # Random Baseline
        y_updates.append(updated_bound)   # DP Safety Bound
        
        step = dict(
            method="update", 
            label=f"ε = {eps}",
            args=[
                {"y": y_updates},
                {
                    "title": f"{base_title_html}<br><span style='font-size:13px; color:gray;'>{meta_text}  |  Effective Epsilon Target: {eps}</span>"
                }
            ]
        )
        slider_steps.append(step)

    
    fig.update_layout(
        title=f"{base_title_html}<br><span style='font-size:13px; color:gray;'>{meta_text}  |  Effective Epsilon Target: {eff_epsilon}</span>",
        xaxis_title="False Positive Rate (FPR - Error Threshold)",
        yaxis_title="True Positive Rate (TPR - Vulnerable Members)",
        xaxis=dict(range=init_range, gridcolor='rgba(230,230,230,0.8)', linecolor='black', linewidth=1.1, mirror=True, showspikes=True, spikethickness=1.5, spikedash="dot", spikemode="across", spikesnap="cursor"),
        yaxis=dict(range=init_range, gridcolor='rgba(230,230,230,0.8)', linecolor='black', linewidth=1.1, mirror=True, showspikes=True, spikethickness=1.5, spikedash="dot", spikemode="across", spikesnap="cursor"),
        plot_bgcolor='white', paper_bgcolor='white',
        updatemenus=[dict(
            buttons=dropdown_buttons, direction="down", pad={"r": 0, "t": 5, "b": 5, "l": 0}, active=default_dropdown_idx, showactive=True,
            x=1.02, xanchor="left", y=1.1, yanchor="top"
        )],
        sliders=[dict(
            active=epsilon_range.index(eff_epsilon) if eff_epsilon in epsilon_range else 3, 
            currentvalue={"prefix": "Select Effective Epsilon: "}, pad={"t": 60}, steps=slider_steps
        )],
        legend=dict(yanchor="top", y=1.0, xanchor="left", x=1.02),
        width=1150, height=750
    )

    if output_path is not None:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        out_path = os.path.join(output_path, f"ROC_curve{current_suffix}.html")
        fig.write_html(out_path)
        



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
                
def plot_asr_per_sensitive_attribute_plotly(data, output_path):
    """Plot AIA related figures using Plotly.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe from the AIAAttackReport class.
    output_path : str
        Path where the figures are to be saved (as HTML files).
    """
    df = data.copy(deep=True)
    df['n_qis'] = df['quasi_identifiers'].apply(lambda x: len(x))

    has_accuracy = 'accuracy' in df.columns and df['accuracy'].notna().any()
    n_qis_levels = sorted(df['n_qis'].unique())

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for qis_val in n_qis_levels:
        qis_df = df[df['n_qis'] == qis_val].copy()
        sensitive_attributes = qis_df['sensitive_attribute'].unique()

        for attr in sensitive_attributes:
            attr_df = qis_df[qis_df['sensitive_attribute'] == attr].copy()

            if not has_accuracy:
                print("Error: No accuracy df found for sensitive attribute or does not apply.")
                continue

            attr_df['attack_label'] = attr_df['attack'].apply(
                lambda x: str(x).split('(')[0]
            )

            # Keep only worst-case (max accuracy) attack per attack type
            attr_df = attr_df.reset_index(drop=True)
            idx = attr_df.groupby('attack_label')['accuracy'].idxmax()
            result = attr_df.loc[idx].reset_index(drop=True)

            has_control  = 'accuracy_control'  in result.columns and result['accuracy_control'].notna().any()
            has_baseline = 'accuracy_baseline' in result.columns and result['accuracy_baseline'].notna().any()

            attack_labels = result['attack_label'].tolist()
            x_pos = list(range(len(result)))
            width = 0.3  # fractional bar width in plotly (0–1 scale)

            traces = []

            if has_control:
                # Two bars — offset them symmetrically
                member_x    = [v - width / 2 for v in x_pos]
                nonmember_x = [v + width / 2 for v in x_pos]

                traces.append(go.Bar(
                    x=member_x,
                    y=result['accuracy'],
                    width=width,
                    name='Member Success',
                    marker_color='#56B4E9',
                    text=[f"{v * 100:.0f}%" for v in result['accuracy']],
                    textposition='outside',
                    textfont=dict(size=12),
                ))

                traces.append(go.Bar(
                    x=nonmember_x,
                    y=result['accuracy_control'],
                    width=width,
                    name='Non Member Success',
                    marker_color='#E69F00',
                    text=[f"{v * 100:.0f}%" for v in result['accuracy_control']],
                    textposition='outside',
                    textfont=dict(size=12),
                ))

            else:
                # Single bar — centered
                traces.append(go.Bar(
                    x=x_pos,
                    y=result['accuracy'],
                    width=width,
                    name='Member Success',
                    marker_color='#56B4E9',
                    text=[f"{v * 100:.0f}%" for v in result['accuracy']],
                    textposition='outside',
                    textfont=dict(size=12),
                ))

            # Naive baseline horizontal line
            if has_baseline:
                baseline_val = attr_df['accuracy_baseline'].mean()
                traces.append(go.Scatter(
                    x=[min(x_pos) - 0.5, max(x_pos) + 0.5],
                    y=[baseline_val, baseline_val],
                    mode='lines',
                    name='Naive Baseline',
                    line=dict(color='#D55E00', dash='dot', width=2),
                ))

            fig = go.Figure(data=traces)

            fig.update_layout(
                barmode='overlay',  
                title=dict(
                    text=(
                        f"KNOWN ATTRIBUTES COUNT: {qis_val} | "
                        f"SENSITIVE ATTRIBUTE: {attr.upper()}"
                    ),
                    x=0,
                    xanchor='left',
                    font=dict(size=12, family='Arial Black, Arial', color='black'),
                ),
                xaxis=dict(
                    tickmode='array',
                    tickvals=x_pos,
                    ticktext=attack_labels,
                    title=dict(text='Attack Name', font=dict(size=13)),
                    showgrid=False,
                    zeroline=False,
                ),
                yaxis=dict(
                    title=dict(text='Attack Success Rate (%)', font=dict(size=13)),
                    range=[0, 1.1],
                    tickformat='.0%',
                    showgrid=True,
                    gridcolor='#DDDDDD',
                    gridwidth=1,
                    zeroline=False,
                ),
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.05,
                    xanchor='center',
                    x=0.5,
                    bgcolor='rgba(0,0,0,0)',
                    borderwidth=0,
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                width=1100,
                height=600,
                margin=dict(l=60, r=40, t=100, b=60),
            )

            
            fig.update_xaxes(showline=True, linecolor='black', mirror=False)
            fig.update_yaxes(showline=True, linecolor='black', mirror=False)

            filename = f"attribute_disclosure_{attr}_nqis{qis_val}.html"
            fig.write_html(os.path.join(output_path, filename))
            print(f"Saved: {filename}")


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
