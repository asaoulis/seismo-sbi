

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.gridspec import GridSpecFromSubplotSpec


def plot_ensemble_results(results_dict, labels, colors=None, savefig=None):
    """
    Create a single figure with 3 vertical subplots:
      1. Bias violins
      2. Z-score violins
      3. Ordered z-score plots vs N(0,1)
    """

    # ---------------------------------------------------------
    # Setup
    # ---------------------------------------------------------
    names = list(results_dict.keys())
    params = ['gamma','delta','Mw','strike','dip','rake']
    pretty_labels = [
        labels['gamma'], labels['delta'], rf"{labels['Mw']} $\times 100$",
        labels['strike'], labels['dip'], labels['rake']
    ]
    K = len(params)

    if colors is None:
        colors = ['cornflowerblue', 'red', 'purple']
    if len(colors) < len(names):
        colors += [colors[-1]] * (len(names) - len(colors))

    log_ticks = [-40, -20, -10, -6, -3, 0, 3, 6, 10, 20, 40]
    tick_labels = [str(t) for t in log_ticks]

    # ---------------------------------------------------------
    # Preprocess data
    # ---------------------------------------------------------
    bias_data, z_data = {}, {}

    for nm in names:
        bias_dict = results_dict[nm][0].copy()
        z_dict    = results_dict[nm][1]

        bias_dict['Mw'] *= 100.
        bias_data[nm] = [bias_dict[p] for p in params]
        z_data[nm]    = [z_dict[p]    for p in params]

    xcenters = np.arange(1, K + 1)
    width = 0.8
    offsets = np.linspace(-width/3, width/3, num=len(names))

    # ==========================================================
    # Figure + main layout
    # ==========================================================
    fig, axs = plt.subplots(
        3, 1, figsize=(13, 12),
        gridspec_kw=dict(height_ratios=[1, 1, 1.8])
    )

    # ==========================================================
    # 1. BIAS VIOLINS
    # ==========================================================
    ax = axs[0]

    for nm, c, off in zip(names, colors, offsets):
        for i, arr in enumerate(bias_data[nm]):
            vp = ax.violinplot(
                arr,
                positions=[xcenters[i] + off],
                widths=width/len(names),
                showmeans=True,
                showextrema=True
            )
            for body in vp['bodies']:
                body.set_facecolor(c)
                body.set_edgecolor('black')
                body.set_alpha(0.55)
            for k in ('cbars','cmins','cmaxes','cmeans'):
                if k in vp:
                    vp[k].set_color(c)

    ax.set_xticks(xcenters)
    ax.set_xticklabels(pretty_labels)
    ax.set_ylabel("Bias")
    ax.set_title("Bias distributions by ensemble")
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_yscale('symlog', linthresh=10)
    ax.set_yticks(log_ticks)
    ax.set_yticklabels(tick_labels)

    ax.legend(
        handles=[plt.Line2D([0],[0], color=c, lw=6, label=nm)
                 for nm,c in zip(names, colors)],
        loc='upper right', frameon=False
    )

    # ==========================================================
    # 2. Z-SCORE VIOLINS
    # ==========================================================
    ax = axs[1]

    for nm, c, off in zip(names, colors, offsets):
        for i, arr in enumerate(z_data[nm]):
            vp = ax.violinplot(
                arr,
                positions=[xcenters[i] + off],
                widths=width/len(names),
                showmeans=True,
                showextrema=True
            )
            for body in vp['bodies']:
                body.set_facecolor(c)
                body.set_edgecolor('black')
                body.set_alpha(0.55)
            for k in ('cbars','cmins','cmaxes','cmeans'):
                if k in vp:
                    vp[k].set_color(c)

    ax.set_xticks(xcenters)
    ax.set_xticklabels(pretty_labels)
    ax.set_ylabel("z-score")
    ax.set_title("z-score distributions by ensemble")
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_yscale('symlog', linthresh=5)
    ax.set_yticks(log_ticks)
    ax.set_yticklabels(tick_labels)
    ax.set_ylim(-10, 10)

    # ==========================================================
    # 3. ORDERED Z-SCORE PANELS (embedded grid)
    # ==========================================================
    sub_gs = GridSpecFromSubplotSpec(
        2, 3, subplot_spec=axs[2].get_subplotspec(),
        hspace=0.35, wspace=0.25
    )
    axs[2].remove()

    sub_axes = [fig.add_subplot(sub_gs[i, j]) for i in range(2) for j in range(3)]

    for ax, param, lab in zip(sub_axes, params, pretty_labels):
        maxN = 0

        for nm, c in zip(names, colors):
            z = results_dict[nm][1][param]
            z = z[~np.isnan(z)]
            if z.size == 0:
                continue

            idx = np.arange(1, len(z) + 1)
            z_sorted = np.sort(z)
            z_plot = np.clip(z_sorted, -5.8, 5.8)
            ax.scatter(idx, z_plot, s=14, alpha=0.85, color=c)
            maxN = max(maxN, z.size)

        if maxN > 0:
            idx = np.arange(1, maxN + 1)
            p = (idx - 0.5) / maxN
            z_exp = np.clip(norm.ppf(p), -5.8, 5.8)
            ax.plot(idx, z_exp, color='grey', lw=1.5)

        ax.axhline(0, color='k', ls=':', lw=0.8)
        ax.set_title(lab)
        ax.set_xlabel("Ordered index")
        ax.set_ylabel("z-score")
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_ylim(-6, 6)

        ax.annotate('', xy=(1.0, 1.0), xycoords='axes fraction',
                    xytext=(1.0, 0.99),
                    arrowprops=dict(arrowstyle='-|>', lw=1))
        ax.annotate('', xy=(1.0, 0.0), xycoords='axes fraction',
                    xytext=(1.0, 0.01),
                    arrowprops=dict(arrowstyle='-|>', lw=1))

    handles = (
        [plt.Line2D([0],[0], color=c, marker='o',
                    linestyle='None', label=nm)
         for nm,c in zip(names, colors)] +
        [plt.Line2D([0],[0], color='k', lw=1.5,
                    label='Expected N(0,1)')]
    )
    sub_axes[2].legend(handles=handles, loc='best', frameon=False)

    fig.suptitle("Ensemble bias and z-score diagnostics", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    if savefig is not None:
        fig.savefig(f"{savefig}",
                    dpi=300, transparent=True)

    plt.show()

# -----------------------------
# Example usage
# -----------------------------
# Compute each ensemble:
# bias_A, z_A, u_A = bias_z_from_posteriors_mt6_parallel(samples_A, truth_mt6, n_jobs=20, max_samples_for_sdr=2000)
# bias_B, z_B, u_B = bias_z_from_posteriors_mt6_parallel(samples_B, truth_mt6, n_jobs=20, max_samples_for_sdr=2000)
# results = {
#   'Ensemble A': (bias_A, z_A, u_A),
#   'Ensemble B': (bias_B, z_B, u_B),
# }
# plot_ensemble_results(results, colors=['cornflowerblue', 'red', 'purple'])