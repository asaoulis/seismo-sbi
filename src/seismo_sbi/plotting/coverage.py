
import matplotlib.pyplot as plt
import scienceplots
def plot_coverage(coverage_dict, colors, savefig=None, title='Inference calibration',ks = [1,2]):

    with plt.style.context('science'):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.set_title(title)
        ax.plot([0, 1], [0, 1], ls='--', color='k', label = "Calibrated", zorder=20)

        for i, (exp_name, coverage) in enumerate(coverage_dict.items()):
            color = colors[i]
            ecp_bootstrap, alpha_bootstrap = coverage
            ax.plot(alpha_bootstrap, ecp_bootstrap.mean(axis=0), label=exp_name, color=color, linewidth=2)
            
            for k in ks:
                ax.fill_between(alpha_bootstrap, ecp_bootstrap.mean(axis=0) - k * ecp_bootstrap.std(axis=0), ecp_bootstrap.mean(axis=0) + k * ecp_bootstrap.std(axis=0), alpha = 0.2, color=color)
        ax.legend(loc='upper left')
        ax.set_ylabel("Expected Coverage")
        ax.set_xlabel("Credibility Level")
        if savefig:
            plt.savefig(savefig, dpi=200, transparent=True)
        plt.show()

# gaussian smoothing
from scipy.ndimage import gaussian_filter1d
def plot_credibility_levels_histograms_dictionary(coverage_dict, colors, savefig=None):

    with plt.style.context('science'):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        num_plots = len(coverage_dict)
        ax.hlines(1, 0, 1, color='k', linestyle='--', label='Calibrated', linewidth=2)

        for i, (exp_name, coverage) in enumerate(coverage_dict.items()):
            color = colors[i]
            ecp_bootstrap, alpha_bootstrap = coverage

            rank_hist = gaussian_filter1d(np.diff(ecp_bootstrap.mean(axis=0)), 1)
            rank_hist /= np.mean(rank_hist)
            errors = np.quantile(np.diff(ecp_bootstrap, axis=0), [0.025, 0.975], axis=0)/ np.mean(rank_hist)
            credibility_levels = alpha_bootstrap[1:]

            bar_width = np.diff(credibility_levels)[0]/num_plots # Define the bar width
            if i == 0:
                offsets = np.linspace(-bar_width, bar_width, num_plots) * num_plots
            # ax.bar(credibility_levels + offset, rank_hist, width=bar_width, color=color, alpha=0.7, label=exp_name)
            ax.plot(credibility_levels, rank_hist, color=color,  label=exp_name, linewidth=2)
            ax.fill_between(credibility_levels, rank_hist + errors[0,:-1], rank_hist + errors[1, :-1], alpha = 0.2, color=color)


        # optimal 

        # Set title and labels
        ax.set_title('Theory Errors Inversion Credibility Level Distribution')
        ax.set_ylabel("Normalised Frequency")
        ax.set_xlabel("Credibility Level")
        # ax.set_yticks([])
        # Add legend
        ax.legend()

        # Adjust layout and show the plot
        fig.tight_layout()
        if savefig:
            plt.savefig(savefig, dpi=300, transparent=True)

        plt.show()

