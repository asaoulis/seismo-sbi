import numpy as np
import torch
from tqdm import tqdm
from copy import copy

from scipy.stats import gaussian_kde

class Coverage:

    def __init__(self, resolution = 10):

        self.extent = [np.array([]),np.array([])]
        self.resolution =resolution

    @torch.no_grad()
    def highest_density_level(self, pdf, alpha, bias=0.0, min_epsilon=10e-17, region=False):
        # Check if a proper bias has been specified.
        if bias >= alpha:
            raise ValueError("The bias cannot be larger or equal to the specified alpha level.")
        # Detect numpy type
        if type(pdf).__module__ != np.__name__:
            pdf = pdf.cpu().clone().numpy()
        else:
            pdf = np.array(pdf)
        
        total_pdf = pdf.sum()
        pdf /= total_pdf
        # Compute highest density level and the corresponding mask
        n = len(pdf)
        optimal_level = pdf.max().item()
        epsilon = 10e-02
        timeout_counter = 0
        while epsilon >= min_epsilon:
            area = float(0)
            while area <= (alpha + bias):
                timeout_counter +=1
                #print(area, optimal_level)
                # Compute the integral
                m = (pdf >= optimal_level).astype(np.float32)
                area = np.sum(m * pdf)
                # Compute the error and apply gradient descent
                optimal_level -= epsilon
                if timeout_counter > 300:
                    break

            optimal_level += 2 * epsilon
            epsilon /= 10

        optimal_level *= total_pdf
        if region:
            return optimal_level, m
        else:
            return optimal_level

    @torch.no_grad()
    def compute_log_posterior(self, posterior):
        # Prepare grid
        epsilon = 0.00001
        ps = [torch.linspace(self.extent [0][i], self.extent[1][i] - epsilon, self.resolution) for i in range(posterior.d)]
        gs = torch.meshgrid(*[p.view(-1) for p in ps])
        # Vectorize
        inputs = torch.cat([g.reshape(-1,1) for g in gs], dim=1)

        inputs = np.swapaxes(inputs.numpy(), 0, 1)
        log_posterior = posterior.logpdf(inputs)
        #log_posterior = torch.stack([posterior.log_prob(inputs[i, :]) for i in range(len(inputs))], axis=0)
        assert (log_posterior.shape == (self.resolution**posterior.d,))

        return log_posterior


    @torch.no_grad()
    def coverage_of_estimator(self, ground_truths, all_posterior_samples, num_samples = 1000, cl_list=[0.95]):

        posteriors = [self.create_gaussian_kde_posterior(posterior_samples)
                              for posterior_samples in all_posterior_samples[:num_samples]]
        alphas = [1 - cl for cl in cl_list]
        empirical_coverage, regions = self.coverage(posteriors, ground_truths[:num_samples], alphas)

        return empirical_coverage, regions

    @torch.no_grad()
    def coverage(self, posteriors, nominals, alphas=[0.05]):
        n = len(nominals)
        covered = [0 for _ in alphas]
        regions = []

        for posterior, nominal in tqdm(zip(posteriors, nominals), "Coverages evaluated"):
            if posterior is not None:
                data = np.hstack([nominal, posterior.dataset])
                self.extent [0], self.extent[1] = np.min(data, axis=1) - 0.1, np.max(data,axis=1) + 0.1

                pdf = np.exp(self.compute_log_posterior(posterior))
                #pdf = compute_log_posterior(posterior, observable).exp()
                nominal_pdf = np.exp(posterior.logpdf(np.swapaxes(nominal.reshape(-1,1).numpy(), 0, 1)))
                #nominal_pdf = posterior.log_prob(nominal.squeeze()).exp()
                for i, alpha in enumerate(alphas):
                    if alpha == 0.95:
                        level, region = self.highest_density_level(pdf, alpha, region=True)
                        regions.append((region, copy(self.extent)))
                    else:
                        level = self.highest_density_level(pdf, alpha)

                    if nominal_pdf >= level:
                        covered[i] += 1
            else:
                n -=1

        return [x / n for x in covered], regions

    def create_gaussian_kde_posterior(self, posterior_samples):

        try:
            posterior = gaussian_kde(np.swapaxes(posterior_samples.numpy(), 0, 1))
        except np.linalg.LinAlgError:
            return None
        return posterior