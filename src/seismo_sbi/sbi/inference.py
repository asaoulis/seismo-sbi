import numpy as np

from sbi.inference import SNLE, SNPE
from sbi.inference import likelihood_estimator_based_potential, MCMCPosterior
from sbi import utils as utils
from sbi import analysis as analysis

from seismo_sbi.sbi.configuration import InvalidConfiguration

class SBI_Inference:
    """
    Class to handle the inference using the sbi library.
    
    Thin wrapper implementing their version of SNPE (SNPE-C).
    
    """

    def __init__(self, inference_mode, num_dim = None) -> None:

        self.inference_mode = inference_mode
        self.posterior = None

        self.prior = None
        self.num_dim = num_dim
        self.likelihood = None

    def build_amortised_estimator(self, train_data):
        
        if self.num_dim is None:
            self.num_dim = train_data.shape[1]//2

        self.prior = utils.BoxUniform(low=np.zeros((self.num_dim)), high=np.ones((self.num_dim)))

        if self.inference_mode == 'posterior':
            inference = SNPE(prior=self.prior)
        elif self.inference_mode == 'likelihood':
            inference = SNLE(density_estimator='maf')
        else:
            raise InvalidConfiguration(f"inference_mode was {self.inference_mode}. Must be either 'posterior' or 'likelihood'.")

        inference = inference.append_simulations(train_data[:, :self.num_dim], train_data[:, self.num_dim:])

        neural_density_estimator = inference.train(dataloader_kwargs = dict(num_workers=3))

        if self.inference_mode == 'posterior':
            posterior = inference.build_posterior(neural_density_estimator)
            self.posterior = posterior
        else:
            self.likelihood = neural_density_estimator 

        

    def sample_posterior(self, x0, num_samples, show_progress_bars = True):

        if self.inference_mode == 'likelihood':
            potential_fn, parameter_transform = likelihood_estimator_based_potential(
                self.likelihood, self.prior, x0
            )
            self.posterior = MCMCPosterior(
                potential_fn, proposal=self.prior, theta_transform=parameter_transform,
                thin=1, method="slice_np_vectorized", num_chains=10, num_workers=1,
                warmup_steps=1000
            )

        posterior_samples = self.posterior.sample((num_samples,), x=x0, show_progress_bars=show_progress_bars)
        posterior_samples = np.array(posterior_samples)

        probabilities = self.posterior.potential(posterior_samples, x=x0)
        return posterior_samples, probabilities