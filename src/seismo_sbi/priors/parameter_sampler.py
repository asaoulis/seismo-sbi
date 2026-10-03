"""Draw source and nuisance parameters from the configured prior.

:class:`ParameterSampler` pairs each parameter of a :class:`~seismo_sbi.sbi.types.parameters.ModelParameters`
with the sampler its ``sampling_method`` entry names (``SAMPLERS``) or with a catalogue prior built at
parse time, and draws whole simulation inputs or one set of nuisance inputs. A sampler takes its
bounds (the fiducial, for ``constant``) and a sample count and yields one draw per sample.
"""

import logging
from itertools import chain

import numpy as np

from seismo_sbi.simulators.cps.CPS import perturb_model
from seismo_sbi.simulators.cps.compatibility import load_velocity_model
from seismo_sbi.simulators.cps.smooth_perturbations import perturb_cps_model

logger = logging.getLogger(__name__)


def constant_sampler(value, num_samples):
    for _ in range(num_samples):
        yield value

def uniform_sampler(bounds, num_samples):
    for _ in range(num_samples):
        yield np.random.uniform(bounds[0], bounds[1])

def flatten_sample(values):
    """One draw per parameter -> the flat sample vector ``vector_to_simulation_inputs`` reads.

    A scalar draw takes one slot, a 1-D draw takes one slot per element, and anything else
    (a velocity model) takes one slot holding the object itself.
    """
    flat = []
    for v in values:
        if np.isscalar(v):
            flat.append(v)
        elif isinstance(v, (list, tuple)):
            flat.extend(v)
        elif isinstance(v, np.ndarray):
            if v.ndim == 1:
                flat.extend(v)
            else:
                flat.append(v)
        else:
            flat.append(v)
    return np.array(flat, dtype=object)

class VelocityModelSampler:
    """Perturbed copies of a layered velocity model shaped ``(6, n_layers)``. ``kappa`` is the
    standard deviation of the perturbation in percent: per layer by default, or with ``"smooth"``
    a fractional standard deviation of kappa/100 in the compressional and shear speeds, correlated
    with depth over ``smooth_correlation_length_km``."""

    #: Depth over which the smooth perturbations are correlated, in km.
    smooth_correlation_length_km = 5.0

    perturbation_methods = {
        "default": perturb_model,
        "smooth": perturb_cps_model
    }
    
    def __init__(self, velocity_model, kappa, num_samples, *args):
        self.velocity_model = velocity_model
        self.kappa = kappa
        self.num_samples = num_samples
        self.kwargs = {}
        if len(args) > 0 and args[0] == "smooth":
            self.perturbation_function = self.perturbation_methods["smooth"]
            self.kwargs = {'corr_length_km': self.smooth_correlation_length_km,
                           'std_vp': kappa/100,
                           'std_vs': kappa/100,}
            logger.info(f"Using smooth perturbations with kappa={kappa}")
        else:
            self.perturbation_function = self.perturbation_methods["default"]
            self.kwargs['kappa'] = kappa

    def __iter__(self):
        for _ in range(self.num_samples):
            yield self.perturbation_function(self.velocity_model, **self.kwargs)

def velocity_model_sampler(velocity_model_args, num_samples):
    """A generator of perturbed velocity models.

    ``velocity_model_args`` is ``(velocity_model, kappa[, "smooth"])``, the model read by
    :meth:`ParameterSampler.from_configuration`.
    """
    velocity_model, kappa, *options = velocity_model_args
    sampler = iter(VelocityModelSampler(velocity_model, kappa, num_samples, *options))
    return sampler



SAMPLERS = {"uniform": uniform_sampler,
            "constant": constant_sampler,
            "velocity model": velocity_model_sampler}


class ParameterSampler:
    """The samplers of every inferred and nuisance parameter, in the order the parameters declare
    them, with the bounds (or fiducial) each is bound to."""

    def __init__(self, parameters, samplers, sampler_args):
        self.parameters = parameters
        self.samplers = samplers
        self.sampler_args = sampler_args

    @classmethod
    def from_configuration(cls, parameters, sampling_method):
        """The sampler ``sampling_method`` names for each parameter of ``parameters``: a key of
        ``SAMPLERS`` or a catalogue-prior closure. A ``velocity model`` parameter's bounds are
        ``(path, kappa[, "smooth"])``; the model at ``path`` is read here."""
        samplers = {key: sampling_method[key] if callable(sampling_method[key]) else SAMPLERS[sampling_method[key]]
                    for key in chain(parameters.names.keys(), parameters.nuisance.keys())}
        # A constant parameter fills len(fiducial) slots of the sample vector, not its two bounds.
        sampler_args = {key: (parameters.get_parameter_values(key) if sampler is constant_sampler
                              else parameters.bounds[key])
                        for key, sampler in samplers.items()}
        for key, sampler in samplers.items():
            if sampler is velocity_model_sampler:
                velocity_model_path, kappa, *options = sampler_args[key]
                sampler_args[key] = (load_velocity_model(velocity_model_path), kappa, *options)
        return cls(parameters, samplers, sampler_args)

    def draw_simulation_inputs(self, num_samples):
        """``num_samples`` simulation-input maps, each holding every inferred and nuisance parameter."""
        draws = zip(*[sampler(self.sampler_args[key], num_samples) for key, sampler in self.samplers.items()])
        return [self.parameters.vector_to_simulation_inputs(flatten_sample(draw)) for draw in draws]

    def draw_nuisance_inputs(self):
        """One draw of every nuisance parameter, as a map from nuisance name to value."""
        draws = [next(self.samplers[key](self.sampler_args[key], 1)) for key in self.parameters.nuisance.keys()]
        return self.parameters.vector_to_nuisance_inputs(flatten_sample(draws))
