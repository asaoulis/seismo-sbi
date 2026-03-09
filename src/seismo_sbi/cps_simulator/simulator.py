from pathlib import Path

import numpy as np

from abc import ABC, abstractmethod
import pyrocko.moment_tensor as mtm


from seismo_sbi.cps_simulator.compatibility import build_objstats
from seismo_sbi.cps_simulator.CPS import update_with_Gtensor

from seismo_sbi.instaseis_simulator.simulator import Simulator
from seismo_sbi.instaseis_simulator.wrapper import GenericPointSource

# convert newtons into dynes
CPS_INPUT_COVERSION = 1.e-13
CPS_OUTPUT_COVERSION = 1.e-2
def convert_mt_convention(mt_rr_phi_theta):
    """(mnn, mee, mdd, mne, mnd, med)"""

    return [mt_rr_phi_theta[0], mt_rr_phi_theta[1], mt_rr_phi_theta[2], mt_rr_phi_theta[3], -mt_rr_phi_theta[4], -mt_rr_phi_theta[5]]

def create_matrix(moment_tensor_sol):
    moment_tensor_matrix = np.array([[moment_tensor_sol[0], moment_tensor_sol[3], moment_tensor_sol[4]],
                                        [moment_tensor_sol[3], moment_tensor_sol[1], moment_tensor_sol[5]],
                                        [moment_tensor_sol[4], moment_tensor_sol[5], moment_tensor_sol[2]]])
                                        
    return moment_tensor_matrix

def enu_to_ned(Mxx, Myy, Mzz, Mxy, Mxz, Myz):
    Mnn = Myy
    Mee = Mxx
    Mdd = Mzz
    Mne = Mxy
    Mnd = -Myz
    Med = -Mxz
    return [Mnn, Mee, Mdd, Mne, Mnd, Med]

import torch

def compute_stft(x, fs=1.0, win_length=20, hop_length=10, n_fft=256, fmin=0.025, fmax=0.1, real_and_imag = False):
    """
    x: torch.Tensor (batch, time)
    fs: sampling frequency (Hz)
    """
    x = torch.tensor(x, dtype=torch.float32)
    window = torch.hann_window(win_length)

    # STFT
    stft = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True
    )  # shape: (batch, freq_bins, frames)
    # Frequency bins
    freqs = torch.fft.rfftfreq(n_fft, d=1/fs)  # shape (freq_bins,)
    # Mask by frequency range
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    stft_band = stft[:, band_mask, :]
    if real_and_imag:
        stft_band = torch.view_as_real(stft_band)
    else:
        # compute the magnitude and phases and stack them
        magnitudes = torch.abs(stft_band)
        phases = torch.angle(stft_band)
        stft_band = torch.stack((magnitudes, phases), dim=-1)
        # stft_band = magnitudes  # shape: (batch, freq_bins, frames, 1)
    # compute magitued
    # print(f"STFT shape: {stft_band.shape}, Frequency bins: {freqs[band_mask].shape}")
    return stft_band.numpy()


class CPSSimulator(Simulator):
    
    def __init__(self, gf_storage_root=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sensitivity_kernels = None
        self.num_traces = len([comp for rec in self.receivers.iterate() for comp in rec.components])
        self.gf_storage_root = gf_storage_root
        self.synthetics_summary = lambda x: x
        # self.synthetics_summary = self.compute_spectrograms
    
    def compute_spectrograms(self, seismograms):
        seismograms = seismograms.reshape(self.num_traces, -1)
        # Compute the spectrograms for each trace
        torch_batch_stfts = compute_stft(seismograms)#.reshape(self.num_traces, -1)
        return torch_batch_stfts
        
    def generic_point_source_simulation(self, source: GenericPointSource, **kwargs):
        
        all_seismograms_map = {}
        velocity_model = kwargs.pop('velocity_model', None)
        self.sensitivity_kernels = self.compute_greens_functions(source, velocity_model, **kwargs)

        seismograms = self._compute_seismograms_from_kernels(source)
        seismograms = self.synthetics_summary(seismograms)
        trace_length = self.sensitivity_kernels.shape[1] // self.num_traces
        seismograms = seismograms.reshape(self.num_traces, -1)

        trace_counter = 0
        for rec_idx, receiver in enumerate(self.receivers.iterate()):
            all_seismograms_map[receiver.station_name] = {}
            for comp_idx, component in enumerate(receiver.components):
                all_seismograms_map[receiver.station_name][component] = seismograms[trace_counter]
                trace_counter +=1
            
        return all_seismograms_map

    def to_enu_convention(self, gf_tensor):
        """
        Convert GF tensor from (Z, E, N) [with Z positive up]
        into (U, S, E) convention, matching moment tensors.

        Input shape: (ns, 3, ne, nt)
            index 0 = Z (Up)
            index 1 = E
            index 2 = N
        Output shape: (ns, 3, ne, nt)
            index 0 = U (Up)
            index 1 = S (South)
            index 2 = E
        """
        Z = gf_tensor[:, 0, :, :]   # up
        N = gf_tensor[:, 2, :, :]   # north
        E = gf_tensor[:, 1, :, :]   # east

        U = Z
        return np.stack([E, N, U], axis=1)
    
    def compute_greens_functions(self, source: GenericPointSource, velocity_model, **kwargs):
        objstats = build_objstats(self.receivers, source, self.seismogram_length)
        greens_functions = self.compute_or_load_greens_functions(objstats, velocity_model, delta=1.0, force_calc=True, verbose=False, rootdir=self.gf_storage_root, return_gf=True, **kwargs)
        # greens_functions = self.to_enu_convention(greens_functions)
        greens_functions = greens_functions.transpose(2, 0, 1, 3)

        trace_counter = 0
        used_greens_functions = []
        comp_ids = ['Z', 'E', 'N']
        for rec_idx, receiver in enumerate(self.receivers.iterate()):
            for component in receiver.components:
                idx = comp_ids.index(component)
                used_greens_functions.append(greens_functions[:, rec_idx, idx, :])

        return np.concatenate(used_greens_functions, axis=1)

    def _compute_seismograms_from_kernels(self, source: GenericPointSource):
        moment_tensor_components = source.moment_tensor.components
        mt = mtm.MomentTensor(m_up_south_east=create_matrix(convert_mt_convention(moment_tensor_components)))
        moment_tensor_components = mt.m6_east_north_up()
        # print('converting to NED')
        moment_tensor_components = np.array(enu_to_ned(*moment_tensor_components))
        seismograms = moment_tensor_components @ self.sensitivity_kernels * CPS_INPUT_COVERSION * CPS_OUTPUT_COVERSION
        # seismograms = np.dot(self.sensitivity_kernels.T, np.array(moment_tensor_components) * CPS_INPUT_COVERSION) * CPS_OUTPUT_COVERSION
        return seismograms
    
    @abstractmethod
    def compute_or_load_greens_functions(self, objstats, velocity_model, delta=1.0, force_calc=True, verbose=False, rootdir='.', return_gf=True, **kwargs):
        """
        Abstract method to compute or load Green's functions.
        Should be implemented in subclasses.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
    
class CPSVariableKernelSimulator(CPSSimulator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # delete everything in gf_storage_root
        if Path(self.gf_storage_root).exists():
            for item in Path(self.gf_storage_root).iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    for sub_item in item.iterdir():
                        sub_item.unlink()
                    item.rmdir()

    
    def compute_or_load_greens_functions(self, objstats, velocity_model, delta=1.0, force_calc=True, verbose=False, rootdir='.', return_gf=True, **kwargs):
        seed = kwargs.get('seed', None)
        use_fiducial = kwargs.pop('use_fiducial', False)
        return update_with_Gtensor(
            objstats, velocity_model, delta=delta, force_calc=force_calc, verbose=verbose, rootdir=rootdir, return_gf=return_gf, filter_params=self.synthetics_processing['filter'], **kwargs)

from pathlib import Path
class CPSPrecomputedSimulator(CPSSimulator):
    
    def __init__(self, fiducial_model_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cps_data_path = self.gf_storage_root
        cps_data_path = Path(self.cps_data_path)
        self.fiducial_model_path = Path(fiducial_model_path)
        # first check if GF.mseed file exists
        single_gf_path = Path(cps_data_path) / 'GF.mseed'
        if single_gf_path.exists():
            self.cps_data_folders = [cps_data_path]
        else:
            self.cps_data_folders = [f for f in Path(cps_data_path).iterdir() if f.is_dir()]
        if not self.cps_data_folders:
            raise FileNotFoundError(f"No CPS data folders found in {cps_data_path}")
        self.num_models = len(self.cps_data_folders)

    def compute_or_load_greens_functions(self, objstats, velocity_model, delta=1.0, force_calc=True, verbose=False, rootdir='.', return_gf=True, **kwargs):
        seed = kwargs.get('seed', None)
        use_fiducial = kwargs.pop('use_fiducial', False)
        if seed is not None:
            np.random.seed(seed)
        
        if use_fiducial:
            cps_data_folder = self.fiducial_model_path
        else:
            # randomly  choose a CPS data folder
            cps_data_folder = np.random.choice(self.cps_data_folders)
        if verbose:
            print(f"Using CPS data folder: {cps_data_folder}")
        return update_with_Gtensor(
            objstats, velocity_model, delta=delta, force_calc=False, verbose=verbose, gf_directory=cps_data_folder, return_gf=return_gf, filter_params=self.synthetics_processing['filter'], **kwargs)

    def get_all_models_array(self):
        """
        Returns an array of all models in the CPS data folders.
        """
        all_models = []
        for folder in self.cps_data_folders:
            model_path = folder / 'vel.mod'
            if model_path.exists():
                model = np.genfromtxt(model_path, skip_header=12)  # Skip header line
                all_models.append(model)
        
        fiducial_model = np.genfromtxt(self.fiducial_model_path / 'vel.mod', skip_header=12)
        return fiducial_model, np.array(all_models)


class MultiModelCPSSimulator(CPSSimulator):
    """Dispatch different receiver subsets to different CPS precomputed models.

    This wraps multiple :class:`CPSPrecomputedSimulator`-like simulators, each
    responsible for a *subset* of the global receiver geometry. All
    simulators share the same components and processing configuration, but
    may use different CPS Green's function roots / fiducial models.

    Parameters
    ----------
    models : list
        A list of dictionaries specifying sub-model configuration. Each dict
        must contain at least:

        - ``receivers``: a :class:`Receivers` object describing the subset of
          receivers handled by this model.
        - either
            * ``simulator``: an already-constructed CPSPrecomputedSimulator
              (in which case ``cps_GFs_path`` / ``cps_GFs_fiducial_path`` are
              ignored), or
            * ``cps_GFs_path`` and ``cps_GFs_fiducial_path``: used to
              construct a new :class:`CPSPrecomputedSimulator`.

        Any additional keys are ignored.

    Notes
    -----
    * ``components`` are assumed identical across all receiver groups.
    * From the outside this behaves like a single :class:`CPSSimulator` over
      the *union* of all receivers.
    """

    def __init__(self, models, *args, **kwargs):
        # CPSSimulator takes: components, receivers, seismogram_duration_in_s,
        # synthetics_processing, gf_storage_root (optional)
        super().__init__(*args, **kwargs)

        if not isinstance(models, (list, tuple)) or len(models) == 0:
            raise ValueError(
                "'models' must be a non-empty list/tuple of configuration dicts."
            )

        self.sub_sims = []
        self.sub_receivers = []

        for cfg in models:
            if not isinstance(cfg, dict):
                raise TypeError(
                    "Each element of 'models' must be a dict with at least "
                    "'receivers' and either 'simulator' or CPS GF paths."
                )

            sub_receivers = cfg.get("receivers")
            if sub_receivers is None:
                raise KeyError(
                    "MultiModelCPSSimulator config dict missing required key 'receivers'."
                )

            existing_sim = cfg.get("simulator")
            if existing_sim is not None:
                if not isinstance(existing_sim, CPSPrecomputedSimulator):
                    raise TypeError(
                        "'simulator' must be a CPSPrecomputedSimulator instance if provided."
                    )
                sim = existing_sim
            else:
                try:
                    fid_path = cfg["cps_GFs_fiducial_path"]
                    root_path = cfg["cps_GFs_path"]
                except KeyError as exc:
                    raise KeyError(
                        "Each MultiModelCPSSimulator config dict must contain either "
                        "'simulator' or both 'cps_GFs_path' and 'cps_GFs_fiducial_path'."
                    ) from exc

                sim = CPSPrecomputedSimulator(
                    fiducial_model_path=fid_path,
                    components=self.components,
                    receivers=sub_receivers,
                    seismogram_duration_in_s=self.seismogram_length,
                    synthetics_processing=self.synthetics_processing,
                    gf_storage_root=root_path,
                )

            self.sub_sims.append(sim)
            self.sub_receivers.append(sub_receivers)

        # num_traces for the *global* simulator still corresponds to the
        # full self.receivers object (inherited from CPSSimulator.__init__).
        # CPSSimulator already set self.num_traces appropriately, so we
        # simply retain it here.
        self.num_models = self.sub_sims[0].num_models

    def compute_or_load_greens_functions(
        self,
        objstats,
        velocity_model,
        delta=1.0,
        force_calc=True,
        verbose=False,
        rootdir='.',
        return_gf=True,
        **kwargs,
    ):
        """Not used directly.

        Each sub-simulator computes its own Green's functions; this method is
        only present to satisfy the abstract interface.
        """
        raise NotImplementedError(
            "MultiModelCPSSimulator does not expose a single global "
            "compute_or_load_greens_functions; it delegates to its sub-simulators."
        )

    def generic_point_source_simulation(self, source: GenericPointSource, **kwargs):
        """Run sub-simulators on their receiver subsets and merge outputs.

        The returned dictionary has the same structure as for a single
        :class:`CPSSimulator` with ``self.receivers``.
        """
        per_model_results = []
        for sim, sub_rec in zip(self.sub_sims, self.sub_receivers):
            sub_map = sim.generic_point_source_simulation(source, **kwargs)
            per_model_results.append((sub_rec, sub_map))

        all_seismograms_map = {}
        # iterate over concrete receiver list to allow membership tests by station_name
        for rec in self.receivers.receivers:
            station_name = rec.station_name
            all_seismograms_map[station_name] = {}

            owning_map = None
            for sub_rec, sub_map in per_model_results:
                if any(r.station_name == station_name for r in sub_rec.receivers):
                    owning_map = sub_map
                    break

            if owning_map is None:
                raise KeyError(
                    f"Station '{station_name}' in global receivers is not covered "
                    "by any sub-model configuration."
                )

            for comp in rec.components:
                try:
                    all_seismograms_map[station_name][comp] = owning_map[station_name][comp]
                except KeyError as exc:
                    raise KeyError(
                        f"Component '{comp}' for station '{station_name}' missing "
                        "from sub-model output. Ensure components are consistent."
                    ) from exc

        return all_seismograms_map