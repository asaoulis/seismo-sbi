from pathlib import Path

import numpy as np

from abc import ABC, abstractmethod

from seismo_sbi.cps_simulator.compatibility import build_objstats
from seismo_sbi.cps_simulator.CPS import update_with_Gtensor

from seismo_sbi.instaseis_simulator.simulator import Simulator
from seismo_sbi.instaseis_simulator.wrapper import GenericPointSource

# convert newtons into dynes
CPS_INPUT_COVERSION = 1.e-7
CPS_OUTPUT_COVERSION = 1.e-2

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
    
    def compute_greens_functions(self, source: GenericPointSource, velocity_model, **kwargs):
        objstats = build_objstats(self.receivers, source, self.seismogram_length)
        greens_functions = self.compute_or_load_greens_functions(objstats, velocity_model, delta=1.0, force_calc=True, verbose=False, rootdir=self.gf_storage_root, return_gf=True, **kwargs)
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
        seismograms = np.dot(self.sensitivity_kernels.T, np.array(moment_tensor_components) * CPS_INPUT_COVERSION) * CPS_OUTPUT_COVERSION
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
