"""Simulators backed by Computer Programs in Seismology Green's functions.

:class:`CPSVariableKernelSimulator` computes the Green's functions for the velocity model it is
handed; :class:`CPSPrecomputedSimulator` draws one from a stored ensemble;
:class:`MultiModelCPSSimulator` gives disjoint receiver regions their own stored model. All of
them contract the Green's-function tensor with the six moment-tensor components.
"""

from pathlib import Path

import numpy as np

from abc import abstractmethod
import pyrocko.moment_tensor as mtm


from seismo_sbi.simulators.cps.compatibility import build_objstats
from seismo_sbi.simulators.cps.CPS import update_with_Gtensor

from seismo_sbi.simulators.base import Simulator
from seismo_sbi.simulators.gf_ensemble import GFEnsembleSimulator
from seismo_sbi.simulators.multi_region import MultiModelSimulator
from seismo_sbi.simulators.sources import GenericPointSource
from seismo_sbi.moment_tensor.conventions import create_matrix
from seismo_sbi.utils.errors import InvalidConfiguration

#: Rate (Hz) at which every CPS backend computes and returns its Green's functions.
CPS_SAMPLING_RATE_HZ = 1.0
CPS_INPUT_COVERSION = 1.e-13
CPS_OUTPUT_COVERSION = 1.e-2
def enu_to_ned(Mxx, Myy, Mzz, Mxy, Mxz, Myz):
    Mnn = Myy
    Mee = Mxx
    Mdd = Mzz
    Mne = Mxy
    Mnd = -Myz
    Med = -Mxz
    return [Mnn, Mee, Mdd, Mne, Mnd, Med]

class CPSSimulator(Simulator):
    
    def __init__(self, gf_storage_root=None, cps_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sampling_rate_hz = float(self.synthetics_processing.get('sampling_rate', CPS_SAMPLING_RATE_HZ))
        if sampling_rate_hz != CPS_SAMPLING_RATE_HZ:
            raise InvalidConfiguration(
                f"CPS synthetics are sampled at {CPS_SAMPLING_RATE_HZ} Hz; "
                f"seismic_context.processing.sampling_rate is {sampling_rate_hz}")
        if self.source_depth_offset_km != 0.0:
            raise InvalidConfiguration(
                "CPS measures source depth from the top of its layered model and applies no "
                f"source_depth_offset_km; got {self.source_depth_offset_km}")
        self.sensitivity_kernels = None
        self.num_traces = len([comp for rec in self.receivers.iterate() for comp in rec.components])
        self.gf_storage_root = gf_storage_root
        self.cps_path = cps_path
        self.synthetics_summary = lambda x: x
    
    def generic_point_source_simulation(self, source: GenericPointSource, **kwargs):
        if source.source_location.time_shift != 0:
            raise InvalidConfiguration(
                "CPS synthetics start at the origin and apply no source time shift; "
                f"got {source.source_location.time_shift} s")
        all_seismograms_map = {}
        velocity_model = kwargs.pop('velocity_model', None)
        # CPS Green's functions carry no source time function, and update_with_Gtensor would
        # raise on the unexpected keyword.
        kwargs.pop('stf_duration', None)
        self.sensitivity_kernels = self.compute_greens_functions(source, velocity_model, **kwargs)

        seismograms = self._compute_seismograms_from_kernels(source)
        seismograms = self.synthetics_summary(seismograms)
        num_traces = len([comp for rec in self.receivers.iterate() for comp in rec.components])
        seismograms = seismograms.reshape(num_traces, -1)

        trace_counter = 0
        for rec_idx, receiver in enumerate(self.receivers.iterate()):
            all_seismograms_map[receiver.station_name] = {}
            for comp_idx, component in enumerate(receiver.components):
                all_seismograms_map[receiver.station_name][component] = seismograms[trace_counter]
                trace_counter +=1
            
        return all_seismograms_map

    def compute_greens_functions(self, source: GenericPointSource, velocity_model, **kwargs):
        objstats = build_objstats(self.receivers, source, self.seismogram_length)
        greens_functions = self.compute_or_load_greens_functions(objstats, velocity_model, delta=1 / CPS_SAMPLING_RATE_HZ, force_calc=True, verbose=False, rootdir=self.gf_storage_root, return_gf=True, **kwargs)
        greens_functions = greens_functions.transpose(2, 0, 1, 3)

        used_greens_functions = []
        comp_ids = ['Z', 'E', 'N']
        for rec_idx, receiver in enumerate(self.receivers.iterate()):
            for component in receiver.components:
                idx = comp_ids.index(component)
                used_greens_functions.append(greens_functions[:, rec_idx, idx, :])

        return np.concatenate(used_greens_functions, axis=1)

    def _compute_seismograms_from_kernels(self, source: GenericPointSource):
        moment_tensor_components = source.moment_tensor.components
        mt = mtm.MomentTensor(m_up_south_east=create_matrix(moment_tensor_components))
        moment_tensor_components = mt.m6_east_north_up()
        moment_tensor_components = np.array(enu_to_ned(*moment_tensor_components))
        seismograms = moment_tensor_components @ self.sensitivity_kernels * CPS_INPUT_COVERSION * CPS_OUTPUT_COVERSION
        return seismograms
    
    @abstractmethod
    def compute_or_load_greens_functions(self, objstats, velocity_model, delta=1.0, force_calc=True, verbose=False, rootdir='.', return_gf=True, **kwargs):
        """Green's functions for this simulator's receivers; implemented by each subclass."""
        raise NotImplementedError("This method should be implemented in subclasses.")
    
class CPSVariableKernelSimulator(CPSSimulator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.gf_storage_root is not None and Path(self.gf_storage_root).exists():
            for item in Path(self.gf_storage_root).iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    for sub_item in item.iterdir():
                        sub_item.unlink()
                    item.rmdir()

    
    def compute_or_load_greens_functions(self, objstats, velocity_model, delta=1.0, force_calc=True, verbose=False, rootdir='.', return_gf=True, **kwargs):
        kwargs.pop('use_fiducial', False)
        kwargs.pop('seed', None)
        return update_with_Gtensor(
            objstats,
            velocity_model,
            delta=delta,
            force_calc=force_calc,
            verbose=verbose,
            rootdir=rootdir,
            return_gf=return_gf,
            filter_params=self.synthetics_processing['filter'],
            cps_path=self.cps_path,
            **kwargs,
        )


class CPSPrecomputedSimulator(GFEnsembleSimulator, CPSSimulator):

    def __init__(self, fiducial_model_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cps_data_path = self.gf_storage_root
        cps_data_path = Path(self.cps_data_path)
        self._fiducial_model_path = Path(fiducial_model_path)
        single_gf_path = Path(cps_data_path) / 'GF.mseed'
        if single_gf_path.exists():
            self._cps_data_folders = [cps_data_path]
        else:
            self._cps_data_folders = [f for f in Path(cps_data_path).iterdir() if f.is_dir()]
        if not self._cps_data_folders:
            raise FileNotFoundError(f"No CPS data folders found in {cps_data_path}")

    @property
    def members(self) -> list:
        return self._cps_data_folders

    @property
    def fiducial_member(self):
        return self._fiducial_model_path

    @property
    def cps_data_folders(self):
        return self._cps_data_folders

    @property
    def fiducial_model_path(self):
        return self._fiducial_model_path

    def compute_or_load_greens_functions(self, objstats, velocity_model, delta=1.0, force_calc=True, verbose=False, rootdir='.', return_gf=True, **kwargs):
        seed = kwargs.pop('seed', None)
        use_fiducial = kwargs.pop('use_fiducial', False)
        member = kwargs.pop('member', None)
        cps_data_folder = self.select_member(use_fiducial=use_fiducial, seed=seed, member=member)
        if verbose:
            print(f"Using CPS data folder: {cps_data_folder}")
        return update_with_Gtensor(
            objstats,
            velocity_model,
            delta=delta,
            force_calc=False,
            verbose=verbose,
            gf_directory=cps_data_folder,
            return_gf=return_gf,
            filter_params=self.synthetics_processing['filter'],
            cps_path=self.cps_path,
            **kwargs,
        )

    def get_all_models_array(self):
        """Every velocity model under the Green's-function storage root, as one array."""
        all_models = []
        for folder in self.cps_data_folders:
            model_path = folder / 'vel.mod'
            if model_path.exists():
                model = np.genfromtxt(model_path, skip_header=12)
                all_models.append(model)
        
        fiducial_model = np.genfromtxt(self.fiducial_model_path / 'vel.mod', skip_header=12)
        return fiducial_model, np.array(all_models)


class MultiModelCPSSimulator(MultiModelSimulator, CPSSimulator):
    """Dispatch receiver subsets to different CPS precomputed models.

    Each entry of ``models`` carries ``receivers`` and either a pre-built ``simulator`` or the
    ``cps_GFs_path`` and ``cps_GFs_fiducial_path`` to build one. Every region shares the
    components and the processing configuration, so from outside this is one CPS simulator
    over the union of the receivers.
    """

    def _build_sub_simulator(self, cfg, sub_receivers):
        try:
            fid_path = cfg["cps_GFs_fiducial_path"]
            root_path = cfg["cps_GFs_path"]
        except KeyError as exc:
            raise KeyError(
                "Each MultiModelCPSSimulator config dict must contain either "
                "'simulator' or both 'cps_GFs_path' and 'cps_GFs_fiducial_path'."
            ) from exc

        return CPSPrecomputedSimulator(
            fiducial_model_path=fid_path,
            components=self.components,
            receivers=sub_receivers,
            seismogram_duration_in_s=self.seismogram_length,
            synthetics_processing=self.synthetics_processing,
            gf_storage_root=root_path,
            cps_path=self.cps_path,
        )

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
        """Never called: each sub-simulator computes its own Green's functions."""
        raise NotImplementedError(
            "MultiModelCPSSimulator does not expose a single global "
            "compute_or_load_greens_functions; it delegates to its sub-simulators."
        )