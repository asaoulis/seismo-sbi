# New imports for PyTorch dataset/dataloader and globbing
import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from seismo_sbi.instaseis_simulator.dataloader import SimulationDataLoader
import numpy as np

# New: Torch dataset that returns (theta, x) where x = D + noise
class TorchSimulationDataset(Dataset):
    def __init__(
        self,
        data_loader: SimulationDataLoader,
        data_folder: str,
        parameter_name_map: dict,
        synthetic_noise_model_sampler,
        data_scaler,
        random_shift_distribution=(0, 0),
        glob_pattern: str = "*.h5",
        return_tensors: bool = True,
        torch_dtype=torch.float32,
    ):
        self.data_loader = data_loader
        receiver_names = [rec.station_name for rec in self.data_loader.receivers.iterate()]

        self.parameter_name_map = parameter_name_map or {}
        self.synthetic_noise_model_sampler = synthetic_noise_model_sampler
        self.data_scaler = data_scaler
        self.return_tensors = return_tensors
        self.torch_dtype = torch_dtype
        def random_shift_sampler():
            # gaussian array-wide random shift
            shift = round(np.random.normal(0, random_shift_distribution[0]))
            return {name: shift + int(np.random.uniform(-random_shift_distribution[1], random_shift_distribution[1])) for name in receiver_names}
        self.random_shift_sampler = random_shift_sampler

        self.paths = sorted(glob.glob(os.path.join(data_folder, glob_pattern)))
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No simulations matched {glob_pattern} under {data_folder}")
        else:
            print(f"Found {len(self.paths)} simulations matching {glob_pattern} under {data_folder}")

        # Preload all sims into memory
        thetas, Ds = [], []
        for sim_path in self.paths:
            theta, D = self._load_sim(sim_path)
            if self.data_scaler is not None and theta.size > 0:
                theta = self.data_scaler.transform(theta[np.newaxis, :]).flatten()
            thetas.append(theta)
            Ds.append(D)
        
        self.thetas = torch.as_tensor(np.stack(thetas), dtype=self.torch_dtype)
        self.Ds = torch.as_tensor(np.stack(Ds), dtype=self.torch_dtype)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        theta = self.thetas[idx]
        D = self.Ds[idx]

        # Add synthetic noise on-the-fly
        noise = self.synthetic_noise_model_sampler()
        if isinstance(noise, tuple):
            noise, _ = noise
        x = D + noise.reshape(*D.shape)
        if self.return_tensors:
            theta = torch.as_tensor(theta, dtype=self.torch_dtype)
            x = torch.as_tensor(x, dtype=self.torch_dtype)
        return theta, x

    def _load_sim(self, sim_path):
        if len(self.parameter_name_map) > 0:
            inputs_dict = self.data_loader.load_input_data(sim_path)
            fixed_keys = dict(
                (param_type, param_names)
                if param_names != ["earthquake_magnitude"]
                else ("moment_tensor", ["earthquake_magnitude"])
                for param_type, param_names in self.parameter_name_map.items()
            )
            theta = np.concatenate(
                [
                    [inputs_dict[param_type][param_name] for param_name in param_names]
                    for param_type, param_names in fixed_keys.items()
                ]
            )
        else:
            theta = np.array([])
        shift_dict = self.random_shift_sampler()
        D = self.data_loader.load_simulation_data_array_with_shifts(sim_path, shift_dict, stacked=True)
        return theta, D


# Convenience factory to create a torch DataLoader for a dataset (no splitting)
def make_torch_dataloader(
    data_loader: SimulationDataLoader,
    data_folder: str,
    parameter_name_map: dict,
    synthetic_noise_model_sampler,
    random_shift_distribution=(0, 0),
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = None,
    glob_pattern: str = "*.h5",
    return_tensors: bool = True,
    torch_dtype=torch.float32,
) -> DataLoader:
    dataset = TorchSimulationDataset(
        data_loader=data_loader,
        data_folder=data_folder,
        parameter_name_map=parameter_name_map,
        synthetic_noise_model_sampler=synthetic_noise_model_sampler,
        random_shift_distribution=random_shift_distribution,
        glob_pattern=glob_pattern,
        return_tensors=return_tensors,
        torch_dtype=torch_dtype,
    )
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

# Build both train and val DataLoaders using a single split parameter train_max_index
def make_torch_dataloaders(
    *,
    data_loader: SimulationDataLoader,
    data_folder: str,
    parameter_name_map: dict,
    synthetic_noise_model_sampler,
    random_shift_distribution=(0, 0),
    data_scaler=None,
    train_max_index: int,
    train_batch_size: int = 32,
    val_batch_size: int  = None,
    train_shuffle: bool = True,
    val_shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = None,
    glob_pattern: str = "*.h5",
    return_tensors: bool = True,
    torch_dtype= torch.float32,
):
    if val_batch_size is None:
        val_batch_size = train_batch_size

    full_dataset = TorchSimulationDataset(
        data_loader=data_loader,
        data_folder=data_folder,
        parameter_name_map=parameter_name_map,
        synthetic_noise_model_sampler=synthetic_noise_model_sampler,
        random_shift_distribution=random_shift_distribution,
        data_scaler=data_scaler,
        glob_pattern=glob_pattern,
        return_tensors=return_tensors,
        torch_dtype=torch_dtype,
    )
    n = len(full_dataset)
    end = max(0, min(train_max_index, n))

    train_subset = Subset(full_dataset, range(0, end))
    val_subset = Subset(full_dataset, range(end, n))

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_subset,
        batch_size=train_batch_size,
        shuffle=train_shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=4
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=val_batch_size,
        shuffle=val_shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=4
    )
    return train_loader, val_loader