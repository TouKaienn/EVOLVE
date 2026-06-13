import os
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import torch
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.fileUtils import nc_to_tensor

class VolumeFolder(Dataset):

    def __init__(self, root_dir, transform=None, split='train', preload_data=False,
                 partial_load_ratio=1.0):
        root_path = Path(root_dir)

        if root_path.is_file():
            if root_path.suffix == '.nc':
                self.samples = [root_path]
            else:
                raise RuntimeError(f'File "{root_path}" is not a .nc file')
        else:
            splitdir = root_path / split
            if not splitdir.is_dir():
                raise RuntimeError(f'Missing directory "{splitdir}"')

            self.samples = sorted(f for f in splitdir.iterdir() if f.is_file() and f.suffix == '.nc')

            if len(self.samples) == 0:
                raise RuntimeError(f'No .nc files found in directory "{splitdir}"')

        self.transform = transform
        self.preload_data = preload_data
        self.partial_load_ratio = min(max(partial_load_ratio, 0.0), 1.0)
        self.split = split

        self.loaded_indices = None
        self.preloaded_volumes = None

        if self.preload_data:
            self._load_data()

    def _load_data(self, indices=None):
        from tqdm import tqdm

        total_samples = len(self.samples)

        if indices is None:
            num_to_load = max(1, int(total_samples * self.partial_load_ratio))
            indices = sorted(np.random.choice(total_samples, num_to_load, replace=False).tolist())

        self.loaded_indices = indices

        print(f"Loading {len(indices)}/{total_samples} volumes ({len(indices)/total_samples*100:.1f}%) into memory...")
        self.preloaded_volumes = []

        for idx in tqdm(indices, desc=f"Loading {self.split} data"):
            nc_file_path = self.samples[idx]
            volume, full_shape = nc_to_tensor(str(nc_file_path), extents=None)
            self.preloaded_volumes.append(volume)

        total_bytes = sum(v.element_size() * v.nelement() for v in self.preloaded_volumes)
        total_mb = total_bytes / (1024 ** 2)
        print(f"Loaded {len(self.preloaded_volumes)} volumes ({total_mb:.2f} MB) into CPU memory")

    def refresh_data(self, refresh_seed=None):
        import gc
        import ctypes

        if not self.preload_data:
            print("Warning: refresh_data() called but preload_data=False, skipping.")
            return

        if self.partial_load_ratio >= 1.0:
            print("Warning: refresh_data() called but partial_load_ratio=1.0, skipping.")
            return

        if self.preloaded_volumes is not None:
            num_volumes = len(self.preloaded_volumes)
            for i in range(num_volumes):
                self.preloaded_volumes[i] = None
            self.preloaded_volumes.clear()
            del self.preloaded_volumes
            self.preloaded_volumes = None
            self.loaded_indices = None

            gc.collect()

            try:
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
            except Exception:
                pass

        if refresh_seed is not None:
            np.random.seed(refresh_seed)

        self._load_data()

    def __getitem__(self, index):
        if self.preload_data and self.preloaded_volumes is not None:
            if self.partial_load_ratio < 1.0:
                volume = self.preloaded_volumes[index]
            else:
                volume = self.preloaded_volumes[index]
        else:
            nc_file_path = str(self.samples[index])
            volume, full_shape = nc_to_tensor(nc_file_path, extents=None)

        if self.transform:
            volume = self.transform(volume)

        return volume

    def get_filename(self, index):
        return self.samples[index].stem

    def __len__(self):
        if self.preload_data and self.partial_load_ratio < 1.0 and self.preloaded_volumes is not None:
            return len(self.preloaded_volumes)
        return len(self.samples)
