import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import torch
import h5py

def nc_to_tensor(location, extents: str = None):
    with h5py.File(location, 'r', swmr=True, libver='latest') as f:
        channels = []
        for var_name in f.keys():
            var = f[var_name]
            if len(var.shape) < 3:
                continue

            full_shape = var.shape

            if extents is None:
                d = np.array(var[:])
            else:
                ext = extents.split(',')
                ext = [eval(i) for i in ext]
                d = np.array(var[ext[0]:ext[1], ext[2]:ext[3], ext[4]:ext[5]])
            channels.append(d)

        if len(channels) > 1:
            d = np.stack(channels)
        else:
            d = channels[0][np.newaxis, ...]
        d = torch.from_numpy(d).float()
        return d, full_shape

def delFilesInDir(dir_path, ext=None):
    if (ext != None and ext[0] != '.'):
        ext = '.' + ext
    for f in os.listdir(dir_path):
        file_extName = os.path.splitext(f)[-1]
        if ((ext == None) or (file_extName == ext)):
            os.remove(os.path.join(dir_path, f))

def ensure_dirs(dir_path, verbose=False, empty=False):
    upperDir = os.path.dirname(dir_path)
    if not os.path.exists(dir_path):
        if not os.path.exists(upperDir):
            ensure_dirs(upperDir, verbose=verbose)
        if verbose: print(f'{dir_path} not exists, create the dir')
        os.mkdir(dir_path)
    else:
        if verbose: print(f'{dir_path} exists, no need to create the dir')
        if empty:
            delFilesInDir(dir_path)
            if verbose: print(f'{dir_path} is empty now')
