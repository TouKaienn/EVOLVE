import os
import argparse
import time
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from dataloader.VolumeFolder import VolumeFolder
from models.scale_hyperprior_3d_context import ScaleHyperprior3DContext, MODEL_CONFIGS
from utils.fileUtils import ensure_dirs

def compute_psnr(original, reconstructed, data_range=1.0):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10((data_range ** 2) / mse)
    return psnr

def scale_intensity(volume, minv=0.0, maxv=1.0):
    v_min = volume.min()
    v_max = volume.max()
    if v_max - v_min < 1e-8:
        return volume.clone().fill_(minv)
    scaled = (volume - v_min) / (v_max - v_min)
    scaled = scaled * (maxv - minv) + minv
    return scaled

def _compute_start_positions(full: int, patch: int, stride: int):
    if patch <= 0 or stride <= 0:
        raise ValueError(f"patch and stride must be > 0, got patch={patch}, stride={stride}")
    if full <= 0:
        raise ValueError(f"full must be > 0, got full={full}")

    if patch > full:
        raise ValueError(f"patch size ({patch}) must be <= full size ({full})")
    if patch == full:
        return [0]

    last = full - patch
    positions = list(range(0, last + 1, stride))
    if positions[-1] != last:
        positions.append(last)
    return positions

def test_compression(model, test_loader, test_dataset, device, args):
    import json
    import struct
    from pathlib import Path

    print("\n" + "="*60)
    print("Testing Compression (Context Model - Patch-based with Overlap)")
    print("="*60)

    model.eval()

    bitstream_dir = Path(args.output_dir) / "bitstream"
    bitstream_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving bitstreams to: {bitstream_dir}")
    print(f"Patch size: {args.patch_size}")
    print(f"Stride: {args.stride}")
    print(f"Model slices: {model.num_slices}")
    print(f"Model groups: {model.groups}")

    if args.factormode:
        print(f"Quality mode: Continuous (factor={args.factor})")
        quality_level = None
        inputscale = args.factor
    else:
        print(f"Quality mode: Discrete (s={args.s})")
        quality_level = args.s
        inputscale = 0

    total_compressed_bytes = 0
    num_volumes = 0
    all_volume_info = []
    all_psnrs = []
    filename_mapping = {}
    all_compression_times = []

    with torch.no_grad():
        for batch_idx, volumes in enumerate(tqdm(test_loader, desc="Compressing volumes")):
            # NOTE: volumes stays on CPU; each crop is streamed to the GPU individually
            # below, so the GPU footprint stays bounded by the crop size rather than the
            # full volume size.
            B, C, D, H, W = volumes.shape

            for b in range(B):
                volume = volumes[b:b+1]
                volume_idx = batch_idx * test_loader.batch_size + b

                original_filename = test_dataset.get_filename(volume_idx)

                volume_start_time = time.time()

                vol_min = volume.min().item()
                vol_max = volume.max().item()
                # Normalize each crop on-GPU using the global vol_min/vol_max scalars
                # instead of materializing a full normalized copy of the volume. This keeps
                # the GPU footprint bounded by the crop size, independent of the volume size.

                patches_list = []
                positions = []

                x_starts = _compute_start_positions(D, args.patch_size[0], args.stride[0])
                y_starts = _compute_start_positions(H, args.patch_size[1], args.stride[1])
                z_starts = _compute_start_positions(W, args.patch_size[2], args.stride[2])

                for x in x_starts:
                    for y in y_starts:
                        for z in z_starts:
                            patch = volume[:, :,
                                          x:x+args.patch_size[0],
                                          y:y+args.patch_size[1],
                                          z:z+args.patch_size[2]]
                            patches_list.append(patch)
                            positions.append((x, y, z))

                output_file = bitstream_dir / f"{original_filename}.bin"
                filename_mapping[original_filename] = {
                    'original_file': f"{original_filename}.nc",
                    'compressed_file': f"{original_filename}.bin"
                }

                patch_data_list = []
                total_volume_size = 0

                for patch_idx, (patch, pos) in enumerate(zip(patches_list, positions)):
                    patch_norm = (patch.to(device) - vol_min) / (vol_max - vol_min)

                    if args.factormode:
                        compressed = model.compress(patch_norm, s=0, inputscale=inputscale)
                    else:
                        compressed = model.compress(patch_norm, s=quality_level, inputscale=0)

                    y_strings_all_slices, z_strings = compressed["strings"]
                    z_shape = compressed["shape"]
                    y_shape = compressed["y_shape"]

                    y_serialized = b''
                    y_slice_info = []
                    for slice_idx, (anchor_str, non_anchor_str) in enumerate(y_strings_all_slices):
                        anchor_bytes = anchor_str[0] if isinstance(anchor_str, list) else anchor_str
                        non_anchor_bytes = non_anchor_str[0] if isinstance(non_anchor_str, list) else non_anchor_str
                        y_slice_info.append((len(anchor_bytes), len(non_anchor_bytes)))
                        y_serialized += anchor_bytes + non_anchor_bytes

                    z_bytes = z_strings[0] if isinstance(z_strings, list) else z_strings
                    z_size = len(z_bytes)
                    y_size = len(y_serialized)

                    _, C_patch, D_patch, H_patch, W_patch = patch.shape

                    patch_data_list.append({
                        'position': pos,
                        'patch_shape': (C_patch, D_patch, H_patch, W_patch),
                        'normalization': (vol_min, vol_max),
                        'y_serialized': y_serialized,
                        'z_bytes': z_bytes,
                        'y_slice_info': y_slice_info,
                        'y_shape': tuple(y_shape) if isinstance(y_shape, torch.Size) else y_shape,
                        'z_shape': tuple(z_shape) if isinstance(z_shape, torch.Size) else z_shape,
                        'y_size': y_size,
                        'z_size': z_size,
                        'num_slices': model.num_slices,
                    })

                    total_volume_size += y_size + z_size

                with open(output_file, 'wb') as f:
                    header = struct.pack(
                        'B4HIB',
                        2,
                        C, D, H, W,
                        len(patch_data_list),
                        model.num_slices
                    )
                    f.write(header)

                    for patch_data in patch_data_list:
                        C_p, D_p, H_p, W_p = patch_data['patch_shape']
                        v_min, v_max = patch_data['normalization']
                        patch_header = struct.pack(
                            '3H4H2fII',
                            *patch_data['position'],
                            C_p, D_p, H_p, W_p,
                            v_min, v_max,
                            patch_data['z_size'],
                            patch_data['y_size']
                        )
                        f.write(patch_header)

                        f.write(struct.pack('3I', *patch_data['y_shape']))
                        f.write(struct.pack('3I', *patch_data['z_shape']))

                        for anchor_len, non_anchor_len in patch_data['y_slice_info']:
                            f.write(struct.pack('2I', anchor_len, non_anchor_len))

                        f.write(patch_data['z_bytes'])
                        f.write(patch_data['y_serialized'])

                file_size = output_file.stat().st_size
                total_compressed_bytes += file_size

                volume_compression_time = time.time() - volume_start_time
                all_compression_times.append(volume_compression_time)

                recon_volume_norm = np.zeros((C, D, H, W), dtype=np.float32)
                weight_volume = np.zeros((C, D, H, W), dtype=np.float32)

                for patch_data in patch_data_list:
                    y_strings_reconstruct = []
                    offset = 0
                    y_ser = patch_data['y_serialized']
                    for anchor_len, non_anchor_len in patch_data['y_slice_info']:
                        anchor_bytes = y_ser[offset:offset + anchor_len]
                        offset += anchor_len
                        non_anchor_bytes = y_ser[offset:offset + non_anchor_len]
                        offset += non_anchor_len
                        y_strings_reconstruct.append([[anchor_bytes], [non_anchor_bytes]])

                    z_shape = patch_data['z_shape']
                    y_shape = patch_data['y_shape']

                    strings = [y_strings_reconstruct, [patch_data['z_bytes']]]
                    if args.factormode:
                        x_hat = model.decompress(strings, z_shape, y_shape, s=0, inputscale=inputscale)
                    else:
                        x_hat = model.decompress(strings, z_shape, y_shape, s=quality_level, inputscale=0)

                    x_hat = torch.clamp(x_hat, 0.0, 1.0)

                    x_pos, y_pos, z_pos = patch_data['position']
                    px, py, pz = args.patch_size
                    x_hat_np = x_hat.squeeze().cpu().numpy()
                    recon_volume_norm[:, x_pos:x_pos+px, y_pos:y_pos+py, z_pos:z_pos+pz] += x_hat_np
                    weight_volume[:, x_pos:x_pos+px, y_pos:y_pos+py, z_pos:z_pos+pz] += 1.0

                mask = weight_volume > 0
                recon_volume_norm[mask] /= weight_volume[mask]

                volume_norm_np = ((volume.squeeze().cpu().numpy() - vol_min) / (vol_max - vol_min)).astype(np.float32)
                psnr = compute_psnr(volume_norm_np, recon_volume_norm, data_range=1.0)
                all_psnrs.append(psnr)

                all_volume_info.append({
                    'volume_idx': volume_idx,
                    'original_filename': original_filename,
                    'volume_shape': (C, D, H, W),
                    'num_patches': len(patch_data_list),
                    'patch_positions': [p['position'] for p in patch_data_list],
                    'file_size_bytes': file_size,
                    'psnr': float(psnr),
                    'compression_time_sec': float(volume_compression_time)
                })

                num_volumes += 1

    print(f"\nSaved {num_volumes} volumes (each with multiple patches) as binary files...")

    total_patches = sum(info['num_patches'] for info in all_volume_info)
    total_voxels = sum(info['volume_shape'][1] * info['volume_shape'][2] * info['volume_shape'][3] for info in all_volume_info)
    actual_bpp = (total_compressed_bytes * 8) / total_voxels if total_voxels > 0 else 0
    avg_psnr = np.mean(all_psnrs) if all_psnrs else 0
    avg_compression_time = np.mean(all_compression_times) if all_compression_times else 0
    total_compression_time = sum(all_compression_times)

    metadata = {
        'num_volumes': num_volumes,
        'total_patches': total_patches,
        'patch_size': args.patch_size,
        'stride': args.stride,
        'volume_info': all_volume_info,
        'filename_mapping': filename_mapping,
        'total_voxels': total_voxels,
        'original_size_bytes': total_voxels * 4,
        'compressed_size_bytes': total_compressed_bytes,
        'compression_ratio': float((total_voxels * 4) / total_compressed_bytes) if total_compressed_bytes > 0 else 0,
        'actual_bpp': float(actual_bpp),
        'avg_psnr_db': float(avg_psnr),
        'avg_compression_time_sec': float(avg_compression_time),
        'total_compression_time_sec': float(total_compression_time),
        'entropy_coding': 'compressai_context',
        'model_config': {
            'model_size': "small",
            'M': model.M,
            'N_hyper': model.N_hyper,
            'num_slices': model.num_slices,
            'groups': model.groups,
        }
    }

    if args.factormode:
        metadata['vbr'] = {
            'mode': 'continuous',
            'factor': args.factor,
            'quality_level': None,
            'gain_value': args.factor
        }
    else:
        metadata['vbr'] = {
            'mode': 'discrete',
            'factor': None,
            'quality_level': args.s,
            'gain_value': float(model.Gain[args.s].item()),
            'lambda': model.lmbda[args.s]
        }

    metadata_file = bitstream_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*60)
    print("Compression Results (Context Model - Patch-based):")
    print("="*60)
    print(f"Total volumes: {num_volumes}")
    print(f"Total patches: {total_patches}")
    print(f"Patch size: {args.patch_size}")
    print(f"Stride: {args.stride}")
    print(f"Original size: {total_voxels * 4 / 1e6:.2f} MB (float32)")
    print(f"Compressed size: {total_compressed_bytes / 1e6:.2f} MB")
    print(f"Compression ratio: {metadata['compression_ratio']:.2f}×")
    print(f"Actual BPP: {actual_bpp:.4f}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average compression time: {avg_compression_time:.3f} sec/volume")
    print(f"Total compression time: {total_compression_time:.3f} sec")
    print(f"\nCompressed data saved to: {bitstream_dir}")
    print(f"  - {num_volumes} compressed files (<filename>.bin)")
    print(f"  - 1 metadata file (metadata.json)")
    print("="*60)

    return {
        "actual_bpp": actual_bpp,
        "avg_psnr": avg_psnr,
        "compressed_size_mb": total_compressed_bytes / 1e6,
        "compression_ratio": metadata['compression_ratio'],
        "avg_compression_time_sec": avg_compression_time,
        "total_compression_time_sec": total_compression_time
    }

def test_decompression(model, bitstream_dir, args):
    import json
    import struct
    from pathlib import Path

    bitstream_dir = Path(bitstream_dir)

    metadata_file = bitstream_dir / "metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    volume_info = metadata['volume_info']
    patch_size = tuple(metadata['patch_size'])
    num_volumes = metadata['num_volumes']
    total_patches = metadata['total_patches']
    num_slices = metadata['model_config']['num_slices']

    vbr_info = metadata.get('vbr', {'mode': 'discrete', 'quality_level': 7, 'factor': None})
    if vbr_info['mode'] == 'continuous':
        quality_level = 0
        inputscale = vbr_info['factor']
        factormode = True
    else:
        quality_level = vbr_info.get('quality_level', 7)
        inputscale = 0
        factormode = False

    print("\n" + "="*60)
    print("Testing Decompression (Context Model - Patch-based with Overlap Averaging)")
    print("="*60)
    print(f"Bitstream directory: {bitstream_dir}")
    print(f"Number of volumes: {num_volumes}")
    print(f"Total patches: {total_patches}")
    print(f"Patch size: {patch_size}")
    print(f"Number of slices: {num_slices}")
    print(f"Compressed size: {metadata['compressed_size_bytes'] / 1e6:.2f} MB")
    print(f"Compression ratio: {metadata['compression_ratio']:.2f}×")
    if factormode:
        print(f"Quality mode: Continuous (factor={inputscale})")
    else:
        print(f"Quality mode: Discrete (s={quality_level})")
    print("="*60 + "\n")

    model.eval()
    ensure_dirs(args.output_dir)

    device = next(model.parameters()).device
    all_decompression_times = []

    with torch.no_grad():
        for vol_info in tqdm(volume_info, desc="Decompressing volumes"):
            volume_idx = vol_info['volume_idx']
            original_filename = vol_info.get('original_filename', f"volume_{volume_idx:04d}")

            volume_start_time = time.time()

            compressed_file = bitstream_dir / f"{original_filename}.bin"

            if not compressed_file.exists():
                print(f"Warning: Missing file {compressed_file}, skipping...")
                continue

            with open(compressed_file, 'rb') as f:
                volume_header = f.read(struct.calcsize('B4HIB'))
                version, C, D, H, W, num_patches_file, num_slices_file = struct.unpack('B4HIB', volume_header)

                recon_patches = []
                patch_positions = []

                for patch_idx in range(num_patches_file):
                    patch_header_size = struct.calcsize('3H4H2fII')
                    patch_header = f.read(patch_header_size)
                    x, y, z, C_p, D_p, H_p, W_p, v_min, v_max, z_len, y_len = struct.unpack('3H4H2fII', patch_header)

                    y_shape = struct.unpack('3I', f.read(12))
                    z_shape = struct.unpack('3I', f.read(12))

                    y_slice_info = []
                    for _ in range(num_slices_file):
                        anchor_len, non_anchor_len = struct.unpack('2I', f.read(8))
                        y_slice_info.append((anchor_len, non_anchor_len))

                    z_bitstream = f.read(z_len)
                    y_serialized = f.read(y_len)

                    y_strings_reconstruct = []
                    offset = 0
                    for anchor_len, non_anchor_len in y_slice_info:
                        anchor_bytes = y_serialized[offset:offset + anchor_len]
                        offset += anchor_len
                        non_anchor_bytes = y_serialized[offset:offset + non_anchor_len]
                        offset += non_anchor_len
                        y_strings_reconstruct.append([[anchor_bytes], [non_anchor_bytes]])

                    strings = [y_strings_reconstruct, [z_bitstream]]
                    if factormode:
                        x_hat = model.decompress(strings, z_shape, y_shape, s=0, inputscale=inputscale)
                    else:
                        x_hat = model.decompress(strings, z_shape, y_shape, s=quality_level, inputscale=0)

                    x_hat = torch.clamp(x_hat, 0.0, 1.0)

                    x_hat = x_hat * (v_max - v_min) + v_min

                    x_hat_np = x_hat.squeeze().cpu().numpy()
                    recon_patches.append(x_hat_np)
                    patch_positions.append((x, y, z))

            recon_volume = np.zeros((C, D, H, W), dtype=np.float32)
            weight_volume = np.zeros((C, D, H, W), dtype=np.float32)

            for patch, (x, y, z) in zip(recon_patches, patch_positions):
                px, py, pz = patch_size

                recon_volume[:, x:x+px, y:y+py, z:z+pz] += patch
                weight_volume[:, x:x+px, y:y+py, z:z+pz] += 1.0

            mask = weight_volume > 0
            recon_volume[mask] /= weight_volume[mask]

            if args.save_output:
                save_path = os.path.join(args.output_dir, f"{original_filename}.raw")
                recon_volume.flatten().astype("<f").tofile(save_path)

            volume_decompression_time = time.time() - volume_start_time
            all_decompression_times.append(volume_decompression_time)

    avg_decompression_time = np.mean(all_decompression_times) if all_decompression_times else 0
    total_decompression_time = sum(all_decompression_times)

    print("\n" + "="*60)
    print("Decompression Complete!")
    print("="*60)
    print(f"Decompressed {num_volumes} volumes")
    print(f"Compressed size: {metadata['compressed_size_bytes'] / 1e6:.2f} MB")
    print(f"Compression ratio: {metadata['compression_ratio']:.2f}×")
    print(f"Average decompression time: {avg_decompression_time:.3f} sec/volume")
    print(f"Total decompression time: {total_decompression_time:.3f} sec")
    if args.save_output:
        print(f"Decompressed volumes saved to: {args.output_dir}/<filename>.raw")
    print("="*60)

    return {
        "metadata": metadata,
        "avg_decompression_time_sec": avg_decompression_time,
        "total_decompression_time_sec": total_decompression_time
    }

def main(args):
    if args.stride is None:
        args.stride = args.patch_size

    args.factormode = (args.quality_mode == "factor")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ScaleHyperprior3DContext(model_size="small", num_slices=5).to(device)

    config = MODEL_CONFIGS["small"]
    print(f"Model size: small")
    print(f"  Channels: {config['channels']}")
    print(f"  Depths: {config['depths']}")
    print(f"  Hyper depths: {config['hyper_depths']}")
    print(f"  Context depths: {config['context_depths']}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
    if "loss" in checkpoint:
        print(f"Checkpoint loss: {checkpoint['loss']:.4f}")

    print(f"Model groups: {model.groups}")
    print(f"Model num_slices: {model.num_slices}")

    model.update()

    if args.mode == "decompression":
        if not args.bitstream_dir:
            raise ValueError("--bitstream_dir is required for decompression mode")

        results = test_decompression(model, args.bitstream_dir, args)
    elif args.mode == "compression":
        print("Loading dataset...")
        test_dataset = VolumeFolder(
            root_dir=args.data_dir,
            split='test'
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        print(f"Test dataset size: {len(test_dataset)} volumes")

        if args.s == -1 and not args.factormode:
            print("\n" + "="*60)
            print("Testing all quality levels (s=0 to s=7)")
            print("="*60)
            base_output_dir = args.output_dir
            all_results = []

            for s in range(model.levels):
                print(f"\n>>> Testing quality level s={s} (lambda={model.lmbda[s]}) <<<")
                args.s = s
                args.output_dir = f"{base_output_dir}_s{s}"
                results = test_compression(model, test_loader, test_dataset, device, args)
                all_results.append({
                    's': s,
                    'lambda': model.lmbda[s],
                    'bpp': results['actual_bpp'],
                    'psnr': results['avg_psnr'],
                    'compression_ratio': results['compression_ratio']
                })

            print("\n" + "="*60)
            print("All Quality Levels Summary:")
            print("="*60)
            print(f"{'Level':>6} | {'Lambda':>8} | {'BPP':>8} | {'PSNR':>10} | {'Ratio':>8}")
            print("-"*55)
            for r in all_results:
                print(f"{r['s']:>6} | {r['lambda']:>8.4f} | {r['bpp']:>8.4f} | {r['psnr']:>8.2f} dB | {r['compression_ratio']:>7.2f}×")
            print("="*60)

            args.output_dir = base_output_dir
            results = all_results
        else:
            results = test_compression(model, test_loader, test_dataset, device, args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    if args.save_results and args.mode != "decompression":
        ensure_dirs(args.output_dir)
        results_path = os.path.join(args.output_dir, f"results_{args.mode}.txt")
        with open(results_path, 'w') as f:
            if isinstance(results, list):
                f.write("All Quality Levels Summary:\n")
                f.write(f"{'Level':>6} | {'Lambda':>10} | {'BPP':>8} | {'PSNR':>10} | {'Ratio':>8}\n")
                f.write("-" * 55 + "\n")
                for r in results:
                    f.write(f"{r['s']:>6} | {r['lambda']:>10.4f} | {r['bpp']:>8.4f} | {r['psnr']:>8.2f} dB | {r['compression_ratio']:>7.2f}x\n")
            else:
                for key, value in results.items():
                    if key not in ["recon_volume", "decompressed_volumes", "metadata"]:
                        f.write(f"{key}: {value}\n")
        print(f"\nSaved results to: {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch-based Inference for 3D Scale Hyperprior with Context Model")

    parser.add_argument("--data_dir", type=str, default='../Data',
                        help="Root directory containing test split with .nc files (required for compression mode)")

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")

    parser.add_argument("--patch_size", type=int, nargs=3, default=[128,128,128],
                        help="Patch size for compression (e.g., 128 128 128)")
    parser.add_argument("--stride", type=int, nargs=3, default=None,
                        help="Stride for overlapping patches (e.g., 32 32 32). Defaults to patch_size if not specified.")

    parser.add_argument("--mode", type=str, default="compression",
                        choices=["compression", "decompression"],
                        help="Inference mode: compression or decompression")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")

    parser.add_argument("--bitstream_dir", type=str, default=None,
                        help="Directory containing compressed bitstreams (required for decompression mode)")

    parser.add_argument("--output_dir", type=str, default="../Exp/output_context_overlap", help="Output directory")
    parser.add_argument("--save_output", action="store_true", help="Save reconstructed/decompressed volume")
    parser.add_argument("--save_results", action="store_true", help="Save results to text file")

    parser.add_argument("--quality_mode", type=str, default="factor",
                        choices=["factor", "discrete"],
                        help="Quality control: 'factor' (continuous gain, primary) or 'discrete' (fixed levels)")
    parser.add_argument("--factor", type=float, default=9.4,
                        help="Continuous gain factor (primary quality control; higher = better quality / larger file; trained range ~0.9-9.4)")
    parser.add_argument("--s", type=int, default=7,
                        help="Discrete quality level 0-7 (only used when --quality_mode discrete; -1 sweeps all levels)")

    args = parser.parse_args()

    main(args)
