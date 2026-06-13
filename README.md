<h1 align="center">EVOLVE</h1>

<p align="center">
  Efficient Learned Volume Compression with Variable-Rate Encoding on a Cross-Domain Database
</p>

<p align="center">
  Kaiyuan Tang &nbsp;·&nbsp; Maizhe Yang &nbsp;·&nbsp; Chaoli Wang
</p>

<p align="center">
  <sub>University of Notre Dame &nbsp;·&nbsp; Department of Computer Science and Engineering</sub>
</p>

<p align="center">
  <a href="https://github.com/TouKaienn/EVOLVE"><img src="https://img.shields.io/badge/Code-GitHub-555?style=flat-square&logo=github&logoColor=white" alt="Code"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/Cite-BibTeX-555?style=flat-square" alt="BibTeX"></a>
</p>

---

EVOLVE is an autoencoder-based framework for compressing large-scale scientific volume
data. Trained on a cross-domain database of 6,376 volumes from 21 simulations, it pairs
context-aware entropy modeling with a learnable gain mechanism, so a single model spans a
continuous range of compression ratios — reaching substantially higher CRs than conventional
compressors at comparable quality, and running orders of magnitude faster than INR methods.

## Directory Layout

```
EVOLVE/
├── train.py            # Multi-stage training (stage 1 → 2 → 3, data loaded once)
├── infer.py            # Patch-based compression / decompression
├── models/
│   ├── scale_hyperprior_3d_context.py   # Model, VBR logic, rate-distortion loss
│   └── context_layers_3d.py             # 3D checkerboard conv, quantizer, masks
├── dataloader/
│   └── VolumeFolder.py # Loads .nc volumes (a directory split, or a single file)
├── utils/
│   └── fileUtils.py    # NetCDF reading + output-dir helpers
├── scripts/
│   ├── inference.sh    # Convenience wrapper around infer.py
│   └── best_model.pth  # Trained checkpoint (downloaded separately, ~342 MB)
├── data/
│   └── H+0161.nc       # Example test volume (600 × 248 × 248)
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/TouKaienn/EVOLVE.git
cd EVOLVE

# Conda environment used for development:
#   ~/anaconda3/envs/ae/bin/python
pip install -r requirements.txt
```

## Pretrained Checkpoint

The trained model weights (~342 MB) are hosted on Google Drive:

<https://drive.google.com/file/d/1AUv_HOzHz7G9zLoeo9t0I0g5SOon53uU/view?usp=sharing>

Download the file from the link above and place it at **`scripts/best_model.pth`**.
All commands below assume the checkpoint lives at that path.

## Data Format

Volumes are stored as **NetCDF (`.nc`)** files, read with h5py. The first variable
with 3 or more dimensions is loaded as the volume (shape `C × D × H × W`, with a
channel dimension added automatically). `VolumeFolder` accepts either:

- a **directory** containing `train/` and `test/` subdirectories of `.nc` files, or
- a **single `.nc` file** (used directly, ignoring the split).

## Quick Test

A sample volume is provided at `data/H+0161.nc`. Run compression on it with the
included checkpoint. Quality is controlled by the continuous **gain factor**
(`--quality_mode factor` is the default):

```bash
# from the repository root (EVOLVE/)
python infer.py \
    --checkpoint scripts/best_model.pth \
    --mode compression \
    --data_dir data/H+0161.nc \
    --patch_size 128 128 128 \
    --stride 128 128 128 \
    --factor 9.4 \
    --output_dir ./output \
    --save_results
```

Expected output on `H+0161.nc` (gain factor mode, `--factor 9.4`):

```
Quality mode:      Continuous (factor=9.4)
Compression ratio: 882.76×
Actual BPP:        0.0362
Average PSNR:      53.30 dB
```

The bitstream is written to `./output/bitstream/H+0161.bin` together with a
`metadata.json`. **Lower the factor for higher compression** (smaller file, lower
PSNR), e.g. `--factor 3.0`; raise it toward `~9.4` for higher quality. The trained
gain range is roughly `0.9–9.4`.

## Inference

`infer.py` operates patch-by-patch with overlapping windows (averaged on overlap).
Quality defaults to the continuous **gain factor** (`--quality_mode factor`).

**Compression (gain factor mode — primary)** — sweep the factor to trace the
rate–quality curve:

```bash
python infer.py --checkpoint scripts/best_model.pth --mode compression \
    --data_dir <dir-with-test-split-or-single.nc> \
    --patch_size 128 128 128 --stride 128 128 128 \
    --factor 9.4 --output_dir ./output --save_results
```

**Discrete preset levels (optional)** — use the 8 fixed quality levels instead; pass
`--s -1` to evaluate all of them in one run:

```bash
python infer.py --checkpoint scripts/best_model.pth --mode compression \
    --data_dir <...> --patch_size 128 128 128 \
    --quality_mode discrete --s 7 --output_dir ./output
```

**Decompression** — reconstruct volumes from a bitstream directory:

```bash
python infer.py --checkpoint scripts/best_model.pth --mode decompression \
    --bitstream_dir ./output/bitstream --output_dir ./recon --save_output
```

Key arguments:

| Argument | Description |
| --- | --- |
| `--checkpoint` | Path to model checkpoint (required) |
| `--mode` | `compression` or `decompression` |
| `--data_dir` | `.nc` directory (test split) or a single `.nc` file (compression) |
| `--bitstream_dir` | Bitstream directory to decode (decompression) |
| `--patch_size` / `--stride` | Sliding-window patch size and stride (e.g. `128 128 128`) |
| `--quality_mode` | `factor` (continuous gain, **default**) or `discrete` (preset levels) |
| `--factor` | Continuous gain factor — primary quality knob (higher = better quality / larger file; trained range ~`0.9–9.4`) |
| `--s` | Discrete level `0–7` (only with `--quality_mode discrete`; `-1` sweeps all 8) |
| `--output_dir` | Output directory |
| `--save_output` | Save the reconstructed volume (`.raw`) on decompression |
| `--save_results` | Write a `results_<mode>.txt` summary |

## Training

`train.py` runs the three VBR training stages sequentially, loading the dataset once:

- **Stage 1** — base training at the highest quality only (noise quantization)
- **Stage 2** — multi-rate training cycling through all quality levels
- **Stage 3** — STE fine-tuning across all quality levels

```bash
python train.py \
    --data_dir <dir-with-train-and-val-splits> \
    --model_size small \
    --stages 1 2 3 \
    --stage1_epochs 1000 --stage2_epochs 1000 --stage3_epochs 500 \
    --batch_size 8 --crop_size 128 128 128 \
    --checkpoint_base ./checkpoints
```

Checkpoints are saved per stage under `checkpoint_base/stage{1,2,3}/best_model.pth`.
The best stage-3 model is the one used for inference.

## Citation

If you use EVOLVE in your research, please cite:

```bibtex
@article{EVOLVE,
  author={Tang, Kaiyuan and Yang, Maizhe and Wang, Chaoli},
  title={EVOLVE: Efficient Learned Volume Compression with Variable-Rate Encoding on a Cross-Domain Database},
  year={2026},
  }
```
