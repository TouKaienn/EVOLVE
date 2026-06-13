import os
import argparse
import time
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import monai.transforms as mt

from dataloader.VolumeFolder import VolumeFolder
from models.scale_hyperprior_3d_context import (
    ScaleHyperprior3DContext,
    RateDistortionLossContext,
    MODEL_CONFIGS,
)
from utils.fileUtils import ensure_dirs

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def configure_optimizers(net, args):
    parameters = {
        n for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    assert len(inter_params) == 0, "Parameters overlap detected"

    optimizer = optim.AdamW(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
        betas=(0.9, 0.999),
    )

    return optimizer, aux_optimizer

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    aux_optimizer,
    criterion,
    device,
    epoch,
    noisequant=True,
    clip_max_norm=1.0,
    stage=3,
):
    model.train()

    total_loss = 0
    total_mse = 0
    total_bpp = 0
    total_aux_loss = 0
    num_batches = 0

    for batch_idx, volumes in enumerate(train_loader):
        volumes = volumes.to(device)

        if stage > 1:
            s = (batch_idx + epoch * len(train_loader)) % model.levels
        else:
            s = model.levels - 1

        lmbda = model.lmbda[s]

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        output = model(volumes, noisequant=noisequant, stage=stage, s=s)

        loss_dict = criterion(output, volumes, lmbda=lmbda)
        loss = loss_dict["loss"]

        loss.backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)

        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        total_loss += loss.item()
        total_mse += loss_dict["mse_loss"].item()
        total_bpp += loss_dict["bpp_loss"].item()
        total_aux_loss += aux_loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_mse = total_mse / num_batches
    avg_bpp = total_bpp / num_batches
    avg_aux_loss = total_aux_loss / num_batches

    return avg_loss, avg_mse, avg_bpp, avg_aux_loss

def validate(model, val_loader, criterion, device, epoch, stage=1):
    model.eval()

    if stage > 1:
        quality_levels = list(range(model.levels))
    else:
        quality_levels = [model.levels - 1]

    all_level_metrics = {s: {'loss': 0, 'mse': 0, 'bpp': 0, 'psnr': 0, 'count': 0}
                         for s in quality_levels}

    with torch.no_grad():
        for batch_idx, volumes in enumerate(tqdm(val_loader, desc="Validation")):
            volumes = volumes.to(device)

            for s in quality_levels:
                lmbda = model.lmbda[s]

                output = model(volumes, noisequant=False, stage=stage, s=s)

                loss_dict = criterion(output, volumes, lmbda=lmbda)

                mse = torch.nn.functional.mse_loss(output["x_hat"], volumes)
                psnr = 10 * torch.log10(1.0 / (mse + 1e-10))

                all_level_metrics[s]['loss'] += loss_dict["loss"].item()
                all_level_metrics[s]['mse'] += loss_dict["mse_loss"].item()
                all_level_metrics[s]['bpp'] += loss_dict["bpp_loss"].item()
                all_level_metrics[s]['psnr'] += psnr.item()
                all_level_metrics[s]['count'] += 1

    level_results = {}
    for s in quality_levels:
        count = all_level_metrics[s]['count']
        level_results[s] = {
            'loss': all_level_metrics[s]['loss'] / count,
            'mse': all_level_metrics[s]['mse'] / count,
            'bpp': all_level_metrics[s]['bpp'] / count,
            'psnr': all_level_metrics[s]['psnr'] / count,
        }

    avg_loss = sum(r['loss'] for r in level_results.values()) / len(quality_levels)
    avg_mse = sum(r['mse'] for r in level_results.values()) / len(quality_levels)
    avg_bpp = sum(r['bpp'] for r in level_results.values()) / len(quality_levels)
    avg_psnr = sum(r['psnr'] for r in level_results.values()) / len(quality_levels)

    if stage > 1:
        print(f"\nValidation (all {len(quality_levels)} quality levels):")
        print(f"{'Level':>6} | {'Lambda':>8} | {'Loss':>8} | {'BPP':>8} | {'PSNR':>8}")
        print("-" * 50)
        for s in quality_levels:
            r = level_results[s]
            print(f"{s:>6} | {model.lmbda[s]:>8.4f} | {r['loss']:>8.4f} | {r['bpp']:>8.4f} | {r['psnr']:>7.2f} dB")
        print("-" * 50)
        print(f"{'Avg':>6} | {'-':>8} | {avg_loss:>8.4f} | {avg_bpp:>8.4f} | {avg_psnr:>7.2f} dB\n")
    else:
        print(f"\nValidation - Loss: {avg_loss:.4f}, MSE: {avg_mse:.6f}, "
              f"BPP: {avg_bpp:.4f}, PSNR: {avg_psnr:.2f} dB\n")

    return avg_loss, avg_psnr

def train_stage(
    model, train_loader, val_loader, criterion, device, args,
    stage, epochs, checkpoint_dir, resume_checkpoint=None,
):
    ensure_dirs(checkpoint_dir)

    optimizer, aux_optimizer = configure_optimizers(model, args)

    warmup_epochs = args.warmup_epochs

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return 0.5 ** ((epoch - warmup_epochs) // args.lr_decay_epochs)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_ste = (stage == 3)

    start_epoch = 1
    best_loss = float('inf')

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"  Resuming stage {stage} from {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location=device)
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'aux_optimizer_state_dict' in ckpt:
            aux_optimizer.load_state_dict(ckpt['aux_optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch'] + 1
        if 'loss' in ckpt:
            best_loss = ckpt['loss']
        print(f"  Resuming from epoch {start_epoch}, best_loss={best_loss:.4f}")

    print(f"\n{'='*50}")
    print(f"Stage {stage}: epochs {start_epoch}-{epochs}, "
          f"quant={'STE' if use_ste else 'noise'}, checkpoint_dir={checkpoint_dir}")
    print(f"  Optimizer: AdamW (weight_decay={args.weight_decay})")
    print(f"  Warmup: {warmup_epochs} epochs, LR decay every {args.lr_decay_epochs} epochs")
    print(f"{'='*50}\n")

    pbar = tqdm(range(start_epoch, epochs + 1), desc=f"Stage {stage}")

    for epoch in pbar:
        if use_ste:
            noisequant = False
        else:
            noisequant = epoch <= args.ste_switch_epoch

        train_loss, train_mse, train_bpp, train_aux = train_one_epoch(
            model, train_loader, optimizer, aux_optimizer,
            criterion, device, epoch,
            noisequant=noisequant,
            clip_max_norm=args.clip_max_norm,
            stage=stage,
        )

        scheduler.step()

        quant_mode = "noise" if noisequant else "STE"
        pbar.set_postfix({
            "loss": f"{train_loss:.4f}",
            "mse": f"{train_mse:.6f}",
            "bpp": f"{train_bpp:.4f}",
            "quant": quant_mode,
        })

        if epoch % args.val_freq == 0:
            val_loss, val_psnr = validate(
                model, val_loader, criterion, device, epoch, stage=stage
            )

            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)

            ckpt_data = {
                "epoch": epoch,
                "stage": stage,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "aux_optimizer_state_dict": aux_optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": val_loss,
                "psnr": val_psnr,
            }

            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch}.pth")
            torch.save(ckpt_data, checkpoint_path)

            if is_best:
                best_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save(ckpt_data, best_path)
                print(f"  Saved best model (stage {stage}) with loss: {best_loss:.4f}")

        if epoch % args.save_freq == 0 and epoch % args.val_freq != 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "stage": stage,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "aux_optimizer_state_dict": aux_optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }, checkpoint_path)

    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    if not os.path.exists(best_model_path):
        best_model_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epochs}.pth")
        torch.save({
            "epoch": epochs,
            "stage": stage,
            "model_state_dict": model.state_dict(),
        }, best_model_path)

    print(f"Stage {stage} completed. Best model: {best_model_path}")
    return best_model_path

def main(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"\n=== Multi-Stage Training (data loaded once) ===")
    print(f"Stages to run: {args.stages}")
    print(f"Model size: {args.model_size}")

    print("\nLoading dataset (shared across all stages)...")
    t0 = time.time()

    train_transform = mt.Compose([
        mt.RandSpatialCrop(roi_size=args.crop_size, random_size=False),
        mt.RandFlip(prob=0.5, spatial_axis=0),
        mt.RandFlip(prob=0.5, spatial_axis=1),
        mt.RandFlip(prob=0.5, spatial_axis=2),
        mt.RandRotate90(prob=0.5, spatial_axes=(0, 1)),
        mt.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
        mt.ScaleIntensity(minv=0.0, maxv=1.0),
    ])

    val_transform = mt.Compose([
        mt.RandSpatialCrop(roi_size=args.crop_size, random_size=False),
        mt.ScaleIntensity(minv=0.0, maxv=1.0),
    ])

    train_dataset = VolumeFolder(
        root_dir=args.data_dir,
        split='train',
        transform=train_transform,
        preload_data=args.preload_data,
        partial_load_ratio=args.partial_load,
    )

    val_dataset = VolumeFolder(
        root_dir=args.data_dir,
        split='val',
        transform=val_transform,
        preload_data=args.preload_data,
        partial_load_ratio=args.partial_load,
    )

    print(f"Train dataset: {len(train_dataset)} volumes")
    print(f"Val dataset: {len(val_dataset)} volumes")
    print(f"Data loading time: {time.time() - t0:.1f}s")

    num_workers_to_use = min(args.num_workers, 2) if args.preload_data else args.num_workers
    persistent_workers_flag = num_workers_to_use > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers_to_use,
        pin_memory=True,
        persistent_workers=persistent_workers_flag,
        timeout=60 if num_workers_to_use > 0 else 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers_to_use,
        pin_memory=True,
        persistent_workers=persistent_workers_flag,
        timeout=60 if num_workers_to_use > 0 else 0,
    )

    model = ScaleHyperprior3DContext(
        model_size=args.model_size,
        num_slices=args.num_slices,
    ).to(device)

    config = MODEL_CONFIGS[args.model_size]
    print(f"\nModel config:")
    print(f"  Channels: {config['channels']}")
    print(f"  Depths: {config['depths']}")
    print(f"  Hyper depths: {config['hyper_depths']}")
    print(f"  Context depths: {config['context_depths']}")
    print(f"  Groups: {model.groups}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params:,}")

    criterion = RateDistortionLossContext()

    if args.checkpoint:
        print(f"\nLoading initial checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

    checkpoint_base = args.checkpoint_base
    stage_dirs = {
        1: os.path.join(checkpoint_base, "stage1"),
        2: os.path.join(checkpoint_base, "stage2"),
        3: os.path.join(checkpoint_base, "stage3"),
    }

    stage_epochs = {
        1: args.stage1_epochs,
        2: args.stage2_epochs,
        3: args.stage3_epochs,
    }

    stages = sorted(args.stages)

    for stage in stages:
        if stage > 1:
            prev_stage = stage - 1
            prev_best = os.path.join(stage_dirs[prev_stage], "best_model.pth")
            if os.path.exists(prev_best):
                print(f"\nLoading stage {prev_stage} best model for stage {stage}: {prev_best}")
                ckpt = torch.load(prev_best, map_location=device)
                model.load_state_dict(ckpt['model_state_dict'])
            else:
                print(f"\nWarning: No best model found for stage {prev_stage} at {prev_best}")
                print(f"  Continuing with current model weights")

        resume_ckpt = None
        latest_ckpt = os.path.join(stage_dirs[stage], "best_model.pth")
        if args.resume and os.path.exists(latest_ckpt):
            resume_ckpt = latest_ckpt

        train_stage(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            args=args,
            stage=stage,
            epochs=stage_epochs[stage],
            checkpoint_dir=stage_dirs[stage],
            resume_checkpoint=resume_ckpt,
        )

    print(f"\n{'='*50}")
    print("All stages completed!")
    print(f"Checkpoints saved to: {checkpoint_base}")
    print(f"{'='*50}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-stage training for 3D Scale Hyperprior with context modeling. "
                    "Loads data once and runs all stages sequentially."
    )

    parser.add_argument("--data_dir", type=str, default='../Data',
                        help="Root directory containing train/val splits with .nc files")
    parser.add_argument("--crop_size", type=int, nargs=3, default=[128, 128, 128],
                        help="Crop size for training (e.g., 128 128 128)")

    parser.add_argument("--model_size", type=str, default="small",
                        choices=["small"],
                        help="Model size configuration")
    parser.add_argument("--num_slices", type=int, default=5,
                        help="Number of channel slices for context modeling")

    parser.add_argument("--stages", type=int, nargs='+', default=[1, 2, 3],
                        help="Which stages to run (e.g., --stages 1 2 3 or --stages 2 3)")

    parser.add_argument("--stage1_epochs", type=int, default=1000,
                        help="Number of epochs for stage 1")
    parser.add_argument("--stage2_epochs", type=int, default=1000,
                        help="Number of epochs for stage 2")
    parser.add_argument("--stage3_epochs", type=int, default=500,
                        help="Number of epochs for stage 3")

    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Main learning rate")
    parser.add_argument("--aux_learning_rate", type=float, default=1e-3,
                        help="Auxiliary learning rate for entropy model quantiles")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--clip_max_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW optimizer")
    parser.add_argument("--warmup_epochs", type=int, default=20,
                        help="Number of warmup epochs at the start of each stage")
    parser.add_argument("--lr_decay_epochs", type=int, default=600,
                        help="LR half-life decay interval (epochs after warmup)")
    parser.add_argument("--ste_switch_epoch", type=int, default=1500,
                        help="Epoch to switch from noise to STE quantization (within stage 1/2)")

    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--preload_data", action="store_true",
                        help="Preload all volumes into CPU memory")
    parser.add_argument("--partial_load", type=float, default=1.0,
                        help="Ratio of data to load into memory (0.0-1.0)")

    parser.add_argument("--checkpoint_base", type=str, default="../Exp/checkpoints_context",
                        help="Base checkpoint directory (stage1/stage2/stage3 subdirs created automatically)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Initial checkpoint to load before starting (for all stages)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from existing stage checkpoints if available")
    parser.add_argument("--val_freq", type=int, default=100,
                        help="Validation frequency (epochs)")
    parser.add_argument("--save_freq", type=int, default=100,
                        help="Checkpoint save frequency (epochs)")

    args = parser.parse_args()
    main(args)
