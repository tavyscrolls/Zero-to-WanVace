import glob
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from einops import rearrange
import torch


@dataclass
class VAEConfig:
    name: str
    spatial_downsample_factor: int
    temporal_downsample_factor: int
    latent_channels: int


VAE_CONFIGS = {
    "wan": VAEConfig(
        name="wan",
        spatial_downsample_factor=8,
        temporal_downsample_factor=4,
        latent_channels=16,
    ),
}


@dataclass
class VideoResolution:
    vae_config: VAEConfig
    height: int
    width: int
    num_frames: int

    def __post_init__(self):
        self.num_channels = self.vae_config.latent_channels

    @property
    def latent_frames(self) -> int:
        num_frames = self.num_frames
        lsize = 1 + math.ceil((num_frames - 1) / self.vae_config.temporal_downsample_factor)
        lsize = int(lsize)
        return lsize

    @property
    def latent_height(self) -> int:
        return self.height // self.vae_config.spatial_downsample_factor

    @property
    def latent_width(self) -> int:
        return self.width // self.vae_config.spatial_downsample_factor


def tuplize(x):
    # handle string input
    if isinstance(x, str):
        x = x.split(",")
        return tuple(int(i) for i in x)
    # handle non-iterable input
    if not isinstance(x, Iterable):
        return (x, x)
    return x


class DirDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dir_path: str,
        video_dir_path: str,
        vae_name: str,
        resolution: Union[Tuple[int, int], int],
        max_sequence_length: int,
        num_frames: int,
        load_conditioned_video_latents=bool,
        load_ref_image_latents=bool,
        load_mask_latents=bool,
        vace_component_dropout_prob=float
    ):
        self.dir_path = dir_path
        self.video_dir_path = video_dir_path
        resolution_tuple = tuplize(resolution)  # resolution is (height, width)
        self.height = resolution_tuple[0]
        self.width = resolution_tuple[1]
        self.num_frames = num_frames
        self.load_conditioned_video_latents = load_conditioned_video_latents
        self.load_ref_image_latents = load_ref_image_latents
        self.load_mask_latents = load_mask_latents
        self.vace_component_dropout_prob = vace_component_dropout_prob
        self.max_sequence_length = max_sequence_length
        all_video_files = glob.glob(f"{self.video_dir_path}/*.mp4")
        self.video_ids = sorted([Path(video_path).stem for video_path in all_video_files])

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        video_id = self.video_ids[index]
        vae_latents_path = Path(self.dir_path) / f"latent_vid_{video_id}.pt"
        text_embeddings_path = Path(self.dir_path) / f"text_embed_{video_id}.pt"
        vae_latents = torch.load(vae_latents_path, map_location="cpu")
        text_embeddings = torch.load(text_embeddings_path, map_location="cpu")

        cond_vid_latent = None
        ref_image_latent = None
        mask_latent = None
        if self.load_conditioned_video_latents:
            cond_vid_path = Path(self.dir_path) / f"latent_cond_vid_{video_id}.pt"
            if cond_vid_path.exists() and torch.rand(1).item() >= self.vace_component_dropout_prob:
                cond_vid_latent = torch.load(cond_vid_path, map_location="cpu")
        if self.load_ref_image_latents:
            ref_image_path = Path(self.dir_path) / f"latent_ref_image_{video_id}.pt"
            if ref_image_path.exists() and torch.rand(1).item() >= self.vace_component_dropout_prob:
                ref_image_latent = torch.load(ref_image_path, map_location="cpu")
        if self.load_mask_latents:
            mask_latent_path = Path(self.dir_path) / f"mask_latent_{video_id}.pt"
            if mask_latent_path.exists() and torch.rand(1).item() >= self.vace_component_dropout_prob:
                mask_latent = torch.load(mask_latent_path, map_location="cpu")

        vace_context = None
        if vace_context is not None:
            latent_shape = (
                vae_latents.shape[0],
                4,
                self.num_frames,
                self.height // 8,
                self.width // 8
            )
            if cond_vid_latent is None:
                cond_vid_latent = torch.zeros(latent_shape, dtype=vae_latents.dtype, device=vae_latents.device)
            if mask_latent is None:
                    mask_latent = torch.ones(
                    1,
                    (cond_vid_latent.shape[1] + 3) // 4,
                    cond_vid_latent.shape[2],
                    cond_vid_latent.shape[3],
                    dtype=cond_vid_latent.dtype,
                    device=cond_vid_latent.device
                )

            inactive = cond_vid_latent * (1 - mask_latent)
            reactive = cond_vid_latent * mask_latent
            vace_context_latents = torch.concat((inactive, reactive), dim=1)

            # Resize and permute the mask to match the required dimensions
            vace_mask_latents = rearrange(mask_latent[0, 0, :, :], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8)
            vace_mask_latents = torch.nn.functional.interpolate(vace_mask_latents, size=((vace_mask_latents.shape[2] + 3) // 4, vace_mask_latents.shape[3], vace_mask_latents.shape[4]), mode='nearest-exact')

            # Handle the reference image latents
            if ref_image_latent is not None:
                vace_reference_latents = torch.concat((ref_image_latent, torch.zeros_like(ref_image_latent)), dim=1)
                vace_context_latents = torch.concat((vace_reference_latents, vace_context_latents), dim=2)
                vace_mask_latents = torch.concat((torch.zeros_like(vace_mask_latents[:, :, :1]), vace_mask_latents), dim=2)

            # Final concatenation to form the complete vace_context
            vace_context = torch.concat((vace_context_latents, vace_mask_latents), dim=1)

        vace_context = torch.zeros(96, 3, 3, 3)
        return {
            "pixel_values": vae_latents,
            "prompt_embeds": text_embeddings,
            "vace_context": vace_context,
        }


def collate_raw_dir_fn(examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate batch of examples into training batch."""
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    if pixel_values.dtype != torch.float32:
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    prompt_embeds = [example["prompt_embeds"] for example in examples]
    vace_contexts = [example["vace_context"] for example in examples]
    if any(ctx is not None for ctx in vace_contexts):
        first_valid_context = next(ctx for ctx in vace_contexts if ctx is not None)
        zero_context = torch.zeros_like(first_valid_context)
        vace_contexts = [ctx if ctx is not None else zero_context for ctx in vace_contexts]
        vace_contexts = torch.stack(vace_contexts)
    else:
        vace_contexts = None
    return_tuple = (pixel_values, prompt_embeds, vace_contexts)
    return return_tuple


def setup_data_modules(
    dir_path: str,
    video_dir_path: str,
    vae_name: str,
    resolution: Union[Tuple[int, int], int],
    max_sequence_length: int,
    num_frames: int,
    batch_size: int,
    num_workers: int,
    load_conditioned_video_latents: bool,
    load_ref_image_latents: bool,
    load_mask_latents: bool,
    vace_component_dropout_prob: float
):
    train_dataset = DirDataset(
        dir_path=dir_path,
        video_dir_path=video_dir_path,
        vae_name=vae_name,
        resolution=resolution,
        max_sequence_length=max_sequence_length,
        num_frames=num_frames,
        load_conditioned_video_latents=load_conditioned_video_latents,
        load_ref_image_latents=load_ref_image_latents,
        load_mask_latents=load_mask_latents,
        vace_component_dropout_prob=vace_component_dropout_prob
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collate_raw_dir_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
        persistent_workers=True,
    )
    return train_dataset, train_dataloader


if __name__ == "__main__":
    train_dataset, train_dataloader = setup_data_modules(
        dir_path="./precache",
        vae_name="wan",
        resolution=(480, 832),
        max_sequence_length=512,
        num_frames=81,
        batch_size=1,
        num_workers=1,
    )
    for i in range(len(train_dataset)):
        print(train_dataset[i])
