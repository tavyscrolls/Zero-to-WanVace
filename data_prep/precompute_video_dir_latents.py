from pathlib import Path

import click
import numpy as np
import torch
import torchvision.transforms.functional as TF
import decord

from tqdm import tqdm

from models.wan.modules.clip import CLIPModel
from models.wan.modules.vae import WanVAE


def get_vae_model(checkpoint_dir: Path):
    vae = WanVAE(vae_pth=checkpoint_dir / "Wan2.1_VAE.pth", device="cuda")
    return vae


def get_clip_model(checkpoint_dir: Path):
    clip_dtype = torch.float16
    clip_checkpoint = "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    clip_tokenizer = "xlm-roberta-large"
    device = torch.device("cuda")
    clip = CLIPModel(
        dtype=clip_dtype,
        device=device,
        checkpoint_path=checkpoint_dir / clip_checkpoint,
        tokenizer_path=checkpoint_dir / clip_tokenizer,
    )
    return clip


@click.command()
@click.option("--video_dir", type=str, help="Path to the video directory")
@click.option("--target_fps", type=float, default=16.0, help="Target fps for the video")
@click.option("--num_frames", type=int, default=81, help="Number of frames to sample")
@click.option("--target_height", type=int, default=480, help="Target height for the video")
@click.option("--target_width", type=int, default=832, help="Target width for the video")
@click.option("--compute_clip_context", type=bool, default=False, help="Whether to compute clip context")
@click.option("--vae_checkpoint_dir", type=str, help="Path to the model checkpoint directory")
@click.option("--clip_checkpoint_dir", type=str, help="Path to the model checkpoint directory")
@click.option("--output_latent_dir", type=str, default=None, help="Directory to save cached latents. Defaults to video_dir if not specified.")
@click.option("--mask_dir", type=str, default=None, help="Directory containing mask images for conditioning.")
@click.option("--ref_image_dir", type=str, default=None, help="Directory containing reference images for conditioning.")
@click.option("--conditioned_video_dir", type=str, default=None, help="Directory containing conditioned videos (e.g., depth, pose) for conditioning.")
@torch.no_grad()
def main(
    video_dir,
    target_fps,
    num_frames,
    target_height,
    target_width,
    compute_clip_context,
    vae_checkpoint_dir,
    clip_checkpoint_dir,
    output_latent_dir,
    mask_dir,
    ref_image_dir,
    conditioned_video_dir,
):
    video_dir = Path(video_dir)
    output_dir = Path(output_latent_dir) if output_latent_dir else video_dir
    target_video_len = (num_frames - 1) / target_fps
    vae = get_vae_model(Path(vae_checkpoint_dir))
    if compute_clip_context:
        assert all(x is None for x in (mask_dir, ref_image_dir, conditioned_video_dir))
        clip = get_clip_model(Path(clip_checkpoint_dir))

    # Create additional folders if needed
    #mask_dir_path = Path(mask_dir) if mask_dir else None
    ref_image_dir_path = Path(ref_image_dir) if ref_image_dir else None
    conditioned_video_dir_path = Path(conditioned_video_dir) if conditioned_video_dir else None
    #if mask_dir_path: mask_dir_path.mkdir(parents=True, exist_ok=True)
    if ref_image_dir_path: ref_image_dir_path.mkdir(parents=True, exist_ok=True)
    if conditioned_video_dir_path: conditioned_video_dir_path.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(list(video_dir.glob("*.mp4")))
    if not video_paths:
        print(f"No .mp4 videos found in {video_dir}. Exiting.")
        return
    for video_path in tqdm(video_paths, desc="Processing videos"):
        stem = video_path.stem
        print(f"processing {stem}")
        latents_output_path = Path(output_dir) / f"latent_vid_{stem}.pt"
        if latents_output_path.exists():
            print(f"latents already exist for {video_path}, skipping")
            continue
        video_reader = decord.VideoReader(str(video_path), height=target_height, width=target_width)
        vid_fps = video_reader.get_avg_fps()
        num_original_frames = min(len(video_reader), int(vid_fps * target_video_len))
        sampled_frames_from_clip_len_in_secs = np.linspace(
            0, num_original_frames-1, min(num_original_frames, num_frames), endpoint=True, dtype=int
        )
        video = video_reader.get_batch(sampled_frames_from_clip_len_in_secs)
        video = video.asnumpy()

        vae_input_vid = torch.stack([TF.to_tensor(img).sub_(0.5).div_(0.5).to("cuda") for img in video], dim=0)
        latent_vid = vae.encode(vae_input_vid.permute(1, 0, 2, 3).unsqueeze(0))[0]
        torch.save(latent_vid, latents_output_path)
        if compute_clip_context:
            clip_context = clip.visual(vae_input_vid[0][:, None, :, :].unsqueeze(0))
            torch.save(clip_context, Path(output_dir) / f"clip_context_{video_path.stem}.pt")

        if conditioned_video_dir_path:
            cond_video_path = conditioned_video_dir_path / f"{stem}.mp4"
            if cond_video_path.exists():
                cond_video_latents_output_path = Path(output_dir) / f"latent_cond_vid_{stem}.pt"
                if cond_video_latents_output_path.exists():
                    print(f"Conditioned video latents already exist for {cond_video_path}, skipping.")
                else:
                    cond_video_reader = decord.VideoReader(str(cond_video_path), height=target_height, width=target_width)
                    cond_total_frames = len(cond_video_reader)

                    # Ensure conditioned video sampling aligns with main video (num_frames)
                    cond_num_frames_to_sample = min(cond_total_frames, num_frames)
                    cond_sampled_frames = np.linspace(
                        0, cond_total_frames - 1, cond_num_frames_to_sample, endpoint=True, dtype=int
                    )
                    cond_video_frames = cond_video_reader.get_batch(cond_sampled_frames).asnumpy()

                    cond_vae_input_vid = torch.stack([TF.to_tensor(img).sub_(0.5).div_(0.5).to("cuda") for img in cond_video_frames], dim=0)
                    cond_latent_vid = vae.encode(cond_vae_input_vid.permute(1, 0, 2, 3).unsqueeze(0))[0]
                    torch.save(cond_latent_vid, cond_video_latents_output_path)
            else:
                print(f"No conditioned video found for {stem} at {cond_video_path}. Skipping conditioned video latent generation.")

        if ref_image_dir_path:
            ref_image_paths = sorted(ref_image_dir_path.glob(f"{stem}_*.png"))
            # Check if any matching files were found
            if not ref_image_paths:
                ref_image_path = ref_image_dir_path / f"{stem}.png"
                if ref_image_path.exists(): ref_image_paths = [ref_image_path]
            if ref_image_paths:
                for ref_image_path in ref_image_paths:
                    file_stem = ref_image_path.stem
                    ref_image_latents_output_path = Path(output_dir) / f"latent_ref_image_{file_stem}.pt"
                    if ref_image_latents_output_path.exists():
                        print(f"Reference image latents already exist for {ref_image_path}, skipping.")
                    else:
                        ref_img, _, _ = read_image(str(ref_image_path), use_type='pil', info=True)
                        if ref_img is not None:
                            ref_img_tensor = TF.to_tensor(ref_img).sub_(0.5).div_(0.5).to("cuda")
                            ref_image_latent = vae.encode(ref_img_tensor.unsqueeze(0).unsqueeze(2))[0]
                            torch.save(ref_image_latent, ref_image_latents_output_path)


if __name__ == "__main__":
    main()
