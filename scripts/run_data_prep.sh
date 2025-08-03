export PYTHONPATH=${PYTHONPATH}:${PWD}                       
#chmod +x data_prep/download_hf_dataset.sh

#huggingface-cli download finetrainers/3dgs-dissolve --repo-type dataset --local-dir ./datasets/3dgs-dissolve
#huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --repo-type model --local-dir ./weights/Wan2.1-T2V-1.3B
#huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --include xlm-roberta-large/ --local-dir ./weights/Wan2.1-I2V-14B-720P/xlm_roberta_large 
#huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --include models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth --local-dir ./weights/Wan2.1-I2V-14B-720P/

#python data_prep/caption_video_dir.py --video_dir ./datasets/3dgs-dissolve/videos

python data_prep/precompute_video_dir_latents.py --video_dir ./datasets/test --vae_checkpoint_dir ./weights/wan --clip_checkpoint_dir ./weghts/wan --target_fps 16 --num_frames 9 --target_height 480 --target_width 848 --ref_image_dir ./datasets/test/ref_images --conditioned_video_dir ./datasets/test/cond_vids --output_latent_dir ./precache

python data_prep/precompute_video_dir_text_embeds.py --video_dir ./datasets/test --output_latent_dir ./precache
