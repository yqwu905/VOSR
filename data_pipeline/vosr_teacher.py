"""Persistent VOSR2 teacher using the repository's one-step inference functions."""

import json
from pathlib import Path
from types import SimpleNamespace


class VOSR2Teacher:
    demo = False
    model = "vosr2"

    def __init__(self, checkpoint, upscale=4, device="cuda", overrides=None):
        # Import only when selected: preprocessing works without torch/diffusers.
        import torch
        from safetensors.torch import load_file
        import inference_vosr_onestep as inference

        checkpoint = Path(checkpoint).resolve()
        config_path = next((p / "args.json" for p in [checkpoint, *list(checkpoint.parents)[:4]]
                            if (p / "args.json").is_file()), None)
        if config_path is None:
            raise FileNotFoundError(f"VOSR2 args.json missing near {checkpoint}")
        values = json.loads(config_path.read_text(encoding="utf-8"))
        values.update({"checkpoint": str(checkpoint), "infer_steps": 1, "tile_size": 0,
                       "tile_overlap": 32, "vae_tile_size": None, "vae_tile_overlap": None,
                       "posterior_mode": True, "align_method": "wavelet"})
        values.update(overrides or {})
        values["auxiliary_time_cond"] = bool(values.get("auxiliary_time_cond", True))
        self.args = args = SimpleNamespace(**values)
        self.device, self.upscale, self.inference = device, int(upscale), inference
        if self.upscale < 1:
            raise ValueError("teacher upscale must be positive")
        if args.ae_type != "qwen":
            raise ValueError("VOSR2 adapter requires the Qwen VAE; use a custom adapter for other teachers")
        from models.qwenimage_vae2d import AutoencoderKLQwenImage2D
        self.vae = AutoencoderKLQwenImage2D.from_pretrained(
            getattr(args, "ae_path", inference.QWEN_AE_PATH)).to(device).eval()
        self.venc = inference.load_dinov2(args, device)
        self.dit = inference.LightningDiT(
            input_size=args.resolution // 8, patch_size=args.patch_size, in_channels=32, out_channels=16,
            hidden_size=args.dim, depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,
            z_dims=args.enc_dim, encdim_ratio=args.encdim_ratio,
            auxiliary_time_cond=args.auxiliary_time_cond, use_qknorm=args.use_qknorm,
            use_swiglu=args.use_swiglu, use_rope=args.use_rope, use_rmsnorm=args.use_rmsnorm,
            wo_shift=args.wo_shift, num_fused_layers=len(args.layer_dinov2b_list))
        weight_path = next((directory / name for directory in
                           (checkpoint / "clean_weights", checkpoint / "checkpoints", checkpoint)
                           for name in ("ema_model.safetensors", "model.safetensors")
                           if (directory / name).is_file()), None)
        if weight_path is None:
            raise FileNotFoundError(f"VOSR2 EMA/model safetensors missing under {checkpoint}")
        self.dit.load_state_dict(load_file(str(weight_path)), strict=True)
        self.dit.to(device).eval()
        self.dit.forward = self.dit.forward_flexible
        self.sampler = inference.VOSR(
            time_dist=args.time_dist, cfg_ratio=args.cfg_ratio, cfg_scale=args.cfg_scale,
            interp_type=args.interp_type, accelerator=SimpleNamespace(device=torch.device(device)),
            t_start=args.t_start, t_end=args.t_end, args=args)

    def restore(self, image, context):
        import torch
        from PIL import Image
        from torchvision.transforms.functional import to_tensor, to_pil_image

        inference, args = self.inference, self.args
        resized = image.resize((image.width * self.upscale, image.height * self.upscale),
                               Image.Resampling.BICUBIC)
        tensor = to_tensor(resized).unsqueeze(0).to(self.device) * 2 - 1
        # fork_rng keeps the pipeline's random state isolated from teacher sampling.
        devices = [torch.device(self.device)] if torch.device(self.device).type == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.manual_seed(context["seed"] % (2 ** 63))
            if args.tile_size > 0:
                output = inference.tiled_latent_inference(
                    self.dit, self.vae, self.venc, tensor, args, device=self.device)
            else:
                latent, mean, std = inference.encode_dispatch(self.vae, tensor, args, self.device)
                features = inference.get_venc_features(self.venc, tensor, args)
                restored = self.sampler.sample_onestep(self.dit, latent, n_steps=args.infer_steps,
                                                       venc_fea=features)
                output = inference.decode_dispatch(self.vae, restored, args, mean, std, None)
            output = to_pil_image((output[0].float().cpu() * 0.5 + 0.5).clamp(0, 1))
        if output.size != resized.size:
            raise ValueError("teacher returned a spatially incompatible image; use supported sizes/tiling")
        if args.align_method == "wavelet":
            output = inference.wavelet_color_fix(output, resized)
        elif args.align_method == "adain":
            output = inference.adain_color_fix(output, resized)
        elif args.align_method != "nofix":
            raise ValueError("unsupported teacher color alignment")
        return output
