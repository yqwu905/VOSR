"""Persistent VOSR2 teacher using the repository's one-step inference functions."""

import json
from pathlib import Path
from types import SimpleNamespace


class VOSR2Teacher:
    demo = False
    model = "vosr2"

    def __init__(self, checkpoint, upscale=4, device="cuda", overrides=None,
                 precision="fp32", max_output_edge=None):
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
        if precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16 or bf16")
        if precision != "fp32" and torch.device(device).type != "cuda":
            raise ValueError("mixed precision requires a CUDA teacher")
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("this GPU does not support bf16")
        if max_output_edge is not None and (isinstance(max_output_edge, bool) or
                not isinstance(max_output_edge, int) or max_output_edge < 1):
            raise ValueError("max_output_edge must be a positive integer")
        self.precision, self.max_output_edge = precision, max_output_edge
        self.dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
        if args.ae_type != "qwen":
            raise ValueError("VOSR2 adapter requires the Qwen VAE; use a custom adapter for other teachers")
        from models.qwenimage_vae2d import AutoencoderKLQwenImage2D
        ae_path = Path(getattr(args, "ae_path", inference.QWEN_AE_PATH))
        if not ae_path.is_dir():
            candidate = config_path.parent.parent / ae_path.name
            if candidate.is_dir():
                ae_path = candidate
        hub_cache = config_path.parent.parent / "torch_cache"
        if hub_cache.is_dir():
            torch.hub.set_dir(str(hub_cache))
        self.vae = AutoencoderKLQwenImage2D.from_pretrained(str(ae_path)).to(device).eval()
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
        for module in (self.vae, self.venc, self.dit):
            module.requires_grad_(False)
        self.dit.forward = self.dit.forward_flexible
        self.sampler = inference.VOSR(
            time_dist=args.time_dist, cfg_ratio=args.cfg_ratio, cfg_scale=args.cfg_scale,
            interp_type=args.interp_type, accelerator=SimpleNamespace(device=torch.device(device)),
            t_start=args.t_start, t_end=args.t_end, args=args)
        self.metadata = {"model": self.model, "checkpoint": str(checkpoint),
                         "weight_path": str(weight_path), "args_path": str(config_path),
                         "precision": precision, "requested_upscale": self.upscale,
                         "max_output_edge": max_output_edge}

    def restore(self, image, context):
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from torchvision.transforms.functional import to_tensor, to_pil_image

        inference, args = self.inference, self.args
        factor = self.upscale
        if self.max_output_edge is not None:
            # Avoid generating 4K only to discard those pixels during 1K normalization.
            # Never reduce the original: the cap limits enlargement only.
            factor = max(1.0, min(factor, self.max_output_edge / max(image.size)))
        target = (round(image.width * factor), round(image.height * factor))
        resized = image.resize(target, Image.Resampling.BICUBIC)
        tensor = to_tensor(resized).unsqueeze(0).to(self.device) * 2 - 1
        # The VAE compresses by 8 and DiT patches by patch_size. Padding only to
        # 8 leaves odd latent dimensions, which fails PatchEmbed on real photos.
        multiple = 8 * args.patch_size
        pad_w, pad_h = (-target[0]) % multiple, (-target[1]) % multiple
        if pad_w or pad_h:
            mode = "reflect" if target[0] > pad_w and target[1] > pad_h else "replicate"
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode=mode)
        # fork_rng keeps the pipeline's random state isolated from teacher sampling.
        devices = [torch.device(self.device)] if torch.device(self.device).type == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.inference_mode(), torch.autocast(
                device_type=torch.device(self.device).type, dtype=self.dtype,
                enabled=self.precision != "fp32"):
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
            if not torch.isfinite(output).all():
                raise RuntimeError("VOSR2 produced non-finite pixels; try precision=fp32")
            output = output[..., :target[1], :target[0]]
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
