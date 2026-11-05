import os
import gc
import torch
from contextlib import contextmanager
from typing import List
from dataclasses import dataclass
from app.utils import rgba_to_rgb
from custum_3d_diffusion.trainings.config_classes import ExprimentConfig, TrainerSubConfig
from custum_3d_diffusion import modules
from custum_3d_diffusion.custum_modules.unifield_processor import AttnConfig, ConfigurableUNet2DConditionModel
from custum_3d_diffusion.trainings.base import BasicTrainer
from custum_3d_diffusion.trainings.utils import load_config


@dataclass
class FakeAccelerator:
    device: torch.device = torch.device("cuda")


def get_diffusion_device() -> str:
    """GPU for the multiview/normal diffusion models. Defaults to the last GPU so the
    first one is left for the SD refiner, upsampler and mesh ops. Override with UNIQUE3D_DIFFUSION_DEVICE."""
    env = os.environ.get("UNIQUE3D_DIFFUSION_DEVICE")
    if env:
        return env
    return f"cuda:{max(torch.cuda.device_count() - 1, 0)}"


# With a single GPU the pipelines are parked in CPU RAM and moved to the GPU only while running.
SWAP_TO_CPU = os.environ.get("UNIQUE3D_SWAP_TO_CPU", "1" if torch.cuda.device_count() <= 1 else "0") == "1"


@contextmanager
def gpu_scope(pipeline):
    if not SWAP_TO_CPU:
        yield
        return
    pipeline.to(pipeline._gpu_device)
    try:
        yield
    finally:
        pipeline.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()


def init_trainers(cfg_path: str, weight_dtype: torch.dtype, extras: dict, device="cuda"):
    accelerator = FakeAccelerator(device=torch.device(device))
    cfg: ExprimentConfig = load_config(ExprimentConfig, cfg_path, extras)
    init_config: AttnConfig = load_config(AttnConfig, cfg.init_config)
    configurable_unet = ConfigurableUNet2DConditionModel(init_config, weight_dtype)
    configurable_unet.enable_xformers_memory_efficient_attention()
    trainer_cfgs: List[TrainerSubConfig] = [load_config(TrainerSubConfig, trainer) for trainer in cfg.trainers]
    trainers: List[BasicTrainer] = [modules.find(trainer.trainer_type)(accelerator, None, configurable_unet, trainer.trainer, weight_dtype, i) for i, trainer in enumerate(trainer_cfgs)]
    return trainers, configurable_unet

from app.utils import make_image_grid, split_image
def process_image(function, img, guidance_scale=2., merged_image=False, remove_bg=True):
    from rembg import remove
    if remove_bg:
        img = remove(img)
    img = rgba_to_rgb(img)
    if merged_image:
        img = split_image(img, rows=2)
    images = function(
        image=img,
        guidance_scale=guidance_scale,
    )
    if len(images) > 1:
        return make_image_grid(images, rows=2)
    else:
        return images[0]


def process_text(trainer, pipeline, img, guidance_scale=2.):
    pipeline.cfg.validation_prompts = [img]
    titles, images = trainer.batched_validation_forward(pipeline, guidance_scale=[guidance_scale])
    return images[0]


def load_pipeline(config_path, ckpt_path, pipeline_filter=lambda x: True, weight_dtype = torch.bfloat16, device=None):
    training_config = config_path
    load_from_checkpoint = ckpt_path
    extras = []
    device = device or get_diffusion_device()
    trainers, configurable_unet = init_trainers(training_config, weight_dtype, extras, device)
    shared_modules = dict()
    for trainer in trainers:
        shared_modules = trainer.init_shared_modules(shared_modules)

    if load_from_checkpoint is not None:
        state_dict = torch.load(load_from_checkpoint)
        configurable_unet.unet.load_state_dict(state_dict, strict=False)
    # Move unet, vae and text_encoder to device and cast to weight_dtype
    configurable_unet.unet.to(device, dtype=weight_dtype)

    pipeline = None
    trainer_out = None
    for trainer in trainers:
        if pipeline_filter(trainer.cfg.trainer_name):
            pipeline = trainer.construct_pipeline(shared_modules, configurable_unet.unet)
            pipeline.set_progress_bar_config(disable=False)
            trainer_out = trainer
    pipeline = pipeline.to(device)
    pipeline._gpu_device = device
    if SWAP_TO_CPU:
        pipeline.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
    return trainer_out, pipeline
