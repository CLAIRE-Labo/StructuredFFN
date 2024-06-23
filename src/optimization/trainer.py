import random
import os
import glob
import wandb
import shutil
import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict
from omegaconf import open_dict
import torch.distributed as dist
from src.modules import get_ckpt_name


class WandbLog:
    def __init__(self, config, metric, x_axis="epoch"):
        self.config = config
        for k, v in metric.items():
            if k == x_axis:
                wandb.define_metric(x_axis)
            else:
                wandb.define_metric(k, step_metric=x_axis)

    def record(self, item):
        wandb.log(item)


class TrainableModel:

    def __init__(self, config):
        self.config = config
        self.epoch = -1
        self.step = -1
        self.max_epoch = self.config.optimization.max_epoch
        self.max_step = None  # define in specific Trainer

        # gpu setting
        self.gpu_id = int(os.getenv("RANK", -1))
        self.device = (
            torch.device("cuda", self.gpu_id)
            if self.gpu_id != -1
            else torch.device("cuda")
        )
        self.ngpus = dist.get_world_size() if self.gpu_id != -1 else 1
        print("The device is {} out of {}".format(self.device, self.ngpus))

        self.global_batch_size = getattr(
            self.config.optimization,
            "global_batch_size",
            self.config.data.train.train_batch,
        )
        assert (
            self.global_batch_size % (self.ngpus * self.config.data.train.train_batch)
            == 0
        )
        self.gradient_accumulation_steps = self.global_batch_size // (
            self.ngpus * self.config.data.train.train_batch
        )

        self.log_interval = getattr(self.config.optimization, "log_interval", False)
        self.check_gradient_norm = getattr(
            self.config.optimization, "check_gradient_norm", False
        )
        self.check_weight_norm = getattr(
            self.config.optimization, "check_weight_norm", False
        )
        self.gradient_clipping = getattr(
            self.config.optimization, "gradient_clipping", False
        )
        self.special_training = (
            self.config.optimization.training.name == "self_guided_training"
        )
        # save
        self.is_save_checkpoint = getattr(
            self.config.optimization, "save_checkpoint", False
        )
        self.is_load_checkpoint = getattr(
            self.config.optimization, "load_checkpoint", False
        )
        self.load_save_mode = getattr(
            self.config.optimization, "load_save_mode", "epoch"
        )

    def prepare_load_save(
        self,
    ):
        if self.is_save_checkpoint or self.is_load_checkpoint:
            long_name = get_ckpt_name(self.config) + "-" + str(self.special_training)
            if self.special_training:
                long_name += (
                    "-"
                    + self.config.optimization.training.kwargs.mode
                    + "-"
                    + str(self.config.optimization.training.kwargs.reduce_flop)
                )
            self.save_dir = os.path.join(self.config.optimization.save_dir, long_name)
            self.save_dir = os.path.join(
                self.save_dir,
                str(self.config.optimization.optimizer.kwargs.lr).replace(".", "x"),
            )
            if self.load_save_mode == "epoch":
                self.save_interval = self.max_epoch // 10
            elif self.load_save_mode == "step":
                self.save_interval = self.max_step // 10
            else:
                raise NotImplementedError
            print(
                "plan to save or load checkpoint in {} for each {} in the mode {}".format(
                    self.save_dir, self.save_interval, self.load_save_mode
                )
            )
            if not self.is_load_checkpoint:
                shutil.rmtree(self.save_dir)
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

    def set_gradient_clipping(
        self,
    ):
        if self.gradient_clipping is not False:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clipping
            )

    def get_info(
        self,
    ):
        nparam = self.get_nparam()
        nflops = self.model.get_flops(
            self.global_batch_size,
            self.block_size,
        )  # we consider all the matrix multiplication including the final logits in the model
        total_flops = nflops * self.max_step
        if self.special_training:
            guide_params = sum(
                [
                    p.guide_linear.numel()
                    for p in self.model.modules()
                    if hasattr(p, "guide_linear")
                ]
            )
            # print("the number of guide parameters are {:.2f}".format(guide_params))
            guide_flops = (
                2 * guide_params * self.global_batch_size * self.block_size
            )  # addition and multiplication
            total_flops -= guide_flops * (self.max_step - self.guided_steps)
            if self.config.optimization.training.kwargs.reduce_flop:
                total_flops -= 0.5 * guide_flops * self.guided_steps
        print("The total parameter is {:.2f} M".format(nparam / 10**6))
        print(
            "FLOPs information: flops per forward step {:.2f}T, total flops {:.2f}T".format(
                nflops / 10**12,
                total_flops * 3 / 10**12,  # backward and forward
            )
        )
        nparam_mlp = self.model.get_params_mlp()
        nflops_mlp = self.model.get_flops_mlp(
            self.global_batch_size,
            self.block_size,
        )
        print(
            "MLP information: params {:.2f}M, flops per step {:.2f}T".format(
                nparam_mlp / 10**6,
                nflops_mlp / 10**12,
            )
        )

        print(self.model)

    def get_nparam(
        self,
    ):
        self.nparam = sum(param.numel() for param in self.model.parameters())
        return self.nparam

    def set_seed(self, seed):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def set_self_guided_training(
        self,
    ):
        self.repeat_steps = 0
        self.guided_steps = 0
        if self.special_training:
            self.guided_steps = int(
                self.max_step * self.config.optimization.training.kwargs.max_step_ratio
            )
            with open_dict(self.config.method.kwargs) as f:
                f.training.enabled = True
                f.training.scheduler = (
                    self.config.optimization.training.kwargs.scheduler
                )
                f.training.max_step = self.guided_steps
                f.training.reduce_flop = (
                    self.config.optimization.training.kwargs.reduce_flop
                )
            if self.config.optimization.training.kwargs.mode == "fixedflop":
                self.repeat_steps = self.guided_steps
                self.max_step += self.repeat_steps
        elif self.config.method.name != "linear":
            with open_dict(self.config.method.kwargs) as f:
                f.training.enabled = False

    def close_self_guided_training(
        self,
    ):
        from src.modules.layer.basiclinear import BasicLinear

        self.special_training = False
        for name, module in self.model.named_modules():
            if isinstance(module, BasicLinear):
                module.training_config.enabled = False

    def get_optimize_param(
        self,
    ):
        params = [{"params": self.model.parameters()}]
        return params

    def save_checkpoint(self, **resume_kwargs):
        # save checkpoint by epoch
        if not self.is_save_checkpoint or self.gpu_id not in [-1, 0]:
            return
        if self.load_save_mode == "epoch":
            cur = self.epoch
            cur_max = self.max_epoch
        elif self.load_save_mode == "step":
            cur = self.step
            cur_max = self.max_step
        if (cur + 1) % self.save_interval == 0 or cur + 1 == cur_max:
            ckpt_path = os.path.join(
                self.save_dir,
                f"{cur}.pth",
            )
            ckpt = {
                "model": (
                    self.model.module.state_dict()
                    if self.gpu_id == 0
                    else self.model.state_dict()
                ),
                self.load_save_mode: cur,
                "config": self.config,
                "nparam": self.nparam,
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": (
                    self.lr_scheduler.state_dict()
                    if getattr(self, "lr_scheduler", None)
                    else None
                ),
                "resume_kwargs": resume_kwargs,
            }
            torch.save(ckpt, ckpt_path)

    def load_checkpoint(self):
        if not self.is_load_checkpoint:
            return {}

        def find_latest_checkpoint():
            checkpoint_files = glob.glob(
                os.path.join(
                    self.save_dir,
                    f"*.pth",
                )
            )
            if not checkpoint_files:
                return None

            latest_checkpoint_file = max(checkpoint_files, key=os.path.getctime)
            return latest_checkpoint_file

        latest_checkpoint = find_latest_checkpoint()
        if latest_checkpoint is not None:
            print("load checkpoint from {}".format(latest_checkpoint))
            ckpt = torch.load(latest_checkpoint, map_location=self.device)
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            if getattr(self, "lr_scheduler", None):
                self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            if self.load_save_mode == "epoch":
                self.epoch = ckpt["epoch"]
            elif self.load_save_mode == "step":
                self.step = ckpt["step"]
            return ckpt["resume_kwargs"]
        return {}
