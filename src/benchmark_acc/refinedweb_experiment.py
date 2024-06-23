import os
import sys
import math
import numpy as np
import time
from tqdm import tqdm


project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

print(project_root)
sys.path.insert(0, project_root)
import wandb
import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from src.optimization import (
    get_optimizer,
    get_lr_scheduler,
    trainer,
)

from src.modules import get_model, update_ratio


class RefinedWebGPT(trainer.TrainableModel):
    # NLP tasks are dominated by step rather than epoch, because we need to consider gradient accumulation
    def __init__(self, config):
        super().__init__(config)
        # get dataset
        self.set_seed(self.config.optimization.seed)
        # get data files
        self.train_file_path = os.path.join(
            self.config.data.train.path,
            self.config.data.tokenizer.name + "-train-tmp.bin",
        )
        self.val_file_path = os.path.join(
            self.config.data.test.path,
            self.config.data.tokenizer.name + "-val-tmp.bin",
        )
        self.block_size = min(
            self.config.data.block_size, self.config.data.tokenizer.model_max_length
        )
        validate_tokens = 512000 * 1024
        self.validate_samples = validate_tokens // self.block_size
        assert (
            self.validate_samples % (self.ngpus * self.config.data.test.test_batch) == 0
        )
        assert self.gpu_id != -1, "we only support torchrun in job submission"

        # get metric
        self.max_step = int(
            self.config.optimization.max_tokens
            / self.global_batch_size
            / self.block_size
        )
        self.set_self_guided_training()
        self.config.optimization.lr_scheduler.kwargs.T_max = self.max_step
        if self.gpu_id in [-1, 0]:
            self.metric = {
                "train_loss": 0.0,
                "train_ppl": 0.0,
                "test_loss": 0.0,
                "test_ppl": 0.0,
                "step": 0,
                "lr": 0.0,
                "fwd+bwd": 0.0,
            }
        # get model
        self.set_seed(self.config.optimization.seed)
        self.model = get_model(self.config, self.device)
        self.get_info()

        # get optimizer
        self.optimizer = get_optimizer(
            self.config.optimization, self.get_optimize_param()
        )
        if getattr(self.config.optimization, "lr_scheduler", None):
            self.lr_scheduler = get_lr_scheduler(
                self.config.optimization, self.optimizer
            )

        # get wandb
        if self.gpu_id in [-1, 0] and self.config.wandb_use:
            self.wandblog = trainer.WandbLog(
                self.config.wandb, self.metric, x_axis="step"
            )

        assert self.load_save_mode == "step"
        self.prepare_load_save()
        self.resume_kwargs = self.load_checkpoint()
        if self.gpu_id != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.gpu_id],
                output_device=self.gpu_id,
                find_unused_parameters=self.special_training,
            )
        if self.gpu_id in [-1, 0]:
            print(self.config)

    def get_batch(self, split, offset_row):
        if split == "train":
            arr = np.memmap(
                self.train_file_path,
                dtype=np.uint16,  # we store in 2 bytes
                mode="r",
                offset=offset_row * (self.block_size + 1) * 2,
                shape=(self.config.data.train.train_batch, (self.block_size + 1)),
            )
        elif split == "val":
            arr = np.memmap(
                self.val_file_path,
                dtype=np.uint16,  # we store in 2 bytes
                mode="r",
                offset=offset_row * (self.block_size + 1) * 2,
                shape=(self.config.data.test.test_batch, (self.block_size + 1)),
            )
        else:
            raise NotImplementedError

        x = torch.from_numpy(arr[:, :-1].astype(np.int64))
        y = torch.from_numpy(arr[:, 1:].astype(np.int64))
        x, y = x.pin_memory().to("cuda", non_blocking=True), y.pin_memory().to(
            "cuda", non_blocking=True
        )
        return x, y

    def _validate(self):
        self.model.eval()
        ddp_loss = torch.tensor(0.0).to(self.device)
        ddp_samples = torch.tensor(0).to(self.device)
        samples_per_gpu = self.validate_samples // self.ngpus
        with torch.no_grad():
            offset_row = self.gpu_id * samples_per_gpu
            for i in range(samples_per_gpu // self.config.data.test.test_batch):
                input_ids, labels = self.get_batch(
                    split="val", offset_row=offset_row + ddp_samples.item()
                )
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    loss = self.model(
                        input_ids=input_ids,
                        labels=labels,
                    )
                if i % 100 == 0 and self.gpu_id in [-1, 0]:
                    print("the loss at batch {} is {}".format(i, loss))
                ddp_loss += loss.item() * input_ids.shape[0]
                ddp_samples += input_ids.shape[0]
        print("The samples on rank {} is {}".format(self.gpu_id, ddp_samples))
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(ddp_samples, op=dist.ReduceOp.SUM)
        var_loss = (ddp_loss / ddp_samples).item()
        var_ppl = math.exp(var_loss)
        return var_loss, var_ppl

    def _train(self, resume_batch, max_step, offset_row=-1):
        if resume_batch >= max_step:
            return
        train_iterator = tqdm(
            range(resume_batch, max_step),
            desc="Steps",
            disable=self.gpu_id not in [-1, 0],
        )
        samples_per_gpu = self.global_batch_size // self.ngpus
        self.model.train()
        self.optimizer.zero_grad()
        train_loss = 0.0
        train_samples = 0
        if offset_row == -1:
            offset_row = resume_batch * self.global_batch_size
        offset_row += self.gpu_id * samples_per_gpu
        for i in train_iterator:
            torch.cuda.synchronize()
            t0 = time.time()
            train_loss = 0.0
            train_samples = 0
            for micro_step in range(self.gradient_accumulation_steps):
                input_ids, labels = self.get_batch(
                    split="train", offset_row=offset_row + train_samples
                )
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    loss = self.model(
                        input_ids=input_ids,
                        labels=labels,
                    )
                train_samples += self.config.data.train.train_batch
                train_loss += loss.item() * self.config.data.train.train_batch
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
            # finish the step
            if self.special_training:
                self.model.apply(lambda module: update_ratio(module=module))
            self.set_gradient_clipping()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            torch.cuda.synchronize()
            t2 = time.time()
            self.step += 1
            offset_row += self.global_batch_size
            if self.gpu_id in [-1, 0] and (self.step + 1) % self.log_interval == 0:
                # test_loss, test_ppl = self._test()
                # self.model.train()
                self.metric.update(
                    {
                        "train_loss": train_loss / train_samples,
                        "train_ppl": math.exp(train_loss / train_samples),
                        "step": self.step,
                        "lr": self.optimizer.param_groups[0]["lr"],
                        "fwd+bwd": (t2 - t0),
                    }
                )
                if self.config.wandb_use:
                    self.wandblog.record(self.metric)
                else:
                    print(self.metric)

            self.save_checkpoint(**{"resume_batch": i + 1})

    def train(self):
        self.set_seed(self.config.optimization.seed)
        print("***** Running training *****")
        num_examples = self.max_step * self.global_batch_size
        print("Num Examples = {}".format(num_examples))
        # Note that epoch would always be zero here
        print("Num Tokens = {}".format(num_examples * self.block_size))
        print("Num Steps = {}".format(self.max_step))
        print("Global batch size = {}".format(self.global_batch_size))
        print(
            "Gradient Accumulation steps = {}".format(self.gradient_accumulation_steps)
        )
        resume_batch = self.resume_kwargs.get("resume_batch", 0)  # next one
        print("resume from batch {}".format(resume_batch))
        # train guided steps
        self._train(resume_batch, self.guided_steps, offset_row=-1)
        self.close_self_guided_training()
        self._train(
            max(self.guided_steps, resume_batch),
            self.max_step - self.repeat_steps,
            offset_row=-1,
        )
        self._train(
            max(self.max_step - self.repeat_steps, resume_batch),
            self.max_step,
            offset_row=max(0, resume_batch + self.repeat_steps - self.max_step),
        )


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="refinedweb_config",
)
def main(config):
    OmegaConf.register_new_resolver("eval", eval)
    config.base_dir = os.path.join(
        config.base_dir, config.data.name + "_" + config.model.name
    )
    config.wandb.dir = config.base_dir
    config.wandb.dir = os.path.join(config.base_dir, config.method.name)
    gpu_id = int(os.getenv("RANK", -1))
    if gpu_id in [-1, 0] and not os.path.exists(config.wandb.dir):
        os.makedirs(config.wandb.dir)

    if gpu_id in [-1, 0] and config.wandb_use:
        wandb.init(
            config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
            entity=config.wandb.entity,
            project=config.wandb.project,
            resume=None if config.optimization.load_checkpoint else "allow",
            anonymous=config.wandb.anonymous,
            mode=config.wandb.mode,
            dir=config.wandb.dir,
        )
    if gpu_id != -1:
        dist.barrier()
    model = RefinedWebGPT(config)
    model.train()

    if gpu_id != -1:
        dist.barrier()
    print("Finish Training!")
    print("Begin to validate!")
    var_loss, var_ppl = model._validate()
    print("The var loss is {:.4f} and var ppl is {:.4f}".format(var_loss, var_ppl))
    if gpu_id in [-1, 0]:
        if config.wandb_use:
            wandb.finish()
    return var_loss, var_ppl


if __name__ == "__main__":
    gpu_id = int(os.getenv("RANK", -1))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    if gpu_id != -1:
        torch.cuda.set_device(gpu_id)
        dist.init_process_group(
            backend="nccl", world_size=world_size, rank=gpu_id, init_method="env://"
        )

    main()
