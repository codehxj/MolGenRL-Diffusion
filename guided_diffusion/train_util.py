import copy
import functools
import os
from torchvision import transforms as tfms
import random
import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from mpi4py import MPI
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision.transforms import transforms

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
import time
from diffusers.models import AutoencoderKL
# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.

from PIL import Image
from omegaconf import OmegaConf
from torchvision.transforms import transforms as tfms
import numpy as np
from vqgan import VQModel
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        #batch,
        batch_size,
        microbatch,
        lr,
        #context,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        #.batch=batch
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        #self.context=context
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model
        self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to("cpu")
        # vqgan_ckpt_path = 'vqgan_jax_strongaug.ckpt'
        # config = OmegaConf.load('vqgan.yaml').model
        # self.vqgan = VQModel(ddconfig=config.params.ddconfig,
        #                 n_embed=config.params.n_embed,
        #                 embed_dim=config.params.embed_dim,
        #                 ckpt_path=vqgan_ckpt_path)

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            start_time = time.perf_counter()
            #batch, cond = next(self.data)
            #batch=self.batch_1

            self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to("cpu")
            batch = Image.open("data/mountain_256.png").convert('RGB')  #加载图像
            init_image = tfms.ToTensor()(batch).unsqueeze(0) * 2.0 - 1.0    #转换成张量，并将像素值缩放到-1到1之间
            batch = init_image.to(device="cpu")
            batch = self.vae.encode(batch).latent_dist.sample() * 0.18215   #乘0.18215，减小像素值的幅度


            # temp = transforms.ToPILImage()(batch[0])
            # temp.save("data/img128_4.png")
            #real = tv.transforms.ToTensor()(Image.open(args.data_dir))[None]


            # batch = Image.open("data/mountain_256.png").convert('RGB')
            # x_0 = tfms.ToTensor()(batch).unsqueeze(0)
            # x_0= x_0.to(device="cpu")
            # batch,_,_ = self.vqgan.encode(x_0)

            cond={}
            self.run_step(batch,cond)          #反向传播，优化
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                #self.save()
                th.save(self.model.state_dict(), f"mountain_chpt/256_try_vqgan_mountain_{self.step}.pth")
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print("Elapsed time: ", elapsed_time)
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
             #self.save()
             th.save(self.model.state_dict(), f"mountain_chpt/256_try_vqgan_mountain_{self.step}.pth")


    def run_step(self, batch, cond):

        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)  #true
        if took_step:
            self._update_ema()
        # self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):

        self.mp_trainer.zero_grad()
        #writer=SummaryWriter("logs")
        #print(batch.shape[0])

        for i in range(0, batch.shape[0], self.microbatch):
            #print(i)
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            # random resize
            # if MPI.COMM_WORLD.Get_rank() == 0:
            #     curr_h = round(micro.shape[2] * random.uniform(0.75, 1.25))        #round(micro.shape[2] * random.uniform(0.75, 1.25))
            #     curr_w = curr_h                  #round(micro.shape[3] * random.uniform(0.75, 1.25))
            #     curr_h, curr_w = 2 * (curr_h // 2), 2 * (curr_w // 2)
            #     MPI.COMM_WORLD.bcast((curr_h, curr_w))
            # else:
            #     curr_h, curr_w = MPI.COMM_WORLD.bcast(None)
            # #writer.add_images("sindiffusion",micro,i)
            # micro = F.interpolate(micro, (curr_h, curr_w), mode="bicubic")

            #micro = self.vae.encode( micro).latent_dist.sample() * 0.18215

            #curr_h, curr_w = 4 * (micro.shape[2] // 4), 4 * (micro.shape[2] // 4)
            #MPI.COMM_WORLD.bcast((curr_h, curr_w))
            #micro = F.interpolate(micro, (curr_h, curr_w), mode="bicubic")

            #micro_cond= self.context
            #writer.add_images("sindiffusion", micro, i)
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                self.context,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:

                losses = compute_losses()

            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()
            #losses=training_losses(self, self.ddp_model,  micro , t, self.context ,model_kwargs=None, noise=None)

            # if isinstance(self.schedule_sampler, LossAwareSampler):
            #     self.schedule_sampler.update_with_local_losses(
            #         t, losses["loss"].detach()
            #     )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):                 #保存模型
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
