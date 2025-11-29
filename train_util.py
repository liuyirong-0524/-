import copy
import functools
import os
import sys
import blobfile as bf
import numpy as np
import torch
import torch as th
import torch.distributed as dist
from scipy.stats import truncnorm
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from collections import Counter
import time
from torch.utils.data import DataLoader
from guided_diffusion import dist_util, logger
#from . import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
#from .fp16_util import MixedPrecisionTrainer
from guided_diffusion.nn import update_ema
#from .nn import update_ema
from guided_diffusion.resample import LossAwareSampler, UniformSampler
#from .resample import LossAwareSampler, UniformSampler
# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0
import torch.nn.functional as F

def count_parameters(model):
    """
    Functionality:
    Calculates the total number of trainable parameters in a neural network model.
    A parameter is considered trainable if its 'requires_grad' attribute is True.

    Inputs:
    :param model: The neural network model whose parameters are to be counted 

    Outputs:
    :return: The total count of trainable parameters in the model.
    :rtype: int
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    """
    Functionality:
    Creates an instance of the Truncated Normal Distribution (truncnorm) from scipy.stats.

    Inputs:
    :param mean: The mean of the underlying normal distribution.
    :type mean: float
    :param sd: The standard deviation of the underlying normal distribution.
    :type sd: float
    :param low: The lower bound for truncation.
    :type low: float
    :param upp: The upper bound for truncation.
    :type upp: float

    Outputs:
    :return: A scipy.stats.truncnorm frozen distribution object, which can be used for sampling.
    :rtype: scipy.stats._distn_infrastructure.rv_frozen
    """
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def np2tensor(arr):
    """
    Functionality:
    Converts a NumPy array or list into a floating-point PyTorch Tensor.

    Inputs:
    :param arr: The input array or list to be converted.
    :type arr: Union[np.ndarray, list]

    Outputs:
    :return: The converted floating-point PyTorch Tensor.
    :rtype: torch.Tensor
    """
    return th.from_numpy(np.array(arr)).float()

import time
import functools
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader


class TrainLoop:

    
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        SR_times=10,
        epoch=1000
    ):
        """
        Functionality:
        Initializes the training loop, setting up the model, diffusion process, data, 
        optimizer, mixed-precision training, logging, and checkpointing configurations.

        Inputs:
        :param model: The neural network model to be trained.
        :param diffusion: The Diffusion Model object, containing the forward/reverse process logic.
        :param data: The training dataset.
        :param batch_size: The total batch size used for training across all devices.
        :param microbatch: The microbatch size used for gradient accumulation (must be <= batch_size).
        :param lr: The initial learning rate.
        :param ema_rate: The EMA (Exponential Moving Average) decay rate(s), as a float or comma-separated string.
        :param log_interval: The step interval for outputting logs.
        :param save_interval: The step interval for saving checkpoints.
        :param resume_checkpoint: The file path to a checkpoint to resume training from.
        :param use_fp16: Flag to enable FP16 mixed-precision training.
        :param fp16_scale_growth: The growth factor for the FP16 loss scaler.
        :param schedule_sampler: The strategy for sampling time steps (e.g., LossAwareSampler or UniformSampler).
        :param weight_decay: The weight decay coefficient for the optimizer.
        :param lr_anneal_steps: The total number of steps over which to anneal the learning rate.
        :param SR_times: Super-resolution related parameter, potentially used for data preprocessing.
        :param epoch: The total number of training epochs.
        """
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.SR_times = SR_times
        self.epoch = epoch
        
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size
        
        self.ema_rate = (
            [ema_rate] if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )
        
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        
        self.sync_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.sync_cuda else "cpu")
        
        self._load_parameters()
        
        print(f'Number of Trainable Parameters: {count_parameters(model)}')
        
        self.opt = AdamW(
            self.mp_trainer.master_params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        self.ddp_model = self.model
        
        self.data_loader = self._create_dataloader_with_prefetch()
    
    def _load_parameters(self):
        """
        Functionality:
        Loads the model parameters from a specified checkpoint file and sets the resume step.
        """
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"Loading model from checkpoint: {resume_checkpoint}...")
            
            state_dict = torch.load(resume_checkpoint, map_location=self.device)
            self.model.load_state_dict(state_dict)
    
    def _load_optimizer_state(self):
        """
        Functionality:
        Loads the optimizer's state (e.g., AdamW momentum buffers) from a corresponding 
        checkpoint file to resume training smoothly.
        """
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint),
            f"opt{self.resume_step:06}.pt"
        )
        
        if bf.exists(opt_checkpoint):
            logger.log(f"Loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = torch.load(opt_checkpoint, map_location=self.device)
            self.opt.load_state_dict(state_dict)
    
    def _create_base_loader(self, deterministic=False):
        """
        Functionality:
        Creates a basic PyTorch DataLoader from the dataset and wraps it into an 
        infinite-loop generator, which yields batches continuously.

        Inputs:
        :param deterministic: Whether to use deterministic loading (i.e., disable shuffling).
        :type deterministic: bool

        Outputs:
        :return: A generator that infinitely yields batches from the DataLoader.
        :rtype: Generator
        """
        num_workers = min(32, os.cpu_count() // 4)
        
        loader = DataLoader(
            self.data,
            batch_size=self.batch_size,
            shuffle=not deterministic,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=8,
            persistent_workers=True,
        )
        
        while True:
            yield from loader
    
    def _create_dataloader_with_prefetch(self, max_prefetch=4):
        """
        Functionality:
        Creates a data loader generator with prefetching capabilities. It maintains 
        a prefetch queue to ensure CPU prepares the next batch while the GPU is processing 
        the current one, improving training throughput.

        Inputs:
        :param max_prefetch: The maximum length of the prefetch queue (number of batches to load ahead).
        :type max_prefetch: int

        Outputs:
        :return: A generator that sequentially yields prefetched batches in the format (SR_ST, model_kwargs).
        :rtype: Generator
        """
        base_loader = self._create_base_loader()
        prefetch_queue = []
        
        for _ in range(max_prefetch):
            try:
                batch_data = self._fetch_and_prepare_batch(base_loader)
                if batch_data:
                    prefetch_queue.append(batch_data)
            except StopIteration:
                break
        
        while prefetch_queue:
            yield prefetch_queue.pop(0)
            
            try:
                batch_data = self._fetch_and_prepare_batch(base_loader)
                if batch_data:
                    prefetch_queue.append(batch_data)
            except StopIteration:
                continue
    
    def _fetch_and_prepare_batch(self, loader):
        """
        Functionality:
        Retrieves the next raw batch from the base loader and processes it into the 
        (target image, conditioning dictionary) format required for the model.

        Inputs:
        :param loader: The base data loader iterator (infinite generator).
        :type loader: Generator

        Outputs:
        :return: A tuple containing (SR_ST, model_kwargs), where SR_ST is the target image 
                 and model_kwargs is the dictionary of conditioning information. 
                 Returns None if the data iteration has stopped.
        :rtype: Optional[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]
        """
        try:
            SR_ST, spot_ST, WSI_5120, WSI_320, gene_class, Gene_index_map, metadata_feature, scale_gt, co_expression, WSI_mask, sc, scgpt, pre_he = next(loader)

   
            model_kwargs = {
                "low_res": spot_ST,
                "WSI_5120": WSI_5120,
                "WSI_320": WSI_320,
                "gene_class": gene_class,
                "Gene_index_map": Gene_index_map,
                "metadata_feature": metadata_feature,
                "scale_gt": scale_gt,
                "co_expression": co_expression,
                "WSI_mask": WSI_mask,
                "sc": sc,
                "scgpt": scgpt,
                "pre_he": pre_he
            }
            
            return (SR_ST, model_kwargs)
        except StopIteration:
            return None
    
    def _calculate_training_ratio(self, current_step, total_steps):
        """
        Functionality:
        Dynamically calculates a data usage ratio based on the current training progress. 
        This is typically used in diffusion models to progressively increase the data's 
        contribution to stabilize training in early stages.

        Inputs:
        :param current_step: The current number of training steps completed.
        :type current_step: int
        :param total_steps: The total number of planned training steps.
        :type total_steps: int

        Outputs:
        :return: The calculated data usage ratio, ranging from 0.1 to 1.0.
        :rtype: float
        """
        ratio_schedule = [
            (0.1, 0.1),
            (0.2, 0.1),
            (0.3, 0.3),
            (0.4, 0.5),
            (0.5, 0.7),
            (0.6, 0.9),
            (0.7, 1.0),
            (0.8, 1.0),
            (0.9, 1.0),
            (1.0, 1.0),
        ]
        
        progress = current_step / total_steps
        
        for threshold, ratio in ratio_schedule:
            if progress < threshold:
                return ratio
        
        return 1.0
    
    def _format_time(self, seconds):
        """
        Functionality:
        Formats a given duration in seconds into a human-readable HH:MM:SS string.

        Inputs:
        :param seconds: The time duration in seconds.
        :type seconds: float or int

        Outputs:
        :return: The formatted time string (HH:MM:SS).
        :rtype: str
        """
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
    
    def run_loop(self):
        """
        Functionality:
        The main training loop. It iterates over training steps, controls logging, 
        learning rate annealing, and checkpoint saving until the total number of 
        iterations is reached.
        """
        # Calculate total iterations
        total_iterations = int(99 * self.epoch / self.batch_size) + 1
        
        # Initialize timing
        loop_start_time = time.time()
        
        logger.log(f"Starting training loop: {total_iterations} iterations")
        
        while self.step <= total_iterations:
            step_start_time = time.time()
            
            # Calculate current training ratio
            ratio = self._calculate_training_ratio(self.step, total_iterations)
            
            # Get batch and run training step
            batch, cond = next(self.data_loader)
            self.run_step(batch, cond, ratio)
            
            current_time = time.time()
            step_duration = current_time - step_start_time
            total_duration = current_time - loop_start_time
            
            remaining_steps = total_iterations - self.step
            avg_time_per_step = total_duration / (self.step + 1)
            estimated_remaining = avg_time_per_step * remaining_steps
            
            if self.step % self.log_interval == 0:
                logger.log(
                    f"Step {self.step}/{total_iterations} | "
                    f"Ratio: {ratio:.1f} | "
                    f"Elapsed: {self._format_time(total_duration)} | "
                    f"Remain: {self._format_time(estimated_remaining)} | "
                    f"Speed: {step_duration:.2f}s/step"
                )
                logger.dumpkvs()
            
            if self.step % self.save_interval == 0 and self.step != 0:
                self.save()
                
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            
            self.step += 1
        
        if (self.step - 1) % self.save_interval != 0:
            self.save()
        
        logger.log("Training completed!")
    
    def run_step(self, batch, cond, ratio):
        """
        Functionality:
        Executes a single, complete training step, including the forward-backward pass, 
        optimizer update, learning rate annealing, and logging.

        Inputs:
        :param batch: The current input image batch [B, C, H, W].
        :type batch: torch.Tensor
        :param cond: The conditioning information dictionary for the batch.
        :type cond: Dict[str, torch.Tensor]
        :param ratio: The data usage ratio for the current step.
        :type ratio: float

        Outputs:
        None (Updates model parameters and optimizer state).
        """
        # Forward and backward pass
        self.forward_backward(batch, cond, ratio)
        
        # Optimizer step
        took_step = self.mp_trainer.optimize(self.opt)
        
        # Learning rate annealing
        self._anneal_lr()
        
        # Log statistics
        self.log_step()
    
    def forward_backward(self, batch, cond, ratio):
        """
        Functionality:
        Performs the forward and backward passes, incorporating micro-batching 
        for gradient accumulation. It calculates the loss for sampled time steps 
        and updates the schedule sampler.

        Inputs:
        :param batch: The full input image batch [B, C, H, W].
        :type batch: torch.Tensor
        :param cond: The full conditioning dictionary.
        :type cond: Dict[str, torch.Tensor]
        :param ratio: The data usage ratio.
        :type ratio: float
        """
        self.mp_trainer.zero_grad()
        
        # Micro-batching loop
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i:i + self.microbatch].to(self.device)
            
            if self.SR_times == 5:
                micro = F.interpolate(micro, size=(256, 256))
        
            micro_cond = {
                k: v[i:i + self.microbatch].to(self.device)
                for k, v in cond.items()
            }
            
            last_batch = (i + self.microbatch) >= batch.shape[0]
            
            t, weights = self.schedule_sampler.sample(micro.shape[0], self.device)
            
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                ratio,
                model_kwargs=micro_cond,
            )
            
            losses = compute_losses()
            
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )
            
            loss = (losses["loss"] * weights).mean()
            
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            
            self.mp_trainer.backward(loss)
    
    def _anneal_lr(self):
        """
        Functionality:
        Performs linear learning rate (LR) annealing, reducing the LR from its initial 
        value to zero over the total number of annealing steps (self.lr_anneal_steps).
        """
        if not self.lr_anneal_steps:
            return
        
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr
    
    def log_step(self):
        """
        Functionality:
        Logs the current global step number and the total number of samples processed 
        since the beginning of training (or resumption).
        """
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
    
    def save(self):
        """
        Functionality:
        Saves the current model parameters as a checkpoint file. 
        It saves the state dictionary of the master parameters.
        """
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            
            if not rate:
                filename = f"model{(self.step + self.resume_step):06d}.pt"
                logger.log(f"Saving model checkpoint: {filename}")
                
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    torch.save(state_dict, f)
        
        save_checkpoint(0, self.mp_trainer.master_params)
    
    def save_optimizer_state(self):
        """
        Functionality:
        Saves the state dictionary of the current optimizer (self.opt) to a file.
        """
        filename = f"opt{(self.step + self.resume_step):06d}.pt"
        logger.log(f"Saving optimizer state: {filename}")
        
        with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
            torch.save(self.opt.state_dict(), f)

def parse_resume_step_from_filename(filename):
    """
    Functionality:
    Parses the training step number (NNNNNN) from a checkpoint filename 
    in the format "path/to/modelNNNNNN.pt".

    Inputs:
    :param filename: The path or filename of the checkpoint.
    :type filename: str

    Outputs:
    :return: The extracted training step number, or 0 if parsing fails.
    :rtype: int
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
    """
    Functionality:
    Retrieves the directory path designated for saving logs and checkpoints.

    Outputs:
    :return: The directory path for logs and checkpoints.
    :rtype: str
    """
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically 
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    """
    Functionality:
    Constructs and checks for the existence of an Exponential Moving Average (EMA) 
    checkpoint file based on the main checkpoint path, step number, and EMA rate.

    Inputs:
    :param main_checkpoint: The file path of the main model checkpoint.
    :type main_checkpoint: Optional[str]
    :param step: The training step corresponding to the checkpoint.
    :type step: int
    :param rate: The decay rate of the EMA.
    :type rate: float

    Outputs:
    :return: The path to the EMA checkpoint file if it exists, otherwise None.
    :rtype: Optional[str]
    """
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    """
    Functionality:
    Logs the mean of each loss term in the provided dictionary to the logging system.

    Inputs:
    :param diffusion: The diffusion model object, used for potential time step analysis (currently commented out).
    :type diffusion: object
    :param ts: The sampled timesteps for the current batch.
    :type ts: torch.Tensor
    :param losses: A dictionary where keys are loss names and values are the corresponding loss tensors.
    :type losses: Dict[str, torch.Tensor]
    """
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        # for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
        #     quartile = int(4 * sub_t / diffusion.num_timesteps)
        #     logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
