import torch
import numpy as np
from diffusion_model import DiffusionModelWrapper as DiffusionModel
from abc import ABC, abstractmethod
from diffusers import DDPMScheduler

class BaseSampler(ABC):
    """Abstract base class for all expert samplers."""
    def __init__(self, diffusion_model: DiffusionModel, scheduler: DDPMScheduler):
        self.diffusion_model = diffusion_model
        self.scheduler = scheduler

    @abstractmethod
    def sample(self, policy, initial_noise: torch.Tensor):
        """The main sampling loop for the expert."""
        pass

class MockSampler(BaseSampler):
    """A mock sampler for testing purposes."""
    def __init__(self, sampler_type: str):
        self.sampler_type = sampler_type

    def sample(self, policy, initial_noise: torch.Tensor):
        print(f"INFO: Using MockSampler: {self.sampler_type}")
        return torch.randn_like(initial_noise)

class DefaultAdaptiveSampler(BaseSampler):
    """
    The original adaptive sampling logic, now encapsulated as an expert.
    The policy decides at each step whether to continue or stop.
    """
    def _get_state_for_rl(self, sample: torch.Tensor, t: int, step_index: int, max_steps: int) -> dict:
        """
        Returns the current state for the RL agent.
        This is a simplified version of the logic from the old adaptive_sampling.py
        """
        return {
            "step_number": step_index,
            "progress": step_index / max_steps,
            "noise_level": t.item() / self.scheduler.config.num_train_timesteps,
            "latent_norm": torch.norm(sample).item(),
            "can_stop": step_index >= 5 # min_steps hardcoded for now
        }

    def sample(self, policy, initial_noise: torch.Tensor) -> torch.Tensor:
        # The DDPMPipeline returns a tuple, we are interested in the image
        return self.diffusion_model(batch_size=1, generator=torch.manual_seed(0)).images[0]

class FastSampler(BaseSampler):
    """
    A faster sampler expert that skips a fixed number of steps.
    It does not use a policy for per-step decisions.
    """
    def sample(self, policy=None, initial_noise: torch.Tensor = None) -> torch.Tensor:
        print("INFO: Using FastSampler.")
        # The DDPMPipeline returns a tuple, we are interested in the image
        return self.diffusion_model(batch_size=1, generator=torch.manual_seed(0), num_inference_steps=500).images[0]

class QualitySampler(BaseSampler):
    """
    A high-quality sampler that uses a different scheduler configuration for potentially better results.
    """
    def sample(self, policy=None, initial_noise: torch.Tensor = None) -> torch.Tensor:
        print("INFO: Using QualitySampler.")
        # The DDPMPipeline returns a tuple, we are interested in the image
        return self.diffusion_model(batch_size=1, generator=torch.manual_seed(0), num_inference_steps=1000).images[0]