#!/usr/bin/env python3
"""
Main script for RL-based adaptive diffusion sampling with advanced Mixture of Experts.
Phase 4: Integration of sophisticated MoE system with RL-guided expert selection.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from pathlib import Path
import logging

# Local Imports
from diffusion_model import DiffusionModelWrapper
from adaptive_sampling import AdvancedExpertSelector
from rl_training import RLTrainer, MoERLTrainer
from mdp_definition import DiffusionMDP
from reward_function import RewardFunction
from diffusers import DDPMScheduler

def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "main_moe.log"),
            logging.StreamHandler()
        ]
    )

def main():
    """Main function to run the MoE-enhanced RL training."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting RL Diffusion Sampling with Advanced Mixture of Experts 🚀")

    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_id = "google/ddpm-cat-256"
    state_dim = 64  # Updated to match MoE state representation
    num_episodes = 1000
    use_moe_system = True  # Toggle between traditional and MoE system

    # 1. Initialize Components
    logger.info("Initializing components...")
    diffusion_model_wrapper = DiffusionModelWrapper(model_id, device=device)
    scheduler = DDPMScheduler.from_pretrained(model_id)
    
    # Initialize advanced expert selector with MoE system
    expert_selector = AdvancedExpertSelector(
        diffusion_model=diffusion_model_wrapper.pipeline, 
        scheduler=scheduler,
        device=device,
        save_dir="moe_checkpoints"
    )
    
    mdp = DiffusionMDP(max_steps=50)
    reward_function = RewardFunction()

    if use_moe_system:
        # 2. Initialize MoE RL Trainer
        logger.info("Initializing MoE RL Trainer...")
        trainer = MoERLTrainer(
            mdp=mdp,
            reward_function=reward_function,
            expert_selector=expert_selector,
            state_dim=state_dim,
            device=device
        )

        # 3. Run MoE Training
        logger.info(f"Starting MoE training for {num_episodes} episodes...")
        trainer.train_moe(num_episodes=num_episodes, save_interval=100)
        
        # 4. Evaluate MoE System
        logger.info("Evaluating MoE system performance...")
        expert_stats = expert_selector.get_expert_statistics()
        logger.info(f"Final Expert Statistics: {expert_stats}")
        
    else:
        # Fallback to traditional RL training
        logger.info("Initializing traditional RL Trainer...")
        trainer = RLTrainer(
            mdp=mdp,
            reward_function=reward_function,
            experts=expert_selector.traditional_experts,
            state_dim=11,  # Traditional state dim
            device=device
        )

        logger.info(f"Starting traditional training for {num_episodes} episodes...")
        trainer.train(num_episodes=num_episodes)

    logger.info("✅ Training finished!")

if __name__ == "__main__":
    main()
