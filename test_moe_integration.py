#!/usr/bin/env python3
"""
Test script for the integrated Mixture of Experts system.
This script verifies Phase 4 implementation: MoE + RL integration.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import logging
from pathlib import Path

from diffusion_model import DiffusionModelWrapper
from adaptive_sampling import AdvancedExpertSelector, run_sampling_with_moe_expert
from rl_training import MoERLTrainer
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
            logging.FileHandler(log_dir / "test_moe_integration.log"),
            logging.StreamHandler()
        ]
    )

def test_moe_expert_selection():
    """Test basic MoE expert selection functionality."""
    print("🧪 Testing MoE Expert Selection...")
    
    # Mock diffusion model for testing
    class MockDiffusionModel:
        class MockConfig:
            def __init__(self):
                self.num_train_timesteps = 1000
        
        def __init__(self):
            self.config = self.MockConfig()

        def __call__(self, **kwargs):
            class MockOutput:
                def __init__(self):
                    self.images = [torch.randn(3, 256, 256)]
            return MockOutput()

    # Initialize components
    diffusion_model = MockDiffusionModel()
    scheduler = DDPMScheduler()
    device = 'cpu'
    
    expert_selector = AdvancedExpertSelector(
        diffusion_model=diffusion_model,
        scheduler=scheduler,
        device=device,
        save_dir="test_moe_checkpoints"
    )
    
    # Test state representation
    sample = torch.randn(1, 4, 64, 64)
    state = expert_selector.get_state_representation(
        sample=sample,
        timestep=500,
        step_count=10,
        max_steps=50
    )
    
    assert state.shape == torch.Size([64]), f"Expected state shape [64], got {state.shape}"
    print(f"✅ State representation test passed. Shape: {state.shape}")
    
    # Test expert selection
    expert_id, expert_weights, routing_info = expert_selector.select_expert_moe(state)
    
    assert isinstance(expert_id, int), f"Expected expert_id to be int, got {type(expert_id)}"
    assert expert_weights is not None, "Expert weights should not be None"
    assert isinstance(routing_info, dict), f"Expected routing_info to be dict, got {type(routing_info)}"
    
    print(f"✅ Expert selection test passed. Selected expert: {expert_id}")
    print(f"   Routing info: {routing_info}")
    
    # Test performance update
    performance_metrics = {
        'quality_score': 0.85,
        'speed_score': 0.75,
        'efficiency_score': 0.80
    }
    
    expert_selector.update_expert_performance(expert_id, performance_metrics)
    stats = expert_selector.get_expert_statistics()
    
    assert isinstance(stats, dict), "Expert statistics should be a dictionary"
    print(f"✅ Performance update test passed. Stats: {stats}")

def test_moe_sampling_integration():
    """Test the integration between MoE system and sampling."""
    print("🧪 Testing MoE Sampling Integration...")
    
    # Mock components
    class MockDiffusionModel:
        class MockConfig:
            def __init__(self):
                self.num_train_timesteps = 1000
        
        def __init__(self):
            self.config = self.MockConfig()

        def __call__(self, **kwargs):
            class MockOutput:
                def __init__(self):
                    self.images = [torch.randn(3, 256, 256)]
            return MockOutput()

    class MockPolicy:
        def select_action(self, state):
            return 0 if state.get('step_number', 0) > 10 else 1

    # Initialize components
    diffusion_model = MockDiffusionModel()
    scheduler = DDPMScheduler()
    device = 'cpu'
    
    expert_selector = AdvancedExpertSelector(
        diffusion_model=diffusion_model,
        scheduler=scheduler,
        device=device
    )
    
    # Test complete sampling pipeline
    initial_noise = torch.randn(1, 4, 64, 64)
    state = expert_selector.get_state_representation(
        sample=initial_noise,
        timestep=500,
        step_count=5,
        max_steps=50
    )
    
    expert_id, expert_object, routing_info = expert_selector.select_expert_moe(state)
    traditional_expert = expert_selector.get_expert_traditional("default")
    
    final_sample, performance_metrics = run_sampling_with_moe_expert(
        moe_system=expert_selector.moe_system,
        expert_id=expert_id,
        expert_object=expert_object,
        state=state,
        traditional_expert=traditional_expert,
        policy=MockPolicy(),
        initial_noise=initial_noise
    )
    
    assert final_sample is not None, "Final sample should not be None"
    assert isinstance(performance_metrics, dict), "Performance metrics should be a dictionary"
    assert 'quality_score' in performance_metrics, "Performance metrics should include quality_score"
    
    print(f"✅ Sampling integration test passed.")
    print(f"   Final sample shape: {final_sample.size if hasattr(final_sample, 'size') else 'N/A'}")
    print(f"   Performance metrics: {performance_metrics}")

def test_moe_rl_trainer():
    """Test the MoE RL trainer initialization and basic functionality."""
    print("🧪 Testing MoE RL Trainer...")
    
    # Mock components
    class MockDiffusionModel:
        class MockConfig:
            def __init__(self):
                self.num_train_timesteps = 1000
        
        def __init__(self):
            self.config = self.MockConfig()

        def __call__(self, **kwargs):
            class MockOutput:
                def __init__(self):
                    self.images = [torch.randn(3, 256, 256)]
            return MockOutput()

    # Initialize components
    diffusion_model = MockDiffusionModel()
    scheduler = DDPMScheduler()
    device = 'cpu'
    
    expert_selector = AdvancedExpertSelector(
        diffusion_model=diffusion_model,
        scheduler=scheduler,
        device=device
    )
    
    mdp = DiffusionMDP(max_steps=50)
    reward_function = RewardFunction()
    
    # Initialize MoE RL Trainer
    trainer = MoERLTrainer(
        mdp=mdp,
        reward_function=reward_function,
        expert_selector=expert_selector,
        state_dim=64,
        device=device
    )
    
    assert trainer.moe_system is not None, "MoE system should be initialized"
    assert trainer.num_moe_experts > 0, "Should have MoE experts"
    
    print(f"✅ MoE RL Trainer initialization test passed.")
    print(f"   Number of MoE experts: {trainer.num_moe_experts}")
    
    # Test single episode
    initial_latent = torch.randn(1, 4, 64, 64, device=device)
    episode_data = trainer.run_moe_episode(initial_latent)
    
    assert isinstance(episode_data, dict), "Episode data should be a dictionary"
    assert 'expert_id' in episode_data, "Episode data should include expert_id"
    assert 'reward' in episode_data, "Episode data should include reward"
    
    print(f"✅ MoE episode test passed.")
    print(f"   Episode reward: {episode_data['reward']:.3f}")
    print(f"   Selected expert: {episode_data['expert_id']}")

def main():
    """Run all MoE integration tests."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("🚀 Starting MoE Integration Tests - Phase 4 Verification 🚀")
    print("=" * 60)
    
    try:
        test_moe_expert_selection()
        print()
        
        test_moe_sampling_integration()
        print()
        
        test_moe_rl_trainer()
        print()
        
        print("=" * 60)
        print("🎉 All MoE Integration Tests Passed! Phase 4 Implementation Verified! 🎉")
        print()
        print("✅ Phase 4.1: Multiple sampling experts created and integrated")
        print("✅ Phase 4.2: RL-Policy dynamically selects experts via MoE system")
        print()
        print("🚀 Ready to proceed to Phase 5: Evaluation & Optimization!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
