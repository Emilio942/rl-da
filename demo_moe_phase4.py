#!/usr/bin/env python3
"""
Demo script showing the Mixture of Experts system in action.
This demonstrates Phase 4 capabilities: Dynamic expert selection with performance tracking.
"""

import torch
import numpy as np
import time
from pathlib import Path

# Mock setup for demonstration
class MockDiffusionModel:
    class MockConfig:
        def __init__(self):
            self.num_train_timesteps = 1000
    
    def __init__(self):
        self.config = self.MockConfig()

    def __call__(self, **kwargs):
        # Simulate different processing times based on num_inference_steps
        steps = kwargs.get('num_inference_steps', 1000)
        time.sleep(max(0.1, steps / 10000))  # Simulate computation time
        
        class MockOutput:
            def __init__(self):
                self.images = [torch.randn(3, 256, 256)]
        return MockOutput()

def main():
    print("🎯 MoE Diffusion Sampling Demo - Phase 4 Complete!")
    print("=" * 60)
    
    # Import our modules
    from adaptive_sampling import AdvancedExpertSelector
    from diffusers import DDPMScheduler
    
    # Initialize components
    diffusion_model = MockDiffusionModel()
    scheduler = DDPMScheduler()
    device = 'cpu'
    
    expert_selector = AdvancedExpertSelector(
        diffusion_model=diffusion_model,
        scheduler=scheduler,
        device=device
    )
    
    print(f"📋 Initialized MoE system with {len(expert_selector.moe_system.experts)} experts:")
    for i, profile in enumerate(expert_selector.moe_system.expert_profiles):
        print(f"   Expert {i}: {profile.name} ({profile.expert_type.value})")
    
    print("\n🔄 Running sampling episodes with dynamic expert selection...")
    
    # Run multiple episodes to show dynamic selection
    for episode in range(5):
        print(f"\n📄 Episode {episode + 1}:")
        
        # Create different scenarios by varying the noise pattern
        if episode % 2 == 0:
            # Complex pattern
            initial_noise = torch.randn(1, 4, 64, 64) * 1.5
            scenario = "complex"
        else:
            # Simple pattern
            initial_noise = torch.randn(1, 4, 64, 64) * 0.8
            scenario = "simple"
        
        # Get state representation
        state = expert_selector.get_state_representation(
            sample=initial_noise,
            timestep=500 + episode * 100,  # Vary timestep
            step_count=episode * 3,
            max_steps=50
        )
        
        # Select expert using MoE
        start_time = time.time()
        expert_id, expert_object, routing_info = expert_selector.select_expert_moe(
            state, 
            context={'scenario': scenario, 'episode': episode}
        )
        selection_time = time.time() - start_time
        
        expert_name = expert_selector.moe_system.expert_profiles[expert_id].name
        confidence = routing_info['confidence']
        
        print(f"   Selected: {expert_name} (ID: {expert_id})")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Selection time: {selection_time*1000:.2f}ms")
        print(f"   Scenario: {scenario}")
        
        # Simulate performance metrics based on expert type and scenario
        if "Speed" in expert_name:
            quality = np.random.uniform(0.7, 0.85)
            speed = np.random.uniform(0.9, 1.0)
        elif "Quality" in expert_name:
            quality = np.random.uniform(0.9, 1.0)
            speed = np.random.uniform(0.6, 0.8)
        else:  # Balanced or other
            quality = np.random.uniform(0.8, 0.9)
            speed = np.random.uniform(0.75, 0.9)
        
        efficiency = (quality + speed) / 2
        
        performance_metrics = {
            'quality_score': quality,
            'speed_score': speed,
            'efficiency_score': efficiency
        }
        
        # Update expert performance
        expert_selector.update_expert_performance(expert_id, performance_metrics)
        
        print(f"   Performance: Q={quality:.3f}, S={speed:.3f}, E={efficiency:.3f}")
    
    print("\n📊 Final Expert Statistics:")
    stats = expert_selector.get_expert_statistics()
    
    print(f"   Total episodes: {stats['total_usage']}")
    print("   Expert Performance:")
    
    for expert_name, perf in stats['expert_performance'].items():
        if perf['total_uses'] > 0:
            print(f"     {expert_name}:")
            print(f"       Uses: {perf['total_uses']}")
            print(f"       Avg Quality: {perf['avg_quality']:.3f}")
            print(f"       Avg Speed: {perf['avg_speed']:.3f}")
            print(f"       Success Rate: {perf['success_rate']:.3f}")
    
    print("\n   Usage Distribution:")
    for expert_name, count in stats['usage_distribution'].items():
        percentage = (count / stats['total_usage']) * 100
        print(f"     {expert_name}: {count} times ({percentage:.1f}%)")
    
    print("\n🏆 Expert Rankings:")
    for i, (expert_name, metrics) in enumerate(stats['expert_rankings']):
        print(f"   {i+1}. {expert_name}: Score {metrics['composite_score']:.3f}")
    
    print("\n" + "=" * 60)
    print("🎉 Phase 4 Demo Complete!")
    print("✅ Dynamic expert selection working")
    print("✅ Performance tracking active")
    print("✅ Adaptive routing based on context")
    print("✅ Ready for Phase 5: Evaluation & Optimization")

if __name__ == "__main__":
    main()
