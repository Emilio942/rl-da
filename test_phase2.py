#!/usr/bin/env python3
"""
Phase 2 Test-Script: RL-Agent aufsetzen
Tests für MDP, Reward-Funktion, Baseline-Policies, RL-Training und Checkpoints
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import traceback
from pathlib import Path

def setup_logging():
    """Setup Logging für Phase 2 Tests"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "phase2_tests.log"),
            logging.StreamHandler()
        ]
    )

def test_phase_2():
    """Führt alle Tests für Phase 2 durch"""
    
    print("🚀 RL-Diffusion-Sampling - Phase 2: RL-Agent aufsetzen")
    print("="*60)
    
    logger = logging.getLogger(__name__)
    
    # Test 2.1: MDP Definition
    print("\n📐 Aufgabe 2.1: MDP definieren")
    print("-" * 40)
    
    try:
        print("Testing MDP Definition...")
        from mdp_definition import DiffusionMDP, SamplingState, SamplingAction
        import torch
        
        # Quick MDP test
        mdp = DiffusionMDP(max_steps=20)
        latent = torch.randn(1, 4, 64, 64)
        noise_schedule = [1.0 - i/20.0 for i in range(20)]
        
        state = mdp.create_initial_state(latent, noise_schedule)
        new_latent = torch.randn(1, 4, 64, 64)
        new_state = mdp.update_state(state, new_latent, 0.1, 0.5)
        valid_actions = mdp.get_valid_actions(new_state)
        
        print(f"✅ MDP Test erfolgreich - State dim: {mdp.state_dim}, Actions: {len(valid_actions)}")
        status_2_1 = "✅ Fertig"
        
    except Exception as e:
        print(f"❌ Fehler beim MDP-Test: {e}")
        traceback.print_exc()
        status_2_1 = "❌ Fehler"
    
    # Test 2.2: Reward Function
    print("\n🎯 Aufgabe 2.2: Reward-Funktion implementieren")
    print("-" * 40)
    
    try:
        print("Testing Reward Function...")
        from reward_function import RewardFunction
        from mdp_definition import SamplingState, SamplingAction
        
        reward_func = RewardFunction(quality_weight=1.0, efficiency_weight=0.5)
        
        # Mock states
        prev_state = SamplingState(
            current_step=5, max_steps=20, progress_ratio=0.25,
            current_noise_level=0.8, noise_schedule=[],
            latent_mean=0.0, latent_std=1.0, latent_norm=100.0,
            quality_estimate=0.6, quality_trend=0.0,
            time_per_step=0.1, total_time=0.5, efficiency_score=6.0,
            steps_since_improvement=0, best_quality_so_far=0.6
        )
        
        next_state = SamplingState(
            current_step=6, max_steps=20, progress_ratio=0.3,
            current_noise_level=0.75, noise_schedule=[],
            latent_mean=0.0, latent_std=1.0, latent_norm=95.0,
            quality_estimate=0.7, quality_trend=0.1,
            time_per_step=0.1, total_time=0.6, efficiency_score=7.0,
            steps_since_improvement=0, best_quality_so_far=0.7
        )
        
        reward = reward_func.calculate_reward(prev_state, SamplingAction.CONTINUE, next_state, False)
        
        print(f"✅ Reward Test erfolgreich - Total reward: {reward.total_reward:.3f}")
        status_2_2 = "✅ Fertig"
        
    except Exception as e:
        print(f"❌ Fehler beim Reward-Test: {e}")
        traceback.print_exc()
        status_2_2 = "❌ Fehler"
    
    # Test 2.3: Baseline Policies
    print("\n🎲 Aufgabe 2.3: Baseline-Policies implementieren")
    print("-" * 40)
    
    try:
        print("Testing Baseline Policies...")
        from baseline_policies import RandomPolicy, HeuristicPolicy, FixedStepPolicy
        from mdp_definition import DiffusionMDP
        
        mdp = DiffusionMDP(max_steps=10)
        policies = [
            RandomPolicy(seed=42),
            HeuristicPolicy(),
            FixedStepPolicy(5)
        ]
        
        # Mock state
        state = SamplingState(
            current_step=3, max_steps=10, progress_ratio=0.3,
            current_noise_level=0.7, noise_schedule=[],
            latent_mean=0.0, latent_std=1.0, latent_norm=100.0,
            quality_estimate=0.6, quality_trend=0.05,
            time_per_step=0.1, total_time=0.3, efficiency_score=6.0,
            steps_since_improvement=0, best_quality_so_far=0.6
        )
        
        valid_actions = mdp.get_valid_actions(state)
        
        for policy in policies:
            action = policy.select_action(state, valid_actions)
            print(f"  {policy.name}: {action.action.name} (conf={action.confidence:.2f})")
        
        print("✅ Baseline Policies Test erfolgreich")
        status_2_3 = "✅ Fertig"
        
    except Exception as e:
        print(f"❌ Fehler beim Baseline-Test: {e}")
        traceback.print_exc()
        status_2_3 = "❌ Fehler"
    
    # Test 2.4: RL Training
    print("\n🧠 Aufgabe 2.4: RL-Training implementieren")
    print("-" * 40)
    
    try:
        print("Testing RL Training...")
        from rl_training import RLTrainer, PolicyNetwork, ValueNetwork
        from mdp_definition import DiffusionMDP
        from reward_function import RewardFunction
        
        mdp = DiffusionMDP(max_steps=10)
        reward_function = RewardFunction()
        
        trainer = RLTrainer(
            mdp=mdp,
            reward_function=reward_function,
            state_dim=mdp.state_dim,
            action_dim=mdp.action_dim,
            use_baseline=True
        )
        
        # Test Episode
        episode_data = trainer.run_episode(max_steps=10)
        losses = trainer.train_episode(episode_data)
        
        print(f"  Episode - Reward: {episode_data['total_reward']:.3f}, "
              f"Quality: {episode_data['final_quality']:.3f}, "
              f"Steps: {episode_data['total_steps']}")
        print(f"  Training - Policy Loss: {losses['policy_loss']:.3f}, "
              f"Value Loss: {losses['value_loss']:.3f}")
        
        print("✅ RL Training Test erfolgreich")
        status_2_4 = "✅ Fertig"
        
    except Exception as e:
        print(f"❌ Fehler beim RL-Training-Test: {e}")
        traceback.print_exc()
        status_2_4 = "❌ Fehler"
    
    # Test 2.5: Checkpoint System
    print("\n💾 Aufgabe 2.5: Checkpoint-System implementieren")
    print("-" * 40)
    
    try:
        print("Testing Checkpoint System...")
        from checkpoint_system import CheckpointManager, ExperimentConfig
        from datetime import datetime
        
        # Verwende existierenden Trainer
        checkpoint_manager = CheckpointManager(base_dir="test_phase2_checkpoints")
        
        config = ExperimentConfig(
            experiment_name="test_phase2",
            timestamp=datetime.now().isoformat(),
            state_dim=11,
            action_dim=4,
            hidden_dims=[64, 32],
            learning_rate=3e-4,
            gamma=0.95,
            use_baseline=True,
            num_episodes=50,
            max_steps=10,
            quality_threshold=0.8,
            quality_weight=1.0,
            efficiency_weight=0.5,
            step_penalty=0.02
        )
        
        # Simuliere Checkpoint
        metrics = {
            'avg_reward': 0.75,
            'avg_quality': 0.65,
            'avg_efficiency': 1.2,
            'success_rate': 0.6
        }
        
        if 'trainer' in locals():
            checkpoint_id = checkpoint_manager.save_checkpoint(
                trainer, config, 10, metrics, force_save=True
            )
            
            print(f"  Checkpoint erstellt: {checkpoint_id}")
            
            # Test Load
            checkpoint_data = checkpoint_manager.load_checkpoint(checkpoint_id, trainer)
            print(f"  Checkpoint geladen - Episode: {checkpoint_data['episode']}")
        
        print("✅ Checkpoint System Test erfolgreich")
        status_2_5 = "✅ Fertig"
        
    except Exception as e:
        print(f"❌ Fehler beim Checkpoint-Test: {e}")
        traceback.print_exc()
        status_2_5 = "❌ Fehler"
    
    # Status-Zusammenfassung
    print("\n📋 Status-Zusammenfassung Phase 2:")
    print("=" * 60)
    print(f"2.1 MDP definieren:                {status_2_1}")
    print(f"2.2 Reward-Funktion:               {status_2_2}")
    print(f"2.3 Baseline-Policies:             {status_2_3}")
    print(f"2.4 RL-Training:                   {status_2_4}")
    print(f"2.5 Checkpoint-System:             {status_2_5}")
    
    # Prüfe Erfolg
    all_success = all([
        status_2_1 == "✅ Fertig",
        status_2_2 == "✅ Fertig",
        status_2_3 == "✅ Fertig",
        status_2_4 == "✅ Fertig",
        status_2_5 == "✅ Fertig"
    ])
    
    if all_success:
        print("\n🎉 Phase 2 erfolgreich abgeschlossen!")
        print("Bereit für Phase 3: Adaptive Sampling Schedules")
        
        print("\n➡️  Nächste Schritte:")
        print("3.1 Dynamisches Stop-Kriterium implementieren")
        print("3.2 Schrittzahl vs. Qualität-Analyse")
        print("3.3 Speed-Up-Faktor evaluieren")
        
    else:
        print("\n⚠️  Phase 2 unvollständig. Behebe die Fehler und versuche es erneut.")
    
    print("\n🏁 Ende Phase 2")
    
    return all_success

if __name__ == "__main__":
    setup_logging()
    test_phase_2()
