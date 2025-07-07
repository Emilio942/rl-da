"""
RL-Training for adaptive diffusion sampling.
This version is updated to select an expert sampler instead of step-by-step actions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import logging
from collections import deque
import time
import pickle
import json
from pathlib import Path

from mdp_definition import DiffusionMDP
from reward_function import RewardFunction
from expert_samplers import BaseSampler
from adaptive_sampling import AdvancedExpertSelector, run_sampling_with_moe_expert
from mixture_of_experts import MixtureOfExperts

logger = logging.getLogger(__name__)

@dataclass
class Experience:
    """A single experience for the replay buffer."""
    state: np.ndarray
    action: int  # Here, action is the index of the chosen expert
    reward: float
    next_state: Optional[np.ndarray]
    done: bool
    log_prob: float

class PolicyNetwork(nn.Module):
    """
    Policy Network for selecting an expert sampler.
    """
    
    def __init__(self, 
                 state_dim: int, 
                 num_experts: int, 
                 hidden_dims: List[int] = [128, 64]):
        super().__init__()
        
        self.state_dim = state_dim
        self.num_experts = num_experts
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_experts))
        
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)
    
    def select_expert(self, state: torch.Tensor) -> Tuple[int, float]:
        """
        Selects an expert based on the current state.
        
        Returns:
            (expert_index, log_probability)
        """
        with torch.no_grad():
            logits = self.forward(state)
            probs = F.softmax(logits, dim=-1)
            action_dist = torch.distributions.Categorical(probs)
            expert_index = action_dist.sample()
            log_prob = action_dist.log_prob(expert_index)
            
            return expert_index.item(), log_prob.item()

class ValueNetwork(nn.Module):
    """
    Value Network for Actor-Critic. Estimates the value of a state.
    """
    
    def __init__(self, state_dim: int, hidden_dims: List[int] = [128, 64]):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

class RLTrainer:
    """
    RL-Trainer that learns to select the best expert sampler.
    """
    
    def __init__(self,
                 mdp: DiffusionMDP,
                 reward_function: RewardFunction,
                 experts: Dict[str, BaseSampler],
                 state_dim: int,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.95,
                 use_baseline: bool = True,
                 device: str = 'auto'):
        
        self.mdp = mdp
        self.reward_function = reward_function
        self.experts = list(experts.values()) # The RL agent will select from this list
        self.num_experts = len(self.experts)
        self.gamma = gamma
        self.use_baseline = use_baseline
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.policy_net = PolicyNetwork(state_dim, self.num_experts).to(self.device)
        self.value_net = ValueNetwork(state_dim).to(self.device) if use_baseline else None
        
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        if self.value_net:
            self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        
        self.experience_buffer = deque(maxlen=10000)
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        self.value_losses = []
        
        logger.info(f"RL-Trainer for expert selection initialized - Device: {self.device}, Experts: {self.num_experts}")
    
    
    
    def run_episode(self, initial_latent: torch.Tensor) -> Dict[str, Any]:
        """
        Runs a single episode where the policy selects an expert and receives a reward.
        """
        # The state for expert selection can be based on the initial noise or other context
        # For now, we use a simplified state representation from the MDP.
        initial_state = self.mdp.create_initial_state(initial_latent, [1.0]) # Dummy schedule
        state_vector = torch.FloatTensor(
            self.mdp.get_state_representation(initial_state)
        ).unsqueeze(0).to(self.device)

        # Select an expert
        expert_index, log_prob = self.policy_net.select_expert(state_vector)
        selected_expert = self.experts[expert_index]

        # The sub-policy for the default sampler (if applicable)
        # In this setup, the main policy only chooses the expert.
        # The DefaultAdaptiveSampler has its own internal logic (or a fixed sub-policy).
        sub_policy = self._get_sub_policy_for_expert(selected_expert)

        # Run the sampling process with the chosen expert
        start_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else time.time()
        final_sample = selected_expert.sample(sub_policy, initial_latent)
        end_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else time.time()

        # Measure performance
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            computation_time = start_time.elapsed_time(end_time) / 1000.0
        else:
            computation_time = end_time - start_time

        # For now, we need a placeholder for quality and reward
        # This should be replaced with a proper reward function call.
        final_quality = 0.8 # Placeholder
        reward = final_quality - (computation_time * 0.1) # Simple reward

        episode_result = {
            'total_reward': reward,
            'final_quality': final_quality,
            'computation_time': computation_time,
            'expert_index': expert_index,
            'log_prob': log_prob,
            'state': state_vector.cpu().numpy().squeeze(),
        }

        return episode_result

    def _get_sub_policy_for_expert(self, expert: BaseSampler):
        """
        Returns a sub-policy to be used by an expert sampler, if needed.
        For now, this is a simple placeholder.
        """
        from expert_samplers import DefaultAdaptiveSampler
        if isinstance(expert, DefaultAdaptiveSampler):
            # The adaptive sampler needs a policy to make step-wise decisions.
            # We can use a simple fixed policy for now.
            class FixedPolicy:
                def select_action(self, state):
                    return 1 # Always continue
            return FixedPolicy()
        return None # Other experts might not need a policy
    
    def calculate_returns(self, rewards: List[float]) -> List[float]:
        """Berechnet diskontierte Returns"""
        returns = []
        R = 0
        
        for reward in reversed(rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)
        
        return returns
    
    def train_on_episode(self, episode_data: Dict[str, Any]):
        """
        Trains the networks based on a single episode's data.
        """
        state = torch.FloatTensor(episode_data['state']).unsqueeze(0).to(self.device)
        action = torch.LongTensor([episode_data['expert_index']]).to(self.device)
        reward = episode_data['total_reward']

        # --- Value Network Update ---
        advantage = reward
        if self.use_baseline and self.value_net:
            value = self.value_net(state).squeeze()
            advantage = reward - value.detach()  # Advantage is reward - baseline
            
            # Update value network
            value_loss = F.mse_loss(value, torch.tensor(reward, device=self.device, dtype=value.dtype))
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            self.value_losses.append(value_loss.item())

        # --- Policy Network Update ---
        # Re-calculate log_prob with gradients to ensure the graph is connected
        logits = self.policy_net(state)
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze()

        # The advantage should not have gradients flowing back from the policy loss.
        # Using .detach() on the value prediction ensures this.
        policy_loss = -(action_log_prob * advantage).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        self.policy_losses.append(policy_loss.item())

    def train(self, 
              num_episodes: int = 1000,
              log_interval: int = 100,
              save_interval: int = 500):
        """
        Main training loop for expert selection.
        """
        logger.info(f"Starting RL training for {num_episodes} episodes...")

        for episode in range(num_episodes):
            # Create a new initial latent vector for each episode
            initial_latent = torch.randn(1, 4, 64, 64, device=self.device)

            # Run a full episode with expert selection
            episode_data = self.run_episode(initial_latent)

            # Perform training update
            self.train_on_episode(episode_data)

            # Logging
            self.episode_rewards.append(episode_data['total_reward'])
            if episode % log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-log_interval:])
                logger.info(f"Episode {episode}: Avg Reward: {avg_reward:.3f}, Expert: {episode_data['expert_index']}")

            # Save checkpoint
            if episode % save_interval == 0 and episode > 0:
                self.save_checkpoint(f"expert_checkpoint_ep_{episode}")

        logger.info("Training finished!")
    
    def save_checkpoint(self, filename: str, checkpoint_dir: str = "checkpoints"):
        """Speichert Trainings-Checkpoint"""
        
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(exist_ok=True)
        
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'policy_losses': self.policy_losses,
            'training_config': {
                'gamma': self.gamma,
                'use_baseline': self.use_baseline,
                'device': str(self.device)
            }
        }
        
        if self.value_net:
            checkpoint['value_net_state_dict'] = self.value_net.state_dict()
            checkpoint['value_optimizer_state_dict'] = self.value_optimizer.state_dict()
            checkpoint['value_losses'] = self.value_losses
        
        torch.save(checkpoint, checkpoint_path / f"{filename}.pt")
        logger.info(f"Checkpoint gespeichert: {checkpoint_path / filename}.pt")
    
    def load_checkpoint(self, filename: str, checkpoint_dir: str = "checkpoints"):
        """Lädt Trainings-Checkpoint"""
        
        checkpoint_path = Path(checkpoint_dir) / f"{filename}.pt"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint nicht gefunden: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        
        if self.value_net and 'value_net_state_dict' in checkpoint:
            self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
            self.value_losses = checkpoint.get('value_losses', [])
        
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])
        self.policy_losses = checkpoint.get('policy_losses', [])
        
        logger.info(f"Checkpoint geladen: {checkpoint_path}")

class MoERLTrainer(RLTrainer):
    """
    Enhanced RL-Trainer that uses the Mixture of Experts system for dynamic expert selection.
    """
    
    def __init__(self,
                 mdp: DiffusionMDP,
                 reward_function: RewardFunction,
                 expert_selector: AdvancedExpertSelector,
                 state_dim: int,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.95,
                 use_baseline: bool = True,
                 device: str = 'auto'):
        
        self.expert_selector = expert_selector
        self.moe_system = expert_selector.moe_system
        
        # Initialize parent with traditional experts for compatibility
        traditional_experts = expert_selector.traditional_experts
        super().__init__(mdp, reward_function, traditional_experts, state_dim, 
                        learning_rate, gamma, use_baseline, device)
        
        # Override num_experts to use MoE experts
        self.num_moe_experts = len(self.moe_system.experts)
        
        # Enhanced policy network for MoE routing
        self.moe_policy_net = PolicyNetwork(state_dim, self.num_moe_experts).to(self.device)
        self.moe_policy_optimizer = optim.Adam(self.moe_policy_net.parameters(), lr=learning_rate)
        
        # Performance tracking
        self.expert_performance_history = []
        self.moe_routing_history = []
        
        logger.info(f"MoE RL-Trainer initialized - MoE Experts: {self.num_moe_experts}")
    
    def run_moe_episode(self, initial_latent: torch.Tensor) -> Dict[str, Any]:
        """
        Run an episode using the MoE system for expert selection.
        """
        # Create initial state representation
        initial_state = self.mdp.create_initial_state(initial_latent, [1.0])
        moe_state = self.expert_selector.get_state_representation(
            sample=initial_latent,
            timestep=500,  # Mid-range timestep as example
            step_count=0,
            max_steps=50
        )
        
        # Select expert using MoE system
        expert_id, expert_object, routing_info = self.expert_selector.select_expert_moe(
            moe_state, context={'episode_type': 'training'}
        )
        
        # Select corresponding traditional expert for actual sampling
        traditional_expert_names = list(self.expert_selector.traditional_experts.keys())
        if expert_id < len(traditional_expert_names):
            traditional_expert_name = traditional_expert_names[expert_id % len(traditional_expert_names)]
        else:
            traditional_expert_name = "default"  # Fallback
        
        selected_traditional_expert = self.expert_selector.get_expert_traditional(traditional_expert_name)
        
        # Run sampling with MoE guidance
        start_time = time.time()
        final_sample, performance_metrics = run_sampling_with_moe_expert(
            moe_system=self.moe_system,
            expert_id=expert_id,
            expert_object=expert_object,
            state=moe_state,
            traditional_expert=selected_traditional_expert,
            policy=self._get_sub_policy_for_expert(selected_traditional_expert),
            initial_noise=initial_latent
        )
        computation_time = time.time() - start_time
        
        # Calculate reward using performance metrics
        reward = self._calculate_moe_reward(performance_metrics, computation_time, routing_info)
        
        # Update expert performance in MoE system
        self.expert_selector.update_expert_performance(expert_id, performance_metrics)
        
        # Store episode data
        episode_data = {
            'expert_id': expert_id,
            'routing_info': routing_info,
            'performance_metrics': performance_metrics,
            'reward': reward,
            'computation_time': computation_time,
            'expert_object': str(expert_object),  # Convert to string for storage
            'state': moe_state.detach().cpu().numpy()
        }
        
        self.expert_performance_history.append(episode_data)
        
        return episode_data
    
    def _calculate_moe_reward(self, 
                             performance_metrics: Dict[str, float], 
                             computation_time: float,
                             routing_info: Dict[str, Any]) -> float:
        """
        Calculate reward based on performance metrics and routing efficiency.
        """
        # Base reward from performance metrics
        quality_weight = 0.4
        speed_weight = 0.3
        efficiency_weight = 0.2
        routing_weight = 0.1
        
        base_reward = (
            quality_weight * performance_metrics.get('quality_score', 0.0) +
            speed_weight * performance_metrics.get('speed_score', 0.0) +
            efficiency_weight * performance_metrics.get('efficiency_score', 0.0)
        )
        
        # Routing efficiency bonus
        routing_efficiency = routing_info.get('confidence', 0.5)
        routing_bonus = routing_weight * routing_efficiency
        
        # Time penalty (encourage faster sampling)
        time_penalty = max(0, (computation_time - 1.0) * 0.1)  # Penalty for taking more than 1 second
        
        total_reward = base_reward + routing_bonus - time_penalty
        return np.clip(total_reward, -1.0, 1.0)
    
    def train_moe(self, num_episodes: int, save_interval: int = 100):
        """
        Train the MoE system with RL.
        """
        logger.info(f"Starting MoE RL training for {num_episodes} episodes")
        
        for episode in range(num_episodes):
            # Generate random initial latent for each episode
            initial_latent = torch.randn(1, 4, 64, 64, device=self.device)
            
            # Run episode with MoE
            episode_data = self.run_moe_episode(initial_latent)
            
            # Update policy (simplified - could be more sophisticated)
            if episode > 0 and episode % 10 == 0:  # Update every 10 episodes
                self._update_moe_policy()
            
            # Logging
            if episode % 50 == 0:
                avg_reward = np.mean([ep['reward'] for ep in self.expert_performance_history[-50:]])
                expert_stats = self.expert_selector.get_expert_statistics()
                logger.info(f"Episode {episode}: Avg Reward: {avg_reward:.3f}, Expert Stats: {expert_stats}")
            
            # Save checkpoints
            if episode % save_interval == 0 and episode > 0:
                self.save_moe_checkpoint(episode)
        
        logger.info("MoE RL training completed!")
    
    def _update_moe_policy(self):
        """
        Update the MoE policy based on recent performance.
        """
        if len(self.expert_performance_history) < 10:
            return
        
        # Simple policy update based on expert performance
        recent_episodes = self.expert_performance_history[-10:]
        
        # Calculate expert preference based on recent rewards
        expert_rewards = {}
        for episode in recent_episodes:
            expert_id = episode['expert_id']
            reward = episode['reward']
            if expert_id not in expert_rewards:
                expert_rewards[expert_id] = []
            expert_rewards[expert_id].append(reward)
        
        # The MoE system itself learns from the performance updates
        # Additional policy learning could be implemented here
        
        logger.debug(f"Updated MoE policy based on {len(recent_episodes)} recent episodes")
    
    def save_moe_checkpoint(self, episode: int):
        """
        Save MoE system and training state.
        """
        checkpoint_dir = Path("moe_training_checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save MoE system state
        self.expert_selector.save_moe_state()
        
        # Save training data
        checkpoint_data = {
            'episode': episode,
            'performance_history': self.expert_performance_history,
            'policy_state_dict': self.moe_policy_net.state_dict(),
            'optimizer_state_dict': self.moe_policy_optimizer.state_dict()
        }
        
        checkpoint_path = checkpoint_dir / f"moe_training_checkpoint_ep{episode}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        logger.info(f"Saved MoE checkpoint at episode {episode}")

# Test-Funktion
def test_rl_training():
    """Testet das RL-Training"""
    logger.info("Teste RL-Training...")
    
    # Setup
    mdp = DiffusionMDP(max_steps=20)
    reward_function = RewardFunction()
    
    # Mock experts for testing
    from expert_samplers import MockSampler
    experts = {
        "mock_expert_1": MockSampler(sampler_type="mock_1"),
        "mock_expert_2": MockSampler(sampler_type="mock_2")
    }
    
    trainer = RLTrainer(
        mdp=mdp,
        reward_function=reward_function,
        experts=experts,
        state_dim=mdp.state_dim,
        use_baseline=True
    )
    
    # Kurzes Training für Test
    trainer.train(num_episodes=50, log_interval=10)
    
    # Test Episode
    initial_latent = torch.randn(1, 4, 64, 64)
    episode_data = trainer.run_episode(initial_latent)
    logger.info(f"Test Episode - Reward: {episode_data['total_reward']:.3f}, "
               f"Quality: {episode_data['final_quality']:.3f}, "
               f"Expert: {episode_data['expert_index']}")
    
    # Checkpoint-Test
    trainer.save_checkpoint("test_checkpoint")
    
    logger.info("✅ RL-Training erfolgreich getestet!")
    return True

if __name__ == "__main__":
    test_rl_training()
