"""
Baseline Policies für RL-basierte adaptive Diffusion-Sampling
Random Policy, Heuristic Policy, und Fixed Policy als Vergleich
"""

import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional, Union
from abc import ABC, abstractmethod
import logging
import random
from dataclasses import dataclass

from mdp_definition import SamplingState, SamplingAction, DiffusionMDP

logger = logging.getLogger(__name__)

@dataclass
class PolicyAction:
    """Aktion mit Wahrscheinlichkeitsverteilung"""
    action: SamplingAction
    probability: float
    confidence: float = 1.0

class BaselinePolicy(ABC):
    """Abstrakte Basis-Klasse für Baseline-Policies"""
    
    def __init__(self, name: str):
        self.name = name
        self.action_history = []
        self.state_history = []
        self.performance_stats = {
            'total_episodes': 0,
            'total_steps': 0,
            'avg_quality': 0.0,
            'avg_efficiency': 0.0,
            'success_rate': 0.0
        }
    
    @abstractmethod
    def select_action(self, state: SamplingState, valid_actions: List[SamplingAction]) -> PolicyAction:
        """Wählt eine Aktion basierend auf dem aktuellen Zustand"""
        pass
    
    def update_stats(self, episode_quality: float, episode_efficiency: float, success: bool):
        """Aktualisiert Performance-Statistiken"""
        self.performance_stats['total_episodes'] += 1
        
        # Gleitender Durchschnitt
        alpha = 0.1
        self.performance_stats['avg_quality'] = (
            alpha * episode_quality + 
            (1 - alpha) * self.performance_stats['avg_quality']
        )
        
        self.performance_stats['avg_efficiency'] = (
            alpha * episode_efficiency + 
            (1 - alpha) * self.performance_stats['avg_efficiency']
        )
        
        # Success Rate
        current_success = self.performance_stats['success_rate'] * (self.performance_stats['total_episodes'] - 1)
        self.performance_stats['success_rate'] = (
            (current_success + (1.0 if success else 0.0)) / self.performance_stats['total_episodes']
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Gibt Performance-Statistiken zurück"""
        return {
            'name': self.name,
            'performance': self.performance_stats,
            'action_distribution': self._get_action_distribution()
        }
    
    def _get_action_distribution(self) -> Dict[str, float]:
        """Berechnet Aktionsverteilung"""
        if not self.action_history:
            return {}
        
        total_actions = len(self.action_history)
        action_counts = {}
        
        for action in self.action_history:
            action_name = action.name
            action_counts[action_name] = action_counts.get(action_name, 0) + 1
        
        return {action: count / total_actions for action, count in action_counts.items()}

class RandomPolicy(BaselinePolicy):
    """
    Zufällige Policy - wählt zufällige gültige Aktionen
    """
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("RandomPolicy")
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def select_action(self, state: SamplingState, valid_actions: List[SamplingAction]) -> PolicyAction:
        """Wählt zufällige Aktion"""
        action = random.choice(valid_actions)
        probability = 1.0 / len(valid_actions)
        
        self.action_history.append(action)
        self.state_history.append(state)
        
        return PolicyAction(action=action, probability=probability, confidence=0.5)

class HeuristicPolicy(BaselinePolicy):
    """
    Heuristische Policy basierend auf einfachen Regeln
    """
    
    def __init__(self, 
                 quality_threshold: float = 0.8,
                 max_stagnation: int = 5,
                 efficiency_threshold: float = 1.0):
        super().__init__("HeuristicPolicy")
        self.quality_threshold = quality_threshold
        self.max_stagnation = max_stagnation
        self.efficiency_threshold = efficiency_threshold
    
    def select_action(self, state: SamplingState, valid_actions: List[SamplingAction]) -> PolicyAction:
        """Wählt Aktion basierend auf Heuristiken"""
        
        # Regel 1: Stoppe wenn Qualität erreicht
        if (state.quality_estimate >= self.quality_threshold and 
            SamplingAction.STOP in valid_actions):
            action = SamplingAction.STOP
            confidence = 0.9
        
        # Regel 2: Stoppe bei zu langer Stagnation
        elif (state.steps_since_improvement >= self.max_stagnation and 
              SamplingAction.STOP in valid_actions):
            action = SamplingAction.STOP
            confidence = 0.8
        
        # Regel 3: Passe Stärke an wenn Effizienz niedrig
        elif (state.efficiency_score < self.efficiency_threshold and 
              SamplingAction.ADJUST_STRENGTH in valid_actions and
              state.current_step > 3):
            action = SamplingAction.ADJUST_STRENGTH
            confidence = 0.7
        
        # Regel 4: Überspringe Schritt wenn fast fertig aber niedrige Qualität
        elif (state.progress_ratio > 0.8 and 
              state.quality_estimate < 0.5 and
              SamplingAction.SKIP_STEP in valid_actions):
            action = SamplingAction.SKIP_STEP
            confidence = 0.6
        
        # Standard: Fortsetzung
        else:
            action = SamplingAction.CONTINUE if SamplingAction.CONTINUE in valid_actions else random.choice(valid_actions)
            confidence = 0.5
        
        # Probability basierend auf Confidence
        probability = confidence if action in valid_actions else 0.0
        
        self.action_history.append(action)
        self.state_history.append(state)
        
        return PolicyAction(action=action, probability=probability, confidence=confidence)

class FixedStepPolicy(BaselinePolicy):
    """
    Fixed-Step Policy - läuft immer eine feste Anzahl von Schritten
    """
    
    def __init__(self, fixed_steps: int = 20):
        super().__init__(f"FixedStepPolicy_{fixed_steps}")
        self.fixed_steps = fixed_steps
    
    def select_action(self, state: SamplingState, valid_actions: List[SamplingAction]) -> PolicyAction:
        """Wählt Aktion basierend auf fester Schrittanzahl"""
        
        if state.current_step >= self.fixed_steps:
            action = SamplingAction.STOP
            confidence = 1.0
        else:
            action = SamplingAction.CONTINUE
            confidence = 1.0
        
        # Fallback auf gültige Aktionen
        if action not in valid_actions:
            action = valid_actions[0]
            confidence = 0.5
        
        probability = 1.0 if action in valid_actions else 0.0
        
        self.action_history.append(action)
        self.state_history.append(state)
        
        return PolicyAction(action=action, probability=probability, confidence=confidence)

class AdaptiveThresholdPolicy(BaselinePolicy):
    """
    Adaptive Threshold Policy - passt Qualitätsschwelle dynamisch an
    """
    
    def __init__(self, 
                 initial_threshold: float = 0.7,
                 adaptation_rate: float = 0.01):
        super().__init__("AdaptiveThresholdPolicy")
        self.initial_threshold = initial_threshold
        self.current_threshold = initial_threshold
        self.adaptation_rate = adaptation_rate
        self.episode_qualities = []
    
    def select_action(self, state: SamplingState, valid_actions: List[SamplingAction]) -> PolicyAction:
        """Wählt Aktion basierend auf adaptiver Schwelle"""
        
        # Adaptive Schwelle basierend auf bisheriger Performance
        if len(self.episode_qualities) > 5:
            recent_avg = np.mean(self.episode_qualities[-5:])
            target_threshold = recent_avg * 1.1  # 10% über Durchschnitt
            
            self.current_threshold += (target_threshold - self.current_threshold) * self.adaptation_rate
            self.current_threshold = max(0.5, min(0.95, self.current_threshold))  # Bounds
        
        # Entscheidungslogik
        if state.quality_estimate >= self.current_threshold:
            action = SamplingAction.STOP
            confidence = 0.9
        elif state.progress_ratio > 0.9:  # Fast am Ende
            action = SamplingAction.STOP
            confidence = 0.8
        elif state.quality_trend < -0.05:  # Verschlechterung
            action = SamplingAction.ADJUST_STRENGTH if SamplingAction.ADJUST_STRENGTH in valid_actions else SamplingAction.CONTINUE
            confidence = 0.7
        else:
            action = SamplingAction.CONTINUE
            confidence = 0.6
        
        # Fallback
        if action not in valid_actions:
            action = valid_actions[0]
            confidence = 0.5
        
        probability = confidence if action in valid_actions else 0.0
        
        self.action_history.append(action)
        self.state_history.append(state)
        
        return PolicyAction(action=action, probability=probability, confidence=confidence)
    
    def update_episode(self, final_quality: float):
        """Aktualisiert Episode-Qualität für Adaptation"""
        self.episode_qualities.append(final_quality)
        if len(self.episode_qualities) > 50:
            self.episode_qualities.pop(0)

class PolicyComparison:
    """
    Vergleicht verschiedene Baseline-Policies
    """
    
    def __init__(self, mdp: DiffusionMDP):
        self.mdp = mdp
        self.policies = [
            RandomPolicy(seed=42),
            HeuristicPolicy(),
            FixedStepPolicy(15),
            FixedStepPolicy(25),
            AdaptiveThresholdPolicy()
        ]
        
        self.comparison_results = []
    
    def run_comparison(self, num_episodes: int = 100) -> Dict[str, Any]:
        """Führt Vergleich zwischen Policies durch"""
        
        logger.info(f"Starte Policy-Vergleich mit {num_episodes} Episoden...")
        
        results = {}
        
        for policy in self.policies:
            logger.info(f"Teste {policy.name}...")
            
            episode_results = []
            
            for episode in range(num_episodes):
                # Simuliere Episode
                result = self._simulate_episode(policy)
                episode_results.append(result)
                
                # Update Policy Stats
                policy.update_stats(
                    result['final_quality'],
                    result['efficiency_score'],
                    result['success']
                )
                
                # Spezielle Updates für adaptive Policies
                if isinstance(policy, AdaptiveThresholdPolicy):
                    policy.update_episode(result['final_quality'])
            
            # Sammle Ergebnisse
            results[policy.name] = {
                'policy_stats': policy.get_stats(),
                'episode_results': episode_results,
                'avg_quality': np.mean([r['final_quality'] for r in episode_results]),
                'avg_efficiency': np.mean([r['efficiency_score'] for r in episode_results]),
                'avg_steps': np.mean([r['total_steps'] for r in episode_results]),
                'success_rate': np.mean([r['success'] for r in episode_results])
            }
        
        self.comparison_results = results
        return results
    
    def _simulate_episode(self, policy: BaselinePolicy) -> Dict[str, Any]:
        """Simuliert eine Episode mit gegebener Policy"""
        
        # Initialer Zustand
        initial_latent = torch.randn(1, 4, 64, 64)
        noise_schedule = [1.0 - i/self.mdp.max_steps for i in range(self.mdp.max_steps)]
        
        state = self.mdp.create_initial_state(initial_latent, noise_schedule)
        
        total_steps = 0
        qualities = []
        
        while not self.mdp.is_terminal(state) and total_steps < self.mdp.max_steps:
            # Gültige Aktionen
            valid_actions = self.mdp.get_valid_actions(state)
            
            # Policy-Aktion
            policy_action = policy.select_action(state, valid_actions)
            
            # Simuliere Zustandsübergang
            new_latent = torch.randn(1, 4, 64, 64)
            step_time = 0.1 + np.random.normal(0, 0.02)
            quality = min(1.0, max(0.0, 
                state.quality_estimate + np.random.normal(0.05, 0.1)))
            
            state = self.mdp.update_state(state, new_latent, step_time, quality)
            qualities.append(quality)
            
            total_steps += 1
            
            # Stoppe wenn STOP-Aktion gewählt
            if policy_action.action == SamplingAction.STOP:
                break
        
        return {
            'total_steps': total_steps,
            'final_quality': state.quality_estimate,
            'efficiency_score': state.efficiency_score,
            'success': state.quality_estimate >= 0.8,
            'qualities': qualities
        }
    
    def get_comparison_summary(self) -> str:
        """Gibt Vergleichszusammenfassung zurück"""
        
        if not self.comparison_results:
            return "Kein Vergleich durchgeführt"
        
        summary = "\n=== POLICY COMPARISON SUMMARY ===\n"
        
        # Sortiere nach Erfolgsrate
        sorted_policies = sorted(
            self.comparison_results.items(),
            key=lambda x: x[1]['success_rate'],
            reverse=True
        )
        
        for policy_name, results in sorted_policies:
            summary += f"\n{policy_name}:\n"
            summary += f"  Success Rate: {results['success_rate']:.2%}\n"
            summary += f"  Avg Quality: {results['avg_quality']:.3f}\n"
            summary += f"  Avg Efficiency: {results['avg_efficiency']:.3f}\n"
            summary += f"  Avg Steps: {results['avg_steps']:.1f}\n"
        
        return summary

# Test-Funktion
def test_baseline_policies():
    """Testet die Baseline-Policies"""
    logger.info("Teste Baseline-Policies...")
    
    # MDP erstellen
    mdp = DiffusionMDP(max_steps=30)
    
    # Policy-Vergleich
    comparison = PolicyComparison(mdp)
    results = comparison.run_comparison(num_episodes=20)  # Weniger für Test
    
    # Zusammenfassung
    summary = comparison.get_comparison_summary()
    logger.info(summary)
    
    logger.info("✅ Baseline-Policies erfolgreich getestet!")
    return True

if __name__ == "__main__":
    test_baseline_policies()
