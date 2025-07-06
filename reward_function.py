"""
Reward-Funktion für RL-basierte adaptive Diffusion-Sampling
Qualität vs. Effizienz Optimierung
"""

import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import logging
from mdp_definition import SamplingState, SamplingAction

logger = logging.getLogger(__name__)

@dataclass
class RewardComponents:
    """Komponenten der Reward-Funktion"""
    quality_reward: float
    efficiency_reward: float
    step_penalty: float
    termination_bonus: float
    total_reward: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'quality_reward': self.quality_reward,
            'efficiency_reward': self.efficiency_reward,
            'step_penalty': self.step_penalty,
            'termination_bonus': self.termination_bonus,
            'total_reward': self.total_reward
        }

class RewardFunction:
    """
    Reward-Funktion für adaptive Diffusion-Sampling
    
    Ziel: Maximiere Qualität bei minimalen Rechenschritten
    """
    
    def __init__(self,
                 quality_weight: float = 1.0,
                 efficiency_weight: float = 0.5,
                 step_penalty: float = 0.02,
                 quality_threshold: float = 0.8,
                 early_stop_bonus: float = 0.1):
        
        self.quality_weight = quality_weight
        self.efficiency_weight = efficiency_weight
        self.step_penalty = step_penalty
        self.quality_threshold = quality_threshold
        self.early_stop_bonus = early_stop_bonus
        
        # Für normalisierte Belohnungen
        self.quality_history = []
        self.efficiency_history = []
        
        logger.info(f"Reward-Funktion initialisiert - Quality weight: {quality_weight}, Efficiency weight: {efficiency_weight}")
    
    def calculate_quality_reward(self, state: SamplingState) -> float:
        """
        Berechnet Qualitäts-Belohnung basierend auf:
        - Aktueller Qualitätsschätzung
        - Qualitätstrend (Verbesserung)
        - Absolute Qualitätsschwelle
        """
        
        # Basis-Qualitätsbelohnung
        base_quality = state.quality_estimate
        
        # Bonus für positive Qualitätsentwicklung
        trend_bonus = max(0, state.quality_trend) * 0.5
        
        # Exponential-Bonus für Überschreitung der Qualitätsschwelle
        threshold_bonus = 0.0
        if state.quality_estimate >= self.quality_threshold:
            threshold_bonus = 0.2 * (state.quality_estimate - self.quality_threshold)
        
        # Konsistenz-Bonus (weniger Schwankungen)
        consistency_bonus = 0.0
        if len(self.quality_history) > 2:
            recent_std = np.std(self.quality_history[-3:])
            consistency_bonus = max(0, 0.1 - recent_std)
        
        total_quality_reward = (base_quality + trend_bonus + 
                               threshold_bonus + consistency_bonus)
        
        # Speichere für Normalisierung
        self.quality_history.append(state.quality_estimate)
        if len(self.quality_history) > 100:
            self.quality_history.pop(0)
        
        return total_quality_reward
    
    def calculate_efficiency_reward(self, state: SamplingState) -> float:
        """
        Berechnet Effizienz-Belohnung basierend auf:
        - Zeit pro Schritt
        - Gesamtzeit
        - Qualität pro Zeit-Einheit
        """
        
        # Basis-Effizienz (Quality/Time)
        base_efficiency = state.efficiency_score
        
        # Bonus für schnelle Schritte
        time_bonus = max(0, 0.1 - state.time_per_step) * 10
        
        # Bonus für frühzeitige Konvergenz
        progress_bonus = 0.0
        if state.quality_estimate >= self.quality_threshold:
            remaining_ratio = 1.0 - state.progress_ratio
            progress_bonus = remaining_ratio * 0.3
        
        # Strafe für zu lange Stagnation
        stagnation_penalty = 0.0
        if state.steps_since_improvement > 5:
            stagnation_penalty = -0.05 * (state.steps_since_improvement - 5)
        
        total_efficiency_reward = (base_efficiency + time_bonus + 
                                  progress_bonus + stagnation_penalty)
        
        # Speichere für Normalisierung
        self.efficiency_history.append(state.efficiency_score)
        if len(self.efficiency_history) > 100:
            self.efficiency_history.pop(0)
        
        return total_efficiency_reward
    
    def calculate_step_penalty(self, state: SamplingState, action: SamplingAction) -> float:
        """
        Berechnet Schritt-Strafe basierend auf:
        - Aktionstyp
        - Fortschritt
        - Qualitätsniveau
        """
        
        # Basis-Strafe pro Schritt
        base_penalty = self.step_penalty
        
        # Verschiedene Aktionen haben verschiedene Kosten
        action_penalties = {
            SamplingAction.CONTINUE: 1.0,
            SamplingAction.STOP: 0.1,  # Stoppen ist günstig
            SamplingAction.ADJUST_STRENGTH: 1.2,  # Anpassung kostet extra
            SamplingAction.SKIP_STEP: 0.8  # Überspringen spart Zeit
        }
        
        action_multiplier = action_penalties.get(action, 1.0)
        
        # Höhere Strafe bei fortgeschrittenem Sampling mit niedriger Qualität
        if state.progress_ratio > 0.7 and state.quality_estimate < 0.5:
            base_penalty *= 2.0
        
        return base_penalty * action_multiplier
    
    def calculate_termination_bonus(self, 
                                   state: SamplingState, 
                                   action: SamplingAction,
                                   is_terminal: bool) -> float:
        """
        Berechnet Terminierungs-Bonus für frühzeitiges Stoppen
        """
        
        if not is_terminal or action != SamplingAction.STOP:
            return 0.0
        
        # Basis-Bonus für rechtzeitiges Stoppen
        base_bonus = self.early_stop_bonus
        
        # Bonus für hohe Qualität bei frühem Stopp
        if state.quality_estimate >= self.quality_threshold:
            early_stop_ratio = 1.0 - state.progress_ratio
            base_bonus += early_stop_ratio * 0.2
        
        # Bonus für Effizienz
        if state.efficiency_score > 1.0:
            base_bonus += min(0.1, state.efficiency_score - 1.0)
        
        return base_bonus
    
    def calculate_reward(self, 
                        prev_state: SamplingState,
                        action: SamplingAction,
                        next_state: SamplingState,
                        is_terminal: bool) -> RewardComponents:
        """
        Hauptfunktion zur Belohnungsberechnung
        """
        
        # Einzelkomponenten berechnen
        quality_reward = self.calculate_quality_reward(next_state)
        efficiency_reward = self.calculate_efficiency_reward(next_state)
        step_penalty = self.calculate_step_penalty(next_state, action)
        termination_bonus = self.calculate_termination_bonus(next_state, action, is_terminal)
        
        # Gewichtete Gesamtbelohnung
        total_reward = (self.quality_weight * quality_reward +
                       self.efficiency_weight * efficiency_reward -
                       step_penalty +
                       termination_bonus)
        
        return RewardComponents(
            quality_reward=quality_reward,
            efficiency_reward=efficiency_reward,
            step_penalty=step_penalty,
            termination_bonus=termination_bonus,
            total_reward=total_reward
        )
    
    def get_reward_stats(self) -> Dict[str, Any]:
        """Gibt Statistiken über die Belohnungsverteilung zurück"""
        stats = {
            'quality_history_length': len(self.quality_history),
            'efficiency_history_length': len(self.efficiency_history),
            'parameters': {
                'quality_weight': self.quality_weight,
                'efficiency_weight': self.efficiency_weight,
                'step_penalty': self.step_penalty,
                'quality_threshold': self.quality_threshold,
                'early_stop_bonus': self.early_stop_bonus
            }
        }
        
        if self.quality_history:
            stats['quality_stats'] = {
                'mean': np.mean(self.quality_history),
                'std': np.std(self.quality_history),
                'min': np.min(self.quality_history),
                'max': np.max(self.quality_history)
            }
        
        if self.efficiency_history:
            stats['efficiency_stats'] = {
                'mean': np.mean(self.efficiency_history),
                'std': np.std(self.efficiency_history),
                'min': np.min(self.efficiency_history),
                'max': np.max(self.efficiency_history)
            }
        
        return stats
    
    def reset_history(self):
        """Setzt die Historie zurück"""
        self.quality_history = []
        self.efficiency_history = []

class AdaptiveRewardFunction(RewardFunction):
    """
    Adaptive Reward-Funktion die sich an die Performance anpasst
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.adaptation_rate = 0.01
        self.min_weight = 0.1
        self.max_weight = 2.0
        
        # Tracking für Anpassungen
        self.reward_history = []
        self.adaptation_history = []
    
    def adapt_weights(self, episode_rewards: List[float]):
        """
        Passt Gewichte basierend auf Episode-Performance an
        """
        
        if len(episode_rewards) < 10:
            return
        
        # Berechne Trend der letzten Episoden
        recent_rewards = episode_rewards[-10:]
        reward_trend = np.mean(np.diff(recent_rewards))
        
        # Anpassung der Gewichte
        if reward_trend < -0.1:  # Verschlechterung
            # Erhöhe Qualitätsgewicht, reduziere Effizienz
            self.quality_weight = min(self.max_weight, 
                                    self.quality_weight + self.adaptation_rate)
            self.efficiency_weight = max(self.min_weight,
                                       self.efficiency_weight - self.adaptation_rate)
        
        elif reward_trend > 0.1:  # Verbesserung
            # Balanciere Gewichte
            target_quality = 1.0
            target_efficiency = 0.5
            
            self.quality_weight += (target_quality - self.quality_weight) * self.adaptation_rate
            self.efficiency_weight += (target_efficiency - self.efficiency_weight) * self.adaptation_rate
        
        # Protokolliere Anpassungen
        self.adaptation_history.append({
            'episode': len(episode_rewards),
            'reward_trend': reward_trend,
            'quality_weight': self.quality_weight,
            'efficiency_weight': self.efficiency_weight
        })
        
        logger.info(f"Gewichte angepasst - Quality: {self.quality_weight:.3f}, Efficiency: {self.efficiency_weight:.3f}")

# Test-Funktion
def test_reward_function():
    """Testet die Reward-Funktion"""
    logger.info("Teste Reward-Funktion...")
    
    # Reward-Funktion initialisieren
    reward_func = RewardFunction(
        quality_weight=1.0,
        efficiency_weight=0.5,
        step_penalty=0.02
    )
    
    # Mock-Zustände erstellen
    from mdp_definition import SamplingState
    
    # Simuliere Sampling-Episode
    prev_state = SamplingState(
        current_step=5,
        max_steps=20,
        progress_ratio=0.25,
        current_noise_level=0.8,
        noise_schedule=[],
        latent_mean=0.0,
        latent_std=1.0,
        latent_norm=100.0,
        quality_estimate=0.6,
        quality_trend=0.05,
        time_per_step=0.1,
        total_time=0.5,
        efficiency_score=6.0,
        steps_since_improvement=0,
        best_quality_so_far=0.6
    )
    
    next_state = SamplingState(
        current_step=6,
        max_steps=20,
        progress_ratio=0.3,
        current_noise_level=0.75,
        noise_schedule=[],
        latent_mean=0.0,
        latent_std=1.0,
        latent_norm=95.0,
        quality_estimate=0.7,
        quality_trend=0.1,
        time_per_step=0.1,
        total_time=0.6,
        efficiency_score=7.0,
        steps_since_improvement=0,
        best_quality_so_far=0.7
    )
    
    # Verschiedene Aktionen testen
    actions = [SamplingAction.CONTINUE, SamplingAction.STOP, SamplingAction.ADJUST_STRENGTH]
    
    for action in actions:
        reward_components = reward_func.calculate_reward(
            prev_state, action, next_state, is_terminal=(action == SamplingAction.STOP)
        )
        
        logger.info(f"Aktion {action.name}: {reward_components.to_dict()}")
    
    # Statistiken
    stats = reward_func.get_reward_stats()
    logger.info(f"Reward-Statistiken: {stats}")
    
    # Adaptive Reward-Funktion testen
    adaptive_reward = AdaptiveRewardFunction()
    
    # Simuliere Episode-Rewards
    episode_rewards = [0.5, 0.6, 0.4, 0.3, 0.7, 0.8, 0.6, 0.9, 1.0, 0.8]
    adaptive_reward.adapt_weights(episode_rewards)
    
    logger.info("✅ Reward-Funktion erfolgreich getestet!")
    return True

if __name__ == "__main__":
    test_reward_function()
