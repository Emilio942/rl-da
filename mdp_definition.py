"""
MDP Definition für RL-basierte adaptive Diffusion-Sampling
Markov Decision Process: States, Actions, Rewards
"""

import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class SamplingAction(Enum):
    """Aktionsraum für den RL-Agent"""
    CONTINUE = 0      # Einen weiteren Denoising-Schritt machen
    STOP = 1          # Sampling beenden
    ADJUST_STRENGTH = 2  # Denoising-Stärke anpassen
    SKIP_STEP = 3     # Aktuellen Schritt überspringen

@dataclass
class SamplingState:
    """
    Zustandsrepräsentation für das MDP
    """
    # Sampling-Fortschritt
    current_step: int
    max_steps: int
    progress_ratio: float
    
    # Noise-Level Information
    current_noise_level: float
    noise_schedule: List[float]
    
    # Latent-Eigenschaften
    latent_mean: float
    latent_std: float
    latent_norm: float
    
    # Qualitäts-Schätzung
    quality_estimate: float
    quality_trend: float  # Änderung der Qualität in letzten Schritten
    
    # Effizienz-Metriken
    time_per_step: float
    total_time: float
    efficiency_score: float
    
    # Historische Information
    steps_since_improvement: int
    best_quality_so_far: float
    
    def to_vector(self) -> np.ndarray:
        """Konvertiert Zustand zu Feature-Vektor für Neural Network"""
        return np.array([
            self.progress_ratio,
            self.current_noise_level,
            self.latent_mean,
            self.latent_std,
            self.latent_norm,
            self.quality_estimate,
            self.quality_trend,
            self.time_per_step,
            self.efficiency_score,
            self.steps_since_improvement / 10.0,  # Normalisiert
            self.best_quality_so_far
        ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary für Logging"""
        return {
            'current_step': self.current_step,
            'max_steps': self.max_steps,
            'progress_ratio': self.progress_ratio,
            'current_noise_level': self.current_noise_level,
            'latent_mean': self.latent_mean,
            'latent_std': self.latent_std,
            'latent_norm': self.latent_norm,
            'quality_estimate': self.quality_estimate,
            'quality_trend': self.quality_trend,
            'time_per_step': self.time_per_step,
            'total_time': self.total_time,
            'efficiency_score': self.efficiency_score,
            'steps_since_improvement': self.steps_since_improvement,
            'best_quality_so_far': self.best_quality_so_far
        }

class DiffusionMDP:
    """
    Markov Decision Process für adaptive Diffusion-Sampling
    """
    
    def __init__(self, 
                 max_steps: int = 50,
                 quality_threshold: float = 0.8,
                 efficiency_weight: float = 0.5):
        
        self.max_steps = max_steps
        self.quality_threshold = quality_threshold
        self.efficiency_weight = efficiency_weight
        
        # Zustandsraum-Dimensionen
        self.state_dim = 11  # Anzahl Features in SamplingState.to_vector()
        self.action_dim = len(SamplingAction)
        
        # MDP-Eigenschaften
        self.discount_factor = 0.95
        self.step_penalty = -0.01  # Kleine Strafe für jeden Schritt
        self.quality_bonus = 1.0   # Bonus für gute Qualität
        
        logger.info(f"MDP initialisiert - State dim: {self.state_dim}, Action dim: {self.action_dim}")
    
    def create_initial_state(self, 
                           latent: torch.Tensor,
                           noise_schedule: List[float]) -> SamplingState:
        """Erstellt initialen Zustand"""
        
        with torch.no_grad():
            latent_mean = torch.mean(latent).item()
            latent_std = torch.std(latent).item()
            latent_norm = torch.norm(latent).item()
        
        return SamplingState(
            current_step=0,
            max_steps=self.max_steps,
            progress_ratio=0.0,
            current_noise_level=noise_schedule[0] if noise_schedule else 1.0,
            noise_schedule=noise_schedule,
            latent_mean=latent_mean,
            latent_std=latent_std,
            latent_norm=latent_norm,
            quality_estimate=0.0,
            quality_trend=0.0,
            time_per_step=0.0,
            total_time=0.0,
            efficiency_score=0.0,
            steps_since_improvement=0,
            best_quality_so_far=0.0
        )
    
    def update_state(self, 
                    state: SamplingState,
                    new_latent: torch.Tensor,
                    step_time: float,
                    quality_estimate: float) -> SamplingState:
        """Aktualisiert Zustand nach einem Schritt"""
        
        with torch.no_grad():
            new_latent_mean = torch.mean(new_latent).item()
            new_latent_std = torch.std(new_latent).item()
            new_latent_norm = torch.norm(new_latent).item()
        
        # Aktualisierte Werte berechnen
        new_step = state.current_step + 1
        new_progress = new_step / state.max_steps
        
        # Noise-Level aktualisieren
        new_noise_level = (state.noise_schedule[new_step] 
                          if new_step < len(state.noise_schedule) 
                          else 0.0)
        
        # Qualitätstrend berechnen
        quality_trend = quality_estimate - state.quality_estimate
        
        # Effizienz-Score berechnen
        total_time = state.total_time + step_time
        avg_time_per_step = total_time / new_step
        efficiency_score = quality_estimate / (avg_time_per_step + 1e-6)
        
        # Verbesserung prüfen
        steps_since_improvement = (state.steps_since_improvement + 1 
                                 if quality_estimate <= state.best_quality_so_far
                                 else 0)
        
        best_quality = max(state.best_quality_so_far, quality_estimate)
        
        return SamplingState(
            current_step=new_step,
            max_steps=state.max_steps,
            progress_ratio=new_progress,
            current_noise_level=new_noise_level,
            noise_schedule=state.noise_schedule,
            latent_mean=new_latent_mean,
            latent_std=new_latent_std,
            latent_norm=new_latent_norm,
            quality_estimate=quality_estimate,
            quality_trend=quality_trend,
            time_per_step=avg_time_per_step,
            total_time=total_time,
            efficiency_score=efficiency_score,
            steps_since_improvement=steps_since_improvement,
            best_quality_so_far=best_quality
        )
    
    def is_terminal(self, state: SamplingState) -> bool:
        """Prüft ob Zustand terminal ist"""
        return (state.current_step >= state.max_steps or 
                state.quality_estimate >= self.quality_threshold)
    
    def get_valid_actions(self, state: SamplingState) -> List[SamplingAction]:
        """Gibt gültige Aktionen für aktuellen Zustand zurück"""
        valid_actions = []
        
        # CONTINUE ist immer möglich, außer bei terminalen Zuständen
        if not self.is_terminal(state):
            valid_actions.append(SamplingAction.CONTINUE)
            valid_actions.append(SamplingAction.ADJUST_STRENGTH)
            
            # SKIP_STEP nur möglich wenn noch genug Schritte übrig
            if state.current_step < state.max_steps - 2:
                valid_actions.append(SamplingAction.SKIP_STEP)
        
        # STOP ist immer möglich
        valid_actions.append(SamplingAction.STOP)
        
        return valid_actions
    
    def get_state_representation(self, state: SamplingState) -> np.ndarray:
        """Gibt normalisierte Zustandsrepräsentation zurück"""
        return state.to_vector()
    
    def get_mdp_info(self) -> Dict[str, Any]:
        """Gibt MDP-Informationen zurück"""
        return {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'max_steps': self.max_steps,
            'quality_threshold': self.quality_threshold,
            'efficiency_weight': self.efficiency_weight,
            'discount_factor': self.discount_factor,
            'step_penalty': self.step_penalty,
            'quality_bonus': self.quality_bonus,
            'actions': [action.name for action in SamplingAction]
        }

# Test-Funktion
def test_mdp():
    """Testet die MDP-Definition"""
    logger.info("Teste MDP-Definition...")
    
    # MDP initialisieren
    mdp = DiffusionMDP(max_steps=20, quality_threshold=0.8)
    
    # Fake Latent und Noise Schedule
    latent = torch.randn(1, 4, 64, 64)
    noise_schedule = [1.0 - i/20.0 for i in range(20)]
    
    # Initialer Zustand
    initial_state = mdp.create_initial_state(latent, noise_schedule)
    logger.info(f"Initialer Zustand: {initial_state.to_dict()}")
    
    # Zustandsvektor
    state_vector = mdp.get_state_representation(initial_state)
    logger.info(f"Zustandsvektor: {state_vector}")
    
    # Gültige Aktionen
    valid_actions = mdp.get_valid_actions(initial_state)
    logger.info(f"Gültige Aktionen: {[a.name for a in valid_actions]}")
    
    # Simuliere einige Schritte
    current_state = initial_state
    for step in range(5):
        # Simuliere neuen Latent
        new_latent = torch.randn(1, 4, 64, 64)
        step_time = 0.1
        quality = 0.1 + step * 0.15
        
        # Zustand aktualisieren
        current_state = mdp.update_state(current_state, new_latent, step_time, quality)
        
        # Terminal prüfen
        is_terminal = mdp.is_terminal(current_state)
        
        logger.info(f"Schritt {step+1}: Quality={quality:.3f}, Terminal={is_terminal}")
        
        if is_terminal:
            break
    
    # MDP-Info
    mdp_info = mdp.get_mdp_info()
    logger.info(f"MDP-Info: {mdp_info}")
    
    logger.info("✅ MDP-Definition erfolgreich getestet!")
    return True

if __name__ == "__main__":
    test_mdp()
