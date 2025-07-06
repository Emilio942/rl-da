"""
Mixture of Experts (MoE) System für Adaptive Sampling
Phase 4: Experten & Hierarchie Implementation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class ExpertType(Enum):
    """Verschiedene Experten-Typen für unterschiedliche Sampling-Strategien"""
    QUALITY_FOCUSED = "quality_focused"      # Fokus auf höchste Qualität
    SPEED_FOCUSED = "speed_focused"          # Fokus auf Geschwindigkeit  
    BALANCED = "balanced"                    # Ausgewogener Ansatz
    EARLY_STOPPING = "early_stopping"       # Aggressives frühes Stoppen
    ADAPTIVE_NOISE = "adaptive_noise"        # Adaptive Rausch-Anpassung

@dataclass
class ExpertProfile:
    """Profil eines Sampling-Experten"""
    expert_type: ExpertType
    name: str
    description: str
    
    # Sampling-Parameter
    min_steps: int = 5
    max_steps: int = 50
    quality_threshold: float = 0.8
    speed_weight: float = 0.5
    quality_weight: float = 0.5
    
    # Spezialisierung
    data_patterns: List[str] = None
    noise_preferences: Dict[str, float] = None
    
    def __post_init__(self):
        if self.data_patterns is None:
            self.data_patterns = []
        if self.noise_preferences is None:
            self.noise_preferences = {}

class SamplingExpert(nn.Module):
    """
    Ein Sampling-Experte mit spezialisierten Strategien
    """
    
    def __init__(self, profile: ExpertProfile, state_dim: int = 128):
        super().__init__()
        self.profile = profile
        self.state_dim = state_dim
        
        # Experten-spezifische Netzwerk-Architektur
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Spezialisierte Ausgabe-Köpfe
        self.action_head = nn.Linear(128, 3)  # stop, continue, adjust
        self.confidence_head = nn.Linear(128, 1)  # Confidence Score
        self.quality_predictor = nn.Linear(128, 1)  # Qualitäts-Vorhersage
        
        # Experten-spezifische Parameter
        self.expertise_score = nn.Parameter(torch.ones(1))
        
        logger.info(f"Experte initialisiert: {profile.name} ({profile.expert_type.value})")
    
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass durch den Experten"""
        features = self.feature_extractor(state)
        
        # Verschiedene Ausgaben
        action_logits = self.action_head(features)
        confidence = torch.sigmoid(self.confidence_head(features))
        quality_pred = torch.sigmoid(self.quality_predictor(features))
        
        return {
            'action_logits': action_logits,
            'confidence': confidence,
            'quality_prediction': quality_pred,
            'features': features,
            'expertise_score': self.expertise_score
        }
    
    def get_action_probabilities(self, state: torch.Tensor) -> torch.Tensor:
        """Berechne Action-Wahrscheinlichkeiten"""
        output = self.forward(state)
        return torch.softmax(output['action_logits'], dim=-1)
    
    def predict_quality(self, state: torch.Tensor) -> float:
        """Vorhersage der erwarteten Qualität"""
        with torch.no_grad():
            output = self.forward(state)
            return output['quality_prediction'].item()
    
    def get_confidence(self, state: torch.Tensor) -> float:
        """Confidence Score für diese Situation"""
        with torch.no_grad():
            output = self.forward(state)
            return output['confidence'].item()

class ExpertRouter(nn.Module):
    """
    Router-Netzwerk zur dynamischen Experten-Auswahl
    """
    
    def __init__(self, state_dim: int, num_experts: int):
        super().__init__()
        self.state_dim = state_dim
        self.num_experts = num_experts
        
        # Router-Netzwerk
        self.router = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts)
        )
        
        # Kontext-Encoder für Experten-Auswahl
        self.context_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        logger.info(f"Router initialisiert für {num_experts} Experten")
    
    def forward(self, state: torch.Tensor, temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        """Wähle Experten basierend auf dem aktuellen Zustand"""
        
        # Router-Gewichte berechnen
        router_logits = self.router(state)
        router_probs = torch.softmax(router_logits / temperature, dim=-1)
        
        # Kontext für Experten-Auswahl
        context = self.context_encoder(state)
        
        return {
            'router_logits': router_logits,
            'router_probs': router_probs,
            'context': context
        }
    
    def select_expert(self, state: torch.Tensor, exploration: float = 0.1) -> Tuple[int, float]:
        """Wähle einen Experten aus (mit optionaler Exploration)"""
        with torch.no_grad():
            output = self.forward(state)
            probs = output['router_probs'].squeeze()
            
            if np.random.random() < exploration:
                # Exploration: zufällige Auswahl
                expert_idx = np.random.randint(self.num_experts)
                confidence = probs[expert_idx].item()
            else:
                # Exploitation: besten Experten wählen
                expert_idx = torch.argmax(probs).item()
                confidence = probs[expert_idx].item()
            
            return expert_idx, confidence

class MixtureOfExperts:
    """
    Hauptklasse für das Mixture of Experts System
    """
    
    def __init__(self, state_dim: int = 128):
        self.state_dim = state_dim
        self.experts: List[SamplingExpert] = []
        self.expert_profiles: List[ExpertProfile] = []
        self.router: Optional[ExpertRouter] = None
        
        # Performance-Tracking
        self.expert_performance = {}
        self.usage_history = []
        
        logger.info("MixtureOfExperts System initialisiert")
        
        # Erstelle Standard-Experten
        self._create_default_experts()
    
    def _create_default_experts(self):
        """Erstelle Standard-Experten mit verschiedenen Spezialisierungen"""
        
        # 1. Quality-Focused Expert
        quality_profile = ExpertProfile(
            expert_type=ExpertType.QUALITY_FOCUSED,
            name="QualityMaster",
            description="Spezialisiert auf höchste Qualität",
            min_steps=10,
            max_steps=50,
            quality_threshold=0.95,
            speed_weight=0.2,
            quality_weight=0.8,
            data_patterns=["complex", "detailed", "high_resolution"],
            noise_preferences={"low_noise": 0.8, "gradual_denoising": 0.9}
        )
        
        # 2. Speed-Focused Expert  
        speed_profile = ExpertProfile(
            expert_type=ExpertType.SPEED_FOCUSED,
            name="SpeedDemon",
            description="Optimiert für maximale Geschwindigkeit",
            min_steps=3,
            max_steps=15,
            quality_threshold=0.7,
            speed_weight=0.8,
            quality_weight=0.2,
            data_patterns=["simple", "low_resolution", "sketch"],
            noise_preferences={"high_noise": 0.7, "aggressive_denoising": 0.8}
        )
        
        # 3. Balanced Expert
        balanced_profile = ExpertProfile(
            expert_type=ExpertType.BALANCED,
            name="BalancedMaster",
            description="Ausgewogener Ansatz",
            min_steps=5,
            max_steps=25,
            quality_threshold=0.8,
            speed_weight=0.5,
            quality_weight=0.5,
            data_patterns=["general", "mixed"],
            noise_preferences={"balanced_noise": 0.8}
        )
        
        # 4. Early Stopping Expert
        early_stop_profile = ExpertProfile(
            expert_type=ExpertType.EARLY_STOPPING,
            name="EarlyStopper",
            description="Aggressives frühes Stoppen",
            min_steps=2,
            max_steps=20,
            quality_threshold=0.75,
            speed_weight=0.7,
            quality_weight=0.3,
            data_patterns=["previews", "drafts"],
            noise_preferences={"early_convergence": 0.9}
        )
        
        # 5. Adaptive Noise Expert
        adaptive_profile = ExpertProfile(
            expert_type=ExpertType.ADAPTIVE_NOISE,
            name="NoiseAdaptive",
            description="Adaptive Rausch-Anpassung",
            min_steps=5,
            max_steps=30,
            quality_threshold=0.85,
            speed_weight=0.4,
            quality_weight=0.6,
            data_patterns=["variable", "adaptive"],
            noise_preferences={"dynamic_noise": 0.95}
        )
        
        # Experten erstellen
        profiles = [quality_profile, speed_profile, balanced_profile, 
                   early_stop_profile, adaptive_profile]
        
        for profile in profiles:
            expert = SamplingExpert(profile, self.state_dim)
            self.add_expert(expert, profile)
        
        # Router erstellen
        self.router = ExpertRouter(self.state_dim, len(self.experts))
        
        logger.info(f"Standard-Experten erstellt: {len(self.experts)} Experten")
    
    def add_expert(self, expert: SamplingExpert, profile: ExpertProfile):
        """Füge einen neuen Experten hinzu"""
        self.experts.append(expert)
        self.expert_profiles.append(profile)
        self.expert_performance[profile.name] = {
            'total_uses': 0,
            'success_rate': 0.0,
            'avg_quality': 0.0,
            'avg_speed': 0.0,
            'confidence_scores': []
        }
        
        logger.info(f"Experte hinzugefügt: {profile.name}")
    
    def select_expert(self, state: torch.Tensor, context: Dict[str, Any] = None) -> Tuple[int, SamplingExpert, float]:
        """Wähle den besten Experten für den aktuellen Zustand"""
        
        if self.router is None:
            # Fallback: ersten Experten wählen
            return 0, self.experts[0], 1.0
        
        # Kontext-basierte Anpassung
        exploration = 0.1
        if context and 'exploration_rate' in context:
            exploration = context['exploration_rate']
        
        # Experten-Auswahl
        expert_idx, confidence = self.router.select_expert(state, exploration)
        expert = self.experts[expert_idx]
        
        # Performance-Tracking
        expert_name = self.expert_profiles[expert_idx].name
        self.expert_performance[expert_name]['total_uses'] += 1
        
        # Usage-History
        self.usage_history.append({
            'expert_idx': expert_idx,
            'expert_name': expert_name,
            'confidence': confidence,
            'state_summary': self._summarize_state(state)
        })
        
        logger.debug(f"Experte ausgewählt: {expert_name} (Confidence: {confidence:.3f})")
        
        return expert_idx, expert, confidence
    
    def _summarize_state(self, state: torch.Tensor) -> Dict[str, float]:
        """Erstelle Zusammenfassung des Zustands"""
        with torch.no_grad():
            return {
                'mean': state.mean().item(),
                'std': state.std().item(),
                'min': state.min().item(),
                'max': state.max().item()
            }
    
    def update_expert_performance(self, expert_idx: int, quality: float, speed: float, success: bool):
        """Update Performance-Metriken eines Experten"""
        expert_name = self.expert_profiles[expert_idx].name
        perf = self.expert_performance[expert_name]
        
        # Update Metriken
        total_uses = perf['total_uses']
        perf['avg_quality'] = (perf['avg_quality'] * (total_uses - 1) + quality) / total_uses
        perf['avg_speed'] = (perf['avg_speed'] * (total_uses - 1) + speed) / total_uses
        
        # Success Rate
        old_success_rate = perf['success_rate']
        perf['success_rate'] = (old_success_rate * (total_uses - 1) + (1.0 if success else 0.0)) / total_uses
        
        logger.debug(f"Performance Update - {expert_name}: Quality={quality:.3f}, Speed={speed:.3f}")
    
    def get_expert_rankings(self) -> List[Tuple[str, Dict[str, float]]]:
        """Hole Experten-Rankings basierend auf Performance"""
        rankings = []
        
        for expert_name, perf in self.expert_performance.items():
            if perf['total_uses'] > 0:
                # Composite Score
                composite_score = (
                    0.4 * perf['avg_quality'] + 
                    0.3 * perf['avg_speed'] + 
                    0.3 * perf['success_rate']
                )
                
                rankings.append((expert_name, {
                    'composite_score': composite_score,
                    'quality': perf['avg_quality'],
                    'speed': perf['avg_speed'],
                    'success_rate': perf['success_rate'],
                    'total_uses': perf['total_uses']
                }))
        
        # Sortiere nach Composite Score
        rankings.sort(key=lambda x: x[1]['composite_score'], reverse=True)
        return rankings
    
    def save_experts(self, save_dir: str):
        """Speichere alle Experten und Router"""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Speichere Router
        if self.router:
            torch.save(self.router.state_dict(), save_path / "router.pt")
        
        # Speichere Experten
        experts_dir = save_path / "experts"
        experts_dir.mkdir(exist_ok=True)
        
        for i, expert in enumerate(self.experts):
            torch.save(expert.state_dict(), experts_dir / f"expert_{i}.pt")
        
        # Speichere Profile und Performance
        with open(save_path / "expert_profiles.json", "w") as f:
            profiles_data = []
            for profile in self.expert_profiles:
                profile_dict = {
                    'expert_type': profile.expert_type.value,
                    'name': profile.name,
                    'description': profile.description,
                    'min_steps': profile.min_steps,
                    'max_steps': profile.max_steps,
                    'quality_threshold': profile.quality_threshold,
                    'speed_weight': profile.speed_weight,
                    'quality_weight': profile.quality_weight,
                    'data_patterns': profile.data_patterns,
                    'noise_preferences': profile.noise_preferences
                }
                profiles_data.append(profile_dict)
            json.dump(profiles_data, f, indent=2)
        
        with open(save_path / "expert_performance.json", "w") as f:
            json.dump(self.expert_performance, f, indent=2)
        
        logger.info(f"Experten gespeichert: {save_path}")
    
    def load_experts(self, load_dir: str):
        """Lade gespeicherte Experten"""
        load_path = Path(load_dir)
        
        # Lade Performance-Daten
        perf_file = load_path / "expert_performance.json"
        if perf_file.exists():
            with open(perf_file, "r") as f:
                self.expert_performance = json.load(f)
        
        logger.info(f"Experten geladen: {load_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Hole detaillierte Statistiken"""
        stats = {
            'num_experts': len(self.experts),
            'total_usage': len(self.usage_history),
            'expert_performance': self.expert_performance,
            'expert_rankings': self.get_expert_rankings(),
            'usage_distribution': {}
        }
        
        # Usage-Verteilung
        for entry in self.usage_history:
            expert_name = entry['expert_name']
            if expert_name not in stats['usage_distribution']:
                stats['usage_distribution'][expert_name] = 0
            stats['usage_distribution'][expert_name] += 1
        
        return stats

def test_mixture_of_experts():
    """Test-Funktion für das MoE System"""
    logger.info("Teste Mixture of Experts System...")
    
    # Erstelle MoE System
    moe = MixtureOfExperts(state_dim=128)
    
    # Test mit verschiedenen Zuständen
    for i in range(10):
        # Simuliere verschiedene Zustände
        state = torch.randn(1, 128)
        
        # Wähle Experten
        expert_idx, expert, confidence = moe.select_expert(state)
        
        # Simuliere Performance
        quality = np.random.uniform(0.5, 1.0)
        speed = np.random.uniform(0.5, 1.0)
        success = quality > 0.7
        
        moe.update_expert_performance(expert_idx, quality, speed, success)
        
        logger.info(f"Test {i+1}: Experte {expert_idx} - Quality: {quality:.3f}, Speed: {speed:.3f}")
    
    # Zeige Statistiken
    stats = moe.get_statistics()
    logger.info(f"Statistiken: {json.dumps(stats['usage_distribution'], indent=2)}")
    
    # Rankings
    rankings = moe.get_expert_rankings()
    logger.info("Experten-Rankings:")
    for rank, (name, scores) in enumerate(rankings[:3], 1):
        logger.info(f"  {rank}. {name}: Score={scores['composite_score']:.3f}")
    
    logger.info("✅ MoE System erfolgreich getestet!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_mixture_of_experts()
