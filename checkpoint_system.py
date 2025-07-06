"""
Checkpoint-System für RL-basierte adaptive Diffusion-Sampling
Modell-Speicherung, -Wiederherstellung und Experiment-Management
"""

import torch
import pickle
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
import logging
import hashlib
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """Experiment-Konfiguration"""
    experiment_name: str
    timestamp: str
    
    # Model parameters
    state_dim: int
    action_dim: int
    hidden_dims: List[int]
    
    # Training parameters
    learning_rate: float
    gamma: float
    use_baseline: bool
    num_episodes: int
    
    # Environment parameters
    max_steps: int
    quality_threshold: float
    
    # Reward parameters
    quality_weight: float
    efficiency_weight: float
    step_penalty: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Erstellt aus Dictionary"""
        return cls(**data)
    
    def get_hash(self) -> str:
        """Berechnet Hash der Konfiguration für Eindeutigkeit"""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

@dataclass
class CheckpointMetadata:
    """Metadaten für Checkpoint"""
    checkpoint_id: str
    experiment_name: str
    episode: int
    timestamp: str
    total_reward: float
    avg_quality: float
    avg_efficiency: float
    success_rate: float
    model_size_mb: float
    config_hash: str

class CheckpointManager:
    """
    Verwaltet Speicherung und Wiederherstellung von RL-Modellen
    """
    
    def __init__(self, 
                 base_dir: str = "experiments",
                 max_checkpoints: int = 10,
                 auto_save_interval: int = 100):
        
        self.base_dir = Path(base_dir)
        self.max_checkpoints = max_checkpoints
        self.auto_save_interval = auto_save_interval
        
        # Erstelle Verzeichnisstruktur
        self.base_dir.mkdir(exist_ok=True)
        (self.base_dir / "checkpoints").mkdir(exist_ok=True)
        (self.base_dir / "configs").mkdir(exist_ok=True)
        (self.base_dir / "logs").mkdir(exist_ok=True)
        (self.base_dir / "results").mkdir(exist_ok=True)
        
        self.checkpoint_registry = self._load_checkpoint_registry()
        
        logger.info(f"CheckpointManager initialisiert - Base dir: {self.base_dir}")
    
    def _generate_checkpoint_id(self, experiment_name: str, episode: int) -> str:
        """Generiert eindeutige Checkpoint-ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{experiment_name}_ep{episode}_{timestamp}"
    
    def _get_model_size(self, checkpoint_path: Path) -> float:
        """Berechnet Modell-Größe in MB"""
        return checkpoint_path.stat().st_size / (1024 * 1024)
    
    def _load_checkpoint_registry(self) -> Dict[str, CheckpointMetadata]:
        """Lädt Checkpoint-Registry"""
        registry_path = self.base_dir / "checkpoint_registry.json"
        
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                registry_data = json.load(f)
            
            registry = {}
            for checkpoint_id, data in registry_data.items():
                registry[checkpoint_id] = CheckpointMetadata(**data)
            
            return registry
        
        return {}
    
    def _save_checkpoint_registry(self):
        """Speichert Checkpoint-Registry"""
        registry_path = self.base_dir / "checkpoint_registry.json"
        
        registry_data = {}
        for checkpoint_id, metadata in self.checkpoint_registry.items():
            registry_data[checkpoint_id] = asdict(metadata)
        
        with open(registry_path, 'w') as f:
            json.dump(registry_data, f, indent=2)
    
    def save_checkpoint(self,
                       trainer,  # RLTrainer instance
                       experiment_config: ExperimentConfig,
                       episode: int,
                       performance_metrics: Dict[str, float],
                       force_save: bool = False) -> str:
        """
        Speichert Checkpoint mit Metadaten
        
        Returns:
            checkpoint_id
        """
        
        # Prüfe Auto-Save-Intervall
        if not force_save and episode % self.auto_save_interval != 0:
            return None
        
        checkpoint_id = self._generate_checkpoint_id(experiment_config.experiment_name, episode)
        checkpoint_path = self.base_dir / "checkpoints" / f"{checkpoint_id}.pt"
        
        # Checkpoint-Daten sammeln
        checkpoint_data = {
            'checkpoint_id': checkpoint_id,
            'experiment_config': experiment_config.to_dict(),
            'episode': episode,
            'model_state': {
                'policy_net': trainer.policy_net.state_dict(),
                'policy_optimizer': trainer.policy_optimizer.state_dict(),
            },
            'training_state': {
                'episode_rewards': trainer.episode_rewards,
                'episode_lengths': trainer.episode_lengths,
                'policy_losses': trainer.policy_losses,
            },
            'performance_metrics': performance_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Value Network falls vorhanden
        if trainer.value_net:
            checkpoint_data['model_state']['value_net'] = trainer.value_net.state_dict()
            checkpoint_data['model_state']['value_optimizer'] = trainer.value_optimizer.state_dict()
            checkpoint_data['training_state']['value_losses'] = trainer.value_losses
        
        # Speichere Checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Metadaten erstellen
        model_size = self._get_model_size(checkpoint_path)
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            experiment_name=experiment_config.experiment_name,
            episode=episode,
            timestamp=checkpoint_data['timestamp'],
            total_reward=performance_metrics.get('avg_reward', 0.0),
            avg_quality=performance_metrics.get('avg_quality', 0.0),
            avg_efficiency=performance_metrics.get('avg_efficiency', 0.0),
            success_rate=performance_metrics.get('success_rate', 0.0),
            model_size_mb=model_size,
            config_hash=experiment_config.get_hash()
        )
        
        # Registry aktualisieren
        self.checkpoint_registry[checkpoint_id] = metadata
        self._save_checkpoint_registry()
        
        # Alte Checkpoints bereinigen
        self._cleanup_old_checkpoints(experiment_config.experiment_name)
        
        logger.info(f"Checkpoint gespeichert: {checkpoint_id} ({model_size:.1f} MB)")
        return checkpoint_id
    
    def load_checkpoint(self, checkpoint_id: str, trainer) -> Dict[str, Any]:
        """
        Lädt Checkpoint und restauriert Trainer-Zustand
        
        Returns:
            Checkpoint-Daten
        """
        
        if checkpoint_id not in self.checkpoint_registry:
            raise ValueError(f"Checkpoint nicht gefunden: {checkpoint_id}")
        
        checkpoint_path = self.base_dir / "checkpoints" / f"{checkpoint_id}.pt"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint-Datei nicht gefunden: {checkpoint_path}")
        
        # Lade Checkpoint
        checkpoint_data = torch.load(checkpoint_path, map_location=trainer.device)
        
        # Restauriere Modell-Zustand
        trainer.policy_net.load_state_dict(checkpoint_data['model_state']['policy_net'])
        trainer.policy_optimizer.load_state_dict(checkpoint_data['model_state']['policy_optimizer'])
        
        if trainer.value_net and 'value_net' in checkpoint_data['model_state']:
            trainer.value_net.load_state_dict(checkpoint_data['model_state']['value_net'])
            trainer.value_optimizer.load_state_dict(checkpoint_data['model_state']['value_optimizer'])
        
        # Restauriere Training-Zustand
        trainer.episode_rewards = checkpoint_data['training_state']['episode_rewards']
        trainer.episode_lengths = checkpoint_data['training_state']['episode_lengths']
        trainer.policy_losses = checkpoint_data['training_state']['policy_losses']
        
        if trainer.value_net and 'value_losses' in checkpoint_data['training_state']:
            trainer.value_losses = checkpoint_data['training_state']['value_losses']
        
        logger.info(f"Checkpoint geladen: {checkpoint_id}")
        return checkpoint_data
    
    def _cleanup_old_checkpoints(self, experiment_name: str):
        """Bereinigt alte Checkpoints"""
        
        # Finde alle Checkpoints für Experiment
        experiment_checkpoints = [
            (checkpoint_id, metadata) 
            for checkpoint_id, metadata in self.checkpoint_registry.items()
            if metadata.experiment_name == experiment_name
        ]
        
        # Sortiere nach Episode (neueste zuerst)
        experiment_checkpoints.sort(key=lambda x: x[1].episode, reverse=True)
        
        # Lösche überschüssige Checkpoints
        if len(experiment_checkpoints) > self.max_checkpoints:
            to_delete = experiment_checkpoints[self.max_checkpoints:]
            
            for checkpoint_id, metadata in to_delete:
                checkpoint_path = self.base_dir / "checkpoints" / f"{checkpoint_id}.pt"
                
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                
                del self.checkpoint_registry[checkpoint_id]
                logger.info(f"Alter Checkpoint gelöscht: {checkpoint_id}")
    
    def get_best_checkpoint(self, 
                           experiment_name: str,
                           metric: str = 'success_rate') -> Optional[str]:
        """
        Findet besten Checkpoint basierend auf Metrik
        
        Args:
            experiment_name: Name des Experiments
            metric: Metrik für Bewertung ('success_rate', 'avg_quality', 'avg_efficiency', 'total_reward')
        
        Returns:
            checkpoint_id des besten Checkpoints
        """
        
        experiment_checkpoints = [
            (checkpoint_id, metadata)
            for checkpoint_id, metadata in self.checkpoint_registry.items()
            if metadata.experiment_name == experiment_name
        ]
        
        if not experiment_checkpoints:
            return None
        
        # Sortiere nach gewählter Metrik
        if metric == 'success_rate':
            best = max(experiment_checkpoints, key=lambda x: x[1].success_rate)
        elif metric == 'avg_quality':
            best = max(experiment_checkpoints, key=lambda x: x[1].avg_quality)
        elif metric == 'avg_efficiency':
            best = max(experiment_checkpoints, key=lambda x: x[1].avg_efficiency)
        elif metric == 'total_reward':
            best = max(experiment_checkpoints, key=lambda x: x[1].total_reward)
        else:
            raise ValueError(f"Unbekannte Metrik: {metric}")
        
        return best[0]
    
    def get_latest_checkpoint(self, experiment_name: str) -> Optional[str]:
        """Findet neuesten Checkpoint für Experiment"""
        
        experiment_checkpoints = [
            (checkpoint_id, metadata)
            for checkpoint_id, metadata in self.checkpoint_registry.items()
            if metadata.experiment_name == experiment_name
        ]
        
        if not experiment_checkpoints:
            return None
        
        latest = max(experiment_checkpoints, key=lambda x: x[1].episode)
        return latest[0]
    
    def list_checkpoints(self, experiment_name: Optional[str] = None) -> List[CheckpointMetadata]:
        """Listet alle Checkpoints auf"""
        
        if experiment_name:
            return [
                metadata for metadata in self.checkpoint_registry.values()
                if metadata.experiment_name == experiment_name
            ]
        
        return list(self.checkpoint_registry.values())
    
    def export_checkpoint(self, checkpoint_id: str, export_path: str):
        """Exportiert Checkpoint in standalone Datei"""
        
        if checkpoint_id not in self.checkpoint_registry:
            raise ValueError(f"Checkpoint nicht gefunden: {checkpoint_id}")
        
        checkpoint_path = self.base_dir / "checkpoints" / f"{checkpoint_id}.pt"
        export_path = Path(export_path)
        
        # Kopiere Checkpoint
        shutil.copy2(checkpoint_path, export_path)
        
        # Exportiere Metadaten
        metadata_path = export_path.with_suffix('.json')
        metadata = self.checkpoint_registry[checkpoint_id]
        
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        
        logger.info(f"Checkpoint exportiert: {export_path}")
    
    def create_experiment_summary(self, experiment_name: str) -> Dict[str, Any]:
        """Erstellt Zusammenfassung eines Experiments"""
        
        checkpoints = self.list_checkpoints(experiment_name)
        
        if not checkpoints:
            return {"error": f"Keine Checkpoints für Experiment {experiment_name}"}
        
        # Statistiken berechnen
        episodes = [cp.episode for cp in checkpoints]
        rewards = [cp.total_reward for cp in checkpoints]
        qualities = [cp.avg_quality for cp in checkpoints]
        efficiencies = [cp.avg_efficiency for cp in checkpoints]
        success_rates = [cp.success_rate for cp in checkpoints]
        
        summary = {
            'experiment_name': experiment_name,
            'total_checkpoints': len(checkpoints),
            'episode_range': {'min': min(episodes), 'max': max(episodes)},
            'performance': {
                'best_reward': max(rewards),
                'best_quality': max(qualities),
                'best_efficiency': max(efficiencies),
                'best_success_rate': max(success_rates),
                'final_reward': rewards[-1] if rewards else 0,
                'final_quality': qualities[-1] if qualities else 0,
                'final_success_rate': success_rates[-1] if success_rates else 0
            },
            'progress': {
                'reward_trend': np.polyfit(episodes, rewards, 1)[0] if len(episodes) > 1 else 0,
                'quality_trend': np.polyfit(episodes, qualities, 1)[0] if len(episodes) > 1 else 0,
                'efficiency_trend': np.polyfit(episodes, efficiencies, 1)[0] if len(episodes) > 1 else 0
            },
            'model_info': {
                'total_size_mb': sum(cp.model_size_mb for cp in checkpoints),
                'avg_size_mb': np.mean([cp.model_size_mb for cp in checkpoints]),
                'config_hash': checkpoints[0].config_hash if checkpoints else None
            }
        }
        
        return summary

# Test-Funktion
def test_checkpoint_system():
    """Testet das Checkpoint-System"""
    logger.info("Teste Checkpoint-System...")
    
    # Setup
    from rl_training import RLTrainer
    from mdp_definition import DiffusionMDP
    from reward_function import RewardFunction
    
    mdp = DiffusionMDP(max_steps=10)
    reward_function = RewardFunction()
    
    trainer = RLTrainer(
        mdp=mdp,
        reward_function=reward_function,
        state_dim=mdp.state_dim,
        action_dim=mdp.action_dim
    )
    
    # Checkpoint Manager
    checkpoint_manager = CheckpointManager(base_dir="test_experiments")
    
    # Experiment Config
    config = ExperimentConfig(
        experiment_name="test_experiment",
        timestamp=datetime.now().isoformat(),
        state_dim=mdp.state_dim,
        action_dim=mdp.action_dim,
        hidden_dims=[64, 32],
        learning_rate=3e-4,
        gamma=0.95,
        use_baseline=True,
        num_episodes=100,
        max_steps=10,
        quality_threshold=0.8,
        quality_weight=1.0,
        efficiency_weight=0.5,
        step_penalty=0.02
    )
    
    # Simuliere Training mit Checkpoints
    for episode in [0, 10, 20, 30]:
        # Simuliere Performance-Metriken
        metrics = {
            'avg_reward': 0.5 + episode * 0.01,
            'avg_quality': 0.6 + episode * 0.005,
            'avg_efficiency': 1.0 + episode * 0.02,
            'success_rate': min(1.0, episode * 0.02)
        }
        
        checkpoint_id = checkpoint_manager.save_checkpoint(
            trainer, config, episode, metrics, force_save=True
        )
        
        logger.info(f"Checkpoint erstellt: {checkpoint_id}")
    
    # Teste Checkpoint-Funktionen
    checkpoints = checkpoint_manager.list_checkpoints("test_experiment")
    logger.info(f"Anzahl Checkpoints: {len(checkpoints)}")
    
    best_checkpoint = checkpoint_manager.get_best_checkpoint("test_experiment", "success_rate")
    logger.info(f"Bester Checkpoint: {best_checkpoint}")
    
    latest_checkpoint = checkpoint_manager.get_latest_checkpoint("test_experiment")
    logger.info(f"Neuester Checkpoint: {latest_checkpoint}")
    
    # Teste Laden
    if latest_checkpoint:
        checkpoint_data = checkpoint_manager.load_checkpoint(latest_checkpoint, trainer)
        logger.info(f"Checkpoint geladen - Episode: {checkpoint_data['episode']}")
    
    # Experiment-Zusammenfassung
    summary = checkpoint_manager.create_experiment_summary("test_experiment")
    logger.info(f"Experiment-Zusammenfassung: {summary}")
    
    logger.info("✅ Checkpoint-System erfolgreich getestet!")
    return True

if __name__ == "__main__":
    test_checkpoint_system()
