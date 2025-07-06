"""
Logging-System für RL-Diffusion-Sampling
"""
import logging
import json
import wandb
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
import torch

@dataclass
class SamplingMetrics:
    """Metriken für einen Sampling-Durchgang"""
    run_id: str
    timestamp: datetime
    total_steps: int
    total_time: float
    final_quality: float
    efficiency_score: float
    steps_saved: int
    model_info: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary für Logging"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

class RLDiffusionLogger:
    """
    Zentrales Logging-System für RL-Diffusion-Experimente
    """
    
    def __init__(self, 
                 log_dir: str = "logs",
                 use_wandb: bool = True,
                 project_name: str = "rl-diffusion-sampling"):
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Erstelle Unterordner
        (self.log_dir / "sampling").mkdir(exist_ok=True)
        (self.log_dir / "training").mkdir(exist_ok=True)
        (self.log_dir / "plots").mkdir(exist_ok=True)
        
        # Setup File Logging
        self.setup_file_logging()
        
        # Setup Wandb
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                wandb.init(
                    project=project_name,
                    config={
                        "experiment_type": "rl_diffusion_sampling",
                        "log_dir": str(self.log_dir)
                    }
                )
                self.logger.info("Wandb initialisiert")
            except Exception as e:
                self.logger.warning(f"Wandb konnte nicht initialisiert werden: {e}")
                self.use_wandb = False
        
        # Metriken-Speicher
        self.sampling_metrics: List[SamplingMetrics] = []
        self.training_metrics: List[Dict[str, Any]] = []
        
    def setup_file_logging(self):
        """Setup für File-basiertes Logging"""
        log_file = self.log_dir / f"rl_diffusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialisiert: {log_file}")
    
    def log_sampling_step(self, 
                         step_data: Dict[str, Any],
                         run_id: str = None):
        """Loggt einen einzelnen Sampling-Schritt"""
        
        if run_id is None:
            run_id = f"sampling_{int(datetime.now().timestamp())}"
        
        # File Logging
        log_file = self.log_dir / "sampling" / f"{run_id}.jsonl"
        with open(log_file, 'a') as f:
            json.dump(step_data, f)
            f.write('\n')
        
        # Wandb Logging
        if self.use_wandb:
            wandb.log({
                f"sampling/{k}": v for k, v in step_data.items()
                if isinstance(v, (int, float, str))
            })
        
        # Console Logging
        self.logger.info(f"Sampling Schritt [{run_id}]: {step_data}")
    
    def log_sampling_summary(self, 
                           summary: Dict[str, Any],
                           run_id: str = None):
        """Loggt Zusammenfassung eines Sampling-Durchgangs"""
        
        if run_id is None:
            run_id = f"sampling_{int(datetime.now().timestamp())}"
        
        # Erstelle Metriken-Objekt
        metrics = SamplingMetrics(
            run_id=run_id,
            timestamp=datetime.now(),
            total_steps=summary.get('total_steps', 0),
            total_time=summary.get('total_computation_time', 0.0),
            final_quality=summary.get('final_quality', 0.0),
            efficiency_score=summary.get('efficiency_score', 0.0),
            steps_saved=summary.get('steps_saved', 0),
            model_info=summary.get('model_info', {})
        )
        
        self.sampling_metrics.append(metrics)
        
        # File Logging
        summary_file = self.log_dir / "sampling" / f"{run_id}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Wandb Logging
        if self.use_wandb:
            wandb.log({
                "sampling_summary/total_steps": metrics.total_steps,
                "sampling_summary/total_time": metrics.total_time,
                "sampling_summary/final_quality": metrics.final_quality,
                "sampling_summary/efficiency_score": metrics.efficiency_score,
                "sampling_summary/steps_saved": metrics.steps_saved
            })
        
        self.logger.info(f"Sampling Zusammenfassung [{run_id}]: {metrics}")
    
    def log_training_step(self, 
                         episode: int,
                         step: int,
                         reward: float,
                         loss: float,
                         additional_metrics: Dict[str, Any] = None):
        """Loggt einen RL-Training-Schritt"""
        
        data = {
            "episode": episode,
            "step": step,
            "reward": reward,
            "loss": loss,
            "timestamp": datetime.now().isoformat()
        }
        
        if additional_metrics:
            data.update(additional_metrics)
        
        self.training_metrics.append(data)
        
        # File Logging
        log_file = self.log_dir / "training" / "training_log.jsonl"
        with open(log_file, 'a') as f:
            json.dump(data, f)
            f.write('\n')
        
        # Wandb Logging
        if self.use_wandb:
            wandb.log({
                "training/episode": episode,
                "training/step": step,
                "training/reward": reward,
                "training/loss": loss,
                **{f"training/{k}": v for k, v in (additional_metrics or {}).items()}
            })
        
        if step % 100 == 0:  # Nicht jeden Schritt loggen
            self.logger.info(f"Training Episode {episode}, Schritt {step}: Reward={reward:.3f}, Loss={loss:.3f}")
    
    def plot_sampling_progress(self, history: List[Dict[str, Any]], save_path: str = None):
        """Erstellt Plots für Sampling-Fortschritt"""
        
        if not history:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        steps = [h['step_number'] for h in history]
        
        # Plot 1: Noise Level
        noise_levels = [h['noise_level'] for h in history]
        axes[0, 0].plot(steps, noise_levels, 'b-', linewidth=2)
        axes[0, 0].set_title('Noise Level vs Steps')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Noise Level')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Qualität
        qualities = [h.get('quality_estimate', 0) for h in history]
        axes[0, 1].plot(steps, qualities, 'g-', linewidth=2)
        axes[0, 1].set_title('Quality Estimate vs Steps')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Quality')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Computation Time
        comp_times = [h['computation_time'] for h in history]
        axes[1, 0].plot(steps, comp_times, 'r-', linewidth=2)
        axes[1, 0].set_title('Computation Time vs Steps')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Time (s)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Latent Norm
        latent_norms = [h.get('latent_norm', 0) for h in history]
        axes[1, 1].plot(steps, latent_norms, 'm-', linewidth=2)
        axes[1, 1].set_title('Latent Norm vs Steps')
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Norm')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.log_dir / "plots" / f"sampling_progress_{int(datetime.now().timestamp())}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Sampling-Plot gespeichert: {save_path}")
        
        # Wandb Logging
        if self.use_wandb:
            wandb.log({"sampling_progress": wandb.Image(str(save_path))})
    
    def plot_training_progress(self, save_path: str = None):
        """Erstellt Plots für Training-Fortschritt"""
        
        if not self.training_metrics:
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        episodes = [m['episode'] for m in self.training_metrics]
        rewards = [m['reward'] for m in self.training_metrics]
        losses = [m['loss'] for m in self.training_metrics]
        
        # Plot 1: Rewards
        axes[0].plot(episodes, rewards, 'b-', linewidth=2)
        axes[0].set_title('Training Rewards')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Losses
        axes[1].plot(episodes, losses, 'r-', linewidth=2)
        axes[1].set_title('Training Losses')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.log_dir / "plots" / f"training_progress_{int(datetime.now().timestamp())}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training-Plot gespeichert: {save_path}")
        
        # Wandb Logging
        if self.use_wandb:
            wandb.log({"training_progress": wandb.Image(str(save_path))})
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Gibt Performance-Zusammenfassung zurück"""
        
        if not self.sampling_metrics:
            return {"message": "Keine Sampling-Metriken verfügbar"}
        
        # Berechne Statistiken
        total_steps = [m.total_steps for m in self.sampling_metrics]
        total_times = [m.total_time for m in self.sampling_metrics]
        final_qualities = [m.final_quality for m in self.sampling_metrics]
        efficiency_scores = [m.efficiency_score for m in self.sampling_metrics]
        
        return {
            "total_runs": len(self.sampling_metrics),
            "avg_steps": np.mean(total_steps),
            "avg_time": np.mean(total_times),
            "avg_quality": np.mean(final_qualities),
            "avg_efficiency": np.mean(efficiency_scores),
            "best_efficiency": max(efficiency_scores),
            "total_steps_saved": sum(m.steps_saved for m in self.sampling_metrics)
        }
    
    def save_experiment_summary(self):
        """Speichert Experiment-Zusammenfassung"""
        
        summary = {
            "experiment_info": {
                "start_time": datetime.now().isoformat(),
                "log_dir": str(self.log_dir),
                "wandb_enabled": self.use_wandb
            },
            "performance_summary": self.get_performance_summary(),
            "sampling_metrics": [m.to_dict() for m in self.sampling_metrics],
            "training_metrics_count": len(self.training_metrics)
        }
        
        summary_file = self.log_dir / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Experiment-Zusammenfassung gespeichert: {summary_file}")
        
        return summary

# Test-Funktion
def test_logging_system():
    """Testet das Logging-System"""
    logger = RLDiffusionLogger(log_dir="test_logs", use_wandb=False)
    
    # Simuliere Sampling-Schritte
    for i in range(5):
        step_data = {
            "step_number": i,
            "noise_level": 1.0 - i * 0.2,
            "quality_estimate": 0.5 + i * 0.1,
            "computation_time": 0.1 + np.random.normal(0, 0.02)
        }
        logger.log_sampling_step(step_data, "test_run_1")
    
    # Simuliere Sampling-Zusammenfassung
    summary = {
        "total_steps": 5,
        "total_computation_time": 0.5,
        "final_quality": 0.9,
        "efficiency_score": 1.8,
        "steps_saved": 5,
        "history": [
            {"step_number": i, "noise_level": 1.0 - i * 0.2, 
             "quality_estimate": 0.5 + i * 0.1, "computation_time": 0.1,
             "latent_norm": 100 + i * 10}
            for i in range(5)
        ]
    }
    logger.log_sampling_summary(summary, "test_run_1")
    
    # Simuliere Training-Schritte
    for episode in range(3):
        for step in range(10):
            reward = np.random.normal(0.5, 0.1)
            loss = np.random.exponential(0.5)
            logger.log_training_step(episode, step, reward, loss)
    
    # Erstelle Plots
    logger.plot_sampling_progress(summary["history"])
    logger.plot_training_progress()
    
    # Speichere Zusammenfassung
    exp_summary = logger.save_experiment_summary()
    print(f"Experiment-Zusammenfassung: {exp_summary}")
    
    print("✅ Logging-System erfolgreich getestet!")
    return True

if __name__ == "__main__":
    test_logging_system()
