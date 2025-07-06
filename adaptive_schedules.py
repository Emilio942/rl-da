"""
Adaptive Sampling Schedules für Phase 3
Implementiert dynamische Stop-Kriterien, Qualitäts-vs-Schritt-Analyse und Speed-Up-Evaluation
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
import logging
from dataclasses import dataclass
from enum import Enum
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
from collections import deque
import statistics

logger = logging.getLogger(__name__)

class StopCriterion(Enum):
    """Verschiedene Stop-Kriterien"""
    QUALITY_THRESHOLD = "quality_threshold"
    QUALITY_PLATEAU = "quality_plateau"
    DIMINISHING_RETURNS = "diminishing_returns"
    CONVERGENCE = "convergence"
    TIME_BUDGET = "time_budget"
    ADAPTIVE_THRESHOLD = "adaptive_threshold"

@dataclass
class QualityMetrics:
    """Qualitäts-Metriken für ein Sample"""
    perceptual_quality: float
    structural_similarity: float
    feature_diversity: float
    artifact_score: float
    overall_quality: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'perceptual_quality': self.perceptual_quality,
            'structural_similarity': self.structural_similarity,
            'feature_diversity': self.feature_diversity,
            'artifact_score': self.artifact_score,
            'overall_quality': self.overall_quality
        }

@dataclass
class ScheduleAnalysis:
    """Analyse-Ergebnisse für ein Sampling-Schedule"""
    total_steps: int
    total_time: float
    final_quality: float
    quality_trajectory: List[float]
    time_trajectory: List[float]
    step_efficiency: List[float]
    convergence_step: Optional[int]
    speed_up_factor: float
    quality_per_step: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_steps': self.total_steps,
            'total_time': self.total_time,
            'final_quality': self.final_quality,
            'quality_trajectory': self.quality_trajectory,
            'time_trajectory': self.time_trajectory,
            'step_efficiency': self.step_efficiency,
            'convergence_step': self.convergence_step,
            'speed_up_factor': self.speed_up_factor,
            'quality_per_step': self.quality_per_step
        }

class AdaptiveStopCriteria:
    """
    Implementiert verschiedene adaptive Stop-Kriterien für Diffusion-Sampling
    """
    
    def __init__(self,
                 quality_threshold: float = 0.85,
                 plateau_patience: int = 3,
                 plateau_threshold: float = 0.01,
                 convergence_threshold: float = 0.005,
                 time_budget: float = 30.0,
                 min_steps: int = 5,
                 max_steps: int = 50):
        
        self.quality_threshold = quality_threshold
        self.plateau_patience = plateau_patience
        self.plateau_threshold = plateau_threshold
        self.convergence_threshold = convergence_threshold
        self.time_budget = time_budget
        self.min_steps = min_steps
        self.max_steps = max_steps
        
        # Tracking
        self.quality_history = deque(maxlen=plateau_patience + 1)
        self.time_history = []
        self.step_count = 0
        self.start_time = None
        
    def reset(self):
        """Reset für neuen Sampling-Durchgang"""
        self.quality_history.clear()
        self.time_history.clear()
        self.step_count = 0
        self.start_time = time.time()
        
    def should_stop(self, 
                   quality: float, 
                   latent: torch.Tensor,
                   criterion: StopCriterion = StopCriterion.QUALITY_THRESHOLD) -> Tuple[bool, str]:
        """
        Prüft ob Sampling gestoppt werden sollte
        
        Returns:
            (should_stop, reason)
        """
        self.step_count += 1
        current_time = time.time()
        
        if self.start_time is None:
            self.start_time = current_time
        
        elapsed_time = current_time - self.start_time
        self.time_history.append(elapsed_time)
        self.quality_history.append(quality)
        
        # Mindest-Schritte prüfen
        if self.step_count < self.min_steps:
            return False, f"Minimum steps not reached ({self.step_count}/{self.min_steps})"
        
        # Maximum-Schritte prüfen
        if self.step_count >= self.max_steps:
            return True, f"Maximum steps reached ({self.max_steps})"
            
        # Zeit-Budget prüfen
        if elapsed_time > self.time_budget:
            return True, f"Time budget exceeded ({elapsed_time:.2f}s > {self.time_budget}s)"
        
        # Kriterium-spezifische Prüfungen
        if criterion == StopCriterion.QUALITY_THRESHOLD:
            if quality >= self.quality_threshold:
                return True, f"Quality threshold reached ({quality:.4f} >= {self.quality_threshold})"
                
        elif criterion == StopCriterion.QUALITY_PLATEAU:
            if len(self.quality_history) >= self.plateau_patience:
                recent_qualities = list(self.quality_history)
                quality_change = max(recent_qualities) - min(recent_qualities)
                if quality_change < self.plateau_threshold:
                    return True, f"Quality plateau detected (change: {quality_change:.4f})"
                    
        elif criterion == StopCriterion.CONVERGENCE:
            if len(self.quality_history) >= 3:
                recent_qualities = list(self.quality_history)[-3:]
                gradient = np.gradient(recent_qualities)
                if np.mean(np.abs(gradient)) < self.convergence_threshold:
                    return True, f"Convergence detected (gradient: {np.mean(gradient):.6f})"
                    
        elif criterion == StopCriterion.DIMINISHING_RETURNS:
            if len(self.quality_history) >= 3:
                recent_qualities = list(self.quality_history)
                efficiency = self._calculate_efficiency(recent_qualities)
                if efficiency < 0.01:  # Weniger als 1% Verbesserung pro Schritt
                    return True, f"Diminishing returns detected (efficiency: {efficiency:.4f})"
                    
        elif criterion == StopCriterion.ADAPTIVE_THRESHOLD:
            adaptive_threshold = self._calculate_adaptive_threshold()
            if quality >= adaptive_threshold:
                return True, f"Adaptive threshold reached ({quality:.4f} >= {adaptive_threshold:.4f})"
        
        return False, "Continue sampling"
    
    def _calculate_efficiency(self, qualities: List[float]) -> float:
        """Berechnet die Effizienz der letzten Schritte"""
        if len(qualities) < 2:
            return 1.0
        
        improvements = np.diff(qualities)
        return np.mean(improvements) if len(improvements) > 0 else 0.0
    
    def _calculate_adaptive_threshold(self) -> float:
        """Berechnet einen adaptiven Qualitäts-Schwellenwert"""
        if len(self.quality_history) < 3:
            return self.quality_threshold
        
        # Nutze Qualitätstrend zur Anpassung des Schwellenwerts
        qualities = list(self.quality_history)
        trend = np.polyfit(range(len(qualities)), qualities, 1)[0]
        
        # Senke Schwellenwert wenn Trend positiv ist
        if trend > 0:
            return self.quality_threshold * (1 - trend * 0.1)
        else:
            return self.quality_threshold

class QualityStepAnalyzer:
    """
    Analysiert Qualität vs. Schrittzahl für verschiedene Sampling-Strategien
    """
    
    def __init__(self, log_dir: str = "analysis_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Sammel-Daten für verschiedene Strategien
        self.strategy_data = {}
        
    def collect_run_data(self, 
                        strategy_name: str,
                        steps: int,
                        quality: float,
                        time_taken: float,
                        quality_trajectory: List[float],
                        time_trajectory: List[float]):
        """Sammelt Daten für einen Sampling-Durchgang"""
        
        if strategy_name not in self.strategy_data:
            self.strategy_data[strategy_name] = {
                'runs': [],
                'avg_steps': [],
                'avg_quality': [],
                'avg_time': [],
                'quality_trajectories': [],
                'time_trajectories': []
            }
        
        self.strategy_data[strategy_name]['runs'].append({
            'steps': steps,
            'quality': quality,
            'time': time_taken,
            'quality_trajectory': quality_trajectory,
            'time_trajectory': time_trajectory
        })
        
        # Aktualisiere Durchschnitte
        runs = self.strategy_data[strategy_name]['runs']
        self.strategy_data[strategy_name]['avg_steps'] = np.mean([r['steps'] for r in runs])
        self.strategy_data[strategy_name]['avg_quality'] = np.mean([r['quality'] for r in runs])
        self.strategy_data[strategy_name]['avg_time'] = np.mean([r['time'] for r in runs])
        
    def analyze_quality_vs_steps(self, baseline_steps: int = 50) -> Dict[str, Any]:
        """Analysiert Qualität vs. Schrittzahl für alle Strategien"""
        
        analysis = {}
        
        for strategy_name, data in self.strategy_data.items():
            if not data['runs']:
                continue
                
            runs = data['runs']
            
            # Grundlegende Statistiken
            steps_list = [r['steps'] for r in runs]
            quality_list = [r['quality'] for r in runs]
            time_list = [r['time'] for r in runs]
            
            # Effizienz-Metriken
            avg_steps = np.mean(steps_list)
            avg_quality = np.mean(quality_list)
            avg_time = np.mean(time_list)
            
            # Speed-up gegenüber Baseline
            speed_up = baseline_steps / avg_steps if avg_steps > 0 else 0
            
            # Qualität pro Schritt
            quality_per_step = avg_quality / avg_steps if avg_steps > 0 else 0
            
            # Effizienz-Score (Qualität / Zeit)
            efficiency_score = avg_quality / avg_time if avg_time > 0 else 0
            
            # Konsistenz (Standardabweichung)
            steps_std = np.std(steps_list)
            quality_std = np.std(quality_list)
            
            analysis[strategy_name] = {
                'avg_steps': avg_steps,
                'avg_quality': avg_quality,
                'avg_time': avg_time,
                'speed_up_factor': speed_up,
                'quality_per_step': quality_per_step,
                'efficiency_score': efficiency_score,
                'steps_std': steps_std,
                'quality_std': quality_std,
                'num_runs': len(runs)
            }
        
        return analysis
    
    def generate_comparison_plots(self, save_path: Optional[str] = None):
        """Generiert Vergleichs-Plots für verschiedene Strategien"""
        
        if not self.strategy_data:
            logger.warning("Keine Daten für Plots verfügbar")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Durchschnittliche Schritte vs. Qualität
        ax1 = axes[0, 0]
        for strategy_name, data in self.strategy_data.items():
            if data['runs']:
                steps = [r['steps'] for r in data['runs']]
                quality = [r['quality'] for r in data['runs']]
                ax1.scatter(steps, quality, label=strategy_name, alpha=0.7)
        
        ax1.set_xlabel('Anzahl Schritte')
        ax1.set_ylabel('Finale Qualität')
        ax1.set_title('Schritte vs. Qualität')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Zeit vs. Qualität
        ax2 = axes[0, 1]
        for strategy_name, data in self.strategy_data.items():
            if data['runs']:
                time_vals = [r['time'] for r in data['runs']]
                quality = [r['quality'] for r in data['runs']]
                ax2.scatter(time_vals, quality, label=strategy_name, alpha=0.7)
        
        ax2.set_xlabel('Zeit (s)')
        ax2.set_ylabel('Finale Qualität')
        ax2.set_title('Zeit vs. Qualität')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Qualitäts-Trajektorien
        ax3 = axes[1, 0]
        for strategy_name, data in self.strategy_data.items():
            if data['runs']:
                # Durchschnittliche Qualitäts-Trajektorie
                trajectories = [r['quality_trajectory'] for r in data['runs']]
                if trajectories:
                    # Interpoliere auf gemeinsame Länge
                    max_len = max(len(t) for t in trajectories)
                    interp_trajectories = []
                    for traj in trajectories:
                        if len(traj) < max_len:
                            # Interpoliere
                            x_old = np.linspace(0, 1, len(traj))
                            x_new = np.linspace(0, 1, max_len)
                            traj_interp = np.interp(x_new, x_old, traj)
                            interp_trajectories.append(traj_interp)
                        else:
                            interp_trajectories.append(traj)
                    
                    avg_trajectory = np.mean(interp_trajectories, axis=0)
                    ax3.plot(avg_trajectory, label=strategy_name, linewidth=2)
        
        ax3.set_xlabel('Schritt')
        ax3.set_ylabel('Qualität')
        ax3.set_title('Durchschnittliche Qualitäts-Trajektorien')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Effizienz-Vergleich
        ax4 = axes[1, 1]
        strategies = list(self.strategy_data.keys())
        efficiencies = []
        
        for strategy_name in strategies:
            data = self.strategy_data[strategy_name]
            if data['runs']:
                runs = data['runs']
                avg_quality = np.mean([r['quality'] for r in runs])
                avg_time = np.mean([r['time'] for r in runs])
                efficiency = avg_quality / avg_time if avg_time > 0 else 0
                efficiencies.append(efficiency)
            else:
                efficiencies.append(0)
        
        ax4.bar(strategies, efficiencies)
        ax4.set_ylabel('Effizienz (Qualität/Zeit)')
        ax4.set_title('Effizienz-Vergleich')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plots gespeichert: {save_path}")
        else:
            save_path = self.log_dir / "quality_step_analysis.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plots gespeichert: {save_path}")
        
        plt.close()
    
    def save_analysis_report(self, filename: Optional[str] = None):
        """Speichert detaillierten Analyse-Bericht"""
        
        if filename is None:
            filename = f"quality_step_analysis_{int(time.time())}.json"
        
        analysis_data = self.analyze_quality_vs_steps()
        
        report = {
            "timestamp": time.time(),
            "analysis_summary": analysis_data,
            "raw_data": self.strategy_data
        }
        
        report_path = self.log_dir / filename
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Analyse-Bericht gespeichert: {report_path}")
        return report_path

class SpeedUpEvaluator:
    """
    Evaluiert Speed-Up-Faktoren für verschiedene adaptive Sampling-Strategien
    """
    
    def __init__(self, baseline_steps: int = 50):
        self.baseline_steps = baseline_steps
        self.evaluations = []
        
    def evaluate_strategy(self,
                         strategy_name: str,
                         adaptive_steps: int,
                         adaptive_quality: float,
                         adaptive_time: float,
                         baseline_quality: float,
                         baseline_time: float) -> Dict[str, Any]:
        """Evaluiert eine Strategie gegen Baseline"""
        
        # Speed-Up-Metriken
        step_speedup = self.baseline_steps / adaptive_steps if adaptive_steps > 0 else 0
        time_speedup = baseline_time / adaptive_time if adaptive_time > 0 else 0
        
        # Qualitäts-Beibehaltung
        quality_retention = adaptive_quality / baseline_quality if baseline_quality > 0 else 0
        
        # Effizienz-Score
        baseline_efficiency = baseline_quality / baseline_time if baseline_time > 0 else 0
        adaptive_efficiency = adaptive_quality / adaptive_time if adaptive_time > 0 else 0
        efficiency_gain = adaptive_efficiency / baseline_efficiency if baseline_efficiency > 0 else 0
        
        # Pareto-Effizienz (Qualität vs. Zeit)
        pareto_score = (quality_retention * time_speedup) / 2
        
        evaluation = {
            'strategy_name': strategy_name,
            'step_speedup': step_speedup,
            'time_speedup': time_speedup,
            'quality_retention': quality_retention,
            'efficiency_gain': efficiency_gain,
            'pareto_score': pareto_score,
            'adaptive_steps': adaptive_steps,
            'adaptive_quality': adaptive_quality,
            'adaptive_time': adaptive_time,
            'baseline_steps': self.baseline_steps,
            'baseline_quality': baseline_quality,
            'baseline_time': baseline_time
        }
        
        self.evaluations.append(evaluation)
        return evaluation
    
    def get_best_strategies(self, metric: str = 'pareto_score', top_k: int = 5) -> List[Dict[str, Any]]:
        """Gibt die besten Strategien nach Metrik zurück"""
        
        if not self.evaluations:
            return []
        
        sorted_evaluations = sorted(self.evaluations, 
                                  key=lambda x: x.get(metric, 0), 
                                  reverse=True)
        
        return sorted_evaluations[:top_k]
    
    def generate_speedup_report(self) -> Dict[str, Any]:
        """Generiert umfassenden Speed-Up-Bericht"""
        
        if not self.evaluations:
            return {"error": "Keine Evaluierungen verfügbar"}
        
        # Durchschnittliche Metriken
        avg_step_speedup = np.mean([e['step_speedup'] for e in self.evaluations])
        avg_time_speedup = np.mean([e['time_speedup'] for e in self.evaluations])
        avg_quality_retention = np.mean([e['quality_retention'] for e in self.evaluations])
        avg_efficiency_gain = np.mean([e['efficiency_gain'] for e in self.evaluations])
        
        # Beste Strategien
        best_step_speedup = self.get_best_strategies('step_speedup', 3)
        best_time_speedup = self.get_best_strategies('time_speedup', 3)
        best_quality_retention = self.get_best_strategies('quality_retention', 3)
        best_pareto = self.get_best_strategies('pareto_score', 3)
        
        report = {
            'summary': {
                'total_strategies_evaluated': len(self.evaluations),
                'avg_step_speedup': avg_step_speedup,
                'avg_time_speedup': avg_time_speedup,
                'avg_quality_retention': avg_quality_retention,
                'avg_efficiency_gain': avg_efficiency_gain
            },
            'best_strategies': {
                'step_speedup': best_step_speedup,
                'time_speedup': best_time_speedup,
                'quality_retention': best_quality_retention,
                'pareto_efficiency': best_pareto
            },
            'all_evaluations': self.evaluations
        }
        
        return report

# Test-Funktionen
def test_adaptive_stop_criteria():
    """Testet die adaptiven Stop-Kriterien"""
    logger.info("Teste adaptive Stop-Kriterien...")
    
    criteria = AdaptiveStopCriteria(
        quality_threshold=0.8,
        plateau_patience=3,
        min_steps=5,
        max_steps=20
    )
    
    # Simuliere Sampling-Verlauf
    quality_trajectory = [0.3, 0.5, 0.65, 0.75, 0.82, 0.83, 0.83, 0.83]
    latent = torch.randn(1, 4, 32, 32)
    
    criteria.reset()
    
    for i, quality in enumerate(quality_trajectory):
        should_stop, reason = criteria.should_stop(quality, latent, StopCriterion.QUALITY_THRESHOLD)
        logger.info(f"Schritt {i+1}: Qualität={quality:.3f}, Stop={should_stop}, Grund='{reason}'")
        
        if should_stop:
            break
    
    logger.info("✅ Adaptive Stop-Kriterien erfolgreich getestet!")

def test_quality_step_analyzer():
    """Testet den Qualitäts-Schritt-Analyzer"""
    logger.info("Teste Qualitäts-Schritt-Analyzer...")
    
    analyzer = QualityStepAnalyzer()
    
    # Simuliere Daten für verschiedene Strategien
    strategies = ['Baseline', 'Adaptive_Threshold', 'Quality_Plateau', 'Convergence']
    
    for strategy in strategies:
        # Simuliere mehrere Durchgänge
        for run in range(5):
            if strategy == 'Baseline':
                steps = 50
                quality = 0.85 + np.random.normal(0, 0.05)
                time_taken = 10.0 + np.random.normal(0, 1.0)
            else:
                steps = np.random.randint(15, 35)
                quality = 0.80 + np.random.normal(0, 0.05)
                time_taken = steps * 0.2 + np.random.normal(0, 0.5)
            
            # Simuliere Trajektorien
            quality_traj = np.linspace(0.2, quality, steps)
            time_traj = np.linspace(0, time_taken, steps)
            
            analyzer.collect_run_data(
                strategy, steps, quality, time_taken, 
                quality_traj.tolist(), time_traj.tolist()
            )
    
    # Analysiere Ergebnisse
    analysis = analyzer.analyze_quality_vs_steps()
    logger.info(f"Analyse-Ergebnisse: {analysis}")
    
    # Generiere Plots
    analyzer.generate_comparison_plots()
    
    # Speichere Bericht
    analyzer.save_analysis_report()
    
    logger.info("✅ Qualitäts-Schritt-Analyzer erfolgreich getestet!")

def test_speedup_evaluator():
    """Testet den Speed-Up-Evaluator"""
    logger.info("Teste Speed-Up-Evaluator...")
    
    evaluator = SpeedUpEvaluator(baseline_steps=50)
    
    # Simuliere Evaluierungen
    strategies = [
        ('Adaptive_Threshold', 25, 0.82, 5.0),
        ('Quality_Plateau', 30, 0.80, 6.0),
        ('Convergence', 20, 0.78, 4.0),
        ('Diminishing_Returns', 35, 0.85, 7.0)
    ]
    
    baseline_quality = 0.85
    baseline_time = 10.0
    
    for name, steps, quality, time_taken in strategies:
        evaluation = evaluator.evaluate_strategy(
            name, steps, quality, time_taken,
            baseline_quality, baseline_time
        )
        logger.info(f"Strategie '{name}': {evaluation}")
    
    # Beste Strategien
    best_strategies = evaluator.get_best_strategies('pareto_score', 3)
    logger.info(f"Beste Strategien: {best_strategies}")
    
    # Vollständiger Bericht
    report = evaluator.generate_speedup_report()
    logger.info(f"Speed-Up-Bericht: {report['summary']}")
    
    logger.info("✅ Speed-Up-Evaluator erfolgreich getestet!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    test_adaptive_stop_criteria()
    test_quality_step_analyzer()
    test_speedup_evaluator()
    
    logger.info("🎉 Alle Tests für Phase 3 erfolgreich!")
