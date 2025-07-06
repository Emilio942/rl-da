"""
Test-Suite für Phase 3: Adaptive Sampling Schedules
Integriert alle Komponenten und testet die Funktionalität
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import logging
import time
import random
from typing import Dict, Any, List
from pathlib import Path

# Importiere alle benötigten Module
from adaptive_schedules import (
    AdaptiveStopCriteria, 
    QualityStepAnalyzer, 
    SpeedUpEvaluator,
    StopCriterion,
    QualityMetrics,
    ScheduleAnalysis
)
from adaptive_sampling import AdaptiveSamplingLoop, SamplingAction
from diffusion_model import DiffusionModelWrapper
from logging_system import RLDiffusionLogger, SamplingMetrics
from mdp_definition import MDPDefinition, SamplingState
from reward_function import RewardFunction
from baseline_policies import RandomPolicy, HeuristicPolicy
from rl_training import RLTrainer

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Phase3TestSuite:
    """Umfassende Test-Suite für Phase 3"""
    
    def __init__(self):
        self.test_logs_dir = Path("test_logs/phase3")
        self.test_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialisiere Komponenten
        self.setup_components()
        
    def setup_components(self):
        """Initialisiert alle Test-Komponenten"""
        logger.info("Initialisiere Test-Komponenten...")
        
        # Diffusion Model (Mock für Tests)
        self.diffusion_model = self.create_mock_diffusion_model()
        
        # Logging System
        self.logger = RLDiffusionLogger(
            log_dir=str(self.test_logs_dir / "logs"),
            use_wandb=False
        )
        self.run_id = f"test_run_{random.randint(1000, 9999)}"
        
        # Adaptive Sampling Loop
        self.sampling_loop = AdaptiveSamplingLoop(
            diffusion_model=self.diffusion_model,
            max_steps=30,
            min_steps=5,
            log_dir=str(self.test_logs_dir / "sampling_logs")
        )
        
        # MDP Definition
        self.mdp = MDPDefinition(
            state_dim=8,
            action_dim=3,
            max_steps=30
        )
        
        # Reward Function
        self.reward_function = RewardFunction(
            quality_weight=1.0,
            efficiency_weight=0.5,
            step_penalty=0.02
        )
        
        # Baseline Policies
        self.random_policy = RandomPolicy(self.mdp)
        self.heuristic_policy = HeuristicPolicy(self.mdp)
        
        # Phase 3 Komponenten
        self.stop_criteria = AdaptiveStopCriteria(
            quality_threshold=0.8,
            plateau_patience=3,
            min_steps=5,
            max_steps=30
        )
        
        self.quality_analyzer = QualityStepAnalyzer(
            log_dir=str(self.test_logs_dir / "analysis")
        )
        
        self.speedup_evaluator = SpeedUpEvaluator(baseline_steps=50)
        
        logger.info("✅ Komponenten erfolgreich initialisiert!")
    
    def create_mock_diffusion_model(self):
        """Erstellt Mock Diffusion Model für Tests"""
        class MockDiffusionModel:
            def __init__(self):
                self.model_id = "test_model_v1"
                self.device = "cpu"
                
            def get_model_info(self):
                return {
                    "model_id": self.model_id,
                    "device": self.device,
                    "scheduler": "DDIM"
                }
            
            def sample(self, prompt, num_steps=50):
                # Simuliere Sampling
                return torch.randn(1, 3, 512, 512)
        
        return MockDiffusionModel()
    
    def test_adaptive_stop_criteria(self) -> Dict[str, Any]:
        """Test 3.1: Dynamische Stop-Kriterien"""
        logger.info("🧪 Test 3.1: Dynamische Stop-Kriterien")
        
        results = {}
        
        # Test verschiedene Stop-Kriterien
        criteria_tests = [
            (StopCriterion.QUALITY_THRESHOLD, "Quality Threshold"),
            (StopCriterion.QUALITY_PLATEAU, "Quality Plateau"),
            (StopCriterion.CONVERGENCE, "Convergence"),
            (StopCriterion.DIMINISHING_RETURNS, "Diminishing Returns"),
            (StopCriterion.ADAPTIVE_THRESHOLD, "Adaptive Threshold")
        ]
        
        for criterion, name in criteria_tests:
            logger.info(f"  Teste {name}...")
            
            # Simuliere verschiedene Qualitäts-Verläufe
            quality_scenarios = [
                # Schnelle Konvergenz
                [0.2, 0.5, 0.7, 0.85, 0.87, 0.87, 0.87],
                # Langsame Steigerung
                [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.78, 0.8, 0.81],
                # Plateau
                [0.3, 0.6, 0.75, 0.76, 0.76, 0.76, 0.76, 0.76],
                # Oszillation
                [0.3, 0.6, 0.5, 0.7, 0.6, 0.75, 0.7, 0.8]
            ]
            
            scenario_results = []
            
            for i, quality_traj in enumerate(quality_scenarios):
                self.stop_criteria.reset()
                stop_step = None
                stop_reason = None
                
                for step, quality in enumerate(quality_traj):
                    latent = torch.randn(1, 4, 32, 32)
                    should_stop, reason = self.stop_criteria.should_stop(
                        quality, latent, criterion
                    )
                    
                    if should_stop:
                        stop_step = step + 1
                        stop_reason = reason
                        break
                
                scenario_results.append({
                    'scenario': i + 1,
                    'total_steps': len(quality_traj),
                    'stop_step': stop_step,
                    'stop_reason': stop_reason,
                    'final_quality': quality_traj[-1],
                    'stopped_early': stop_step is not None and stop_step < len(quality_traj)
                })
            
            results[name] = {
                'criterion': criterion.value,
                'scenarios': scenario_results,
                'early_stop_rate': sum(1 for r in scenario_results if r['stopped_early']) / len(scenario_results)
            }
            
            logger.info(f"    {name}: {results[name]['early_stop_rate']:.1%} early stops")
        
        # Speichere Ergebnisse
        self.save_test_results("adaptive_stop_criteria", results)
        
        logger.info("✅ Test 3.1 abgeschlossen!")
        return results
    
    def test_quality_step_analysis(self) -> Dict[str, Any]:
        """Test 3.2: Qualitäts-vs-Schritt-Analyse"""
        logger.info("🧪 Test 3.2: Qualitäts-vs-Schritt-Analyse")
        
        # Simuliere verschiedene Sampling-Strategien
        strategies = {
            'Baseline_Fixed': {'steps': 50, 'quality_range': (0.82, 0.88)},
            'Adaptive_Threshold': {'steps': (20, 35), 'quality_range': (0.78, 0.85)},
            'Quality_Plateau': {'steps': (15, 40), 'quality_range': (0.75, 0.82)},
            'Convergence': {'steps': (10, 30), 'quality_range': (0.70, 0.85)},
            'Diminishing_Returns': {'steps': (25, 45), 'quality_range': (0.80, 0.87)}
        }
        
        # Sammle Daten für jede Strategie
        for strategy_name, config in strategies.items():
            logger.info(f"  Sammle Daten für {strategy_name}...")
            
            # Simuliere mehrere Durchgänge
            for run in range(10):
                if isinstance(config['steps'], tuple):
                    steps = np.random.randint(config['steps'][0], config['steps'][1])
                else:
                    steps = config['steps']
                
                quality = np.random.uniform(config['quality_range'][0], config['quality_range'][1])
                time_taken = steps * (0.15 + np.random.normal(0, 0.02))
                
                # Simuliere realistische Qualitäts-Trajektorie
                quality_traj = self.simulate_quality_trajectory(steps, quality)
                time_traj = np.linspace(0, time_taken, steps)
                
                self.quality_analyzer.collect_run_data(
                    strategy_name, steps, quality, time_taken,
                    quality_traj, time_traj.tolist()
                )
        
        # Analysiere gesammelte Daten
        analysis_results = self.quality_analyzer.analyze_quality_vs_steps(baseline_steps=50)
        
        # Generiere Visualisierungen
        self.quality_analyzer.generate_comparison_plots(
            save_path=str(self.test_logs_dir / "quality_step_comparison.png")
        )
        
        # Speichere detaillierten Bericht
        report_path = self.quality_analyzer.save_analysis_report("phase3_quality_analysis.json")
        
        logger.info("✅ Test 3.2 abgeschlossen!")
        logger.info(f"  Analyse-Bericht: {report_path}")
        
        return analysis_results
    
    def test_speedup_evaluation(self, quality_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Test 3.3: Speed-Up-Faktor-Evaluation"""
        logger.info("🧪 Test 3.3: Speed-Up-Faktor-Evaluation")
        
        # Baseline-Werte (Fixed-Step Sampling)
        baseline_steps = 50
        baseline_quality = 0.85
        baseline_time = 10.0
        
        # Evaluiere jede Strategie
        speedup_results = {}
        
        for strategy_name, metrics in quality_analysis.items():
            if strategy_name == 'Baseline_Fixed':
                continue
                
            evaluation = self.speedup_evaluator.evaluate_strategy(
                strategy_name=strategy_name,
                adaptive_steps=int(metrics['avg_steps']),
                adaptive_quality=metrics['avg_quality'],
                adaptive_time=metrics['avg_time'],
                baseline_quality=baseline_quality,
                baseline_time=baseline_time
            )
            
            speedup_results[strategy_name] = evaluation
            
            logger.info(f"  {strategy_name}:")
            logger.info(f"    Step Speed-Up: {evaluation['step_speedup']:.2f}x")
            logger.info(f"    Time Speed-Up: {evaluation['time_speedup']:.2f}x")
            logger.info(f"    Quality Retention: {evaluation['quality_retention']:.3f}")
            logger.info(f"    Pareto Score: {evaluation['pareto_score']:.3f}")
        
        # Generiere umfassenden Bericht
        full_report = self.speedup_evaluator.generate_speedup_report()
        
        # Beste Strategien identifizieren
        best_strategies = self.speedup_evaluator.get_best_strategies('pareto_score', 3)
        
        logger.info("\n🏆 Beste Strategien (Pareto-Effizienz):")
        for i, strategy in enumerate(best_strategies, 1):
            logger.info(f"  {i}. {strategy['strategy_name']}: {strategy['pareto_score']:.3f}")
        
        # Speichere Ergebnisse
        self.save_test_results("speedup_evaluation", full_report)
        
        logger.info("✅ Test 3.3 abgeschlossen!")
        return full_report
    
    def test_integrated_sampling_workflow(self) -> Dict[str, Any]:
        """Test: Integrierter Sampling-Workflow mit adaptiven Schedules"""
        logger.info("🧪 Test: Integrierter Sampling-Workflow")
        
        # Test verschiedene Policies mit adaptiven Stop-Kriterien
        policies = [
            ('Random', self.random_policy),
            ('Heuristic', self.heuristic_policy)
        ]
        
        criteria = [
            StopCriterion.QUALITY_THRESHOLD,
            StopCriterion.QUALITY_PLATEAU,
            StopCriterion.CONVERGENCE
        ]
        
        workflow_results = []
        
        for policy_name, policy in policies:
            for criterion in criteria:
                logger.info(f"  Teste {policy_name} mit {criterion.value}...")
                
                # Simuliere Sampling mit RL-Policy und adaptiven Stop-Kriterien
                result = self.run_adaptive_sampling_episode(
                    policy, criterion, f"{policy_name}_{criterion.value}"
                )
                
                workflow_results.append(result)
        
        # Aggregiere Ergebnisse
        summary = self.aggregate_workflow_results(workflow_results)
        
        logger.info("✅ Integrierter Workflow-Test abgeschlossen!")
        return summary
    
    def run_adaptive_sampling_episode(self, policy, stop_criterion, run_id: str) -> Dict[str, Any]:
        """Führt einen adaptiven Sampling-Durchgang durch"""
        
        # Reset Komponenten
        self.sampling_loop.reset_state()
        self.stop_criteria.reset()
        
        # Initialer Zustand
        latent = torch.randn(1, 4, 32, 32)
        episode_data = {
            'run_id': run_id,
            'policy': policy.__class__.__name__,
            'stop_criterion': stop_criterion.value,
            'steps': [],
            'rewards': [],
            'quality_trajectory': [],
            'time_trajectory': [],
            'total_reward': 0.0,
            'final_quality': 0.0,
            'total_steps': 0,
            'total_time': 0.0
        }
        
        start_time = time.time()
        
        for step in range(self.sampling_loop.max_steps):
            # Aktuelle RL-State
            rl_state = self.sampling_loop.get_state_for_rl()
            state = SamplingState(
                step_number=rl_state['step_number'],
                progress=rl_state['progress'],
                noise_level=rl_state['noise_level'],
                latent_norm=rl_state['latent_norm'],
                quality_estimate=rl_state.get('quality_estimate', 0.5),
                computation_time=rl_state['avg_computation_time'],
                steps_remaining=rl_state['steps_remaining']
            )
            
            # Policy-Entscheidung
            action_idx = policy.select_action(state)
            action = SamplingAction(list(SamplingAction)[action_idx])
            
            # Sampling-Schritt
            timestep = 1000 - step * 20  # Vereinfachter Timestep
            latent, sampling_finished = self.sampling_loop.step(latent, timestep, action)
            
            # Qualität bewerten
            quality = self.sampling_loop.quality_estimator(latent)
            
            # Reward berechnen
            reward_components = self.reward_function.calculate_reward(
                state, action_idx, quality, step >= 5
            )
            
            # Adaptive Stop-Kriterium prüfen
            should_stop, stop_reason = self.stop_criteria.should_stop(quality, latent, stop_criterion)
            
            # Daten sammeln
            episode_data['steps'].append(step)
            episode_data['rewards'].append(reward_components.total_reward)
            episode_data['quality_trajectory'].append(quality)
            episode_data['time_trajectory'].append(time.time() - start_time)
            episode_data['total_reward'] += reward_components.total_reward
            
            # Beenden wenn nötig
            if should_stop or sampling_finished or action == SamplingAction.STOP:
                episode_data['stop_reason'] = stop_reason if should_stop else 'policy_stop'
                break
        
        # Finale Metriken
        episode_data['final_quality'] = quality
        episode_data['total_steps'] = len(episode_data['steps'])
        episode_data['total_time'] = time.time() - start_time
        
        return episode_data
    
    def aggregate_workflow_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregiert Workflow-Ergebnisse"""
        
        # Gruppiere nach Policy und Kriterium
        grouped = {}
        for result in results:
            key = f"{result['policy']}_{result['stop_criterion']}"
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result)
        
        # Berechne Durchschnitte
        summary = {}
        for key, group in grouped.items():
            avg_steps = np.mean([r['total_steps'] for r in group])
            avg_quality = np.mean([r['final_quality'] for r in group])
            avg_time = np.mean([r['total_time'] for r in group])
            avg_reward = np.mean([r['total_reward'] for r in group])
            
            summary[key] = {
                'avg_steps': avg_steps,
                'avg_quality': avg_quality,
                'avg_time': avg_time,
                'avg_reward': avg_reward,
                'num_runs': len(group),
                'efficiency': avg_quality / avg_time if avg_time > 0 else 0
            }
        
        return summary
    
    def simulate_quality_trajectory(self, steps: int, final_quality: float) -> List[float]:
        """Simuliert realistische Qualitäts-Trajektorie"""
        
        # Verschiedene Muster
        patterns = ['linear', 'exponential', 'sigmoid', 'plateau']
        pattern = np.random.choice(patterns)
        
        x = np.linspace(0, 1, steps)
        
        if pattern == 'linear':
            trajectory = 0.2 + (final_quality - 0.2) * x
        elif pattern == 'exponential':
            trajectory = final_quality * (1 - np.exp(-3 * x))
        elif pattern == 'sigmoid':
            trajectory = final_quality / (1 + np.exp(-10 * (x - 0.5)))
        else:  # plateau
            plateau_start = int(steps * 0.7)
            trajectory = np.concatenate([
                np.linspace(0.2, final_quality, plateau_start),
                np.full(steps - plateau_start, final_quality)
            ])
        
        # Füge Noise hinzu
        noise = np.random.normal(0, 0.02, steps)
        trajectory += noise
        
        # Stelle sicher, dass Werte im gültigen Bereich sind
        trajectory = np.clip(trajectory, 0.1, 1.0)
        
        return trajectory.tolist()
    
    def save_test_results(self, test_name: str, results: Dict[str, Any]):
        """Speichert Test-Ergebnisse"""
        
        results_path = self.test_logs_dir / f"{test_name}_results.json"
        with open(results_path, 'w') as f:
            import json
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Ergebnisse gespeichert: {results_path}")
    
    def run_full_phase3_test(self) -> Dict[str, Any]:
        """Führt alle Phase 3 Tests durch"""
        logger.info("🚀 Starte vollständige Phase 3 Test-Suite")
        
        start_time = time.time()
        
        # Test 3.1: Adaptive Stop-Kriterien
        stop_criteria_results = self.test_adaptive_stop_criteria()
        
        # Test 3.2: Qualitäts-vs-Schritt-Analyse
        quality_analysis_results = self.test_quality_step_analysis()
        
        # Test 3.3: Speed-Up-Evaluation
        speedup_results = self.test_speedup_evaluation(quality_analysis_results)
        
        # Integrierter Workflow-Test
        workflow_results = self.test_integrated_sampling_workflow()
        
        # Gesamtbericht
        total_time = time.time() - start_time
        
        final_report = {
            'phase': 3,
            'test_suite': 'Adaptive Sampling Schedules',
            'execution_time': total_time,
            'tests': {
                'adaptive_stop_criteria': stop_criteria_results,
                'quality_analysis': quality_analysis_results,
                'speedup_evaluation': speedup_results,
                'integrated_workflow': workflow_results
            },
            'summary': {
                'total_strategies_tested': len(quality_analysis_results),
                'best_speedup_strategy': max(speedup_results['all_evaluations'], 
                                           key=lambda x: x['pareto_score'])['strategy_name'],
                'avg_speedup_factor': np.mean([e['step_speedup'] for e in speedup_results['all_evaluations']]),
                'quality_retention_rate': np.mean([e['quality_retention'] for e in speedup_results['all_evaluations']])
            }
        }
        
        # Speichere Gesamtbericht
        self.save_test_results("phase3_final_report", final_report)
        
        logger.info("🎉 Phase 3 Test-Suite erfolgreich abgeschlossen!")
        logger.info(f"⏱️  Gesamtzeit: {total_time:.2f}s")
        logger.info(f"🏆 Beste Strategie: {final_report['summary']['best_speedup_strategy']}")
        logger.info(f"📈 Ø Speed-Up: {final_report['summary']['avg_speedup_factor']:.2f}x")
        logger.info(f"🎯 Qualitäts-Retention: {final_report['summary']['quality_retention_rate']:.3f}")
        
        return final_report

def main():
    """Hauptfunktion für Phase 3 Tests"""
    logger.info("=" * 60)
    logger.info("🧪 PHASE 3 TEST-SUITE: ADAPTIVE SAMPLING SCHEDULES")
    logger.info("=" * 60)
    
    try:
        # Erstelle Test-Suite
        test_suite = Phase3TestSuite()
        
        # Führe alle Tests durch
        results = test_suite.run_full_phase3_test()
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ PHASE 3 TESTS ERFOLGREICH ABGESCHLOSSEN!")
        logger.info("=" * 60)
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Fehler in Phase 3 Tests: {e}")
        raise

if __name__ == "__main__":
    results = main()
