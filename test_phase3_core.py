"""
Phase 3 Quick Test - Simplified version focusing on adaptive schedules
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from adaptive_schedules import (
    AdaptiveStopCriteria,
    QualityStepAnalyzer,
    SpeedUpEvaluator,
    StopCriterion,
    test_adaptive_stop_criteria,
    test_quality_step_analyzer,
    test_speedup_evaluator
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_phase3_core_tests():
    """Führt die Kern-Tests für Phase 3 durch"""
    logger.info("🚀 PHASE 3 CORE TESTS: ADAPTIVE SAMPLING SCHEDULES")
    logger.info("=" * 60)
    
    # Test 3.1: Adaptive Stop Criteria
    logger.info("\n📍 Test 3.1: Dynamische Stop-Kriterien")
    test_adaptive_stop_criteria()
    
    # Test 3.2: Quality Step Analysis
    logger.info("\n📍 Test 3.2: Qualitäts-vs-Schritt-Analyse")
    test_quality_step_analyzer()
    
    # Test 3.3: SpeedUp Evaluation
    logger.info("\n📍 Test 3.3: Speed-Up-Faktor-Evaluation")
    test_speedup_evaluator()
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ PHASE 3 CORE TESTS ERFOLGREICH ABGESCHLOSSEN!")
    logger.info("=" * 60)
    
    return True

if __name__ == "__main__":
    success = run_phase3_core_tests()
    if success:
        print("\n🎉 Phase 3 implementation ready!")
    else:
        print("\n❌ Phase 3 tests failed!")
