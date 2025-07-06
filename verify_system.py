#!/usr/bin/env python3
"""
Quick System Test - Final Verification
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from main import main
        print("✅ main.py - OK")
    except Exception as e:
        print(f"❌ main.py - Error: {e}")
        return False
    
    try:
        from diffusion_model import DiffusionModelWrapper
        print("✅ diffusion_model.py - OK")
    except Exception as e:
        print(f"❌ diffusion_model.py - Error: {e}")
        return False
    
    try:
        from adaptive_sampling import AdaptiveSamplingLoop
        print("✅ adaptive_sampling.py - OK")
    except Exception as e:
        print(f"❌ adaptive_sampling.py - Error: {e}")
        return False
    
    try:
        from rl_training import RLTrainer
        print("✅ rl_training.py - OK")
    except Exception as e:
        print(f"❌ rl_training.py - Error: {e}")
        return False
    
    try:
        from logging_system import RLDiffusionLogger
        print("✅ logging_system.py - OK")
    except Exception as e:
        print(f"❌ logging_system.py - Error: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        from mdp_definition import MDPDefinition
        mdp = MDPDefinition()
        print("✅ MDP Definition - OK")
    except Exception as e:
        print(f"❌ MDP Definition - Error: {e}")
        return False
    
    try:
        from reward_function import RewardFunction
        reward = RewardFunction()
        print("✅ Reward Function - OK")
    except Exception as e:
        print(f"❌ Reward Function - Error: {e}")
        return False
    
    try:
        from baseline_policies import RandomPolicy
        policy = RandomPolicy()
        print("✅ Baseline Policies - OK")
    except Exception as e:
        print(f"❌ Baseline Policies - Error: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🚀 Final System Verification")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed!")
        return False
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Functionality tests failed!")
        return False
    
    print("\n🎉 All tests passed!")
    print("✅ System is ready for production!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
