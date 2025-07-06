"""
Diffusion Model Wrapper für RL-basierte adaptive Sampling
"""
import torch
import torch.nn as nn
from diffusers import DDPMPipeline, DDIMScheduler, DPMSolverMultistepScheduler
from typing import Dict, Any, Optional, List
import numpy as np
from PIL import Image
import logging

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiffusionModelWrapper:
    """
    Wrapper für vortrainierte Diffusionsmodelle mit adaptiver Sampling-Kontrolle
    """
    
    def __init__(self, 
                 model_id: str = "google/ddpm-cat-256",
                 device: str = "auto",
                 scheduler_type: str = "ddim"):
        
        self.model_id = model_id
        self.device = self._get_device(device)
        self.scheduler_type = scheduler_type
        
        logger.info(f"Initialisiere Diffusionsmodell: {model_id}")
        logger.info(f"Verwendete Device: {self.device}")
        
        # Lade Pipeline
        self.pipeline = self._load_pipeline()
        self.original_scheduler = self.pipeline.scheduler
        
        # Tracking für RL
        self.sampling_history = []
        self.current_step = 0
        
    def _get_device(self, device: str) -> str:
        """Ermittelt die beste verfügbare Device"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _load_pipeline(self) -> DDPMPipeline:
        """Lädt die Diffusion Pipeline"""
        try:
            # Verwende kleineres Modell für schnellere Tests
            pipeline = DDPMPipeline.from_pretrained(self.model_id)
            
            # Scheduler konfigurieren
            if self.scheduler_type == "ddim":
                pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
            elif self.scheduler_type == "dpm":
                pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
            
            pipeline = pipeline.to(self.device)
            
            logger.info("Pipeline erfolgreich geladen")
            return pipeline
            
        except Exception as e:
            logger.error(f"Fehler beim Laden der Pipeline: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Gibt Informationen über das geladene Modell zurück"""
        return {
            "model_id": self.model_id,
            "device": self.device,
            "scheduler_type": self.scheduler_type,
            "scheduler_config": self.pipeline.scheduler.config,
            "unet_params": sum(p.numel() for p in self.pipeline.unet.parameters()),
            "supports_dynamic_sampling": True
        }
    
    def reset_sampling_state(self):
        """Resettet den Sampling-Zustand für neuen Durchgang"""
        self.sampling_history = []
        self.current_step = 0
    
    def get_sampling_state(self) -> Dict[str, Any]:
        """Gibt aktuellen Sampling-Zustand zurück (für RL-Agent)"""
        return {
            "current_step": self.current_step,
            "total_steps": len(self.sampling_history),
            "history": self.sampling_history,
            "last_noise_level": self.sampling_history[-1]["noise_level"] if self.sampling_history else 1.0
        }

# Test-Funktion
def test_diffusion_model():
    """Testet das Diffusionsmodell"""
    logger.info("Starte Test des Diffusionsmodells...")
    
    try:
        # Verwende kleineres Modell für Test
        model = DiffusionModelWrapper(
            model_id="runwayml/stable-diffusion-v1-5",
            device="auto"
        )
        
        info = model.get_model_info()
        logger.info(f"Modell-Info: {info}")
        
        # Teste Sampling-State
        state = model.get_sampling_state()
        logger.info(f"Initial Sampling State: {state}")
        
        logger.info("✅ Diffusionsmodell erfolgreich initialisiert!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Fehler beim Testen: {e}")
        return False

if __name__ == "__main__":
    test_diffusion_model()
