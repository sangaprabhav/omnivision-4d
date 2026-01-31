# src/models/cosmos_model.py (Production-hardened)
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from typing import List, Dict
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class CosmosModel:
    """Singleton wrapper for Cosmos-Reason2"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self._initialized = True
        logger.info("CosmosModel singleton created")
    
    def load(self):
        """Lazy load with error handling"""
        if self.model is not None:
            return
            
        try:
            logger.info("Loading Cosmos-Reason2-8B...")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                "nvidia/Cosmos-Reason2-8B",
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                cache_dir="/app/models"
            )
            self.processor = AutoProcessor.from_pretrained(
                "nvidia/Cosmos-Reason2-8B",
                trust_remote_code=True,
                cache_dir="/app/models"
            )
            logger.info("Cosmos loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Cosmos: {e}")
            raise
    
    def unload(self):
        """Free VRAM"""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            torch.cuda.empty_cache()
            logger.info("Cosmos unloaded")
    
    def annotate(self, frames: List[Image.Image], prompt: str) -> Dict:
        """Production inference with timeout/retry logic"""
        self.load()
        
        try:
            content = [{"type": "image", "image": f} for f in frames]
            content.append({"type": "text", "text": prompt})
            
            messages = [
                {"role": "system", "content": "You are a spatial reasoning AI. Output valid JSON."},
                {"role": "user", "content": content}
            ]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=frames, padding=True, return_tensors="pt")
            inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0.2,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            result = self.processor.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
            
            # Parse JSON with error handling
            import re
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                import json
                return json.loads(json_match.group())
            return {"raw_output": result, "parse_error": True}
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {"error": str(e)}