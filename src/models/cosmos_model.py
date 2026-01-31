# src/models/cosmos_model.py (Production-hardened)
import torch
import numpy as np
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from typing import List, Dict
from dataclasses import dataclass
from PIL import Image
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class CosmosResult:
    """Cosmos annotation result with confidence metrics"""
    annotation: Dict  # Parsed JSON annotation
    raw_output: str  # Raw model output
    confidence_score: float  # Overall confidence (0-1)
    parse_success: bool  # Whether JSON parsing succeeded

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
    
    def annotate(
        self,
        frames: List[Image.Image],
        prompt: str,
        return_logits: bool = False
    ) -> CosmosResult:
        """
        Production inference with confidence estimation

        Args:
            frames: List of PIL images
            prompt: Text prompt for annotation
            return_logits: If True, compute confidence from output logits

        Returns:
            CosmosResult with annotation, confidence, and metadata
        """
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
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=return_logits
                )

            # Decode output
            if hasattr(outputs, 'sequences'):
                output_ids = outputs.sequences
            else:
                output_ids = outputs

            result_text = self.processor.batch_decode(
                output_ids[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )[0]

            # Compute confidence from logits if available
            if return_logits and hasattr(outputs, 'scores') and outputs.scores:
                confidence = self._compute_confidence_from_logits(outputs.scores)
            else:
                # Heuristic confidence based on output quality
                confidence = self._estimate_confidence_heuristic(result_text)

            # Parse JSON with error handling
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)

            if json_match:
                try:
                    annotation = json.loads(json_match.group())
                    parse_success = True

                    # Further validate JSON structure
                    if not isinstance(annotation, dict):
                        logger.warning("Parsed JSON is not a dictionary")
                        confidence *= 0.5

                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse error: {e}")
                    annotation = {"raw_output": result_text, "parse_error": str(e)}
                    parse_success = False
                    confidence *= 0.3
            else:
                logger.warning("No JSON found in output")
                annotation = {"raw_output": result_text, "parse_error": "No JSON found"}
                parse_success = False
                confidence *= 0.2

            logger.info(f"Cosmos annotation complete (confidence: {confidence:.3f}, parsed: {parse_success})")

            return CosmosResult(
                annotation=annotation,
                raw_output=result_text,
                confidence_score=float(confidence),
                parse_success=parse_success
            )

        except Exception as e:
            logger.error(f"Cosmos inference failed: {e}")
            return CosmosResult(
                annotation={"error": str(e)},
                raw_output="",
                confidence_score=0.0,
                parse_success=False
            )

    def _compute_confidence_from_logits(self, scores: tuple) -> float:
        """
        Compute confidence from output logits

        Args:
            scores: Tuple of logits for each generated token

        Returns:
            Confidence score in [0, 1]
        """
        import torch.nn.functional as F

        # Compute average token probability
        token_probs = []

        for token_logits in scores:
            # Convert logits to probabilities
            probs = F.softmax(token_logits, dim=-1)

            # Get probability of selected token (max probability)
            max_prob = probs.max().item()
            token_probs.append(max_prob)

        # Average probability across all tokens
        avg_prob = sum(token_probs) / len(token_probs) if token_probs else 0.5

        # Convert to confidence (higher average prob = more confident)
        confidence = avg_prob

        return float(np.clip(confidence, 0.0, 1.0))

    def _estimate_confidence_heuristic(self, text: str) -> float:
        """
        Estimate confidence using heuristics when logits unavailable

        Factors:
        - Length (too short or too long = lower confidence)
        - Presence of JSON structure
        - Absence of error indicators
        """
        import re

        # Base confidence
        confidence = 0.7

        # Check length (optimal: 100-2000 characters)
        text_len = len(text)
        if text_len < 50:
            confidence *= 0.5  # Too short
        elif text_len > 2000:
            confidence *= 0.8  # Too long

        # Check for JSON structure
        if '{' in text and '}' in text:
            confidence *= 1.1  # Bonus for JSON
        else:
            confidence *= 0.5

        # Check for error indicators
        error_patterns = [
            r'error',
            r'failed',
            r'unable',
            r'cannot',
            r'sorry',
            r'don\'t know'
        ]

        for pattern in error_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                confidence *= 0.6

        # Check for positive indicators
        positive_patterns = [
            r'object',
            r'trajectory',
            r'position',
            r'movement',
            r'detected'
        ]

        positive_count = sum(
            1 for pattern in positive_patterns
            if re.search(pattern, text, re.IGNORECASE)
        )

        confidence *= (1.0 + 0.05 * positive_count)  # Bonus for relevant terms

        return float(np.clip(confidence, 0.0, 1.0))