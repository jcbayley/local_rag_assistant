from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
from typing import Generator, Union, Optional
import threading


class TransformersBackend:
    """Backend for HuggingFace Transformers local inference."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small", device: Optional[str] = None):
        """
        Initialize Transformers backend.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._model_loaded = False
        self._load_lock = threading.Lock()
    
    def _load_model(self):
        """Load model and tokenizer if not already loaded."""
        with self._load_lock:
            if not self._model_loaded:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        device_map="auto" if self.device == "cuda" else None
                    )
                    if self.device == "cpu":
                        self.model = self.model.to(self.device)
                    
                    # Set pad token if not present
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                    self._model_loaded = True
                except Exception as e:
                    raise RuntimeError(f"Failed to load model {self.model_name}: {e}")
    
    def generate_response(self, prompt: str, max_tokens: int = 500, 
                         temperature: float = 0.7, top_p: float = 0.9,
                         stream: bool = False, **kwargs) -> Union[str, Generator]:
        """
        Generate response using Transformers.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stream: Whether to stream response (limited streaming support)
            **kwargs: Additional generation parameters
            
        Returns:
            Complete response string or streaming generator
        """
        self._load_model()
        
        try:
            # Encode input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            inputs = inputs.to(self.device)
            
            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                **kwargs
            }
            
            if stream:
                return self._stream_generate(inputs, generation_kwargs)
            else:
                return self._complete_generate(inputs, generation_kwargs, prompt)
                
        except Exception as e:
            error_msg = f"Generation failed: {e}"
            if stream:
                return self._error_generator(error_msg)
            return error_msg
    
    def _complete_generate(self, inputs, generation_kwargs, original_prompt):
        """Generate complete response."""
        with torch.no_grad():
            outputs = self.model.generate(inputs, **generation_kwargs)
        
        # Decode only the new tokens (skip the input prompt)
        new_tokens = outputs[0][len(inputs[0]):]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()
    
    def _stream_generate(self, inputs, generation_kwargs):
        """
        Simulate streaming by generating in chunks.
        Note: True streaming requires more complex implementation with custom generation loops.
        """
        try:
            # For simplicity, generate complete response and yield it in chunks
            response = self._complete_generate(inputs, generation_kwargs, "")
            chunk_size = max(1, len(response) // 10)  # Divide into ~10 chunks
            
            for i in range(0, len(response), chunk_size):
                yield response[i:i + chunk_size]
                
        except Exception as e:
            yield f"Error in streaming: {str(e)}"
    
    def _error_generator(self, error_msg: str) -> Generator[str, None, None]:
        """Generate error message for streaming."""
        yield error_msg
    
    def is_available(self) -> bool:
        """Check if the model can be loaded."""
        try:
            if not self._model_loaded:
                # Try to load tokenizer as a quick check
                AutoTokenizer.from_pretrained(self.model_name)
            return True
        except Exception:
            return False
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "loaded": self._model_loaded,
            "cuda_available": torch.cuda.is_available()
        }