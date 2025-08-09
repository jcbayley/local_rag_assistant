import requests
import json
from typing import Generator, Union, Dict, Any


class OllamaBackend:
    """Backend for Ollama API integration."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama backend.
        
        Args:
            base_url: Base URL for Ollama API
        """
        self.base_url = base_url.rstrip('/')
        self.generate_url = f"{self.base_url}/api/generate"
    
    def generate_response(self, prompt: str, model: str = "smollm2:135m", 
                         temperature: float = 0.4, max_tokens: int = 500,
                         stream: bool = False, **kwargs) -> Union[str, Generator]:
        """
        Generate response using Ollama API.
        
        Args:
            prompt: Input prompt
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response
            **kwargs: Additional model parameters
            
        Returns:
            Complete response string or streaming generator
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **kwargs
            },
            "stream": stream
        }
        
        try:
            response = requests.post(
                self.generate_url, 
                json=payload, 
                stream=stream, 
                timeout=120
            )
            response.raise_for_status()
            
            if stream:
                return self._stream_response(response)
            else:
                return self._complete_response(response)
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Ollama request failed: {e}"
            if stream:
                return self._error_generator(error_msg)
            return error_msg
    
    def _stream_response(self, response) -> Generator[str, None, None]:
        """Stream response chunks from Ollama."""
        try:
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if 'response' in data:
                            yield data['response']
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            yield f"Error in streaming: {str(e)}"
    
    def _complete_response(self, response) -> str:
        """Get complete response from Ollama."""
        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    if 'response' in data:
                        full_response += data['response']
                except json.JSONDecodeError:
                    continue
        return full_response
    
    def _error_generator(self, error_msg: str) -> Generator[str, None, None]:
        """Generate error message for streaming."""
        yield error_msg
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False