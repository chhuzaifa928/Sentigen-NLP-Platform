"""
Text Generation Module
Generates coherent text using GPT-2 transformer model
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextGenerator:
    def __init__(self):
        """Initialize text generation model"""
        try:
            logger.info("Loading GPT-2 model...")
            
            self.model_name = "gpt2"
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
            
            # Set padding token
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Move to GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            
            logger.info(f"Text generator initialized successfully on {self.device}!")
            
        except Exception as e:
            logger.error(f"Error initializing text generator: {e}")
            raise
    
    def generate(self, prompt: str, max_length: int = 100, 
                temperature: float = 0.7, top_k: int = 50, 
                top_p: float = 0.9, num_return_sequences: int = 1) -> Dict:
        """
        Generate text based on a prompt
        
        Args:
            prompt: Starting text
            max_length: Maximum length of generated text
            temperature: Randomness (0.0 = deterministic, 1.0+ = creative)
            top_k: Consider top k tokens
            top_p: Nucleus sampling threshold
            num_return_sequences: Number of different completions to generate
            
        Returns:
            Dictionary containing generated text and metadata
        """
        try:
            # Encode prompt
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2,  # Avoid repetition
                    early_stopping=True
                )
            
            # Decode generated text
            generated_texts = []
            for output in outputs:
                text = self.tokenizer.decode(output, skip_special_tokens=True)
                generated_texts.append(text)
            
            return {
                "prompt": prompt,
                "generated_texts": generated_texts,
                "num_sequences": num_return_sequences,
                "parameters": {
                    "max_length": max_length,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p
                },
                "model": self.model_name
            }
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return {
                "error": str(e),
                "prompt": prompt,
                "generated_texts": [prompt],
                "num_sequences": 0
            }
    
    def complete_sentence(self, partial_sentence: str, max_new_tokens: int = 50) -> str:
        """Complete a partial sentence"""
        result = self.generate(
            partial_sentence,
            max_length=len(partial_sentence.split()) + max_new_tokens,
            temperature=0.7,
            num_return_sequences=1
        )
        
        return result['generated_texts'][0] if result['generated_texts'] else partial_sentence
    
    def generate_variations(self, prompt: str, num_variations: int = 3, 
                          max_length: int = 100) -> List[str]:
        """Generate multiple variations of text from the same prompt"""
        result = self.generate(
            prompt,
            max_length=max_length,
            temperature=0.8,  # Higher temperature for more variation
            num_return_sequences=num_variations
        )
        
        return result['generated_texts']
    
    def creative_writing(self, prompt: str, max_length: int = 200) -> str:
        """
        Generate creative text with higher temperature
        Good for storytelling and creative content
        """
        result = self.generate(
            prompt,
            max_length=max_length,
            temperature=0.9,  # High creativity
            top_k=40,
            top_p=0.95,
            num_return_sequences=1
        )
        
        return result['generated_texts'][0] if result['generated_texts'] else prompt
    
    def factual_completion(self, prompt: str, max_length: int = 100) -> str:
        """
        Generate more deterministic, factual text
        Good for completing factual statements
        """
        result = self.generate(
            prompt,
            max_length=max_length,
            temperature=0.3,  # Low temperature for consistency
            top_k=10,
            top_p=0.8,
            num_return_sequences=1
        )
        
        return result['generated_texts'][0] if result['generated_texts'] else prompt
    
    def get_next_word_probabilities(self, text: str, top_n: int = 10) -> List[Dict]:
        """Get probability distribution for next word"""
        try:
            input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids)
                predictions = outputs.logits
            
            # Get probabilities for next token
            next_token_logits = predictions[0, -1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            
            # Get top N tokens
            top_probs, top_indices = torch.topk(next_token_probs, top_n)
            
            results = []
            for prob, idx in zip(top_probs, top_indices):
                token = self.tokenizer.decode([idx])
                results.append({
                    "token": token,
                    "probability": float(prob)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting word probabilities: {e}")
            return []


# Test function
if __name__ == "__main__":
    generator = TextGenerator()
    
    test_prompts = [
        "The future of artificial intelligence is",
        "Once upon a time in a distant galaxy",
        "The most important aspect of machine learning is"
    ]
    
    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}")
        
        result = generator.generate(prompt, max_length=80, num_return_sequences=2)
        
        for i, text in enumerate(result['generated_texts'], 1):
            print(f"\nVariation {i}:")
            print(text)
