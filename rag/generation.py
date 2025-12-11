"""LLM answer generation with RAG - Research-grade implementation."""

import logging
import re
from typing import List, Tuple, Union, Optional
import torch

logger = logging.getLogger(__name__)


class RAGGenerator:
    """
    Research-grade RAG generator with balanced faithfulness and fluency.
    
    Design principles:
    - Encourage synthesis rather than extraction
    - Balance faithfulness with naturalness
    - Avoid hallucination without being overly conservative
    """
    
    def __init__(self, model_interface, config: dict):
        """
        Initialize RAG generator.
        
        Args:
            model_interface: ModelInterface instance
            config: Generation config
        """
        self.model_interface = model_interface
        
        self.max_new_tokens = config.get('max_new_tokens', 64)
        self.temperature = config.get('temperature', 0.3)
        self.top_p = config.get('top_p', 0.9)
        self.do_sample = config.get('do_sample', True)
        self.repetition_penalty = config.get('repetition_penalty', 1.15)
        
        self.model_type = model_interface.model_type or "instruct"
        self.use_chat_template = config.get('use_chat_template', True)
        
        logger.info(f"Initialized RAG generator for {self.model_type} model")
        logger.info(f"Temperature: {self.temperature}, Max tokens: {self.max_new_tokens}")
    
    def generate(
        self,
        query: str,
        context: str,
        return_prompt: bool = False
    ) -> Union[str, Tuple[str, str]]:
        """
        Generate answer given query and context.
        
        Args:
            query: User question
            context: Retrieved context
            return_prompt: If True, return (answer, prompt) tuple
            
        Returns:
            Generated answer or (answer, prompt) tuple
        """
        context = self._truncate_context(context, max_chars=2000)
        
        prompt = self._format_prompt(query, context)
        
        try:
            from models.model_interface import GenerationConfig
            
            gen_config = GenerationConfig(
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample
            )
            
            output = self.model_interface.generate(prompt, gen_config)
            answer = output.generated_text
        except Exception as e:
            logger.warning(f"Generation failed, using fallback: {e}")
            answer = self.model_interface.generate(
                prompt=prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample
            )
            if hasattr(answer, 'generated_text'):
                answer = answer.generated_text
        
        answer = self._clean_answer(answer)
        
        if return_prompt:
            return answer, prompt
        return answer
    
    def generate_batch(
        self,
        queries: List[str],
        contexts: List[str],
        show_progress: bool = True
    ) -> List[str]:
        """Generate answers for multiple queries."""
        if len(queries) != len(contexts):
            raise ValueError(f"Queries and contexts must have same length")
        
        answers = []
        
        iterator = zip(queries, contexts)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=len(queries), desc="Generating")
            except ImportError:
                pass
        
        for query, context in iterator:
            try:
                answer = self.generate(query, context)
                answers.append(answer)
            except Exception as e:
                logger.error(f"Generation failed for query: {e}")
                answers.append("")
        
        return answers
    
    def generate_without_context(self, query: str) -> str:
        """Generate answer without RAG context."""
        prompt = self._format_prompt_no_context(query)
        
        try:
            from models.model_interface import GenerationConfig
            
            gen_config = GenerationConfig(
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample
            )
            
            output = self.model_interface.generate(prompt, gen_config)
            answer = output.generated_text
        except Exception:
            answer = self.model_interface.generate(
                prompt=prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample
            )
            if hasattr(answer, 'generated_text'):
                answer = answer.generated_text
        
        return self._clean_answer(answer)
    
    def generate_batch_without_context(
        self,
        queries: List[str],
        show_progress: bool = True
    ) -> List[str]:
        """Generate answers without context."""
        answers = []
        
        iterator = queries
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(queries, desc="Generating (no RAG)")
            except ImportError:
                pass
        
        for query in iterator:
            try:
                answer = self.generate_without_context(query)
                answers.append(answer)
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                answers.append("")
        
        return answers
    
    def get_tokenizer(self):
        """Get model tokenizer."""
        return self.model_interface.tokenizer
    
    def _truncate_context(self, context: str, max_chars: int = 2000) -> str:
        """Truncate context to prevent overwhelming the model."""
        if len(context) <= max_chars:
            return context
        
        truncated = context[:max_chars]
        last_period = truncated.rfind('.')
        
        if last_period > max_chars * 0.7:
            return truncated[:last_period + 1]
        
        return truncated + "..."
    
    def _format_prompt(self, query: str, context: str) -> str:
        """Format prompt based on model type."""
        if self.model_type == "instruct":
            return self._format_instruct_prompt(query, context)
        return self._format_base_prompt(query, context)
    
    def _format_instruct_prompt(self, query: str, context: str) -> str:
        """Format prompt for instruct models."""
        tokenizer = self.model_interface.tokenizer
        
        if self.use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
            messages = [{
                "role": "user",
                "content": f"""Answer the question based on the context provided.

Context:
{context}

Question: {query}

Provide a clear, concise answer:"""
            }]
            
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                logger.debug(f"Chat template failed: {e}")
        
        return f"""Context: {context}

Question: {query}

Answer:"""
    
    def _format_base_prompt(self, query: str, context: str) -> str:
        """Format prompt for base models."""
        return f"""Context: {context}

Question: {query}

Answer:"""
    
    def _format_prompt_no_context(self, query: str) -> str:
        """Format prompt without context."""
        tokenizer = self.model_interface.tokenizer
        
        if self.model_type == "instruct" and self.use_chat_template:
            if hasattr(tokenizer, 'apply_chat_template'):
                messages = [{"role": "user", "content": query}]
                try:
                    return tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except Exception:
                    pass
        
        return f"Question: {query}\n\nAnswer:"
    
    def _clean_answer(self, answer: str) -> str:
        """Clean and normalize generated answer."""
        if not answer or not answer.strip():
            return ""
        
        answer = re.sub(r'\s+', ' ', answer).strip()
        answer = re.sub(r'^(Answer:|A:)\s*', '', answer, flags=re.IGNORECASE)
        answer = re.sub(r'^Based on (the )?(context|information)( provided)?,?\s*', '', answer, flags=re.IGNORECASE)
        
        if not answer.strip():
            return ""
        
        if len(answer) > 500:
            sentences = re.split(r'([.!?])\s+', answer)
            if len(sentences) > 6:
                answer = ''.join(sentences[:6])
        
        if answer and answer[-1] not in '.!?':
            answer += '.'
        
        return answer.strip()