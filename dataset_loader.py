"""
Unified dataset loader for evaluation benchmarks.

Provides standardized interface for loading QA datasets.
"""

import logging
from typing import Dict, List, Any, Optional
from datasets import load_dataset as hf_load_dataset

logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Unified dataset loader with standardized output format.
    
    All datasets return format:
    {
        'question': str,
        'answer': str or List[str],
        'context': Optional[str],
        'id': str,
        'metadata': Dict[str, Any]
    }
    """
    
    SUPPORTED_DATASETS = {
        'squad': 'squad_v2',
        'squad_v2': 'squad_v2',
        'natural_questions': 'nq_open',
        'nq_open': 'nq_open',
        'triviaqa': 'trivia_qa',
        'hotpotqa': 'hotpot_qa',
        'ms_marco': 'ms_marco',
        'pubmedqa': 'pubmed_qa',
        'asqa': 'asqa'
    }
    
    def __init__(self):
        self.cache = {}
    
    def load(
        self,
        dataset_name: str,
        split: str = "validation",
        max_samples: Optional[int] = None,
        config: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Load dataset with standardized format.
        
        Args:
            dataset_name: Name of dataset (squad, natural_questions, etc.)
            split: Dataset split to load
            max_samples: Maximum number of samples
            config: Optional dataset configuration
            
        Returns:
            List of standardized sample dicts
        """
        cache_key = f"{dataset_name}:{split}:{max_samples}"
        
        if cache_key in self.cache:
            logger.info(f"Using cached {dataset_name}")
            return self.cache[cache_key]
        
        logger.info(f"Loading {dataset_name} ({split})")
        
        if dataset_name.lower() in ['squad', 'squad_v2']:
            samples = self._load_squad(split, max_samples)
        elif dataset_name.lower() in ['natural_questions', 'nq_open']:
            samples = self._load_nq(split, max_samples)
        elif dataset_name.lower() == 'triviaqa':
            samples = self._load_triviaqa(split, max_samples, config)
        elif dataset_name.lower() == 'hotpotqa':
            samples = self._load_hotpotqa(split, max_samples)
        elif dataset_name.lower() == 'ms_marco':
            samples = self._load_ms_marco(split, max_samples)
        elif dataset_name.lower() == 'pubmedqa':
            samples = self._load_pubmedqa(split, max_samples)
        elif dataset_name.lower() == 'asqa':
            samples = self._load_asqa(split, max_samples)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        logger.info(f"Loaded {len(samples)} samples")
        
        self.cache[cache_key] = samples
        return samples
    
    def _load_ms_marco(self, split: str, max_samples: Optional[int]) -> List[Dict[str, Any]]:
        """Load MS MARCO dataset."""
        dataset = hf_load_dataset("ms_marco", "v2.1", split=split if split != "validation" else "dev")
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        samples = []
        for i, item in enumerate(dataset):
            query = item.get('query', '')
            passages = item.get('passages', {})
            answers = item.get('answers', [])
            
            if not query or not answers:
                continue
            
            passage_texts = []
            if 'passage_text' in passages:
                passage_texts = passages['passage_text']
            
            samples.append({
                'question': query,
                'answer': answers[0] if answers else '',
                'context': '\n\n'.join(passage_texts) if passage_texts else None,
                'id': f"msmarco_{i}",
                'metadata': {
                    'all_answers': answers,
                    'passages': passage_texts
                }
            })
        
        return samples
    
    def _load_pubmedqa(self, split: str, max_samples: Optional[int]) -> List[Dict[str, Any]]:
        """Load PubMedQA dataset."""
        dataset = hf_load_dataset("pubmed_qa", "pqa_labeled", split=split if split == "train" else "test")
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        samples = []
        for i, item in enumerate(dataset):
            question = item.get('question', '')
            context_text = ' '.join(item.get('context', {}).get('contexts', []))
            final_decision = item.get('final_decision', '')
            
            if not question or not final_decision:
                continue
            
            samples.append({
                'question': question,
                'answer': final_decision,
                'context': context_text if context_text else None,
                'id': item.get('pubid', f"pubmed_{i}"),
                'metadata': {
                    'long_answer': item.get('long_answer', '')
                }
            })
        
        return samples
    
    def _load_asqa(self, split: str, max_samples: Optional[int]) -> List[Dict[str, Any]]:
        """Load ASQA dataset (Answer Sentence QA with citations)."""
        try:
            dataset = hf_load_dataset("din0s/asqa", split=split if split == "train" else "dev")
        except:
            logger.warning("ASQA dataset not available via datasets library")
            return []
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        samples = []
        for i, item in enumerate(dataset):
            question = item.get('ambiguous_question', '')
            qa_pairs = item.get('qa_pairs', [])
            
            if not question or not qa_pairs:
                continue
            
            answers = [pair.get('short_answers', [''])[0] for pair in qa_pairs if pair.get('short_answers')]
            
            samples.append({
                'question': question,
                'answer': answers[0] if answers else '',
                'context': None,
                'id': f"asqa_{i}",
                'metadata': {
                    'all_answers': answers,
                    'qa_pairs': qa_pairs,
                    'annotations': item.get('annotations', [])
                }
            })
        
        return samples
    
    def _load_squad(self, split: str, max_samples: Optional[int]) -> List[Dict[str, Any]]:
        """Load SQuAD v2 dataset."""
        dataset = hf_load_dataset("squad_v2", split=split)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        samples = []
        for i, item in enumerate(dataset):
            answers = item.get('answers', {}).get('text', [])
            
            if not answers:
                continue
            
            samples.append({
                'question': item['question'],
                'answer': answers[0] if len(answers) == 1 else answers,
                'context': item.get('context', ''),
                'id': item.get('id', f"squad_{i}"),
                'metadata': {
                    'title': item.get('title', ''),
                    'all_answers': answers
                }
            })
        
        return samples
    
    def _load_nq(self, split: str, max_samples: Optional[int]) -> List[Dict[str, Any]]:
        """Load Natural Questions Open dataset."""
        if split not in ["train", "validation"]:
            logger.warning(f"NQ Open doesn't have '{split}', using 'validation'")
            split = "validation"
        
        try:
            dataset = hf_load_dataset("nq_open", split=split, trust_remote_code=True)
        except Exception as e:
            logger.error(f"Failed to load NQ Open: {e}")
            return []
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        samples = []
        for i, item in enumerate(dataset):
            answers = item.get('answer', [])
            
            if not answers:
                continue
            
            samples.append({
                'question': item['question'],
                'answer': answers[0] if len(answers) == 1 else answers,
                'context': None,
                'id': f"nq_{i}",
                'metadata': {
                    'all_answers': answers
                }
            })
        
        return samples
    
    def _load_triviaqa(
        self,
        split: str,
        max_samples: Optional[int],
        config: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Load TriviaQA dataset."""
        config = config or "unfiltered"
        dataset = hf_load_dataset("trivia_qa", config, split=split)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        samples = []
        for i, item in enumerate(dataset):
            answer = item.get('answer', {})
            answer_text = answer.get('value', '')
            
            if not answer_text:
                continue
            
            samples.append({
                'question': item['question'],
                'answer': answer_text,
                'context': None,
                'id': item.get('question_id', f"trivia_{i}"),
                'metadata': {
                    'aliases': answer.get('aliases', [])
                }
            })
        
        return samples
    
    def _load_hotpotqa(self, split: str, max_samples: Optional[int]) -> List[Dict[str, Any]]:
        """Load HotpotQA dataset."""
        dataset = hf_load_dataset("hotpot_qa", "distractor", split=split)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        samples = []
        for i, item in enumerate(dataset):
            answer = item.get('answer', '')
            
            if not answer:
                continue
            
            context_titles = item.get('context', {}).get('title', [])
            context_sentences = item.get('context', {}).get('sentences', [])
            
            context_parts = []
            for title, sentences in zip(context_titles, context_sentences):
                context_parts.append(f"{title}: {' '.join(sentences)}")
            
            samples.append({
                'question': item['question'],
                'answer': answer,
                'context': '\n\n'.join(context_parts) if context_parts else None,
                'id': item.get('id', f"hotpot_{i}"),
                'metadata': {
                    'type': item.get('type', ''),
                    'level': item.get('level', ''),
                    'supporting_facts': item.get('supporting_facts', {})
                }
            })
        
        return samples
    
    def clear_cache(self):
        """Clear dataset cache."""
        self.cache.clear()
        logger.info("Dataset cache cleared")