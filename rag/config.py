"""Standard document corpus for RAG evaluation."""

import logging
from typing import Dict, List
from datasets import load_dataset

logger = logging.getLogger(__name__)


class StandardCorpus:
    """
    Standard document corpus for research-grade RAG evaluation.
    Uses widely-adopted benchmarks for reproducibility.
    """
    
    AVAILABLE_CORPORA = {
        'squad_v2': {
            'dataset': 'squad_v2',
            'split': 'train',
            'description': 'SQuAD 2.0 contexts - gold standard for QA',
            'context_field': 'context',
            'size': 'small'
        },
        'hotpotqa': {
            'dataset': 'hotpot_qa',
            'config': 'distractor',
            'split': 'train',
            'description': 'HotpotQA - multi-hop reasoning',
            'context_field': 'context',
            'size': 'medium'
        },
        'natural_questions': {
            'dataset': 'natural_questions',
            'config': 'default',
            'split': 'train',
            'description': 'Natural Questions - real Google queries',
            'context_field': 'document',
            'size': 'large'
        },
        'ms_marco': {
            'dataset': 'ms_marco',
            'config': 'v2.1',
            'split': 'train',
            'description': 'MS MARCO - web passages',
            'context_field': 'passages',
            'size': 'large'
        }
    }
    
    @staticmethod
    def load_corpus(
        corpus_name: str = 'squad_v2',
        max_documents: int = 500,
        min_length: int = 100
    ) -> List[Dict]:
        """
        Load standard corpus for RAG evaluation.
        
        Args:
            corpus_name: Name of corpus to load
            max_documents: Maximum number of documents
            min_length: Minimum document length
            
        Returns:
            List of document dicts with 'text', 'doc_id', 'metadata'
        """
        if corpus_name not in StandardCorpus.AVAILABLE_CORPORA:
            raise ValueError(f"Unknown corpus: {corpus_name}")
        
        corpus_info = StandardCorpus.AVAILABLE_CORPORA[corpus_name]
        logger.info(f"Loading corpus: {corpus_name}")
        logger.info(f"Description: {corpus_info['description']}")
        
        if corpus_name == 'squad_v2':
            return StandardCorpus._load_squad(max_documents, min_length)
        elif corpus_name == 'hotpotqa':
            return StandardCorpus._load_hotpotqa(max_documents, min_length)
        else:
            raise NotImplementedError(f"Loader for {corpus_name} not yet implemented")
    
    @staticmethod
    def _load_squad(max_documents: int, min_length: int) -> List[Dict]:
        """Load SQuAD v2 contexts."""
        dataset = load_dataset('squad_v2', split='train', trust_remote_code=True)
        
        documents = []
        seen_contexts = set()
        
        for i, item in enumerate(dataset):
            if len(documents) >= max_documents:
                break
            
            context = item['context']
            
            if len(context) < min_length:
                continue
            
            if context in seen_contexts:
                continue
            
            seen_contexts.add(context)
            
            documents.append({
                'text': context,
                'doc_id': f"squad_{item['id']}",
                'metadata': {
                    'title': item.get('title', ''),
                    'source': 'squad_v2',
                    'length': len(context)
                }
            })
        
        logger.info(f"Loaded {len(documents)} unique documents from SQuAD v2")
        return documents
    
    @staticmethod
    def _load_hotpotqa(max_documents: int, min_length: int) -> List[Dict]:
        """Load HotpotQA contexts."""
        dataset = load_dataset('hotpot_qa', 'distractor', split='train', trust_remote_code=True)
        
        documents = []
        seen_texts = set()
        
        for i, item in enumerate(dataset):
            if len(documents) >= max_documents:
                break
            
            context_titles = item['context']['title']
            context_sentences = item['context']['sentences']
            
            for title, sentences in zip(context_titles, context_sentences):
                text = f"{title}. " + ' '.join(sentences)
                
                if len(text) < min_length:
                    continue
                
                if text in seen_texts:
                    continue
                
                seen_texts.add(text)
                
                documents.append({
                    'text': text,
                    'doc_id': f"hotpot_{item['id']}_{title}",
                    'metadata': {
                        'title': title,
                        'source': 'hotpotqa',
                        'length': len(text),
                        'type': item.get('type', '')
                    }
                })
                
                if len(documents) >= max_documents:
                    break
        
        logger.info(f"Loaded {len(documents)} documents from HotpotQA")
        return documents