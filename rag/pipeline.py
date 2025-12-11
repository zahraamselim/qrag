"""End-to-end RAG pipeline orchestration."""

import time
import logging
from typing import List, Dict, Optional, Union

from rag.chunking import TextChunker
from rag.embedding import EmbeddingModel
from rag.indexing import VectorStore
from rag.retrieval import ContextRetriever
from rag.generation import RAGGenerator

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    End-to-end RAG pipeline for evaluation.
    Simplified for research use with pre-cleaned datasets.
    """
    
    def __init__(self, config: dict):
        """
        Initialize RAG pipeline.
        
        Args:
            config: RAG config dict
        """
        self.config = config
        
        self.chunker = None
        self.embedding_model = None
        self.vector_store = None
        self.retriever = None
        self.generator = None
        
        logger.info("RAG Pipeline initialized")
    
    def setup(self, model_interface):
        """
        Setup all pipeline components.
        
        Args:
            model_interface: ModelInterface instance for generation
        """
        logger.info("Setting up RAG pipeline components...")
        
        chunk_config = self.config.get('chunking', {})
        self.chunker = TextChunker(chunk_config)
        
        embed_config = self.config.get('embedding', {})
        self.embedding_model = EmbeddingModel(embed_config)
        
        store_config = self.config.get('vector_store', {})
        self.vector_store = VectorStore(store_config)
        
        retrieval_config = self.config.get('retrieval', {})
        self.retriever = ContextRetriever(
            vector_store=self.vector_store,
            embedding_model=self.embedding_model,
            config=retrieval_config
        )
        
        generation_config = self.config.get('generation', {})
        self.generator = RAGGenerator(
            model_interface=model_interface,
            config=generation_config
        )
        
        logger.info("Pipeline setup complete")
    
    def index_documents(
        self,
        documents: List[str],
        show_progress: bool = True
    ) -> float:
        """
        Index documents into vector store.
        
        Args:
            documents: List of document strings (pre-cleaned)
            show_progress: Show progress bars
            
        Returns:
            Processing time in seconds
        """
        start_time = time.time()
        
        logger.info(f"Processing {len(documents)} documents")
        
        all_chunks = []
        for i, doc in enumerate(documents):
            if not doc or not doc.strip():
                continue
            
            chunks = self.chunker.chunk(doc, page_num=i+1)
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks")
        
        if not all_chunks:
            raise ValueError("No chunks created from documents")
        
        embeddings = self.embedding_model.embed_chunks(
            all_chunks,
            show_progress=show_progress
        )
        
        self.vector_store.create_index(all_chunks, embeddings)
        
        processing_time = time.time() - start_time
        logger.info(f"Indexing complete in {processing_time:.2f}s")
        
        return processing_time
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Retrieve relevant contexts for a query.
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve
            
        Returns:
            List of context dicts
        """
        return self.retriever.retrieve(query, top_k=top_k)
    
    def generate_answer(
        self,
        query: str,
        contexts: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate answer for a query.
        
        Args:
            query: Query string
            contexts: Pre-retrieved contexts (if None, retrieves automatically)
            
        Returns:
            Generated answer string
        """
        if contexts is None:
            contexts = self.retriever.retrieve(query)
        
        if contexts and len(contexts) > 0:
            context_str = '\n\n'.join([
                ctx.get('text', ctx.get('content', ''))
                for ctx in contexts
            ])
        else:
            context_str = ""
        
        if context_str:
            return self.generator.generate(query, context_str)
        else:
            return self.generator.generate_without_context(query)
    
    def evaluate(
        self,
        test_questions: List[Dict[str, str]],
        compare_no_rag: bool = True,
        show_progress: bool = True
    ) -> Dict:
        """
        Evaluate RAG system on test questions.
        
        Args:
            test_questions: List of {'question': ..., 'answer': ...}
            compare_no_rag: Whether to compare with no-RAG baseline
            show_progress: Show progress bars
            
        Returns:
            Dict with predictions and metadata
        """
        logger.info(f"Evaluating on {len(test_questions)} questions")
        
        questions = [qa['question'] for qa in test_questions]
        references = [qa['answer'] for qa in test_questions]
        
        logger.info("Retrieving contexts...")
        contexts = []
        contexts_list = []
        
        for q in questions:
            retrieved = self.retriever.retrieve(q)
            contexts_list.append(retrieved)
            context_str = '\n\n'.join([chunk['text'] for chunk in retrieved])
            contexts.append(context_str)
        
        logger.info("Generating RAG answers...")
        predictions = self.generator.generate_batch(
            queries=questions,
            contexts=contexts,
            show_progress=show_progress
        )
        
        predictions_no_rag = None
        if compare_no_rag:
            logger.info("Generating no-RAG baseline answers...")
            predictions_no_rag = self.generator.generate_batch_without_context(
                queries=questions,
                show_progress=show_progress
            )
        
        return {
            'questions': questions,
            'references': references,
            'predictions': predictions,
            'contexts': contexts,
            'retrieved_chunks': contexts_list,
            'predictions_no_rag': predictions_no_rag
        }
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        stats = {
            'vector_store': self.vector_store.get_stats() if self.vector_store else {},
            'embedding_dim': self.embedding_model.get_dimension() if self.embedding_model else None,
            'config': self.config
        }
        
        if self.embedding_model:
            stats['embedding'] = {
                'model_name': self.embedding_model.model_name,
                'dimension': self.embedding_model.get_dimension(),
                'device': self.embedding_model.device,
                'batch_size': self.embedding_model.batch_size,
                'normalize': self.embedding_model.normalize
            }
        
        if self.retriever:
            stats['retrieval'] = {
                'top_k': self.retriever.top_k,
                'similarity_threshold': self.retriever.similarity_threshold,
                'rerank': self.retriever.rerank,
                'diversity_penalty': self.retriever.diversity_penalty
            }
        
        return stats