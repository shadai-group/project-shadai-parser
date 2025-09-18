"""
Document processing agent for PDFs and text documents.
"""

from typing import List, Dict, Any, Optional, Union
import os
from dataclasses import dataclass

from ..llm_providers.base import BaseLLMProvider
from ..parsers.pdf_parser import PDFParser
from .metadata_schemas import DocumentType, MetadataSchema, ChunkNode, ChunkProcessor
from .language_config import get_language_prompt
from .text_chunker import TextChunker, ChunkConfig, SmartChunker
from parser_shadai.agents.language_detector import LanguageDetector


@dataclass
class ProcessingConfig:
    """Configuration for document processing."""
    chunk_size: int = 1000
    overlap_size: int = 200
    use_smart_chunking: bool = True
    extract_images: bool = True
    max_pages: Optional[int] = None
    temperature: float = 0.3  # Lower temperature for more consistent metadata extraction
    language: str = "en"
    auto_detect_language: bool = False


class DocumentAgent:
    """
    Intelligent document processing agent.
    
    This agent follows a structured workflow:
    1. Extract text and metadata from PDF
    2. Determine document type and metadata schema
    3. Chunk text into manageable pieces
    4. Extract metadata for each chunk using LLM
    5. Return structured results
    """
    
    def __init__(self, llm_provider: BaseLLMProvider, config: Optional[ProcessingConfig] = None):
        """
        Initialize document agent.
        
        Args:
            llm_provider: LLM provider for processing
            config: Processing configuration
        """
        self.llm_provider = llm_provider
        self.config = config or ProcessingConfig()
        self.pdf_parser = PDFParser(llm_provider)
        
        # Initialize chunker
        chunk_config = ChunkConfig(
            chunk_size=self.config.chunk_size,
            overlap_size=self.config.overlap_size
        )
        
        if self.config.use_smart_chunking:
            self.chunker = SmartChunker(chunk_config)
        else:
            self.chunker = TextChunker(chunk_config)

    def set_language(self, text: str):
        self.language_detector = LanguageDetector()
        self.language = self.language_detector.detect_language_with_llm(text, self.llm_provider) if self.config.auto_detect_language else self.config.language
        print(f"Language: {self.language}")

    def process_document(self, 
                        document_path: str, 
                        document_type: DocumentType = DocumentType.GENERAL,
                        auto_detect_type: bool = True) -> Dict[str, Any]:
        """
        Process a document following the complete workflow.
        
        Args:
            document_path: Path to the document file
            document_type: Type of document (used if auto_detect_type is False)
            auto_detect_type: Whether to automatically detect document type
            
        Returns:
            Dictionary containing processed document data
        """
        try:
            # Step 1: Extract text and basic metadata from PDF
            print("Step 1: Extracting text and metadata from PDF...")
            pdf_text = self.pdf_parser.extract_text(document_path)
            pdf_metadata = self.pdf_parser.extract_metadata(document_path)
            page_count = self.pdf_parser.get_page_count(document_path)
            self.set_language(pdf_text[:1000] if len(pdf_text) > 1000 else pdf_text)
            # Step 2: Determine document type if auto-detection is enabled
            if auto_detect_type:
                print("Step 2: Detecting document type...")
                document_type = self._detect_document_type(pdf_text, pdf_metadata)
                print(f"Detected document type: {document_type.value}")
            else:
                print(f"Using specified document type: {document_type.value}")
            
            # Step 3: Chunk the text
            print("Step 3: Chunking text...")
            chunks_data = self._chunk_document_text(pdf_text, page_count)
            print(f"Created {len(chunks_data)} chunks")
            
            # Step 4: Extract metadata for each chunk
            print("Step 4: Extracting metadata for each chunk...")
            chunk_nodes = self._extract_chunk_metadata(chunks_data, document_type)
            
            # Step 5: Compile results
            print("Step 5: Compiling results...")
            results = self._compile_results(
                document_path, 
                pdf_metadata, 
                chunk_nodes, 
                document_type
            )
            
            print("Document processing completed successfully!")
            return results
            
        except Exception as e:
            print(f"Error processing document: {str(e)}")
            raise RuntimeError(f"Document processing failed: {str(e)}")
    
    def _detect_document_type(self, text: str, metadata: Dict[str, Any]) -> DocumentType:
        """
        Detect document type using LLM analysis.
        
        Args:
            text: Document text content
            metadata: PDF metadata
            
        Returns:
            Detected document type
        """
        # Create prompt for document type detection
        prompt = f"""
Analyze the following document and determine its type. Consider both the content and metadata.

Document metadata: {metadata}
Document content (first 2000 characters): {text[:2000]}

Available document types:
- legal: Legal documents, contracts, court papers, legal briefs
- medical: Medical reports, patient records, clinical studies
- financial: Financial statements, invoices, banking documents
- technical: Technical manuals, documentation, specifications
- academic: Research papers, theses, academic articles
- business: Business plans, reports, proposals
- general: General documents that don't fit other categories

Return only the document type (e.g., "legal", "medical", etc.) based on the content and context.
"""
        
        try:
            response = self.llm_provider.generate_text(
                prompt, 
                temperature=self.config.temperature,
                max_tokens=50
            )
            
            detected_type = response.content.strip().lower()
            
            # Map response to DocumentType enum
            type_mapping = {
                "legal": DocumentType.LEGAL,
                "medical": DocumentType.MEDICAL,
                "financial": DocumentType.FINANCIAL,
                "technical": DocumentType.TECHNICAL,
                "academic": DocumentType.ACADEMIC,
                "business": DocumentType.BUSINESS,
                "general": DocumentType.GENERAL
            }
            
            return type_mapping.get(detected_type, DocumentType.GENERAL)
            
        except Exception as e:
            print(f"Warning: Could not detect document type: {e}")
            return DocumentType.GENERAL
    
    def _chunk_document_text(self, text: str, page_count: int) -> List[Dict[str, Any]]:
        """
        Chunk document text into manageable pieces.
        
        Args:
            text: Full document text
            page_count: Number of pages in document
            
        Returns:
            List of chunk data dictionaries
        """
        if self.config.use_smart_chunking:
            return self.chunker.chunk_with_context(text)
        else:
            return self.chunker.chunk_text(text)
    
    def _extract_chunk_metadata(self, chunks_data: List[Dict[str, Any]], document_type: DocumentType) -> List[ChunkNode]:
        """
        Extract metadata for each chunk using LLM.
        
        Args:
            chunks_data: List of chunk data
            document_type: Type of document
            
        Returns:
            List of ChunkNode objects with metadata
        """
        chunk_processor = ChunkProcessor(self.llm_provider, document_type, self.config.language)
        chunk_nodes = []
        
        for i, chunk_data in enumerate(chunks_data):
            print(f"Processing chunk {i+1}/{len(chunks_data)}...")
            
            try:
                chunk_node = chunk_processor.extract_metadata(
                    chunk=chunk_data["content"],
                    chunk_id=chunk_data["chunk_id"],
                    page_number=chunk_data.get("page_number")
                )
                
                # Add chunk processing metadata
                chunk_node.chunk_index = chunk_data["chunk_index"]
                chunk_node.total_chunks = chunk_data["total_chunks"]
                
                chunk_nodes.append(chunk_node)
                
            except Exception as e:
                print(f"Warning: Error processing chunk {i+1}: {e}")
                # Create fallback chunk node
                fallback_node = ChunkNode(
                    chunk_id=chunk_data["chunk_id"],
                    content=chunk_data["content"],
                    metadata={
                        "summary": chunk_data["content"][:200] + "..." if len(chunk_data["content"]) > 200 else chunk_data["content"],
                        "document_type": document_type.value,
                        "processing_error": str(e)
                    },
                    page_number=chunk_data.get("page_number"),
                    chunk_index=chunk_data["chunk_index"],
                    total_chunks=chunk_data["total_chunks"]
                )
                chunk_nodes.append(fallback_node)
        
        return chunk_nodes
    
    def _compile_results(self, 
                        document_path: str, 
                        pdf_metadata: Dict[str, Any], 
                        chunk_nodes: List[ChunkNode], 
                        document_type: DocumentType) -> Dict[str, Any]:
        """
        Compile final results from processing.
        
        Args:
            document_path: Path to original document
            pdf_metadata: PDF metadata
            chunk_nodes: Processed chunk nodes
            document_type: Detected document type
            
        Returns:
            Compiled results dictionary
        """
        # Get chunk statistics
        chunk_stats = self.chunker.get_chunk_statistics([
            {"char_count": node.metadata.get("char_count", len(node.content)),
            "word_count": node.metadata.get("word_count", len(node.content.split())),
            "page_number": node.page_number}
            for node in chunk_nodes
        ])
        
        # Extract summary from all chunks
        all_summaries = [node.metadata.get("summary", "") for node in chunk_nodes if node.metadata.get("summary")]
        combined_summary = " ".join(all_summaries)[:1000] + "..." if len(" ".join(all_summaries)) > 1000 else " ".join(all_summaries)
        
        return {
            "document_info": {
                "file_path": document_path,
                "file_name": os.path.basename(document_path),
                "document_type": document_type.value,
                "page_count": pdf_metadata.get("page_count", 0),
                "processing_timestamp": self._get_timestamp()
            },
            "pdf_metadata": pdf_metadata,
            "chunk_statistics": chunk_stats,
            "document_summary": combined_summary,
            "chunks": [node.to_dict() for node in chunk_nodes],
            "total_chunks": len(chunk_nodes),
            "processing_config": {
                "chunk_size": self.config.chunk_size,
                "overlap_size": self.config.overlap_size,
                "use_smart_chunking": self.config.use_smart_chunking,
                "temperature": self.config.temperature
            }
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def process_text_document(self, 
                             text: str, 
                             document_type: DocumentType = DocumentType.GENERAL,
                             auto_detect_type: bool = True) -> Dict[str, Any]:
        """
        Process a text document (not from PDF).
        
        Args:
            text: Document text content
            document_type: Type of document
            auto_detect_type: Whether to auto-detect type
            
        Returns:
            Dictionary containing processed document data
        """
        try:
            # Step 1: Detect document type if needed
            if auto_detect_type:
                print("Detecting document type...")
                document_type = self._detect_document_type(text, {})
                print(f"Detected document type: {document_type.value}")
            
            # Step 2: Chunk the text
            print("Chunking text...")
            chunks_data = self._chunk_document_text(text, 1)
            print(f"Created {len(chunks_data)} chunks")
            
            # Step 3: Extract metadata for each chunk
            print("Extracting metadata for each chunk...")
            chunk_nodes = self._extract_chunk_metadata(chunks_data, document_type)
            
            # Step 4: Compile results
            print("Compiling results...")
            results = self._compile_results(
                "text_document", 
                {"page_count": 1}, 
                chunk_nodes, 
                document_type
            )
            
            print("Text document processing completed successfully!")
            return results
            
        except Exception as e:
            print(f"Error processing text document: {str(e)}")
            raise RuntimeError(f"Text document processing failed: {str(e)}")
    
    def get_chunk_by_id(self, results: Dict[str, Any], chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific chunk by ID from results."""
        for chunk in results.get("chunks", []):
            if chunk["chunk_id"] == chunk_id:
                return chunk
        return None
