"""
Text chunking functionality for processing documents.
"""

from typing import List, Dict, Any, Optional, Tuple
import re
import hashlib
from dataclasses import dataclass


@dataclass
class ChunkConfig:
    """Configuration for text chunking."""
    chunk_size: int = 1000
    overlap_size: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    split_by_sentences: bool = True
    split_by_paragraphs: bool = False
    preserve_whitespace: bool = True


class TextChunker:
    """Text chunking utility for breaking documents into manageable pieces."""
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        """
        Initialize text chunker.
        
        Args:
            config: Chunking configuration (uses default if None)
        """
        self.config = config or ChunkConfig()
    
    def chunk_text(self, text: str, page_number: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks based on configuration.
        
        Args:
            text: Text to chunk
            page_number: Page number for reference
            
        Returns:
            List of chunk dictionaries with metadata
        """
        if not text or not text.strip():
            return []
        
        # Clean and normalize text
        cleaned_text = self._clean_text(text)
        
        # Split text into chunks
        if self.config.split_by_sentences:
            chunks = self._chunk_by_sentences(cleaned_text)
        elif self.config.split_by_paragraphs:
            chunks = self._chunk_by_paragraphs(cleaned_text)
        else:
            chunks = self._chunk_by_size(cleaned_text)
        
        # Add metadata to chunks
        chunked_data = []
        for i, chunk in enumerate(chunks):
            chunk_id = self._generate_chunk_id(chunk, i, page_number)
            chunked_data.append({
                "chunk_id": chunk_id,
                "content": chunk,
                "page_number": page_number,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "char_count": len(chunk),
                "word_count": len(chunk.split())
            })
        
        return chunked_data
    
    def chunk_pdf_pages(self, pdf_text: str, page_breaks: List[int]) -> List[Dict[str, Any]]:
        """
        Chunk PDF text while preserving page boundaries.
        
        Args:
            pdf_text: Full PDF text
            page_breaks: List of character positions where pages break
            
        Returns:
            List of chunk dictionaries with page information
        """
        if not pdf_text or not page_breaks:
            return self.chunk_text(pdf_text)
        
        all_chunks = []
        current_pos = 0
        
        for page_num, page_end in enumerate(page_breaks):
            page_start = current_pos
            page_text = pdf_text[page_start:page_end].strip()
            
            if page_text:
                page_chunks = self.chunk_text(page_text, page_num + 1)
                all_chunks.extend(page_chunks)
            
            current_pos = page_end
        
        # Update chunk indices
        for i, chunk in enumerate(all_chunks):
            chunk["chunk_index"] = i
            chunk["total_chunks"] = len(all_chunks)
        
        return all_chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere with chunking
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Normalize line breaks
        text = re.sub(r'\r\n|\r|\n', '\n', text)
        
        return text.strip()
    
    def _chunk_by_sentences(self, text: str) -> List[str]:
        """Split text by sentences with size constraints."""
        # Split by sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed max size, start new chunk
            if (len(current_chunk) + len(sentence) > self.config.max_chunk_size and 
                len(current_chunk) >= self.config.min_chunk_size):
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add remaining text as final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _chunk_by_paragraphs(self, text: str) -> List[str]:
        """Split text by paragraphs with size constraints."""
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If adding this paragraph would exceed max size, start new chunk
            if (len(current_chunk) + len(paragraph) > self.config.max_chunk_size and 
                len(current_chunk) >= self.config.min_chunk_size):
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Add remaining text as final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _chunk_by_size(self, text: str) -> List[str]:
        """Split text by fixed size with overlap."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.config.chunk_size
            
            # Try to break at word boundary
            if end < len(text):
                # Look for the last space before the end
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.config.overlap_size
            if start >= len(text):
                break
        
        return chunks
    
    def _generate_chunk_id(self, chunk: str, index: int, page_number: Optional[int] = None) -> str:
        """Generate unique ID for chunk."""
        # Create hash from chunk content
        content_hash = hashlib.md5(chunk.encode('utf-8')).hexdigest()[:8]
        
        # Include page number if available
        if page_number:
            return f"chunk_{page_number}_{index}_{content_hash}"
        else:
            return f"chunk_{index}_{content_hash}"
    
    def get_chunk_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the chunked text."""
        if not chunks:
            return {}
        
        char_counts = [chunk["char_count"] for chunk in chunks]
        word_counts = [chunk["word_count"] for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "total_characters": sum(char_counts),
            "total_words": sum(word_counts),
            "avg_chunk_size": sum(char_counts) / len(chunks),
            "min_chunk_size": min(char_counts),
            "max_chunk_size": max(char_counts),
            "avg_words_per_chunk": sum(word_counts) / len(word_counts),
            "pages_covered": len(set(chunk["page_number"] for chunk in chunks if chunk["page_number"]))
        }


class SmartChunker(TextChunker):
    """Enhanced chunker with intelligent splitting strategies."""
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        """Initialize smart chunker."""
        super().__init__(config)
        self.sentence_endings = r'[.!?]+(?:\s|$)'
        self.paragraph_breaks = r'\n\s*\n'
    
    def chunk_with_context(self, text: str, page_number: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Chunk text with context preservation and smart boundaries.
        
        Args:
            text: Text to chunk
            page_number: Page number for reference
            
        Returns:
            List of chunk dictionaries with enhanced metadata
        """
        if not text or not text.strip():
            return []
        
        # Clean text
        cleaned_text = self._clean_text(text)
        
        # Use smart chunking strategy
        chunks = self._smart_chunk(cleaned_text)
        
        # Add enhanced metadata
        chunked_data = []
        for i, chunk in enumerate(chunks):
            chunk_id = self._generate_chunk_id(chunk, i, page_number)
            
            # Analyze chunk content
            analysis = self._analyze_chunk_content(chunk)
            
            chunked_data.append({
                "chunk_id": chunk_id,
                "content": chunk,
                "page_number": page_number,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "char_count": len(chunk),
                "word_count": len(chunk.split()),
                "sentence_count": analysis["sentence_count"],
                "has_numbers": analysis["has_numbers"],
                "has_dates": analysis["has_dates"],
                "has_entities": analysis["has_entities"],
                "complexity_score": analysis["complexity_score"]
            })
        
        return chunked_data
    
    def _smart_chunk(self, text: str) -> List[str]:
        """Smart chunking that considers content structure."""
        # First, try to split by paragraphs
        paragraphs = re.split(self.paragraph_breaks, text)
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If paragraph is too large, split it further
            if len(paragraph) > self.config.max_chunk_size:
                # Split large paragraph by sentences
                sentences = re.split(self.sentence_endings, paragraph)
                paragraph_chunks = self._chunk_sentences(sentences)
                chunks.extend(paragraph_chunks)
            else:
                # If adding paragraph would exceed size, start new chunk
                if (len(current_chunk) + len(paragraph) > self.config.max_chunk_size and 
                    len(current_chunk) >= self.config.min_chunk_size):
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Add remaining text
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _chunk_sentences(self, sentences: List[str]) -> List[str]:
        """Chunk sentences with size constraints."""
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if (len(current_chunk) + len(sentence) > self.config.max_chunk_size and 
                len(current_chunk) >= self.config.min_chunk_size):
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _analyze_chunk_content(self, chunk: str) -> Dict[str, Any]:
        """Analyze chunk content for additional metadata."""
        # Count sentences
        sentences = re.split(self.sentence_endings, chunk)
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Check for numbers
        has_numbers = bool(re.search(r'\d+', chunk))
        
        # Check for dates (basic pattern)
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b'
        has_dates = bool(re.search(date_pattern, chunk))
        
        # Check for entities (basic patterns)
        entity_patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Names
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\$\d+',  # Money
            r'\b\d+%\b'  # Percentages
        ]
        has_entities = any(re.search(pattern, chunk) for pattern in entity_patterns)
        
        # Calculate complexity score (simple heuristic)
        words = chunk.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        complexity_score = min(avg_word_length / 10, 1.0)  # Normalize to 0-1
        
        return {
            "sentence_count": sentence_count,
            "has_numbers": has_numbers,
            "has_dates": has_dates,
            "has_entities": has_entities,
            "complexity_score": complexity_score
        }
