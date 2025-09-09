"""
Main processing agent that orchestrates document and image processing.
"""

from typing import List, Dict, Any, Optional
import os
from dataclasses import dataclass
from pathlib import Path

from ..llm_providers.base import BaseLLMProvider
from .document_agent import DocumentAgent, ProcessingConfig
from .image_agent import ImageAgent, ImageProcessingConfig
from .metadata_schemas import DocumentType
from .utils_files import pdf_extensions, image_extensions, pdf_patterns, image_patterns


@dataclass
class AgentConfig:
    """Configuration for the main processing agent."""
    # Document processing config
    chunk_size: int = 1000
    overlap_size: int = 200
    use_smart_chunking: bool = True
    extract_images: bool = True
    
    # Image processing config
    max_image_size: tuple = (1024, 1024)
    extract_text_from_images: bool = True
    describe_images: bool = True
    classify_images: bool = True
    
    # General config
    temperature: float = 0.3
    auto_detect_document_type: bool = True
    parallel_processing: bool = False
    folder_processing: bool = False
    
    # Language config
    language: str = "en"
    auto_detect_language: bool = False


class MainProcessingAgent:
    """
    Main processing agent that coordinates document and image processing.
    
    This agent provides a unified interface for processing various types of content:
    - PDF documents with text and images
    - Individual images
    - Mixed content processing
    - Batch processing
    """
    
    def __init__(self, llm_provider: BaseLLMProvider, config: Optional[AgentConfig] = None):
        """
        Initialize main processing agent.
        
        Args:
            llm_provider: LLM provider for processing
            config: Agent configuration
        """
        self.llm_provider = llm_provider
        self.config = config or AgentConfig()
        
        # Initialize sub-agents
        doc_config = ProcessingConfig(
            chunk_size=self.config.chunk_size,
            overlap_size=self.config.overlap_size,
            use_smart_chunking=self.config.use_smart_chunking,
            extract_images=self.config.extract_images,
            temperature=self.config.temperature,
            language=self.config.language,
            auto_detect_language=self.config.auto_detect_language
        )
        
        img_config = ImageProcessingConfig(
            max_image_size=self.config.max_image_size,
            extract_text=self.config.extract_text_from_images,
            describe_content=self.config.describe_images,
            classify_type=self.config.classify_images,
            temperature=self.config.temperature,
            language=self.config.language,
            auto_detect_language=self.config.auto_detect_language
        )
        
        self.document_agent = DocumentAgent(llm_provider, doc_config)
        self.image_agent = ImageAgent(llm_provider, img_config)
    
    def process_file(self, 
                    file_path: str, 
                    document_type: DocumentType = DocumentType.GENERAL,
                    auto_detect_type: bool = None) -> Dict[str, Any]:
        """
        Process a single file (PDF or image).
        
        Args:
            file_path: Path to the file to process
            document_type: Type of document (used if auto_detect_type is False)
            auto_detect_type: Whether to auto-detect document type (uses config default if None)
            
        Returns:
            Dictionary containing processing results
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = Path(file_path).suffix.lower()
        auto_detect = auto_detect_type if auto_detect_type is not None else self.config.auto_detect_document_type
        
        if self.config.folder_processing is True:
            return self.process_directory(file_path, document_type)
        elif file_extension in pdf_extensions:
            return self._process_pdf_file(file_path, document_type, auto_detect)
        elif file_extension in image_extensions:
            return self._process_image_file(file_path, document_type)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    def _process_pdf_file(self, 
                         file_path: str, 
                         document_type: DocumentType, 
                         auto_detect_type: bool) -> Dict[str, Any]:
        """Process a PDF file."""
        print("Processing PDF document...")
        
        # Process the PDF document
        doc_results = self.document_agent.process_document(
            file_path, 
            document_type, 
            auto_detect_type
        )
        print("CONFIG EXTRACT IMAGES: ", self.config)
        # If PDF contains images and we want to process them
        if self.config.extract_images is True:
            print("Extracting and processing images from PDF...")
            try:
                # Extract images from PDF
                from ..parsers.pdf_parser import PDFParser
                pdf_parser = PDFParser(self.llm_provider)
                images = pdf_parser.extract_images(file_path)
                
                if images:
                    print(f"Found {len(images)} images in PDF")
                    
                    # Process each image
                    image_results = []
                    for i, image in enumerate(images):
                        print(f"Processing image {i+1}/{len(images)} from PDF...")
                        try:
                            # Save image temporarily for processing
                            temp_path = f"temp_pdf_image_{i}.png"
                            image.save(temp_path)
                            
                            # Process image
                            img_result = self.image_agent.process_image(temp_path, document_type)
                            image_results.append(img_result)
                            
                            # Clean up temp file
                            os.remove(temp_path)
                            
                        except Exception as e:
                            print(f"Warning: Error processing image {i+1}: {e}")
                            image_results.append({
                                "error": str(e),
                                "image_index": i,
                                "processed": False
                            })
                    
                    # Add image results to document results
                    doc_results["extracted_images"] = {
                        "total_images": len(images),
                        "processed_images": len([img for img in image_results if img.get("processed", True)]),
                        "images": image_results
                    }
                else:
                    print("No images found in PDF")
                    doc_results["extracted_images"] = {
                        "total_images": 0,
                        "processed_images": 0,
                        "images": []
                    }
            except Exception as e:
                print(f"Warning: Error extracting images from PDF: {e}")
                doc_results["extracted_images"] = {
                    "error": str(e),
                    "total_images": 0,
                    "processed_images": 0,
                    "images": []
                }
        
        return doc_results
    
    def _process_image_file(self, file_path: str, document_type: DocumentType) -> Dict[str, Any]:
        """Process an image file."""
        print("Processing image file...")
        
        # Process the image
        img_results = self.image_agent.process_image(file_path, document_type)
        
        return img_results
    
    def process_directory(self, 
                         directory_path: str, 
                         document_type: DocumentType = DocumentType.GENERAL,
                         file_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process all supported files in a directory.
        
        Args:
            directory_path: Path to directory to process
            document_type: Type of document
            file_patterns: List of file patterns to include (e.g., ['*.pdf', '*.jpg'])
            
        Returns:
            Dictionary containing processing results for all files
        """
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"Directory not found: {directory_path}")
        
        # Default file patterns
        if file_patterns is None:
            file_patterns = pdf_patterns + image_patterns       
        
        # Find all matching files
        all_files = []
        for pattern in file_patterns:
            pattern_path = os.path.join(directory_path, pattern)
            import glob
            all_files.extend(glob.glob(pattern_path))
        
        if not all_files:
            print(f"No supported files found in {directory_path}")
            return {
                "directory_path": directory_path,
                "total_files": 0,
                "processed_files": 0,
                "files": [],
                "processing_timestamp": self._get_timestamp()
            }
        
        print(f"Found {len(all_files)} files to process")
        print("-" * 50)
        
        # Process each file
        processed_files = []
        for i, file_path in enumerate(all_files):
            print(f"\nProcessing file {i+1}/{len(all_files)}: {os.path.basename(file_path)}")
            try:
                result = self.process_file(file_path, document_type)
                result["file_path"] = file_path
                result["processed"] = True
                processed_files.append(result)
                print(f"✓ Successfully processed {os.path.basename(file_path)}")
            except Exception as e:
                print(f"✗ Error processing {os.path.basename(file_path)}: {e}")
                processed_files.append({
                    "file_path": file_path,
                    "error": str(e),
                    "processed": False
                })
        
        # Compile directory results
        return {
            "directory_path": directory_path,
            "total_files": len(all_files),
            "processed_files": len([f for f in processed_files if f.get("processed", False)]),
            "files": processed_files,
            "processing_timestamp": self._get_timestamp(),
            "summary": self._generate_directory_summary(processed_files)
        }
    
    def _generate_directory_summary(self, processed_files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of directory processing results."""
        successful_files = [f for f in processed_files if f.get("processed", False)]
        failed_files = [f for f in processed_files if not f.get("processed", False)]
        
        # Count by file type
        file_types = {}
        for file in processed_files:
            ext = Path(file["file_path"]).suffix.lower()
            file_types[ext] = file_types.get(ext, 0) + 1
        
        # Count by document type
        doc_types = {}
        for file in successful_files:
            if "document_info" in file:
                doc_type = file["document_info"].get("document_type", "unknown")
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        return {
            "successful_files": len(successful_files),
            "failed_files": len(failed_files),
            "file_types": file_types,
            "document_types": doc_types,
            "success_rate": len(successful_files) / len(processed_files) if processed_files else 0
        }
    
    def batch_process(self, 
                     file_paths: List[str], 
                     document_type: DocumentType = DocumentType.GENERAL) -> Dict[str, Any]:
        """
        Process multiple files in batch.
        
        Args:
            file_paths: List of file paths to process
            document_type: Type of document
            
        Returns:
            Dictionary containing batch processing results
        """
        print(f"Starting batch processing of {len(file_paths)} files...")
        print("-" * 50)
        
        processed_files = []
        for i, file_path in enumerate(file_paths):
            print(f"\nProcessing file {i+1}/{len(file_paths)}: {os.path.basename(file_path)}")
            try:
                result = self.process_file(file_path, document_type)
                result["file_path"] = file_path
                result["processed"] = True
                processed_files.append(result)
                print(f"✓ Successfully processed {os.path.basename(file_path)}")
            except Exception as e:
                print(f"✗ Error processing {os.path.basename(file_path)}: {e}")
                processed_files.append({
                    "file_path": file_path,
                    "error": str(e),
                    "processed": False
                })
        
        return {
            "total_files": len(file_paths),
            "processed_files": len([f for f in processed_files if f.get("processed", False)]),
            "files": processed_files,
            "processing_timestamp": self._get_timestamp(),
            "summary": self._generate_directory_summary(processed_files)
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()

