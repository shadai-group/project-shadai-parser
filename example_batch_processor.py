"""
Batch PDF Processor Example - Professional Production-Ready Script.

This script demonstrates production-grade PDF processing:
- Reads all PDF files from data/ folder
- Uses Gemini API from .env file (GOOGLE_API_KEY)
- Processes each PDF with optimized parsing
- Saves structured JSON outputs to output/ folder
- Comprehensive error handling and logging
- Progress tracking and performance metrics

Usage:
    python example_batch_processor.py

Requirements:
    - .env file with GOOGLE_API_KEY (or AZURE credentials)
    - data/ folder with PDF files
    - output/ folder (created automatically)
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from parser_shadai.infrastructure import Container, ParserConfiguration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(filename="batch_processing.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


class BatchPDFProcessor:
    """
    Professional batch PDF processor with comprehensive error handling.

    Features:
    - Automatic provider detection (Gemini/Azure/OpenAI)
    - Progress tracking with ETA
    - Detailed metrics (processing time, token usage, success rate)
    - Structured JSON output
    - Graceful error handling
    """

    def __init__(
        self,
        data_folder: str = "data",
        output_folder: str = "output",
        use_optimized: bool = True,
    ):
        """
        Initialize batch processor.

        Args:
            data_folder: Folder containing PDF files
            output_folder: Folder for JSON outputs
            use_optimized: Use optimized extraction (recommended)
        """
        self.data_folder = Path(data_folder)
        self.output_folder = Path(output_folder)
        self.use_optimized = use_optimized

        # Create output folder if not exists
        self.output_folder.mkdir(exist_ok=True)

        # Metrics tracking
        self.metrics: Dict[str, Any] = {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "total_chunks": 0,
            "total_tokens": 0,
            "total_time_seconds": 0,
            "files_processed": [],
        }

        # Load environment variables
        load_dotenv()

        # Initialize parser container
        self.container = self._create_container()

        logger.info("=" * 80)
        logger.info("Batch PDF Processor Initialized")
        logger.info("=" * 80)
        logger.info(f"Data folder: {self.data_folder.absolute()}")
        logger.info(f"Output folder: {self.output_folder.absolute()}")
        logger.info(f"Optimized extraction: {self.use_optimized}")

    def _create_container(self) -> Container:
        """
        Create parser container with auto-detected provider.

        Priority:
        1. GOOGLE_API_KEY (Gemini)
        2. AZURE_API_KEY + AZURE_ENDPOINT (Azure OpenAI)
        3. OPENAI_API_KEY (OpenAI)
        4. ANTHROPIC_API_KEY (Claude)

        Returns:
            Configured Container instance

        Raises:
            EnvironmentError: If no valid API key is found
        """
        # Try Gemini first (recommended for this example)
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if google_api_key:
            logger.info("✓ Using Google Gemini (GOOGLE_API_KEY found)")
            config = ParserConfiguration(
                llm_provider_name="gemini",
                llm_credentials=google_api_key,
                llm_model="gemini-2.0-flash-exp",
                temperature=0.2,
                chunk_size=4000,
                chunk_overlap=400,
                chunker_type="smart",
                language="multilingual",
                auto_detect_language=False,
                extract_text_from_images=True,
                enable_caching=True,
                cache_ttl_seconds=86400,  # 24 hours
            )
            return Container(config=config)

        # Try Azure OpenAI
        azure_api_key = os.getenv("AZURE_API_KEY")
        azure_endpoint = os.getenv("AZURE_ENDPOINT")
        azure_deployment = os.getenv("AZURE_DEPLOYMENT")
        if azure_api_key and azure_endpoint and azure_deployment:
            logger.info("✓ Using Azure OpenAI (AZURE credentials found)")
            config = ParserConfiguration(
                llm_provider_name="azure",
                llm_credentials={
                    "api_key": azure_api_key,
                    "azure_endpoint": azure_endpoint,
                    "azure_deployment": azure_deployment,
                    "api_version": os.getenv("API_VERSION", "2023-05-15"),
                },
                llm_model=azure_deployment,
                temperature=0.2,
                chunk_size=4000,
                chunk_overlap=400,
                chunker_type="smart",
                language="multilingual",
                auto_detect_language=False,
                extract_text_from_images=True,
                enable_caching=True,
                cache_ttl_seconds=86400,
            )
            return Container(config=config)

        # Try OpenAI
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            logger.info("✓ Using OpenAI (OPENAI_API_KEY found)")
            config = ParserConfiguration(
                llm_provider_name="openai",
                llm_credentials=openai_api_key,
                llm_model="gpt-4o-mini",
                temperature=0.2,
                chunk_size=4000,
                chunk_overlap=400,
                chunker_type="smart",
                language="multilingual",
                auto_detect_language=False,
                extract_text_from_images=True,
                enable_caching=True,
                cache_ttl_seconds=86400,
            )
            return Container(config=config)

        # Try Anthropic
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_api_key:
            logger.info("✓ Using Anthropic Claude (ANTHROPIC_API_KEY found)")
            config = ParserConfiguration(
                llm_provider_name="anthropic",
                llm_credentials=anthropic_api_key,
                llm_model="claude-3-5-sonnet-20241022",
                temperature=0.2,
                chunk_size=4000,
                chunk_overlap=400,
                chunker_type="smart",
                language="multilingual",
                auto_detect_language=False,
                extract_text_from_images=True,
                enable_caching=True,
                cache_ttl_seconds=86400,
            )
            return Container(config=config)

        # No valid API key found
        raise EnvironmentError(
            "No valid API key found in .env file. Please set one of:\n"
            "  - GOOGLE_API_KEY (Gemini)\n"
            "  - AZURE_API_KEY + AZURE_ENDPOINT + AZURE_DEPLOYMENT (Azure)\n"
            "  - OPENAI_API_KEY (OpenAI)\n"
            "  - ANTHROPIC_API_KEY (Anthropic)"
        )

    def get_pdf_files(self) -> List[Path]:
        """
        Get all PDF files from data folder.

        Returns:
            List of PDF file paths

        Raises:
            FileNotFoundError: If data folder doesn't exist
        """
        if not self.data_folder.exists():
            raise FileNotFoundError(
                f"Data folder not found: {self.data_folder.absolute()}\n"
                f"Please create the folder and add PDF files."
            )

        pdf_files = sorted(self.data_folder.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"⚠️  No PDF files found in {self.data_folder.absolute()}")

        return pdf_files

    def process_single_pdf(
        self, pdf_path: Path
    ) -> tuple[bool, Optional[Dict[str, Any]]]:
        """
        Process a single PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Tuple of (success: bool, result: dict or None)
        """
        logger.info("─" * 80)
        logger.info(f"Processing: {pdf_path.name}")
        logger.info(f"  Size: {pdf_path.stat().st_size / 1024:.1f} KB")

        start_time = time.time()

        try:
            # Create document agent for this file
            agent = self.container.create_document_agent(file_path=str(pdf_path))

            # Process document
            result = agent.process_document(
                document_path=str(pdf_path),
                auto_detect_type=True,
            )

            processing_time = time.time() - start_time

            # Extract metrics
            total_chunks = result.get("total_chunks", 0)
            total_tokens = result.get("total_tokens", 0)

            logger.info(f"✓ Success in {processing_time:.2f}s")
            logger.info(f"  Chunks: {total_chunks}")
            logger.info(f"  Tokens: {total_tokens:,}")
            logger.info(f"  Speed: {total_chunks / processing_time:.1f} chunks/sec")

            # Update metrics
            self.metrics["successful"] += 1
            self.metrics["total_chunks"] += total_chunks
            self.metrics["total_tokens"] += total_tokens
            self.metrics["total_time_seconds"] += processing_time
            self.metrics["files_processed"].append(
                {
                    "filename": pdf_path.name,
                    "status": "success",
                    "chunks": total_chunks,
                    "tokens": total_tokens,
                    "processing_time_seconds": processing_time,
                }
            )

            # Add processing metadata
            result["processing_metadata"] = {
                "filename": pdf_path.name,
                "processing_time_seconds": processing_time,
                "processed_at": datetime.now().isoformat(),
                "parser_version": "phase-2-di",
                "optimized_extraction": self.use_optimized,
            }

            return True, result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Failed in {processing_time:.2f}s: {str(e)}")

            self.metrics["failed"] += 1
            self.metrics["files_processed"].append(
                {
                    "filename": pdf_path.name,
                    "status": "failed",
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                }
            )

            return False, None

    def save_result(self, pdf_path: Path, result: Dict[str, Any]) -> None:
        """
        Save processing result as JSON.

        Args:
            pdf_path: Original PDF path
            result: Processing result dictionary
        """
        # Create output filename: document.pdf → document.json
        output_filename = pdf_path.stem + ".json"
        output_path = self.output_folder / output_filename

        try:
            with open(file=output_path, mode="w", encoding="utf-8") as f:
                json.dump(obj=result, fp=f, ensure_ascii=False, indent=2)

            logger.info(f"  Saved: {output_path.name}")

        except Exception as e:
            logger.error(f"  Failed to save JSON: {str(e)}")

    def save_metrics(self) -> None:
        """Save processing metrics to JSON."""
        metrics_path = self.output_folder / "_processing_metrics.json"

        try:
            with open(file=metrics_path, mode="w", encoding="utf-8") as f:
                json.dump(obj=self.metrics, fp=f, ensure_ascii=False, indent=2)

            logger.info(f"\n✓ Metrics saved: {metrics_path.name}")

        except Exception as e:
            logger.error(f"Failed to save metrics: {str(e)}")

    def process_all(self) -> None:
        """
        Process all PDF files in data folder.

        Main workflow:
        1. Get PDF files
        2. Process each file
        3. Save results as JSON
        4. Generate summary report
        """
        logger.info("\n" + "=" * 80)
        logger.info("BATCH PROCESSING STARTED")
        logger.info("=" * 80 + "\n")

        # Get PDF files
        pdf_files = self.get_pdf_files()
        self.metrics["total_files"] = len(pdf_files)

        if not pdf_files:
            logger.warning("No PDF files to process. Exiting.")
            return

        logger.info(f"Found {len(pdf_files)} PDF file(s)\n")

        # Process each file
        for i, pdf_path in enumerate(iterable=pdf_files, start=1):
            logger.info(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")

            success, result = self.process_single_pdf(pdf_path=pdf_path)

            if success and result:
                self.save_result(pdf_path=pdf_path, result=result)

        # Save metrics
        self.save_metrics()

        # Print summary
        self._print_summary()

    def _print_summary(self) -> None:
        """Print processing summary report."""
        logger.info("\n" + "=" * 80)
        logger.info("BATCH PROCESSING COMPLETED")
        logger.info("=" * 80)

        total_files = self.metrics["total_files"]
        successful = self.metrics["successful"]
        failed = self.metrics["failed"]
        total_chunks = self.metrics["total_chunks"]
        total_tokens = self.metrics["total_tokens"]
        total_time = self.metrics["total_time_seconds"]

        logger.info("\n📊 Summary:")
        logger.info(f"  Total files: {total_files}")
        logger.info(f"  ✓ Successful: {successful}")
        logger.info(f"  ❌ Failed: {failed}")
        logger.info(f"  Success rate: {successful / total_files * 100:.1f}%")

        if successful > 0:
            logger.info("\n📈 Processing stats:")
            logger.info(f"  Total chunks: {total_chunks:,}")
            logger.info(f"  Total tokens: {total_tokens:,}")
            logger.info(f"  Total time: {total_time:.2f}s")
            logger.info(f"  Avg chunks/file: {total_chunks / successful:.1f}")
            logger.info(f"  Avg tokens/file: {total_tokens / successful:,.0f}")
            logger.info(f"  Avg time/file: {total_time / successful:.2f}s")
            logger.info(
                f"  Processing speed: {total_chunks / total_time:.1f} chunks/sec"
            )

        logger.info(f"\n📁 Output location: {self.output_folder.absolute()}")
        logger.info("\n" + "=" * 80 + "\n")


def main() -> None:
    """Main entry point."""
    try:
        # Create processor
        processor = BatchPDFProcessor(
            data_folder="data",
            output_folder="output",
            use_optimized=True,
        )

        # Process all PDFs
        processor.process_all()

    except EnvironmentError as e:
        logger.error(f"\n❌ Environment error: {e}")
        sys.exit(1)

    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
