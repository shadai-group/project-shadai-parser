"""
Test script for TableExtractor - demonstrates table extraction from PDFs.

This script shows:
1. Automatic table detection
2. Markdown conversion
3. Header detection
4. Table statistics
"""

import json
import logging
from pathlib import Path

from parser_shadai.parsers.table_extractor import TableExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_table_extraction() -> None:
    """Test table extraction on PDF files in data/ folder."""
    logger.info("=" * 80)
    logger.info("TABLE EXTRACTION TEST")
    logger.info("=" * 80)

    # Get PDF files from data folder
    data_folder = Path("data")
    pdf_files = list(data_folder.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {data_folder}")
        return

    logger.info(f"\nFound {len(pdf_files)} PDF file(s) in {data_folder}")

    # Create table extractor
    extractor = TableExtractor()

    # Test each PDF
    for pdf_path in pdf_files[:3]:  # Limit to first 3 PDFs
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing: {pdf_path.name}")
        logger.info(f"{'=' * 80}")

        try:
            # Extract tables
            tables = extractor.extract_tables(pdf_path=str(pdf_path))

            if not tables:
                logger.info(f"  ℹ️  No tables found in {pdf_path.name}")
                continue

            # Show table summary
            summary = extractor.get_table_summary(tables=tables)

            logger.info("\n📊 Summary:")
            logger.info(f"  Total tables: {summary['total_tables']}")
            logger.info(f"  Total rows: {summary['total_rows']}")
            logger.info(f"  Total cells: {summary['total_cells']}")
            logger.info(f"  Pages with tables: {summary['pages_with_tables']}")
            logger.info(f"  Tables with headers: {summary['tables_with_headers']}")
            logger.info(f"  Avg rows/table: {summary['avg_rows_per_table']:.1f}")
            logger.info(f"  Avg cols/table: {summary['avg_cols_per_table']:.1f}")
            logger.info(f"  Cell fill rate: {summary['cell_fill_rate']:.1%}")

            # Show first 2 tables
            for i, table in enumerate(tables[:2], start=1):
                logger.info(f"\n{'─' * 80}")
                logger.info(f"Table {i} Details:")
                logger.info(f"{'─' * 80}")
                logger.info(f"  Page: {table['page_number']}")
                logger.info(f"  Table ID: {table['table_id']}")
                logger.info(
                    f"  Size: {table['row_count']} rows × {table['col_count']} columns"
                )
                logger.info(f"  Has header: {table['has_header']}")

                if table["has_header"] and table["header_row"]:
                    logger.info(f"  Headers: {' | '.join(table['header_row'])}")

                logger.info("\n  Markdown Preview:")
                # Show first 10 lines of markdown
                markdown_lines = table["markdown"].split("\n")[:10]
                for line in markdown_lines:
                    logger.info(f"    {line}")

                if len(markdown_lines) < table["row_count"]:
                    logger.info(
                        f"    ... ({table['row_count'] - len(markdown_lines)} more rows)"
                    )

            # Save tables to JSON
            output_folder = Path("output")
            output_folder.mkdir(exist_ok=True)

            output_file = output_folder / f"{pdf_path.stem}_tables.json"
            with open(file=output_file, mode="w", encoding="utf-8") as f:
                json.dump(obj=tables, fp=f, ensure_ascii=False, indent=2)

            logger.info(f"\n✓ Saved tables to: {output_file.name}")

        except Exception as e:
            logger.error(f"❌ Error processing {pdf_path.name}: {e}")
            import traceback

            traceback.print_exc()

    logger.info(f"\n{'=' * 80}")
    logger.info("TABLE EXTRACTION COMPLETE")
    logger.info(f"{'=' * 80}\n")


def main() -> None:
    """Main entry point."""
    try:
        test_table_extraction()

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
