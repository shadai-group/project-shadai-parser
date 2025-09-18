"""
Metadata schemas and document type definitions for the processing agents.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import json
from parser_shadai.agents.language_config import get_language_prompt


class DocumentType(Enum):
    """Enumeration of supported document types."""
    LEGAL = "legal"
    MEDICAL = "medical"
    FINANCIAL = "financial"
    TECHNICAL = "technical"
    ACADEMIC = "academic"
    BUSINESS = "business"
    GENERAL = "general"


@dataclass
class MetadataSchema:
    """Schema definition for metadata extraction based on document type."""
    document_type: DocumentType
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    description: str = ""
    
    def get_all_fields(self) -> List[str]:
        """Get all fields (required + optional)."""
        return self.required_fields + self.optional_fields


@dataclass
class ChunkNode:
    """Represents a processed chunk with its metadata."""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    page_number: Optional[int] = None
    chunk_index: int = 0
    total_chunks: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "metadata": self.metadata,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class MetadataSchemas:
    """Collection of predefined metadata schemas for different document types."""
    
    SCHEMAS = {
        DocumentType.LEGAL: MetadataSchema(
            document_type=DocumentType.LEGAL,
            required_fields=[
                "summary",
                "document_type",
                "parties_involved",
                "legal_issues",
                "key_terms",
                "jurisdiction",
                "date_references"
            ],
            optional_fields=[
                "case_number",
                "court_name",
                "legal_precedents",
                "statutes_cited",
                "outcome",
                "legal_advice",
                "compliance_requirements"
            ],
            description="Legal document metadata extraction"
        ),
        
        DocumentType.MEDICAL: MetadataSchema(
            document_type=DocumentType.MEDICAL,
            required_fields=[
                "summary",
                "document_type",
                "patient_info",
                "medical_conditions",
                "treatments",
                "medications",
                "vital_signs"
            ],
            optional_fields=[
                "diagnosis",
                "symptoms",
                "medical_history",
                "allergies",
                "procedures",
                "recommendations",
                "follow_up_required"
            ],
            description="Medical document metadata extraction"
        ),
        
        DocumentType.FINANCIAL: MetadataSchema(
            document_type=DocumentType.FINANCIAL,
            required_fields=[
                "summary",
                "document_type",
                "financial_institution",
                "account_info",
                "amounts",
                "dates",
                "transaction_types"
            ],
            optional_fields=[
                "account_holder",
                "balance",
                "interest_rates",
                "fees",
                "terms_conditions",
                "regulatory_info",
                "tax_implications"
            ],
            description="Financial document metadata extraction"
        ),
        
        DocumentType.TECHNICAL: MetadataSchema(
            document_type=DocumentType.TECHNICAL,
            required_fields=[
                "summary",
                "document_type",
                "technology_stack",
                "system_requirements",
                "implementation_steps",
                "technical_specifications"
            ],
            optional_fields=[
                "code_examples",
                "dependencies",
                "performance_metrics",
                "troubleshooting",
                "best_practices",
                "security_considerations",
                "scalability_notes"
            ],
            description="Technical document metadata extraction"
        ),
        
        DocumentType.ACADEMIC: MetadataSchema(
            document_type=DocumentType.ACADEMIC,
            required_fields=[
                "summary",
                "document_type",
                "research_topic",
                "methodology",
                "findings",
                "conclusions",
                "references"
            ],
            optional_fields=[
                "authors",
                "institution",
                "publication_date",
                "abstract",
                "keywords",
                "research_questions",
                "limitations",
                "future_work"
            ],
            description="Academic document metadata extraction"
        ),
        
        DocumentType.BUSINESS: MetadataSchema(
            document_type=DocumentType.BUSINESS,
            required_fields=[
                "summary",
                "document_type",
                "business_objective",
                "stakeholders",
                "key_metrics",
                "recommendations",
                "timeline"
            ],
            optional_fields=[
                "market_analysis",
                "competitors",
                "financial_projections",
                "risks",
                "opportunities",
                "action_items",
                "success_criteria"
            ],
            description="Business document metadata extraction"
        ),
        
        DocumentType.GENERAL: MetadataSchema(
            document_type=DocumentType.GENERAL,
            required_fields=[
                "summary",
                "document_type",
                "main_topic",
                "key_points",
                "purpose",
                "audience"
            ],
            optional_fields=[
                "author",
                "creation_date",
                "keywords",
                "sentiment",
                "language",
                "complexity_level",
                "action_items"
            ],
            description="General document metadata extraction"
        )
    }
    
    @classmethod
    def get_schema(cls, document_type: DocumentType) -> MetadataSchema:
        """Get metadata schema for a document type."""
        return cls.SCHEMAS.get(document_type, cls.SCHEMAS[DocumentType.GENERAL])
    
    @classmethod
    def get_available_types(cls) -> List[DocumentType]:
        """Get list of available document types."""
        return list(cls.SCHEMAS.keys())
    
    @classmethod
    def get_schema_fields(cls, document_type: DocumentType) -> List[str]:
        """Get all fields for a document type."""
        schema = cls.get_schema(document_type)
        return schema.get_all_fields()


class ChunkProcessor:
    """Utility class for processing chunks and extracting metadata."""
    
    def __init__(self, llm_provider, document_type: DocumentType, language: str = "en"):
        """
        Initialize chunk processor.
        
        Args:
            llm_provider: LLM provider instance
            document_type: Type of document being processed
            language: Language code for processing (default: "en")
        """
        self.llm_provider = llm_provider
        self.document_type = document_type
        self.language = language
        self.schema = MetadataSchemas.get_schema(document_type)
    
    def extract_metadata(self, chunk: str, chunk_id: str, page_number: Optional[int] = None) -> ChunkNode:
        """
        Extract metadata from a text chunk.
        
        Args:
            chunk: Text content of the chunk
            chunk_id: Unique identifier for the chunk
            page_number: Page number where chunk was found
            
        Returns:
            ChunkNode with extracted metadata
        """
        # Create prompt for metadata extraction
        prompt = self._create_metadata_prompt(chunk)
        
        try:
            # Call LLM to extract metadata
            response = self.llm_provider.generate_text(prompt)
            metadata = self._parse_metadata_response(response.content)
            
            return ChunkNode(
                chunk_id=chunk_id,
                content=chunk,
                metadata=metadata,
                page_number=page_number
            )
        except Exception as e:
            # Fallback metadata if extraction fails
            return ChunkNode(
                chunk_id=chunk_id,
                content=chunk,
                metadata={
                    "summary": chunk[:200] + "..." if len(chunk) > 200 else chunk,
                    "document_type": self.document_type.value,
                    "extraction_error": str(e)
                },
                page_number=page_number
            )
    
    def _create_metadata_prompt(self, chunk: str) -> str:
        """Create prompt for metadata extraction."""
        fields = self.schema.get_all_fields()
        fields_str = ", ".join(fields)
        language_prompt = get_language_prompt(self.language)
        
        prompt = f"""
{language_prompt}

Extract the following metadata from the given text chunk. Return the result as a JSON object with the specified fields.

Required fields: {fields_str}

Text chunk:
{chunk}

Instructions:
1. Extract information for each field based on the document type: {self.document_type.value}
2. If a field cannot be determined from the text, use null or an empty string
3. For the summary field, provide a concise summary of the chunk content
4. Return only valid JSON, no additional text

JSON Response:
"""
        return prompt
    
    def _parse_metadata_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract metadata."""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                # If no JSON found, create basic metadata
                return {
                    "summary": response[:200] + "..." if len(response) > 200 else response,
                    "document_type": self.document_type.value,
                    "raw_response": response
                }
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "summary": response[:200] + "..." if len(response) > 200 else response,
                "document_type": self.document_type.value,
                "raw_response": response
            }
