"""
Predefined image categories for classification.
"""

from enum import Enum
from typing import List

class ImageCategory(Enum):
    """Enumeration of image categories for classification."""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    CHART = "chart"
    DIAGRAM = "diagram"
    GRAPH = "graph"
    PICTURE = "picture"
    PHOTO = "photo"
    DRAWING = "drawing"
    ILLUSTRATION = "illustration"
    MAP = "map"
    SIGNATURE = "signature"
    SEAL = "seal"
    STAMP = "stamp"
    LOGO = "logo"
    BANNER = "banner"
    HEADER = "header"
    FOOTER = "footer"


def get_all_categories() -> List[str]:
    """
    Get all available image categories.
    
    Returns:
        List of all image categories
    """
    return [category.value for category in ImageCategory]
