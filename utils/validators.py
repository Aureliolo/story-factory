"""Input validation utilities for Story Factory."""

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def validate_story_id(story_id: str) -> bool:
    """Validate that a story ID is safe to use in file paths.
    
    Args:
        story_id: The story ID to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If the story ID is invalid
    """
    if not story_id:
        raise ValidationError("Story ID cannot be empty")
    
    # Check for path traversal attempts
    if ".." in story_id or "/" in story_id or "\\" in story_id:
        raise ValidationError("Story ID contains invalid characters")
    
    # Ensure it's a valid UUID format (8-4-4-4-12 hex)
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    if not re.match(uuid_pattern, story_id, re.IGNORECASE):
        raise ValidationError(f"Story ID must be a valid UUID format, got: {story_id}")
    
    return True


def validate_file_path(file_path: str, base_dir: Optional[Path] = None) -> Path:
    """Validate and normalize a file path to prevent path traversal attacks.
    
    Args:
        file_path: The file path to validate
        base_dir: Optional base directory to constrain the path to
        
    Returns:
        Resolved Path object
        
    Raises:
        ValidationError: If the path is unsafe
    """
    if not file_path:
        raise ValidationError("File path cannot be empty")
    
    path = Path(file_path).resolve()
    
    # If base_dir is provided, ensure the path is within it
    if base_dir:
        base = Path(base_dir).resolve()
        try:
            # This will raise ValueError if path is not relative to base
            path.relative_to(base)
        except ValueError:
            raise ValidationError(f"Path {file_path} is outside allowed directory {base_dir}")
    
    return path


def validate_model_name(model_name: str) -> bool:
    """Validate that a model name is properly formatted.
    
    Args:
        model_name: The model name to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If the model name is invalid
    """
    if not model_name:
        raise ValidationError("Model name cannot be empty")
    
    # Check for special "auto" value
    if model_name == "auto":
        return True
    
    # Model names should follow pattern: username/model:tag or username/model
    # Allow alphanumeric, underscore, hyphen, dot, slash, and colon
    pattern = r'^[a-zA-Z0-9_\-\.]+/[a-zA-Z0-9_\-\.:]+$'
    if not re.match(pattern, model_name):
        raise ValidationError(
            f"Model name '{model_name}' is invalid. "
            "Expected format: 'username/model' or 'username/model:tag'"
        )
    
    return True


def validate_temperature(temperature: float) -> bool:
    """Validate temperature parameter.
    
    Args:
        temperature: The temperature value to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If the temperature is invalid
    """
    if not isinstance(temperature, (int, float)):
        raise ValidationError(f"Temperature must be a number, got {type(temperature).__name__}")
    
    if temperature < 0.0 or temperature > 2.0:
        raise ValidationError(f"Temperature must be between 0.0 and 2.0, got {temperature}")
    
    return True


def validate_chapter_number(chapter_number: int, max_chapters: Optional[int] = None) -> bool:
    """Validate chapter number.
    
    Args:
        chapter_number: The chapter number to validate
        max_chapters: Optional maximum number of chapters
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If the chapter number is invalid
    """
    if not isinstance(chapter_number, int):
        raise ValidationError(f"Chapter number must be an integer, got {type(chapter_number).__name__}")
    
    if chapter_number < 1:
        raise ValidationError(f"Chapter number must be positive, got {chapter_number}")
    
    if max_chapters and chapter_number > max_chapters:
        raise ValidationError(
            f"Chapter number {chapter_number} exceeds maximum {max_chapters}"
        )
    
    return True


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename to remove unsafe characters.
    
    Args:
        filename: The filename to sanitize
        
    Returns:
        Sanitized filename
    """
    if not filename:
        return "untitled"
    
    # Remove path separators and other unsafe characters
    # Keep only alphanumeric, spaces, hyphens, underscores, and dots
    sanitized = re.sub(r'[^\w\s\-\.]', '', filename)
    
    # Replace multiple spaces/underscores with single ones
    sanitized = re.sub(r'[\s_]+', '_', sanitized)
    
    # Remove leading/trailing spaces, dots, and underscores
    sanitized = sanitized.strip(' ._')
    
    # Ensure it's not empty after sanitization
    if not sanitized:
        return "untitled"
    
    # Limit length
    max_length = 200
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized
