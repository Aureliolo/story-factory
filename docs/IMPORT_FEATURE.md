# Import Feature

The import feature allows you to extract entities and relationships from existing story text and populate your world database.

## Overview

Use the **Import from Text** button in the World Builder page to analyze existing prose and automatically extract:
- **Characters** (name, role, description, relationships)
- **Locations** (name, description, significance)
- **Items** (name, description, properties, significance)
- **Relationships** (connections between characters with descriptions)

## How to Use

1. **Navigate to World Builder**: Go to the World Builder tab in the application
2. **Click "Import from Text"**: Find the button in the toolbar (blue button with upload icon)
3. **Input Your Text**: Either:
   - Paste text directly into the textarea
   - Upload a `.txt` or `.md` file
4. **Extract Entities**: Click "Extract Entities" to analyze the text
5. **Review Results**: The AI will show extracted entities organized by type
6. **Select Items**: Review each entity and uncheck any you don't want to import
7. **Import**: Click "Import Selected" to add entities to your world

## Features

### AI-Powered Extraction

The system uses your configured LLM (via Ollama) to intelligently identify:
- Named characters with their roles and descriptions
- Important locations mentioned in the story
- Significant items and objects
- Relationships between characters

### Confidence Scoring

Each extracted entity includes a confidence score (0-100%):
- **High confidence (70%+)**: Clearly mentioned with details
- **Low confidence (<70%)**: Flagged with ⚠️ for review
- **Very low confidence (<40%)**: May be uncertain or ambiguous

Items flagged for review are marked with ⚠️ and shown in orange.

### Conflict Resolution

The import process automatically:
- Checks for duplicate entities by name
- Skips entities that already exist in your world
- Only creates relationships between existing or newly-imported entities

### Multiple Format Support

Currently supports:
- Plain text (`.txt`)
- Markdown (`.md`)

## Best Practices

1. **Provide Context**: Longer, more detailed text produces better extraction results
2. **Review Carefully**: Always review low-confidence items (marked with ⚠️)
3. **Incremental Import**: You can import multiple times - duplicates are automatically skipped
4. **Edit After Import**: Imported entities can be edited like any other entity in the World Builder

## Technical Details

### Extraction Process

1. **Character Extraction**: Identifies names, roles (protagonist/antagonist/supporting), descriptions
2. **Location Extraction**: Finds places mentioned, their descriptions and significance
3. **Item Extraction**: Identifies significant objects with properties
4. **Relationship Inference**: Analyzes interactions between characters

### Prompt Engineering

The extraction uses similar prompt structures to world generation in `world_quality_service.py`:
- Low temperature (0.3) for consistent extraction
- Structured JSON output format
- Confidence scoring for each extracted item

### Performance

- Extraction speed depends on text length and your LLM
- Longer texts may take 30-60 seconds to fully analyze
- Results are shown progressively (characters → locations → items → relationships)

## Troubleshooting

**No entities extracted**: Make sure your text is substantial (50+ characters) and contains character names, locations, or items.

**Low confidence items**: The AI may be uncertain. Review these carefully and edit after import if needed.

**Missing relationships**: Relationship inference is conservative - it only includes clearly stated or strongly implied connections.

**Duplicates**: If you see "duplicate" messages, entities with the same name already exist in your world. The import skips these automatically.

## Example Workflow

```
1. Write or paste your story draft (500+ words recommended)
2. Click "Import from Text"
3. Review the 10-15 entities typically extracted from a short story
4. Uncheck any incorrect or unwanted items
5. Import selected entities
6. Edit or enhance entities in the World Builder as needed
7. Continue writing with a populated world database
```

## Future Enhancements

Planned improvements:
- Support for more file formats (DOCX, EPUB, etc.)
- Faction/organization extraction
- Concept/theme extraction
- Batch import from multiple files
- Custom extraction rules/patterns
