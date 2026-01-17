---
applyTo: "agents/*.py"
---

## AI Agent Implementation Guidelines

When creating or modifying AI agents, follow these guidelines:

### Agent Architecture

1. **Extend BaseAgent**: All agents must extend `agents/base.py`
   ```python
   from agents.base import BaseAgent

   class MyAgent(BaseAgent):
       def __init__(self, settings: Settings):
           super().__init__(
               name="MyAgent",
               role="agent_role",
               settings=settings
           )
   ```

2. **Agent Responsibilities**: Each agent has a specific role
   - **Interviewer**: Gathers story requirements from users
   - **Architect**: Designs world, characters, and plot structure
   - **Writer**: Generates prose content
   - **Editor**: Polishes and refines writing
   - **Continuity**: Detects plot holes and inconsistencies
   - Don't mix agent responsibilities

### Implementation Requirements

1. **Temperature Settings**: Configure appropriate temperature for agent's role
   - Writer: 0.9 (high creativity)
   - Editor: 0.6 (balanced)
   - Continuity: 0.3 (strict, analytical)
   - Architect: 0.85 (creative but structured)
   - Interviewer: 0.7 (conversational)

2. **Prompts**: Use structured prompts with clear instructions
   - Include role definition and constraints
   - Specify output format (JSON when needed)
   - Provide context from story state
   - Use system prompts for role definition

3. **Error Handling**: Use decorators from `utils/error_handling.py`
   ```python
   from utils.error_handling import handle_ollama_errors, retry_with_fallback

   @handle_ollama_errors
   @retry_with_fallback(max_retries=3)
   async def generate_content(self, prompt: str):
       # Implementation
   ```

4. **Logging**: Log all significant operations
   ```python
   import logging
   logger = logging.getLogger(__name__)

   logger.debug(f"Generating content with prompt: {prompt[:100]}...")
   logger.info(f"Successfully generated {len(result)} characters")
   logger.warning("Retrying due to incomplete response")
   logger.error(f"Failed to generate content: {error}")
   ```

### Ollama Integration

1. **Use Base Class Methods**: Leverage methods from BaseAgent
   - `self.generate(prompt)` - Make LLM calls with retry logic (synchronous)
   - `self.settings` - Access model configuration
   - `self.client` - Ollama client instance

2. **Rate Limiting**: BaseAgent handles rate limiting (max 2 concurrent)
   - Don't implement custom rate limiting
   - Respect the configured timeout (default from settings)

3. **Context Management**: Pass story context appropriately
   - Use `StoryState` from `memory/story_state.py`
   - Include relevant history for continuity
   - Balance context size vs. token limits

### Output Processing

1. **JSON Extraction**: Use `utils/json_parser.py` for parsing LLM responses
   ```python
   from utils.json_parser import extract_json, parse_json_response

   json_str = extract_json(llm_response)
   data = parse_json_response(json_str)
   ```

2. **Validation**: Validate all LLM outputs
   - Check for required fields
   - Verify data types match expectations
   - Handle malformed or incomplete responses
   - Use Pydantic models for structured data

3. **Error Recovery**: Implement graceful degradation
   - Retry with different prompts if needed
   - Fall back to simpler approaches on failure
   - Don't fail silently - log and raise appropriate exceptions

### State Management

1. **Story State**: Update story state by directly modifying fields (Pydantic model)
   ```python
   # Add a chapter
   story_state.chapters.append(chapter)

   # Update a character by finding and modifying it
   for i, char in enumerate(story_state.characters):
       if char.id == character_id:
           story_state.characters[i] = updated_character
           break
   ```

2. **Thread Safety**: Be aware that agents may be called concurrently
   - Don't modify shared state without synchronization
   - Use thread-safe data structures when needed

### Testing

1. **Mock Ollama**: Always mock Ollama API calls in tests
   ```python
   from unittest.mock import patch, MagicMock

   @pytest.fixture
   def mock_ollama(self):
       with patch("agents.base.ollama.Client") as mock_client:
           mock_instance = MagicMock()
           mock_client.return_value = mock_instance
           yield mock_instance
   ```

2. **Test Cases**: Cover key scenarios
   - Successful generation with valid output
   - Handling of malformed LLM responses
   - Error conditions (network errors, timeouts)
   - Validation of output format
   - Integration with story state

### Example Agent Implementation

```python
from agents.base import BaseAgent
from settings import Settings
from memory.story_state import StoryState
from utils.json_parser import extract_json
from utils.error_handling import handle_ollama_errors
import logging

logger = logging.getLogger(__name__)

WRITER_SYSTEM_PROMPT = """You are a creative writer specializing in storytelling.
Your task is to generate engaging prose content for stories.
Follow the style and tone established in the story context."""

class WriterAgent(BaseAgent):
    """Agent responsible for generating prose content."""

    def __init__(self, settings: Settings):
        super().__init__(
            name="Writer",
            role="writer",
            system_prompt=WRITER_SYSTEM_PROMPT,  # Required parameter
            agent_role="writer",  # For auto model/temperature selection
            settings=settings,
        )

    @handle_ollama_errors
    def write_chapter(
        self,
        story_state: StoryState,
        chapter_number: int
    ) -> str:
        """Generate a chapter based on story state."""
        logger.info(f"Writing chapter {chapter_number}")

        # Build context from story state
        context = self._build_context(story_state, chapter_number)

        # Create prompt
        prompt = self._create_prompt(context)
        logger.debug(f"Prompt: {prompt[:200]}...")

        # Call LLM (synchronous)
        response = self.generate(prompt, context)

        logger.info(f"Generated {len(response)} characters for chapter {chapter_number}")
        return response

    def _build_context(self, story_state: StoryState, chapter_number: int) -> str:
        """Build context for the chapter."""
        return f"Chapter {chapter_number} of {story_state.title}"

    def _create_prompt(self, context: str) -> str:
        """Create the writing prompt."""
        return f"Write the next chapter. Context: {context}"
```

### Best Practices

1. **Separation of Concerns**: Keep agent logic focused on its role
2. **Reusability**: Use base class functionality when possible
3. **Maintainability**: Write clear, documented code
4. **Performance**: Be mindful of token usage and API calls
5. **Reliability**: Handle errors gracefully and provide informative messages
