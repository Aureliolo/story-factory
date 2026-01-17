---
applyTo: "ui/pages/*.py"
---

## NiceGUI Page Component Requirements

When creating or modifying NiceGUI page components, follow these guidelines:

### Page Structure

1. **Class-Based Pages**: All pages should be classes with a `build()` method
   ```python
   class MyPage:
       def __init__(self, app_state: AppState, services: ServiceContainer):
           self.app_state = app_state
           self.services = services

       def build(self):
           """Build the page UI."""
           # UI construction code here
   ```

2. **Dependency Injection**: Pages receive dependencies via constructor:
   - `app_state: AppState` - Centralized UI state from `ui/state.py`
   - `services: ServiceContainer` - Service layer access from `services/__init__.py`

3. **No Business Logic**: Pages should only handle UI concerns
   - Delegate business logic to services
   - Don't import from `agents/`, `workflows/`, or `memory/` directly
   - Use `self.services.<service_name>` for all data operations

### NiceGUI Best Practices

1. **UI Elements**:
   - Import: `from nicegui import ui`
   - Use built-in components: `ui.card()`, `ui.button()`, `ui.input()`, `ui.label()`
   - For custom HTML/JS: `ui.html(content, sanitize=False)` (only for trusted content)

2. **Layout**:
   - Use `ui.card()` for content grouping
   - Use `ui.row()` and `ui.column()` for layout
   - Apply classes for styling (defined in `ui/theme.py`)
   - Use `ui.expansion()` for collapsible sections

3. **User Interactions**:
   - Use `.on()` for event handlers: `button.on('click', handler)`
   - Bind data with `.bind_value()` for two-way binding
   - Show notifications: `ui.notify(message, type='positive|negative|warning|info')`

4. **Async Operations**:
   - Use `async def` for handlers that call async services
   - Example: `async def on_save(): await self.services.story_service.save_story()`
   - Use `ui.run_javascript()` for client-side operations when needed

5. **State Management**:
   - Use `self.app_state` for cross-page state
   - Use local variables for page-specific state
   - Subscribe to state changes: `self.app_state.current_project.on_value_change(handler)`

6. **Dialogs and Modals**:
   - Use `ui.dialog()` for modal dialogs
   - Use `ui.menu()` for dropdown menus
   - Close dialogs properly to prevent memory leaks

### Performance

1. **Lazy Loading**: Load heavy data only when needed
2. **Update Optimization**: Use `.update()` on containers instead of rebuilding entire UI
3. **Debouncing**: Debounce rapid user input (search, live updates)

### Example Page Pattern

```python
from nicegui import ui
from ui.state import AppState
from services import ServiceContainer

class SettingsPage:
    """Settings page for configuring the application."""

    def __init__(self, app_state: AppState, services: ServiceContainer):
        self.app_state = app_state
        self.services = services

    def build(self):
        """Build the settings page UI."""
        with ui.card().classes('w-full'):
            ui.label('Settings').classes('text-2xl font-bold')

            with ui.row().classes('w-full gap-4'):
                self._build_model_settings()
                self._build_preferences()

    def _build_model_settings(self):
        """Build model configuration section."""
        with ui.column().classes('flex-1'):
            ui.label('Model Settings').classes('text-xl')

            models = self.services.model_service.list_models()
            ui.select(
                models,
                label='Default Model',
                on_change=self._on_model_change
            ).bind_value(self.app_state, 'selected_model')

    async def _on_model_change(self, event):
        """Handle model selection change."""
        try:
            await self.services.model_service.validate_model(event.value)
            ui.notify('Model updated successfully', type='positive')
        except Exception as e:
            ui.notify(f'Error: {str(e)}', type='negative')
```

### Styling

1. **Use Theme**: Import colors and styles from `ui/theme.py`
2. **Tailwind Classes**: NiceGUI supports Tailwind CSS classes
3. **Custom CSS**: Add to `ui/styles.css` if needed
4. **Responsive Design**: Use responsive Tailwind classes (`sm:`, `md:`, `lg:`)

### Testing

- UI changes should be manually tested by running `python main.py`
- Check the page at http://localhost:7860
- Test on different screen sizes if applicable
- Verify async operations complete successfully
