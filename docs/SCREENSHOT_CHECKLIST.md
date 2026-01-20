# Screenshot Update Checklist

This document lists all screenshots that need to be refreshed to reflect the current state of the application.

## Current Screenshots in README.md

The README.md currently uses the following GitHub-hosted screenshots:

1. **Main Interface / Write Page**
   - Current URL: `https://github.com/user-attachments/assets/9056d5cb-b696-4510-89ba-3fbfd2fe067e`
   - Location in README: Line 33
   - Description: *Clean, modern interface with dark mode support and intuitive navigation*
   - **Status**: Needs refresh ⚠️

2. **Projects Page**
   - Current URL: `https://github.com/user-attachments/assets/8b3b165a-96ef-4dd8-940e-ae1917b7c232`
   - Location in README: Line 37
   - Description: *Manage multiple story projects with backup support*
   - **Status**: Needs refresh ⚠️

3. **Settings Page**
   - Current URL: `https://github.com/user-attachments/assets/42800304-97b9-43b0-9d05-a77bdedc64a9`
   - Location in README: Line 41
   - Description: *Fine-tune every aspect: models, temperatures, workflow, and more*
   - **Status**: Needs refresh ⚠️

4. **Models Page**
   - Current URL: `https://github.com/user-attachments/assets/6cf02131-5876-4b1f-9b71-e75adecf7397`
   - Location in README: Line 45
   - Description: *Download and manage Ollama models with VRAM-aware filtering*
   - **Status**: Needs refresh ⚠️

5. **Analytics Page**
   - Current URL: `https://github.com/user-attachments/assets/1c17001c-18a9-4cd3-886f-611293508937`
   - Location in README: Line 49
   - Description: *Track model performance, quality metrics, and get recommendations*
   - **Status**: Needs refresh ⚠️

6. **Templates Page**
   - Current URL: `https://github.com/user-attachments/assets/1bfd94aa-fa49-4db5-a985-6e6e924f1a05`
   - Location in README: Line 53
   - Description: *Built-in genre templates to jumpstart your story creation*
   - **Status**: Needs refresh ⚠️

## Additional Screenshots Needed

The following pages/features are not currently screenshotted but should be:

7. **World Builder Page**
   - Shows: Visual graph explorer with entities and relationships
   - Features: Filters, search, import, undo/redo
   - **Status**: Missing screenshot ❌

8. **Timeline Page**
   - Shows: Event timeline visualization for story
   - Features: Event tracking and sequencing
   - **Status**: Missing screenshot ❌

9. **Comparison Page**
   - Shows: Side-by-side model comparison interface
   - Features: Multi-model testing and performance comparison
   - **Status**: Missing screenshot ❌

## How to Update Screenshots

### Prerequisites
1. Ensure Python 3.14+ is installed
2. Install dependencies: `pip install -r requirements.txt`
3. Copy settings: `cp settings.example.json settings.json`
4. Ensure Ollama is running with at least one model pulled

### Steps to Capture

1. **Start the application**:
   ```bash
   python main.py
   ```
   Access at http://localhost:7860

2. **Create a sample project** (if needed for screenshots)

3. **Navigate to each page and capture screenshots**:
   - Use full browser window (recommended: 1920x1080)
   - Enable dark mode for consistency
   - Ensure sample data is visible (not empty states)
   - Crop to show relevant UI without excess whitespace

4. **Upload screenshots to GitHub**:
   - Create a draft GitHub issue or PR
   - Drag and drop screenshots to upload
   - Copy the generated URLs

5. **Update README.md**:
   Replace the old screenshot URLs with new ones:
   ```markdown
   ![Description](NEW_URL_HERE)
   ```

### Screenshot Quality Standards

- **Resolution**: At least 1920x1080 (Full HD)
- **Format**: PNG for UI screenshots (better quality than JPEG)
- **Theme**: Dark mode enabled for consistency
- **Content**: Show actual features, not empty states
- **Annotations**: None needed - let the UI speak for itself
- **File size**: Compress if > 1MB (use tools like TinyPNG)

### Screenshot Content Guidelines

Each screenshot should demonstrate:

1. **Write Page**: 
   - Interview chat with sample messages
   - Story structure visible
   - Live writing tab with chapter content

2. **World Builder**: 
   - Graph with multiple entities and connections
   - Sidebar showing entity details
   - Search/filter functionality

3. **Projects Page**: 
   - List of sample projects
   - Action buttons (create, load, delete)
   - Project metadata visible

4. **Templates Page**: 
   - Template cards showing different genres
   - Preview of template details
   - Create custom template button

5. **Timeline Page**: 
   - Event timeline with multiple events
   - Story progression visible
   - Timeline controls

6. **Comparison Page**: 
   - Side-by-side comparison interface
   - Model selection dropdowns
   - Comparison results

7. **Settings Page**: 
   - Model configuration section
   - Temperature sliders
   - Workflow preferences

8. **Models Page**: 
   - List of installed models
   - VRAM usage indicators
   - Pull/delete actions

9. **Analytics Page**: 
   - Performance metrics dashboard
   - Quality score charts
   - Model recommendations

## Notes

- All screenshots should be consistent in style and theme
- Dark mode is preferred for all screenshots
- Ensure no sensitive information is visible
- Screenshots should represent the current state (v1.0+)
- After updating, delete this checklist file or mark all items as ✅

## Verification

After updating all screenshots:
- [ ] All 6 existing screenshots replaced with current versions
- [ ] 3 new screenshots added (World, Timeline, Comparison)
- [ ] All screenshots are in dark mode
- [ ] All screenshots show actual content (not empty states)
- [ ] README.md updated with new URLs
- [ ] All screenshots load correctly on GitHub
