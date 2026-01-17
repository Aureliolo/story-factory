# Chapter Regeneration with Feedback - UI Features

## Overview
This document describes the new UI features for intelligent chapter regeneration with natural language feedback.

## New UI Components

### 1. Regenerate with Feedback Panel
**Location**: Live Writing tab → Controls panel → "Regenerate with Feedback" expansion

**Features**:
- Expandable panel with autorenew icon
- Large textarea for detailed feedback input
- Placeholder text with helpful examples:
  - "e.g., 'Add more dialogue between the characters' or 'Make the action sequence more suspenseful'"
- "Regenerate Chapter" button (blue, with autorenew icon)

**Behavior**:
- Only enabled when a chapter has content
- Shows validation warnings if:
  - No chapter is selected
  - Chapter has no content yet
  - Feedback textarea is empty
- Streams regeneration progress just like normal chapter writing
- Automatically saves the current version before regenerating
- Clears feedback input on successful regeneration
- Updates version history automatically

### 2. Version History Panel
**Location**: Live Writing tab → Controls panel → "Version History" expansion

**Features**:
- Expandable panel with history icon
- Shows all saved versions of the current chapter
- Versions sorted newest first

**Per-Version Display**:
- Version badge (e.g., "v1", "v2", "v3")
- "Current" badge (green) for the active version
- Timestamp (e.g., "Jan 17, 2:30 PM")
- Feedback text (if provided when creating this version)
- Word count
- Action buttons:
  - **Restore** (restore icon) - Rollback to this version
  - **View** (visibility icon) - Open full content in dialog

**Empty States**:
- "Select a chapter to see its version history" (when no chapter selected)
- "No previous versions yet" (when chapter has no versions)

### 3. Version Viewer Dialog
**Triggered by**: Clicking "View" button on any version

**Features**:
- Full-screen maximized dialog
- Header showing:
  - Chapter number and title
  - Version badge
  - Current indicator (if applicable)
  - Timestamp (full date and time)
  - Word count
  - Feedback that prompted this version (if any)
- Full chapter content displayed in markdown format with prose styling
- Footer actions:
  - "Restore This Version" button (primary, blue) - Only shown for non-current versions
  - "Close" button

## User Workflows

### Workflow 1: Regenerate with Feedback
1. User writes a chapter
2. User reads the chapter and wants improvements
3. User expands "Regenerate with Feedback" panel
4. User enters specific feedback (e.g., "Add more tension in the dialogue")
5. User clicks "Regenerate Chapter"
6. System:
   - Saves current version with the feedback
   - Shows generation status
   - Calls AI writer with feedback
   - Saves new content as a version
   - Updates display
7. User sees regenerated chapter with improvements

### Workflow 2: Compare Versions
1. User expands "Version History" panel
2. User sees list of all versions with timestamps and feedback
3. User clicks "View" on an older version
4. System shows full content in dialog
5. User reads and compares mentally with current version
6. User can restore if preferred

### Workflow 3: Rollback to Previous Version
1. User expands "Version History" panel
2. User finds desired version
3. User clicks "Restore" button (or "Restore This Version" in viewer)
4. System:
   - Updates chapter content to that version
   - Marks that version as current
   - Updates word count
   - Refreshes all displays
5. User sees restored content

## Technical Implementation

### Frontend (ui/pages/write.py)
- New UI references added to `__init__`:
  - `_regenerate_feedback_input`: Textarea for feedback
  - `_version_history_container`: Container for version list
- New methods:
  - `_regenerate_with_feedback()`: Async handler for regeneration
  - `_build_version_history()`: Builds version list UI
  - `_refresh_version_history()`: Refreshes version display
  - `_rollback_to_version()`: Handles version restoration
  - `_view_version()`: Shows version in dialog
- Auto-refresh triggers:
  - Version history refreshes on chapter selection change
  - Version history refreshes after successful regeneration
  - Version history refreshes after rollback

### Backend (services/story_service.py)
- New method: `regenerate_chapter_with_feedback()`
- Workflow:
  1. Validates chapter exists and has content
  2. Saves current version with feedback
  3. Calls orchestrator with feedback parameter
  4. Saves regenerated content as new version
  5. Handles cancellation with rollback

### Data Model (memory/story_state.py)
- New model: `ChapterVersion`
  - `id`: Unique identifier
  - `created_at`: Timestamp
  - `content`: Chapter prose
  - `word_count`: Word count
  - `feedback`: User feedback that prompted this version
  - `version_number`: Sequential number
  - `is_current`: Boolean flag
- Extended `Chapter` model:
  - `versions`: List of ChapterVersion
  - `current_version_id`: Active version ID
  - Methods:
    - `save_current_as_version(feedback)`
    - `rollback_to_version(version_id)`
    - `get_version_by_id(version_id)`
    - `get_current_version()`
    - `compare_versions(version_id_a, version_id_b)`

## Acceptance Criteria ✅

- [x] **Users can provide specific feedback**: Large textarea with helpful placeholder examples
- [x] **AI incorporates feedback meaningfully**: Feedback passed to writer agent via orchestrator
- [x] **Previous versions accessible**: Version history panel shows all versions with view/restore actions
- [x] **Feedback history visible**: Each version displays the feedback that prompted its creation

## Testing

### Unit Tests (17 total, all passing)
- **ChapterVersion model** (11 tests)
  - Version creation and defaults
  - Version management (save, rollback, get)
  - Version comparison
- **StoryService regeneration** (6 tests)
  - Version saving before regeneration
  - Feedback passing to orchestrator
  - Validation (content exists, valid chapter, feedback required)
  - Cancellation with rollback

### Manual Testing Checklist
- [ ] Create a project and write a chapter
- [ ] Enter feedback and regenerate
- [ ] Verify version saved with feedback
- [ ] Check version history shows both versions
- [ ] View old version in dialog
- [ ] Rollback to old version
- [ ] Verify content restored correctly
- [ ] Test with multiple regenerations (3+ versions)
- [ ] Test cancellation during regeneration
- [ ] Test with different feedback types

## Future Enhancements (Not in Scope)
- Side-by-side A/B comparison view
- Diff highlighting between versions
- Version annotations/notes
- Export specific version
- Merge content from multiple versions
- Auto-save versions at intervals
