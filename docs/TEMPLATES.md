# Story Templates and Presets

Story Factory provides a comprehensive template system to accelerate story creation. Templates include pre-configured story structures, character archetypes, world-building elements, and plot frameworks.

## Overview

Templates help you:
- **Start faster** - Begin with proven story structures and archetypes
- **Learn storytelling** - Understand genre conventions and narrative patterns
- **Maintain consistency** - Follow established structures and frameworks
- **Share ideas** - Create and share custom templates with others

## Features

### Built-in Templates

Story Factory includes five professionally-crafted genre templates:

1. **Mystery / Detective**
   - Classic whodunit structure
   - Detective, suspects, and victim archetypes
   - Fair-play clue system
   - Red herrings and revelations

2. **Contemporary Romance**
   - Emotional character-driven narrative
   - Meet-cute to happily-ever-after
   - Relationship development arcs
   - Supporting cast and obstacles

3. **Science Fiction / Space Opera**
   - Epic space adventure
   - Alien species and technology
   - Interstellar politics and exploration
   - Hero's journey structure

4. **Epic Fantasy**
   - High fantasy with magic systems
   - Chosen one and mentor archetypes
   - Quest narrative structure
   - World-changing stakes

5. **Action Thriller**
   - Fast-paced suspense
   - Conspiracy and espionage
   - Ticking clock urgency
   - Save the Cat structure

### Structure Presets

Three proven narrative frameworks:

1. **Three-Act Structure**
   - Classic setup, confrontation, resolution
   - 7 key plot points
   - Ideal for most genres

2. **Hero's Journey**
   - Joseph Campbell's monomyth
   - 12-stage adventure structure
   - Perfect for epic narratives

3. **Save The Cat**
   - Blake Snyder's beat sheet
   - 15 precise story beats
   - Proven commercial structure

## Using Templates

### Creating a New Project with a Template

1. Click **"+ New Project"** on the Projects page
2. Enter a project name (optional)
3. Select a template from the dropdown (or choose "Blank Project")
4. Click **"Create"**

The template will pre-populate:
- Genre, tone, and themes
- Character archetypes (names are placeholders you can customize)
- Plot points and story beats
- World-building elements and rules
- Story structure

### Browsing Templates

Visit the **Templates** page from the navigation menu to:
- View all built-in templates
- See template details (characters, plot points, structure)
- Explore structure presets
- Create custom templates

### Template Details

Click **"View Details"** on any template to see:
- Full description and genre information
- Story settings (tone, themes, time period)
- World description and rules
- Character archetypes with traits and goals
- Complete plot point sequence
- Associated structure preset

## Custom Templates

### Creating a Custom Template

Save your project as a template for reuse:

1. Open the **Templates** page
2. Click **"Create from Project"**
3. Enter a template name and description
4. Click **"Create"**

Your custom template includes:
- All story brief settings
- Character configurations
- Plot points structure
- World-building elements

### Managing Custom Templates

Custom templates can be:
- **Edited** - Modify and save changes
- **Exported** - Share as JSON file
- **Deleted** - Remove from your library
- **Imported** - Add templates from others

### Importing Templates

To import a shared template:

1. Obtain a template JSON file
2. Click **"Import Template"**
3. Select the file
4. The template appears in your library

### Exporting Templates

To share your custom template:

1. Click **"Export"** on a custom template
2. Save the JSON file
3. Share the file with others

## Template Components

### Story Brief

Templates configure:
- **Genre** - Primary genre classification
- **Subgenres** - Additional genre tags
- **Tone** - Overall mood and atmosphere
- **Themes** - Central ideas to explore
- **Setting** - Time period and location
- **Target Length** - Short story, novella, or novel

### Characters

Character archetypes include:
- **Name** - Placeholder or role designation
- **Role** - Protagonist, antagonist, mentor, etc.
- **Description** - Character overview
- **Personality Traits** - Key characteristics
- **Goals** - What the character wants
- **Arc Notes** - How the character should develop

### Plot Points

Plot frameworks provide:
- **Description** - What happens at this point
- **Act** - Which story act it belongs to
- **Percentage** - Approximate story position (0-100%)

### World Building

Templates may include:
- **World Description** - Setting overview
- **World Rules** - Constraints and logic of the world

## Best Practices

### Starting with Templates

1. **Use templates as starting points** - Customize freely
2. **Modify character names** - Templates use placeholder names
3. **Adapt plot points** - Adjust to fit your specific story
4. **Add your premise** - Templates don't include specific premises

### Customizing Templates

1. **Keep what works** - Use template elements that fit your vision
2. **Remove what doesn't** - Templates are suggestions, not requirements
3. **Add your unique elements** - Blend template structure with original ideas

### Creating Reusable Templates

1. **Start from a complete project** - Best templates come from finished work
2. **Generalize specifics** - Replace unique elements with archetypes
3. **Document your template** - Write clear descriptions
4. **Test your template** - Use it to start a new project

## Structure Presets in Detail

### Three-Act Structure

**Act 1: Setup (25%)**
- Opening image
- Inciting incident
- First plot point

**Act 2: Confrontation (50%)**
- Midpoint revelation
- Rising complications
- Second plot point (all is lost)

**Act 3: Resolution (25%)**
- Climax
- Resolution
- Closing image

### Hero's Journey

**Departure**
- Ordinary world
- Call to adventure
- Refusal of call
- Meeting the mentor
- Crossing the threshold

**Initiation**
- Tests, allies, enemies
- Approach to inmost cave
- The ordeal
- Reward

**Return**
- The road back
- Resurrection
- Return with elixir

### Save The Cat

**Act 1**
- Opening image (1%)
- Theme stated (5%)
- Setup (10%)
- Catalyst (12%)
- Debate (20%)
- Break into two (25%)

**Act 2A**
- B story (30%)
- Fun and games (40%)
- Midpoint (50%)

**Act 2B**
- Bad guys close in (60%)
- All is lost (75%)
- Dark night (80%)

**Act 3**
- Break into three (85%)
- Finale (95%)
- Final image (100%)

## Technical Details

### Template File Format

Templates are stored as JSON files with the following structure:

```json
{
  "id": "unique-template-id",
  "name": "Template Name",
  "description": "Template description",
  "is_builtin": false,
  "genre": "Genre",
  "tone": "Tone description",
  "themes": ["Theme 1", "Theme 2"],
  "characters": [
    {
      "name": "Character Name",
      "role": "protagonist",
      "description": "Character description",
      "personality_traits": ["Trait 1", "Trait 2"],
      "goals": ["Goal 1", "Goal 2"],
      "arc_notes": "Development notes"
    }
  ],
  "plot_points": [
    {
      "description": "Plot point description",
      "act": 1,
      "percentage": 25
    }
  ],
  "structure_preset_id": "three-act",
  "world_description": "World description",
  "world_rules": ["Rule 1", "Rule 2"]
}
```

### Template Storage

- **Built-in templates**: Embedded in application code
- **Custom templates**: Stored in `output/templates/` directory
- **Structure presets**: Embedded in application code

### Template Application

When a template is applied to a new project:

1. Story brief is populated with template settings
2. Character archetypes are added to the project
3. Plot points are created from template
4. World description and rules are applied
5. If template references a structure preset, additional plot points are added

## FAQ

**Q: Can I modify built-in templates?**
A: No, but you can create a project from a built-in template, modify it, and save it as a custom template.

**Q: Do templates include the actual story premise?**
A: No, templates provide structure and archetypes, but you provide the specific premise and details during the interview phase.

**Q: Can I combine multiple templates?**
A: Not directly, but you can start with one template and manually add elements from others by editing your project.

**Q: What happens if I delete a template I created a project from?**
A: Nothing - projects are independent copies. Deleting a template doesn't affect existing projects.

**Q: Can templates be updated after creation?**
A: Custom templates can be deleted and recreated. Built-in templates may be updated in new versions of Story Factory.

**Q: Do I need to use a template?**
A: No, you can always create a blank project and build everything from scratch.

## See Also

- [Architecture Documentation](ARCHITECTURE.md) - System design details
- [Models Guide](MODELS.md) - AI model recommendations
- [UX/UI Improvements](UX_UI_IMPROVEMENTS.md) - Interface features
