#!/usr/bin/env python3
"""Story Factory - AI-Powered Story Production Team.

A multi-agent system for generating stories with:
- Interviewer: Gathers story requirements
- Architect: Designs structure and characters
- Writer: Writes prose
- Editor: Polishes and refines
- Continuity Checker: Detects plot holes

Usage:
    python main.py          # Launch NiceGUI web UI
    python main.py --cli    # Run in CLI mode (basic)
"""

import argparse
import logging

from src.utils.environment import check_environment
from src.utils.logging_config import setup_logging

# Check Python version and dependencies before running app logic
check_environment()

logger = logging.getLogger(__name__)


def run_web_ui(host: str = "127.0.0.1", port: int = 7860, reload: bool = False) -> None:
    """Launch the NiceGUI web interface.

    Args:
        host: Host to bind to.
        port: Port to listen on.
        reload: Enable auto-reload for development.
    """
    from src.services import ServiceContainer
    from src.settings import Settings
    from src.ui import create_app

    logger.info("Starting Story Factory web UI...")

    # Load settings and create services
    settings = Settings.load()
    services = ServiceContainer(settings)

    # Create and run app
    app = create_app(services)
    app.run(host=host, port=port, reload=reload)


def run_cli(load_story: str | None = None, list_stories: bool = False) -> None:
    """Run a simple CLI version.

    Args:
        load_story: Path to a saved story to load.
        list_stories: If True, list saved stories and exit.
    """
    if list_stories:
        # List stories without creating orchestrator/agents (avoids Ollama connection)
        from src.services.project_service import ProjectService
        from src.settings import Settings

        settings = Settings.load()
        project_service = ProjectService(settings)
        stories = project_service.list_projects()
        if not stories:
            print("No saved stories found.")
            logger.info("No saved stories found")
        else:
            print("Saved Stories:")
            print("-" * 40)
            for i, s in enumerate(stories, 1):
                premise = (s.premise or "Untitled")[:50]
                print(f"{i}. {premise}...")
                print(f"   Status: {s.status} | ID: {s.id}")
                print()
            logger.info(f"Listed {len(stories)} saved stories")
        return

    from src.services.orchestrator import StoryOrchestrator

    orchestrator = StoryOrchestrator()

    # Load existing story
    if load_story:
        logger.info(f"Loading story from: {load_story}")
        try:
            state = orchestrator.load_story(load_story)
            logger.info(f"Loaded story: {state.id} (status: {state.status})")
            print(f"Loaded story: {state.id}")
            print(f"Status: {state.status}")
            print()
            print(orchestrator.get_outline_summary())
            print()
            print("-" * 40)
            print("STORY CONTENT:")
            print("-" * 40)
            print(orchestrator.get_full_story())
            return
        except FileNotFoundError as e:
            logger.error(f"Story file not found: {load_story}")
            print(f"Error: {e}")
            return

    orchestrator.create_new_story()
    logger.info("Created new story")

    # Interview phase
    logger.debug("Starting interview phase")
    print("INTERVIEWER:")
    print("-" * 40)
    questions = orchestrator.start_interview()
    print(questions)
    print()

    while True:
        response = input("\nYour response (or 'done' to finish interview): ").strip()
        if response.lower() == "done":
            break
        followup, is_complete = orchestrator.process_interview_response(response)
        print(f"\nINTERVIEWER: {followup}")
        if is_complete:
            break

    # Extract brief
    print("\nProcessing your responses...")
    logger.info("Processing interview responses")
    brief = orchestrator.finalize_interview()
    print("\nStory Brief:")
    print(f"  Premise: {brief.premise}")
    print(f"  Genre: {brief.genre}")
    print(f"  Tone: {brief.tone}")
    print()

    # Architecture phase
    print("Creating story architecture...")
    logger.info("Starting architecture phase")
    orchestrator.build_story_structure()

    print("\nOutline:")
    print(orchestrator.get_outline_summary())

    # Write story
    proceed = input("\nProceed with writing? (yes/no): ").strip().lower()
    if proceed != "yes":
        print("Story saved as outline.")
        orchestrator.save_story()
        logger.info("Story saved as outline")
        return

    print("Writing story...")
    logger.info("Starting story writing")

    if not orchestrator.story_state or not orchestrator.story_state.brief:
        logger.error("No story state or brief available")
        print("Error: No story state or brief available.")
        return

    is_short_story = orchestrator.story_state.brief.target_length == "short_story"
    if is_short_story:
        logger.info("Writing short story")
        for event in orchestrator.write_short_story():
            print(f"  [{event.agent_name}] {event.message}")
    else:
        logger.info("Writing full story")
        for event in orchestrator.write_all_chapters():
            print(f"  [{event.agent_name}] {event.message}")

    logger.info("Story writing complete")

    print("\n" + "=" * 60)
    print("FINAL STORY")
    print("=" * 60)
    print(orchestrator.get_full_story())

    stats = orchestrator.get_statistics()
    logger.info(
        f"Story statistics: {stats['total_words']} words, {stats['total_chapters']} chapters"
    )
    print("\n" + "-" * 40)
    print(f"Statistics: {stats['total_words']} words, {stats['total_chapters']} chapters")

    # Offer to save
    save = input("\nSave story? (yes/no): ").strip().lower()
    if save == "yes":
        filepath = orchestrator.save_story()
        logger.info(f"Story saved to: {filepath}")
        print(f"Story saved to: {filepath}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Story Factory - AI-Powered Story Production Team")
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in CLI mode instead of web UI",
    )
    parser.add_argument(
        "--list-stories",
        action="store_true",
        help="List saved stories (CLI mode)",
    )
    parser.add_argument(
        "--load",
        type=str,
        metavar="PATH",
        help="Load a saved story by path (CLI mode)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for web UI (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for web UI (default: 7860)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="default",
        help="Log file path (default: output/logs/story_factory.log, use 'none' to disable)",
    )

    args = parser.parse_args()

    # Configure logging
    log_file = None if args.log_file.lower() == "none" else args.log_file
    setup_logging(level=args.log_level, log_file=log_file)

    if args.cli or args.list_stories or args.load:
        run_cli(load_story=args.load, list_stories=args.list_stories)
    else:
        run_web_ui(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
