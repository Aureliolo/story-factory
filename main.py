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

from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def run_web_ui(host: str = "127.0.0.1", port: int = 7860, reload: bool = False) -> None:
    """Launch the NiceGUI web interface.

    Args:
        host: Host to bind to.
        port: Port to listen on.
        reload: Enable auto-reload for development.
    """
    from services import ServiceContainer
    from settings import Settings
    from ui import create_app

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
        load_story: Path to story file to load.
        list_stories: If True, list all saved stories.
    """
    from workflows.orchestrator import StoryOrchestrator

    print("=" * 60)
    print("STORY FACTORY - CLI Mode")
    print("=" * 60)
    print()

    # List saved stories
    if list_stories:
        stories = StoryOrchestrator.list_saved_stories()
        if not stories:
            print("No saved stories found.")
        else:
            print("Saved Stories:")
            print("-" * 40)
            for i, s in enumerate(stories, 1):
                premise = (s.get("premise") or "Untitled")[:50]
                status = s.get("status") or "?"
                path = s.get("path") or "?"
                print(f"{i}. {premise}...")
                print(f"   Status: {status} | Path: {path}")
                print()
        return

    orchestrator = StoryOrchestrator()

    # Load existing story
    if load_story:
        try:
            state = orchestrator.load_story(load_story)
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
            print(f"Error: {e}")
            return

    orchestrator.create_new_story()

    # Interview phase
    print("INTERVIEWER:")
    print("-" * 40)
    questions = orchestrator.start_interview()
    print(questions)
    print()

    while True:
        response = input("\nYour response (or 'done' to finish interview): ").strip()
        if response.lower() == "done":
            orchestrator.finalize_interview()
            break

        follow_up, is_complete = orchestrator.process_interview_response(response)
        print("\n" + follow_up)

        if is_complete:
            break

    print("\n" + "=" * 60)
    print("Building story structure...")
    orchestrator.build_story_structure()

    print("\nOUTLINE:")
    print("-" * 40)
    print(orchestrator.get_outline_summary())

    proceed = input("\n\nProceed with writing? (yes/no): ").strip().lower()
    if proceed != "yes":
        print("Aborted.")
        return

    print("\n" + "=" * 60)
    print("Writing story...")

    if not orchestrator.story_state or not orchestrator.story_state.brief:
        print("Error: No story state or brief available.")
        return

    is_short_story = orchestrator.story_state.brief.target_length == "short_story"
    if is_short_story:
        for event in orchestrator.write_short_story():
            print(f"  [{event.agent_name}] {event.message}")
    else:
        for event in orchestrator.write_all_chapters():
            print(f"  [{event.agent_name}] {event.message}")

    print("\n" + "=" * 60)
    print("FINAL STORY")
    print("=" * 60)
    print(orchestrator.get_full_story())

    stats = orchestrator.get_statistics()
    print("\n" + "-" * 40)
    print(f"Statistics: {stats['total_words']} words, {stats['total_chapters']} chapters")

    # Offer to save
    save = input("\nSave story? (yes/no): ").strip().lower()
    if save == "yes":
        filepath = orchestrator.save_story()
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
        help="Log file path (default: logs/story_factory.log, use 'none' to disable)",
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
