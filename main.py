#!/usr/bin/env python3
"""Story Factory - AI-Powered Story Production Team.

A multi-agent system for generating stories with:
- Interviewer: Gathers story requirements
- Architect: Designs structure and characters
- Writer: Writes prose
- Editor: Polishes and refines
- Continuity Checker: Detects plot holes

Usage:
    python main.py          # Launch Gradio web UI
    python main.py --cli    # Run in CLI mode (basic)
"""

import argparse
import sys


def run_web_ui():
    """Launch the Gradio web interface."""
    from ui.gradio_app import main
    main()


def run_cli():
    """Run a simple CLI version."""
    from workflows.orchestrator import StoryOrchestrator

    print("=" * 60)
    print("STORY FACTORY - CLI Mode")
    print("=" * 60)
    print()

    orchestrator = StoryOrchestrator()
    orchestrator.create_new_story()

    # Interview phase
    print("INTERVIEWER:")
    print("-" * 40)
    questions = orchestrator.start_interview()
    print(questions)
    print()

    while True:
        response = input("\nYour response (or 'done' to finish interview): ").strip()
        if response.lower() == 'done':
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
    if proceed != 'yes':
        print("Aborted.")
        return

    print("\n" + "=" * 60)
    print("Writing story...")

    state = orchestrator.story_state
    if state.brief.target_length == "short_story":
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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Story Factory - AI-Powered Story Production Team"
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in CLI mode instead of web UI",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for web UI (default: 7860)",
    )

    args = parser.parse_args()

    if args.cli:
        run_cli()
    else:
        run_web_ui()


if __name__ == "__main__":
    main()
