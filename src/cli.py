"""
SAIL Command Line Interface

Entry point for the SAIL application.
"""

import asyncio
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .config import (
    ConfigurationError,
    create_default_config,
    get_config,
    load_config,
)

console = Console()

# Typed at the prompt these end the session. Without them "quit" was sent to
# the model as a question, which is never what the user meant.
EXIT_COMMANDS = frozenset({"/quit", "/exit", "quit", "exit", ":q"})


def print_banner() -> None:
    """Print the SAIL banner."""
    banner = """
███████╗ █████╗ ██╗██╗
██╔════╝██╔══██╗██║██║
███████╗███████║██║██║
╚════██║██╔══██║██║██║
███████║██║  ██║██║███████╗
╚══════╝╚═╝  ╚═╝╚═╝╚══════╝
    """
    console.print(Panel(banner, title="Situational Awareness Interaction Layer", subtitle=f"v{__version__}"))


@click.group()
@click.version_option(version=__version__, prog_name="SAIL")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=False),
    help="Path to configuration file",
    envvar="SAIL_CONFIG_PATH",
)
@click.pass_context
def cli(ctx: click.Context, config: str | None) -> None:
    """SAIL - Situational Awareness Interaction Layer

    A privacy-first, locally-hosted AI companion for contextual guidance.
    """
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config


@cli.command()
@click.option(
    "--text-mode",
    is_flag=True,
    default=False,
    help="Use text I/O instead of voice (no audio hardware needed)",
)
@click.option(
    "--hub",
    is_flag=True,
    default=False,
    help="Start in hub mode (accepts connections from room nodes)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose logging",
)
@click.pass_context
def run(ctx: click.Context, text_mode: bool, hub: bool, verbose: bool) -> None:
    """Start the SAIL assistant."""
    print_banner()

    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    config_path = ctx.obj.get("config_path")

    try:
        if config_path:
            config = load_config(config_path)
        else:
            config = get_config()

        console.print(f"[green]Environment:[/green] {config.sail.environment.value}")
        console.print(f"[green]LLM Provider:[/green] {config.llm.provider}")
        console.print(f"[green]Model:[/green] {config.llm.model}")
        if hub:
            console.print(f"[green]Mode:[/green] Hub (accepting node connections)")
        console.print()

        if hub:
            asyncio.run(_run_hub_mode(config, text_mode))
        elif text_mode:
            asyncio.run(_run_text_mode(config))
        else:
            asyncio.run(_run_voice_mode(config))

    except ConfigurationError as e:
        console.print(f"[red]Configuration Error:[/red] {e}")
        sys.exit(1)


async def _run_text_mode(config) -> None:
    """Run SAIL in interactive text mode."""
    from .pipeline import Pipeline

    console.print("[yellow]Starting SAIL in text mode...[/yellow]")

    pipeline = Pipeline(config)

    try:
        await pipeline.start()
    except Exception as e:
        console.print(f"[red]Failed to start pipeline:[/red] {e}")
        console.print("[dim]Is Ollama running? Try: ollama serve[/dim]")
        return

    console.print(
        "[green]SAIL ready.[/green] Type your question "
        "(or 'quit' / Ctrl+C to exit)."
    )
    if config.users:
        console.print("[dim]Commands: /user <name> to switch user, /users to list users[/dim]")
    console.print()

    try:
        while True:
            # Build prompt with current user
            current_user = pipeline.get_current_user()
            prompt_name = current_user.name if current_user else "You"

            try:
                user_input = input(f"{prompt_name}: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input.lower() in EXIT_COMMANDS:
                break

            # Handle /user command
            if user_input.startswith("/user "):
                name = user_input[6:].strip()
                user = await pipeline.set_active_user(name)
                if user:
                    console.print(f"[green]Switched to user:[/green] {user.name} ({user.role.value})\n")
                else:
                    console.print(f"[red]User not found:[/red] {name}\n")
                continue

            if user_input == "/users":
                if pipeline.user_manager:
                    all_users = pipeline.user_manager.get_all_users()
                    if all_users:
                        for u in all_users:
                            marker = " *" if current_user and u.user_id == current_user.user_id else ""
                            console.print(f"  {u.name} ({u.role.value}){marker}")
                    else:
                        console.print("[dim]No users registered.[/dim]")
                else:
                    console.print("[dim]User management not enabled.[/dim]")
                console.print()
                continue

            try:
                result = await pipeline.query(user_input)
                console.print(f"\n[cyan]SAIL:[/cyan] {result.response}")
                if result.citations:
                    console.print(f"[dim]Sources: {', '.join(result.citations)}[/dim]")
                if result.intervention_mode != "ambient":
                    console.print(f"[dim]Mode: {result.intervention_mode}[/dim]")
                if result.user_name:
                    console.print(f"[dim]User: {result.user_name}[/dim]")
                if result.guardian_alerts:
                    for alert in result.guardian_alerts:
                        console.print(f"[yellow]Guardian Alert:[/yellow] {alert.message}")
                console.print()
            except Exception as e:
                console.print(f"\n[red]Error:[/red] {e}\n")

    except KeyboardInterrupt:
        console.print()

    await pipeline.stop()
    console.print("[yellow]SAIL shut down.[/yellow]")


async def _run_voice_mode(config) -> None:
    """Run SAIL with full voice I/O pipeline."""
    from .pipeline import Pipeline

    console.print("[yellow]Starting SAIL in voice mode...[/yellow]")

    pipeline = Pipeline(config)

    try:
        await pipeline.start()
    except Exception as e:
        console.print(f"[red]Failed to start pipeline:[/red] {e}")
        console.print("[dim]Is Ollama running? Try: ollama serve[/dim]")
        return

    # Import voice components
    try:
        from .voice.manager import VoiceInputManager, create_voice_input_manager
        from .voice.tts.manager import VoiceOutputManager, create_voice_output_manager

        voice_in = await create_voice_input_manager(config.voice)
        voice_out = await create_voice_output_manager(config.voice)
    except Exception as e:
        console.print(f"[red]Voice initialization failed:[/red] {e}")
        console.print("[dim]Try --text-mode if audio hardware is unavailable.[/dim]")
        await pipeline.stop()
        return

    console.print("[green]SAIL listening.[/green] Say 'hey sail' to activate. Ctrl+C to exit.\n")

    try:
        await voice_out.start()
        async for utterance in voice_in.run():
            if utterance.was_stopped or not utterance.has_text:
                if utterance.was_stopped:
                    voice_out.interrupt()
                continue

            console.print(f"[dim]Heard:[/dim] {utterance.text}")

            # Identify speaker from audio if user management is active
            audio_data = getattr(utterance, "audio", None)
            if pipeline.user_manager and audio_data is not None:
                user = await pipeline.identify_speaker(
                    audio=audio_data,
                    sample_rate=getattr(utterance, "sample_rate", 16000),
                )
                if user:
                    console.print(f"[dim]Speaker:[/dim] {user.name}")

            try:
                result = await pipeline.query(utterance.text)
                console.print(f"[cyan]SAIL:[/cyan] {result.response}")
                if result.guardian_alerts:
                    for alert in result.guardian_alerts:
                        console.print(f"[yellow]Guardian Alert:[/yellow] {alert.message}")
                console.print()
                await voice_out.speak(result.response, wait=True)
            except Exception as e:
                console.print(f"[red]Error:[/red] {e}\n")

    except KeyboardInterrupt:
        console.print()

    await voice_out.stop()
    await pipeline.stop()
    console.print("[yellow]SAIL shut down.[/yellow]")


async def _run_hub_mode(config, text_mode: bool = False) -> None:
    """Run SAIL in hub mode: pipeline + hub coordinator accepting node connections."""
    from .hub.coordinator import HubCoordinator
    from .pipeline import Pipeline

    console.print("[yellow]Starting SAIL in hub mode...[/yellow]")

    pipeline = Pipeline(config)

    try:
        await pipeline.start()
    except Exception as e:
        console.print(f"[red]Failed to start pipeline:[/red] {e}")
        console.print("[dim]Is Ollama running? Try: ollama serve[/dim]")
        return

    # Start hub coordinator and wire query handler to pipeline
    hub = HubCoordinator(context_manager=pipeline._context_mgr)

    async def pipeline_query_handler(request, node) -> dict:
        """Route node queries through the full SAIL pipeline."""
        query_text = request.payload.get("text", "")
        user_id = request.user_id or (node.info.name if node else None)
        if user_id and pipeline.user_manager:
            await pipeline.set_active_user(user_id)
        result = await pipeline.query(query_text)
        return {
            "response": result.response,
            "citations": result.citations,
            "intervention_mode": result.intervention_mode,
        }

    hub.register_request_handler("query", pipeline_query_handler)

    try:
        await hub.start()
    except Exception as e:
        console.print(f"[red]Failed to start hub:[/red] {e}")
        await pipeline.stop()
        return

    status = hub.get_status()
    console.print(f"[green]Hub running.[/green] Accepting node connections.")
    console.print(f"[dim]Hub state: {status['state']}, nodes: {status['total_nodes']}[/dim]")

    # Also run local text/voice loop alongside hub
    if text_mode:
        console.print("[green]Local text mode active.[/green] Type your question (Ctrl+C to exit).\n")
        try:
            while True:
                try:
                    user_input = input("You: ").strip()
                except EOFError:
                    break
                if not user_input:
                    continue

                if user_input == "/nodes":
                    nodes = hub.get_online_nodes()
                    if nodes:
                        for n in nodes:
                            console.print(f"  {n.info.name} ({n.info.node_type.value}) - {n.state.value}")
                    else:
                        console.print("[dim]No nodes connected.[/dim]")
                    console.print()
                    continue

                if user_input == "/status":
                    s = hub.get_status()
                    console.print(f"  State: {s['state']}")
                    console.print(f"  Nodes: {s['total_nodes']} ({s['online_nodes']} online)")
                    console.print(f"  Requests: {s['stats']['total_requests']}")
                    console.print()
                    continue

                try:
                    result = await pipeline.query(user_input)
                    console.print(f"\n[cyan]SAIL:[/cyan] {result.response}")
                    if result.citations:
                        console.print(f"[dim]Sources: {', '.join(result.citations)}[/dim]")
                    console.print()
                except Exception as e:
                    console.print(f"\n[red]Error:[/red] {e}\n")
        except KeyboardInterrupt:
            console.print()
    else:
        # Hub-only mode: just wait for Ctrl+C
        console.print("[dim]Hub running. Press Ctrl+C to stop.[/dim]\n")
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            console.print()

    await hub.stop()
    await pipeline.stop()
    console.print("[yellow]SAIL hub shut down.[/yellow]")


@cli.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="config/config.yaml",
    help="Output path for configuration file",
)
def init(output: str) -> None:
    """Initialize a new SAIL configuration file."""
    output_path = Path(output)

    if output_path.exists():
        if not click.confirm(f"Configuration file {output} already exists. Overwrite?"):
            console.print("[yellow]Aborted.[/yellow]")
            return

    try:
        create_default_config(output_path)
        console.print(f"[green]Created configuration file:[/green] {output_path}")
        console.print("[dim]Edit this file to customize SAIL settings.[/dim]")
    except ConfigurationError as e:
        console.print(f"[red]Error creating configuration:[/red] {e}")
        sys.exit(1)


@cli.command()
@click.pass_context
def config_show(ctx: click.Context) -> None:
    """Display current configuration."""
    config_path = ctx.obj.get("config_path")

    try:
        if config_path:
            config = load_config(config_path)
        else:
            config = get_config()

        # Create configuration table
        table = Table(title="SAIL Configuration")
        table.add_column("Section", style="cyan")
        table.add_column("Setting", style="green")
        table.add_column("Value", style="yellow")

        # SAIL settings
        table.add_row("sail", "version", config.sail.version)
        table.add_row("sail", "environment", config.sail.environment.value)

        # LLM settings
        table.add_row("llm", "provider", config.llm.provider)
        table.add_row("llm", "model", config.llm.model)
        table.add_row("llm", "base_url", config.llm.base_url)

        # Voice settings
        table.add_row("voice", "wake_word", config.voice.wake_word.phrase)
        table.add_row("voice", "stt_engine", config.voice.stt.engine)
        table.add_row("voice", "tts_engine", config.voice.tts.engine)

        # Context settings
        table.add_row("context", "immediate_duration", f"{config.context.immediate_duration_seconds}s")
        table.add_row("context", "short_term_duration", f"{config.context.short_term_duration_seconds}s")

        # Intervention settings
        table.add_row("intervention", "default_mode", config.intervention.default_mode.value)

        # Users
        table.add_row("users", "count", str(len(config.users)))

        console.print(table)

    except ConfigurationError as e:
        console.print(f"[red]Configuration Error:[/red] {e}")
        sys.exit(1)


@cli.command()
@click.pass_context
def users(ctx: click.Context) -> None:
    """List configured users."""
    config_path = ctx.obj.get("config_path")

    try:
        if config_path:
            config = load_config(config_path)
        else:
            config = get_config()

        if not config.users:
            console.print("[yellow]No users configured.[/yellow]")
            return

        table = Table(title="Configured Users")
        table.add_column("Name", style="cyan")
        table.add_column("Role", style="green")
        table.add_column("Access", style="yellow")
        table.add_column("Guardian", style="magenta")

        for user in config.users:
            table.add_row(
                user.name,
                user.role.value,
                user.access.value,
                user.guardian or "-",
            )

        console.print(table)

    except ConfigurationError as e:
        console.print(f"[red]Configuration Error:[/red] {e}")
        sys.exit(1)


@cli.command()
def check() -> None:
    """Check system requirements and dependencies."""
    console.print("[bold]SAIL System Check[/bold]\n")

    checks = [
        ("Python version", _check_python),
        ("Configuration file", _check_config),
        ("Ollama connection", _check_ollama),
        ("Audio devices", _check_audio),
        ("Knowledge embeddings", _check_embeddings),
    ]

    all_passed = True
    for name, check_fn in checks:
        try:
            result, message = check_fn()
            status = "[green]PASS[/green]" if result else "[red]FAIL[/red]"
            console.print(f"  {status} {name}: {message}")
            if not result:
                all_passed = False
        except Exception as e:
            console.print(f"  [red]ERROR[/red] {name}: {e}")
            all_passed = False

    console.print()
    if all_passed:
        console.print("[green]All checks passed![/green]")
    else:
        console.print("[yellow]Some checks failed. SAIL may not function correctly.[/yellow]")
        # Exit non-zero so scripts and CI can detect a failed environment check.
        sys.exit(1)


def _check_python() -> tuple[bool, str]:
    """Check Python version."""
    version = sys.version_info
    if version >= (3, 11):
        return True, f"{version.major}.{version.minor}.{version.micro}"
    return False, f"{version.major}.{version.minor} (requires 3.11+)"


def _check_config() -> tuple[bool, str]:
    """Check configuration file."""
    from .config.loader import find_config_file
    path = find_config_file()
    if path:
        return True, str(path)
    return False, "No configuration file found"


def _check_embeddings() -> tuple[bool, str]:
    """Report which embedding backend semantic search will use.

    Never fails the check: the fallback is functional, just weaker. It is
    reported because the difference is otherwise invisible -- the fallback is
    chosen silently at runtime.
    """
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        # Backslash-escaped: console.print parses [...] as Rich markup and
        # would otherwise swallow "[embeddings]" from the install hint.
        return True, (
            r"TF-IDF fallback (install 'sail\[embeddings]' for semantic search)"
        )

    # Installed is not the same as working: with no cached model and no
    # network the encoder fails to load and retrieval silently falls back.
    # Load it for real so this reports what SAIL will actually use.
    from .knowledge.retrieval import LocalEmbeddingProvider

    provider = LocalEmbeddingProvider()
    asyncio.run(provider.initialize())
    if provider.uses_semantic_model:
        return True, f"sentence-transformers ({provider.model_name})"
    return True, "TF-IDF fallback (model installed but could not be loaded)"


def _check_ollama() -> tuple[bool, str]:
    """Check Ollama connection."""
    try:
        import httpx
        config = get_config()
        response = httpx.get(f"{config.llm.base_url}/api/tags", timeout=5.0)
        if response.status_code == 200:
            return True, "Connected"
        return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, f"Not reachable ({type(e).__name__})"


def _check_audio() -> tuple[bool, str]:
    """Check audio devices."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devices = sum(1 for d in devices if d["max_input_channels"] > 0)
        output_devices = sum(1 for d in devices if d["max_output_channels"] > 0)
        return True, f"{input_devices} input, {output_devices} output"
    except Exception as e:
        return False, str(e)


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
