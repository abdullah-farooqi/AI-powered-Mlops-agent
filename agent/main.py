import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from agent.orchestrator import MLOpsAgentOrchestrator

console = Console()

def display_response(resp):
    console.print(Panel(f"[bold cyan]User Request:[/bold cyan] {resp.query}", expand=False))

    # Tool Execution Audit Table
    table = Table(title="Structured Tools Executed by Agent", show_header=True, header_style="bold magenta")
    table.add_column("Tool Name", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Duration", justify="right")
    table.add_column("Request ID", style="dim")

    for t in resp.tool_results:
        if t.status == "success":
            status_style = "[green]SUCCESS[/green]"
        elif t.status == "needs_confirmation":
            status_style = "[yellow]NEEDS CONFIRMATION[/yellow]"
        elif t.status == "denied":
            status_style = "[red]DENIED[/red]"
        else:
            status_style = "[red]ERROR[/red]"
        table.add_row(t.tool_name, status_style, f"{t.duration_seconds}s", t.request_id)
    console.print(table)

    # Known Facts (Empirical Evidence)
    if resp.known_facts:
        facts_md = "\n".join([f"- {f}" for f in resp.known_facts])
        console.print(Panel(Markdown(facts_md), title="[bold green]Known Facts (Evidence-Based)[/bold green]", border_style="green"))

    # Uncertainties & Tool Warnings
    if resp.uncertainties:
        unc_md = "\n".join([f"- {u}" for u in resp.uncertainties])
        console.print(Panel(Markdown(unc_md), title="[bold yellow]Uncertainties & Tool Warnings[/bold yellow]", border_style="yellow"))

    # Recommended Actions
    if resp.recommended_actions:
        actions_md = "\n".join([f"- {a}" for a in resp.recommended_actions])
        console.print(Panel(Markdown(actions_md), title="[bold blue]Recommended Next Actions[/bold blue]", border_style="blue"))

    # Synthesis Narrative
    console.print(Panel(resp.final_narrative, title="[bold white]Agent Synthesis & Diagnosis[/bold white]", border_style="white"))

def main():
    parser = argparse.ArgumentParser(description="Production-Ready MLOps AI Agent CLI")
    parser.add_argument("--query", type=str, help="Single natural language request to evaluate", default=None)
    parser.add_argument("--confirm", action="store_true", help="Automatically confirm state-changing actions", default=False)
    args = parser.parse_args()

    agent = MLOpsAgentOrchestrator()

    if args.query:
        resp = agent.process_query(args.query, user_confirmation=args.confirm)
        display_response(resp)
        return

    console.print("[bold green]=== MLOps AI Agent Interactive Console ===[/bold green]")
    console.print("Type your request below, or type 'exit' or 'quit' to close.\n")

    while True:
        try:
            user_input = input("\nMLOps-Agent > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                console.print("[bold yellow]Exiting MLOps Agent. Goodbye![/bold yellow]")
                break

            resp = agent.process_query(user_input, user_confirmation=False)
            display_response(resp)

            # Interactive Human-in-the-Loop Confirmation Gate
            if resp.confirmation_required:
                act = resp.confirmation_required
                confirm_input = input(f"\n[CONFIRMATION GATE] Authorize action '{act.get('action') or act.get('tool')}'? (yes/no): ").strip().lower()
                if confirm_input in ["y", "yes"]:
                    console.print("[cyan]User confirmed. Re-executing with authorization...[/cyan]")
                    resp_confirmed = agent.process_query(user_input, user_confirmation=True)
                    display_response(resp_confirmed)
                else:
                    console.print("[yellow]Action cancelled by user.[/yellow]")

        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold yellow]Session ended.[/bold yellow]")
            break

if __name__ == "__main__":
    main()
