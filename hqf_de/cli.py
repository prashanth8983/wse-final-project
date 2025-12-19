import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import logging

from .config import config
from .pipeline.expander import Expander
from .pipeline.indexer_bridge import Bridge
from .evaluation.evaluator import Evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(name="hqf-de", help="HQF-DE Document Expansion")
console = Console()


@app.command()
def expand(
    input_file: Path = typer.Option(None, "--input", "-i"),
    output_file: Path = typer.Option(None, "--output", "-o"),
    limit: int = typer.Option(None, "--limit", "-n"),
    use_llm: bool = typer.Option(True, "--llm/--no-llm"),
    use_nli: bool = typer.Option(True, "--nli/--no-nli"),
    use_d2q: bool = typer.Option(True, "--d2q/--no-d2q"),
    d2q_only: bool = typer.Option(False, "--d2q-only")
):
    input_path = input_file or config.data_dir / config.input_tsv
    output_path = output_file or config.output_dir / config.output_tsv

    if not input_path.exists():
        console.print(f"[red]Error: {input_path}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold blue]HQF-DE[/bold blue]")
    console.print(f"Input:  {input_path}")
    console.print(f"Output: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bridge = Bridge()

    if d2q_only:
        console.print("[yellow]d2q-only mode[/yellow]")
        exp = Expander(use_llm=False, use_nli=False, use_d2q=True)
    else:
        exp = Expander(use_llm=use_llm, use_nli=use_nli, use_d2q=use_d2q)

    docs = list(bridge.read(limit=limit))
    console.print(f"Loaded [green]{len(docs)}[/green] documents")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Loading...", total=None)

        with exp:
            progress.update(task, description="Expanding...")
            results = []
            for doc_id, text in docs:
                if d2q_only:
                    r = exp.d2q_only(doc_id, text)
                else:
                    r = exp.expand(doc_id, text)
                results.append(r)

    n, path = bridge.write(iter(results), filename=output_path.name)

    console.print(f"[green]Done![/green] {n} docs -> {path}")
    avg = sum(len(r.final) for r in results) / len(results) if results else 0
    console.print(f"Avg expansions: {avg:.1f}")


@app.command()
def demo(
    text: str = typer.Argument(..., help="Document text"),
    details: bool = typer.Option(True, "--details/--no-details")
):
    console.print("[bold blue]HQF-DE Demo[/bold blue]")
    console.print(f"[bold]Original:[/bold] {text[:200]}{'...' if len(text) > 200 else ''}")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Loading...", total=None)

        exp = Expander()
        exp.load()

        progress.update(task, description="Expanding...")
        result = exp.expand("demo", text)
        exp.unload()

    if details:
        console.print("[bold]Gaps:[/bold]")
        for g in result.gaps:
            console.print(f"  - {g}")

        console.print("[bold]Valid:[/bold]")
        for e in result.valid:
            console.print(f"  + {e}")

        if result.rejected:
            console.print("[bold]Rejected:[/bold]")
            for e in result.rejected:
                console.print(f"  x {e}")

        console.print("[bold]Queries:[/bold]")
        for q in result.queries:
            console.print(f"  ? {q}")

    console.print(f"[bold green]Expanded:[/bold green] {result.expanded}")
    console.print(f"[bold]Stats:[/bold] {len(result.final)} expansions")


@app.command()
def evaluate(
    num_queries: int = typer.Option(100, "--queries", "-q"),
    num_docs: int = typer.Option(1000, "--docs", "-d"),
    report: Path = typer.Option(None, "--report", "-r")
):
    console.print("[bold blue]HQF-DE Evaluation[/bold blue]")
    console.print(f"Docs: {num_docs}, Queries: {num_queries}")

    bridge = Bridge()
    docs = list(bridge.read(limit=num_docs))

    console.print(f"Expanding {len(docs)} docs...")

    exp = Expander()
    with exp:
        results = [exp.expand(doc_id, text) for doc_id, text in docs]

    ev = Evaluator()
    eval_results = ev.compare(results, num_queries=num_queries)

    ev.save(eval_results)
    rep = ev.report(eval_results)

    if report:
        with open(report, "w") as f:
            f.write(rep)
        console.print(f"Saved: {report}")
    else:
        console.print(rep)


@app.command()
def info():
    console.print("[bold blue]HQF-DE Config[/bold blue]")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Setting")
    table.add_column("Value")

    table.add_row("Root", str(config.project_root))
    table.add_row("Data", str(config.data_dir))
    table.add_row("Output", str(config.output_dir))
    table.add_row("Device", config.device)
    table.add_row("LLM", config.llm_model_name)
    table.add_row("D2Q", config.doc2query_model)
    table.add_row("NLI", config.nli_model)

    console.print(table)

    indexer = config.project_root / "indexer" / "indexer"
    if indexer.exists():
        console.print("[green]+[/green] Indexer found")
    else:
        console.print("[yellow]![/yellow] Indexer not found")

    bridge = Bridge()
    if bridge.health():
        console.print("[green]+[/green] API running")
    else:
        console.print("[yellow]![/yellow] API not running")


if __name__ == "__main__":
    app()
