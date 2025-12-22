import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import config
from .pipeline.expander import Expander
from .pipeline.indexer_bridge import Bridge
from .evaluation.evaluator import Evaluator

app = typer.Typer(name="hqf-de", help="HQF-DE Document Expansion")
console = Console()


@app.command()
def expand(
    input_file: Path = typer.Option(None, "-i"),
    output_file: Path = typer.Option(None, "-o"),
    limit: int = typer.Option(None, "-n"),
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

    console.print(f"[bold]HQF-DE[/bold] {input_path} -> {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bridge = Bridge()
    exp = Expander(use_llm=False if d2q_only else use_llm, use_nli=False if d2q_only else use_nli, use_d2q=use_d2q)
    docs = list(bridge.read(limit=limit))

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        progress.add_task("Expanding...", total=None)
        with exp:
            results = [exp.d2q_only(doc_id, text) if d2q_only else exp.expand(doc_id, text) for doc_id, text in docs]

    n, path = bridge.write(iter(results), filename=output_path.name)
    console.print(f"[green]Done![/green] {n} docs -> {path}")


@app.command()
def demo(text: str = typer.Argument(...)):
    console.print(f"[bold]Input:[/bold] {text[:200]}{'...' if len(text) > 200 else ''}")

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        progress.add_task("Expanding...", total=None)
        exp = Expander()
        exp.load()
        result = exp.expand("demo", text)
        exp.unload()

    console.print(f"[bold]Gaps:[/bold] {result.get('gaps', [])}")
    console.print(f"[bold]Expansions:[/bold] {result.get('final', [])}")
    console.print(f"[bold green]Result:[/bold green] {result.get('expanded', text)}")


@app.command()
def evaluate(num_queries: int = typer.Option(100, "-q"), num_docs: int = typer.Option(1000, "-d")):
    console.print(f"[bold]Evaluating[/bold] {num_docs} docs, {num_queries} queries")

    bridge = Bridge()
    docs = list(bridge.read(limit=num_docs))
    exp = Expander()

    with exp:
        results = [exp.expand(doc_id, text) for doc_id, text in docs]

    ev = Evaluator()
    eval_results = ev.compare(results, num_queries=num_queries)
    ev.save(eval_results)
    console.print(ev.report(eval_results))


@app.command()
def info():
    table = Table(show_header=True, header_style="bold")
    table.add_column("Setting")
    table.add_column("Value")
    table.add_row("Root", str(config.project_root))
    table.add_row("Data", str(config.data_dir))
    table.add_row("Device", config.device)
    table.add_row("LLM", config.llm_model_name)
    console.print(table)


if __name__ == "__main__":
    app()
