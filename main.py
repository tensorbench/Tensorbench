"""Точка входа CLI для TensorBench"""
import click
from src.tensorbench.hw_detect import get_system_info, print_report
from src.tensorbench.benchmark import download_model, run_benchmark, print_results

@click.group()
def cli():
    """TensorBench: Benchmark your AI hardware"""
    pass

@cli.command()
def detect():
    """Собрать и вывести информацию о железе"""
    info = get_system_info()
    print_report(info)

@cli.command()
def bench():
    """Запустить тест скорости генерации ИИ"""
    model_path = download_model()
    results = run_benchmark(model_path)
    print_results(results)

if __name__ == "__main__":
    cli()
