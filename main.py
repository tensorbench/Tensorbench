"""Точка входа CLI для TensorBench"""
import click
from src.tensorbench.hw_detect import get_system_info, print_report

@click.group()
def cli():
    """TensorBench: Benchmark your AI hardware"""
    pass

@cli.command()
def detect():
    """Собрать и вывести информацию о железе"""
    info = get_system_info()
    print_report(info)

if __name__ == "__main__":
    cli()
