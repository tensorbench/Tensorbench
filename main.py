"""Точка входа CLI для TensorBench"""
import click
from src.tensorbench.hw_detect import get_system_info, print_report
from src.tensorbench.benchmark import download_model, run_benchmark, print_results
from src.tensorbench.analyzer import analyze_performance, print_analysis

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
    """Запустить тест скорости и получить рекомендации"""
    # 1. Детект железа
    hw_info = get_system_info()
    
    # 2. Запуск бенчмарка
    model_path = download_model()
    bench_result = run_benchmark(model_path)
    if not bench_result:
        return
        
    # 3. Вывод сырых метрик
    print_results(bench_result)
    
    # 4. Аналитика и рекомендации
    analysis = analyze_performance(hw_info, bench_result)
    print_analysis(analysis)

if __name__ == "__main__":
    cli()
