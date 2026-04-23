"""Точка входа CLI для TensorBench v2"""
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
@click.option('--model-size', default='0.5b', help='Размер модели: 0.5b, 3b, 7b')
def bench(model_size):
    """Запустить тест и получить сценарные рекомендации"""
    # 1. Детект железа
    hw_info = get_system_info()
    
    # 2. Скачиваем нужную модель
    model_path = download_model(size=model_size)
    bench_result = run_benchmark(model_path)
    if not bench_result:
        return
        
    # 3. Вывод метрик
    print_results(bench_result)
    
    # 4. Сценарная аналитика
    analysis = analyze_performance(hw_info, bench_result)
    print_analysis(analysis)

if __name__ == "__main__":
    cli()
