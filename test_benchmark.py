# test_benchmark.py
import asyncio
import json
from src.tensorbench.benchmark.runner import BenchmarkRunner, BenchmarkConfig
from src.tensorbench.hardware.detector import detect_hardware

async def main():
    print("🔍 Сканирую железо...")
    hw = detect_hardware()
    
    print("🚀 Запускаю бенчмарк...")
    runner = BenchmarkRunner()
    
    # Прогресс-бар для консоли
    def progress(value: float, text: str):
        print(f"[{int(value*100)}%] {text}")
    
    runner.set_progress_callback(progress)
    
    try:
        result = await runner.run_async(hw)
        
        print(f"\n✅ Результаты:")
        print(f"   TPS: {result.tps} ток/сек")
        print(f"   Модель: {result.model_name}")
        
        # Прогноз для большей модели
        print(f"\n🔮 Прогноз для Llama-3-8B-Q4:")
        prediction = runner.predict_for_model(result, "llm_7b_q4")
        print(f"   Ожидаемая скорость: ~{prediction.predicted_tps} TPS")
        print(f"   Уверенность: {prediction.confidence}")
        if prediction.upgrade_potential:
            print(f"   💡 Апгрейд: {prediction.upgrade_potential}")
            
    finally:
        runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())