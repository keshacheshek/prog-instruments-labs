"""
Модуль для профилирования производительности алгоритмов
"""

import cProfile
import pstats
import io
import time
import tracemalloc
import matplotlib.pyplot as plt
from memory_profiler import profile
import numpy as np

# Импортируем функции из основного модуля
from main import (
    planeSweepCover,
    planeSweepPacking,
    bruteForcePacking,
    bruteForceCover,
    computeIntersection,
    Point,
    Circle
)


def run_cprofile(func, *args, **kwargs):
    """Запускает функцию под профилировщиком cProfile"""
    pr = cProfile.Profile()
    pr.enable()

    result = func(*args, **kwargs)

    pr.disable()

    # Сохраняем результаты профилирования
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Топ 20 функций

    print("\n" + "=" * 80)
    print(f"cProfile Results for {func.__name__} (Top 20 by cumulative time):")
    print("=" * 80)
    print(s.getvalue())

    # Сохраняем в файл
    with open(f'profiling_{func.__name__}.txt', 'w') as f:
        f.write(s.getvalue())

    return result


@profile
def profile_memory_planeSweepCover(x, y, circles, animation, scaleFactor):
    """Обертка для профилирования памяти planeSweepCover"""
    return planeSweepCover(x, y, circles, animation, scaleFactor)


@profile
def profile_memory_planeSweepPacking(x, y, circles, animation, scaleFactor):
    """Обертка для профилирования памяти planeSweepPacking"""
    return planeSweepPacking(x, y, circles, animation, scaleFactor)


def profile_compute_intersection_heavy(circles, scaleFactor=50, iterations=1000):
    """Тест производительности для computeIntersection"""
    print("\n" + "=" * 80)
    print("Профилирование computeIntersection (тяжелый тест):")
    print("=" * 80)

    pr = cProfile.Profile()
    pr.enable()

    # Имитируем нагрузку - много вызовов computeIntersection
    for _ in range(iterations):
        for i in range(len(circles)):
            for j in range(i + 1, len(circles)):
                computeIntersection(circles[i], circles[j], scaleFactor)

    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('time')
    ps.print_stats(10)

    results = s.getvalue()
    print(results)

    # Сохраняем в файл
    with open('profiling_computeIntersection.txt', 'w') as f:
        f.write(results)

    return results


def measure_performance_baseline(x, y, circles, scaleFactor):
    """Измерение базовой производительности всех алгоритмов"""
    print("\n" + "=" * 80)
    print("БАЗОВЫЕ ИЗМЕРЕНИЯ ПРОИЗВОДИТЕЛЬНОСТИ (ДО ОПТИМИЗАЦИИ)")
    print("=" * 80)

    # Трассировка памяти
    tracemalloc.start()

    results = {}

    # 1. BruteForce Packing
    print("\n1. BruteForce Packing:")
    start = time.time()
    bruteForcePacking(x, y, circles, scaleFactor)
    results['bruteForcePacking'] = time.time() - start
    print(f"Время выполнения: {results['bruteForcePacking']:.4f} сек")

    # 2. Plane Sweep Packing
    print("\n2. Plane Sweep Packing:")
    start = time.time()
    planeSweepPacking(x, y, circles, animation=False, scaleFactor=scaleFactor)
    results['planeSweepPacking'] = time.time() - start
    print(f"Время выполнения: {results['planeSweepPacking']:.4f} сек")

    # 3. Plane Sweep Cover
    print("\n3. Plane Sweep Cover:")
    start = time.time()
    planeSweepCover(x, y, circles, animation=False, scaleFactor=scaleFactor)
    results['planeSweepCover'] = time.time() - start
    print(f"Время выполнения: {results['planeSweepCover']:.4f} сек")

    # Анализ использования памяти
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\nИспользование памяти:")
    print(f"Текущее: {current / 10 ** 6:.2f} MB")
    print(f"Пиковое: {peak / 10 ** 6:.2f} MB")

    results['memory_current_mb'] = current / 10 ** 6
    results['memory_peak_mb'] = peak / 10 ** 6

    # Визуализация результатов
    if len(results) > 0:
        create_performance_chart(results, 'performance_baseline.png', 'До оптимизации')

    # Сохраняем результаты в файл
    with open('performance_baseline.json', 'w') as f:
        import json
        json.dump(results, f, indent=2)

    return results


def create_performance_chart(results, filename, title):
    """Создает график производительности"""
    plt.figure(figsize=(12, 8))

    # График времени выполнения
    plt.subplot(2, 1, 1)
    algorithms = [k for k in results.keys() if not k.startswith('memory')]
    times = [results[k] for k in algorithms]

    bars = plt.bar(algorithms, times, color=['red', 'green', 'blue'])
    plt.xlabel('Алгоритм')
    plt.ylabel('Время выполнения (сек)')
    plt.title(f'Сравнение производительности алгоритмов - {title}')
    plt.xticks(rotation=45)

    # Добавляем значения на столбцы
    for bar, time_val in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{time_val:.4f}', ha='center', va='bottom')

    # График использования памяти (если есть данные)
    if 'memory_peak_mb' in results:
        plt.subplot(2, 1, 2)
        memory_labels = ['Текущая', 'Пиковая']
        memory_values = [results.get('memory_current_mb', 0), results.get('memory_peak_mb', 0)]

        bars_mem = plt.bar(memory_labels, memory_values, color=['orange', 'purple'])
        plt.xlabel('Тип памяти')
        plt.ylabel('Память (MB)')
        plt.title('Использование памяти')

        for bar, mem_val in zip(bars_mem, memory_values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{mem_val:.2f} MB', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    print(f"\nГрафик сохранен как '{filename}'")

    # Также показываем график
    plt.show()


def generate_profiling_report(results_before, results_after=None):
    """Генерирует отчет о профилировании"""
    print("\n" + "=" * 80)
    print("ОТЧЕТ О ПРОФИЛИРОВАНИИ")
    print("=" * 80)

    print("\nПроизводительность алгоритмов:")
    print("-" * 50)
    print(f"{'Алгоритм':<25} {'Время (сек)':<15} {'Примечания':<30}")
    print("-" * 50)

    for algo, time_val in results_before.items():
        if not algo.startswith('memory'):
            print(f"{algo:<25} {time_val:<15.4f} {'Базовое измерение':<30}")

    if results_after:
        print("\n" + "=" * 80)
        print("СРАВНЕНИЕ ДО/ПОСЛЕ ОПТИМИЗАЦИИ")
        print("=" * 80)

        for algo in results_before:
            if not algo.startswith('memory') and algo in results_after:
                before = results_before[algo]
                after = results_after[algo]
                improvement = ((before - after) / before) * 100

                print(f"\n{algo}:")
                print(f"  До оптимизации: {before:.4f} сек")
                print(f"  После оптимизации: {after:.4f} сек")
                print(f"  Улучшение: {improvement:.2f}%")


def run_comprehensive_profiling(x, y, circles, scaleFactor):
    """Запускает все виды профилирования"""

    # 1. Базовые измерения производительности
    baseline_results = measure_performance_baseline(x, y, circles, scaleFactor)

    # 2. Детальное профилирование CPU для основных алгоритмов
    print("\n" + "=" * 80)
    print("ДЕТАЛЬНОЕ ПРОФИЛИРОВАНИЕ CPU (cProfile):")
    print("=" * 80)

    print("\n1. Профилирование planeSweepCover:")
    run_cprofile(planeSweepCover, x, y, circles, animation=False, scaleFactor=scaleFactor)

    print("\n2. Профилирование planeSweepPacking:")
    run_cprofile(planeSweepPacking, x, y, circles, animation=False, scaleFactor=scaleFactor)

    # 3. Профилирование наиболее тяжелой функции
    profile_compute_intersection_heavy(circles, scaleFactor, iterations=100)

    # 4. Профилирование памяти (опционально, можно раскомментировать)
    # print("\n" + "="*80)
    # print("ПРОФИЛИРОВАНИЕ ПАМЯТИ:")
    # print("="*80)
    # print("\nПрофилирование памяти для planeSweepCover:")
    # profile_memory_planeSweepCover(x, y, circles, animation=False, scaleFactor=scaleFactor)

    # 5. Генерация отчета
    generate_profiling_report(baseline_results)

    return baseline_results


def compare_with_bruteforce(x, y, circles, scaleFactor):
    """Сравнение алгоритмов plane sweep с bruteforce"""
    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ ALGORITHM PLANE SWEEP VS BRUTEFORCE")
    print("=" * 80)

    results = {}

    # BruteForce Packing
    start = time.time()
    bruteForcePacking(x, y, circles, scaleFactor)
    results['bruteForcePacking'] = time.time() - start

    # Plane Sweep Packing
    start = time.time()
    planeSweepPacking(x, y, circles, animation=False, scaleFactor=scaleFactor)
    results['planeSweepPacking'] = time.time() - start

    # Расчет ускорения
    if results['bruteForcePacking'] > 0:
        speedup_packing = results['bruteForcePacking'] / results['planeSweepPacking']
        print(f"\nУскорение plane sweep packing: {speedup_packing:.2f}x")

    return results


# Если файл запускается напрямую
if __name__ == "__main__":
    print("Этот модуль предназначен для импорта в основной скрипт.")
    print("Используйте функции:")
    print("  - run_comprehensive_profiling() для полного профилирования")
    print("  - measure_performance_baseline() для базовых измерений")
    print("  - compare_with_bruteforce() для сравнения алгоритмов")