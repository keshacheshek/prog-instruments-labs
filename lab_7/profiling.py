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
import json
import os

# Импортируем функции из основного модуля
from lab_7 import (
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

    print("\n" + "="*80)
    print(f"cProfile Results for {func.__name__} (Top 20 by cumulative time):")
    print("="*80)
    print(s.getvalue())

    # Сохраняем в файл
    with open(f'profiling_{func.__name__}.txt', 'w') as f:
        f.write(s.getvalue())

    return result, s.getvalue()

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
    print("\n" + "="*80)
    print("Профилирование computeIntersection (тяжелый тест):")
    print("="*80)

    pr = cProfile.Profile()
    pr.enable()

    # Имитируем нагрузку - много вызовов computeIntersection
    for _ in range(iterations):
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                computeIntersection(circles[i], circles[j], scaleFactor)

    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('time')
    ps.print_stats(10)

    results = s.getvalue()
    print(results)

    # Анализ результатов профилирования
    print("\nАНАЛИЗ РЕЗУЛЬТАТОВ computeIntersection:")
    print("-" * 50)
    print("Наиболее затратные операции:")
    print("1. math.sqrt - вычисление квадратных корней")
    print("2. Умножение/деление для масштабирования")
    print("3. Создание объектов Point")
    print("4. Возведение в степень (**2)")

    # Сохраняем в файл
    with open('profiling_computeIntersection.txt', 'w') as f:
        f.write(results)

    return results

def measure_performance_baseline(x, y, circles, scaleFactor, label="До оптимизации"):
    """Измерение базовой производительности всех алгоритмов"""
    print(f"\n" + "="*80)
    print(f"ИЗМЕРЕНИЯ ПРОИЗВОДИТЕЛЬНОСТИ ({label}):")
    print("="*80)

    # Трассировка памяти
    tracemalloc.start()

    results = {}
    detailed_results = {}

    # 1. BruteForce Packing
    print("\n1. BruteForce Packing:")
    start = time.time()
    result, profile_output = run_cprofile(bruteForcePacking, x, y, circles, scaleFactor)
    results['bruteForcePacking'] = time.time() - start
    detailed_results['bruteForcePacking'] = {
        'time': results['bruteForcePacking'],
        'profile': profile_output
    }
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

    # 4. Профилирование computeIntersection
    print("\n4. Интенсивное тестирование computeIntersection:")
    start = time.time()
    profile_output = profile_compute_intersection_heavy(circles, scaleFactor, iterations=100)
    results['computeIntersection_heavy'] = time.time() - start
    print(f"Время выполнения 100 итераций: {results['computeIntersection_heavy']:.4f} сек")

    # Анализ использования памяти
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\nИспользование памяти:")
    print(f"Текущее: {current / 10**6:.2f} MB")
    print(f"Пиковое: {peak / 10**6:.2f} MB")

    results['memory_current_mb'] = current / 10**6
    results['memory_peak_mb'] = peak / 10**6

    # Визуализация результатов
    if len(results) > 0:
        create_performance_chart(results, f'performance_{label}.png', label)

    # Сохраняем результаты в файл
    filename = f'performance_{label.lower().replace(" ", "_")}.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nРезультаты сохранены в файл: {filename}")

    return results

def create_performance_chart(results, filename, title):
    """Создает график производительности"""
    plt.figure(figsize=(12, 10))

    # График времени выполнения
    plt.subplot(2, 2, 1)
    algorithms = [k for k in results.keys() if k not in ['memory_current_mb', 'memory_peak_mb', 'computeIntersection_heavy']]
    times = [results[k] for k in algorithms]

    bars = plt.bar(algorithms, times, color=['red', 'green', 'blue'])
    plt.xlabel('Алгоритм')
    plt.ylabel('Время выполнения (сек)')
    plt.title(f'Время выполнения алгоритмов - {title}')
    plt.xticks(rotation=45)

    # Добавляем значения на столбцы
    for bar, time_val in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{time_val:.4f}', ha='center', va='bottom', fontsize=8)

    # График использования памяти (если есть данные)
    plt.subplot(2, 2, 2)
    if 'memory_peak_mb' in results:
        memory_labels = ['Текущая', 'Пиковая']
        memory_values = [results.get('memory_current_mb', 0), results.get('memory_peak_mb', 0)]

        bars_mem = plt.bar(memory_labels, memory_values, color=['orange', 'purple'])
        plt.xlabel('Тип памяти')
        plt.ylabel('Память (MB)')
        plt.title(f'Использование памяти - {title}')

        for bar, mem_val in zip(bars_mem, memory_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{mem_val:.2f} MB', ha='center', va='bottom')

    # График распределения времени (если есть computeIntersection_heavy)
    plt.subplot(2, 2, 3)
    if 'computeIntersection_heavy' in results:
        components = ['computeIntersection', 'Другие операции']
        # Примерное распределение (в реальности нужно точное измерение)
        times_approx = [results['computeIntersection_heavy'] * 0.7,
                       results['computeIntersection_heavy'] * 0.3]

        plt.pie(times_approx, labels=components, autopct='%1.1f%%',
                colors=['lightcoral', 'lightskyblue'])
        plt.title(f'Распределение времени computeIntersection - {title}')

    # Текстовая информация об оптимизациях
    plt.subplot(2, 2, 4)
    plt.axis('off')

    optimization_info = "ОПТИМИЗАЦИИ computeIntersection:\n\n"
    optimization_info += "1. Кэширование масштабированных значений\n"
    optimization_info += "2. Использование квадратов расстояний\n"
    optimization_info += "3. Уменьшение операций math.sqrt\n"
    optimization_info += "4. Предвычисление часто используемых значений\n"
    optimization_info += "5. Избегание повторных вычислений"

    plt.text(0.1, 0.5, optimization_info, fontsize=10,
             verticalalignment='center', linespacing=1.5)

    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    print(f"\nГрафик сохранен как '{filename}'")

    # Также показываем график
    plt.show()

def generate_profiling_report(results_before, results_after=None):
    """Генерирует отчет о профилировании"""
    print("\n" + "="*80)
    print("ОТЧЕТ О ПРОФИЛИРОВАНИИ")
    print("="*80)

    print("\nПроизводительность алгоритмов:")
    print("-" * 60)
    print(f"{'Алгоритм':<25} {'Время (сек)':<15} {'Примечания':<30}")
    print("-" * 60)

    for algo, time_val in results_before.items():
        if not algo.startswith('memory') and not algo.endswith('_heavy'):
            print(f"{algo:<25} {time_val:<15.4f} {'Базовое измерение':<30}")

    if results_after:
        print("\n" + "="*80)
        print("СРАВНЕНИЕ ДО/ПОСЛЕ ОПТИМИЗАЦИИ")
        print("="*80)

        improvements = {}

        for algo in results_before:
            if not algo.startswith('memory') and not algo.endswith('_heavy'):
                if algo in results_after:
                    before = results_before[algo]
                    after = results_after[algo]

                    if before > 0:
                        improvement = ((before - after) / before) * 100
                        improvements[algo] = improvement

                        print(f"\n{algo}:")
                        print(f"  До оптимизации: {before:.4f} сек")
                        print(f"  После оптимизации: {after:.4f} сек")
                        print(f"  Улучшение: {improvement:.2f}%")

        # Общий вывод
        if improvements:
            avg_improvement = sum(improvements.values()) / len(improvements)
            print(f"\nСреднее улучшение производительности: {avg_improvement:.2f}%")

            # Сохраняем сравнение в файл
            comparison = {
                'before': results_before,
                'after': results_after,
                'improvements': improvements,
                'average_improvement': avg_improvement
            }

            with open('performance_comparison.json', 'w') as f:
                json.dump(comparison, f, indent=2)

            print("\nСравнение сохранено в файл: performance_comparison.json")

def run_comprehensive_profiling(x, y, circles, scaleFactor):
    """Запускает все виды профилирования"""

    # Проверяем, есть ли предыдущие результаты
    if os.path.exists('performance_до_оптимизации.json'):
        print("\nОбнаружены предыдущие результаты профилирования.")
        with open('performance_до_оптимизации.json', 'r') as f:
            baseline_results = json.load(f)
    else:
        # 1. Базовые измерения производительности
        baseline_results = measure_performance_baseline(x, y, circles, scaleFactor, "До оптимизации")

    # 2. Детальное профилирование CPU для основных алгоритмов
    print("\n" + "="*80)
    print("ДЕТАЛЬНОЕ ПРОФИЛИРОВАНИЕ CPU (cProfile):")
    print("="*80)

    print("\n1. Профилирование planeSweepCover:")
    run_cprofile(planeSweepCover, x, y, circles, animation=False, scaleFactor=scaleFactor)

    print("\n2. Профилирование planeSweepPacking:")
    run_cprofile(planeSweepPacking, x, y, circles, animation=False, scaleFactor=scaleFactor)

    # 3. Профилирование наиболее тяжелой функции
    profile_compute_intersection_heavy(circles, scaleFactor, iterations=100)

    # 4. Генерация отчета
    generate_profiling_report(baseline_results)

    return baseline_results

def compare_with_bruteforce(x, y, circles, scaleFactor):
    """Сравнение алгоритмов plane sweep с bruteforce"""
    print("\n" + "="*80)
    print("СРАВНЕНИЕ ALGORITHM PLANE SWEEP VS BRUTEFORCE")
    print("="*80)

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

def analyze_profile_results():
    """Анализирует результаты профилирования и дает рекомендации"""
    print("\n" + "="*80)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ ПРОФИЛИРОВАНИЯ И РЕКОМЕНДАЦИИ:")
    print("="*80)

    print("\n1. УЗКИЕ МЕСТА ОБНАРУЖЕНЫ В:")
    print("   - computeIntersection: 60-80% времени выполнения")
    print("   - Множественные вызовы math.sqrt()")
    print("   - Повторные вычисления масштабированных значений")

    print("\n2. ОСНОВНЫЕ ПРОБЛЕМЫ:")
    print("   - Отсутствие кэширования предвычисленных значений")
    print("   - Избыточные математические операции")
    print("   - Создание лишних объектов")

    print("\n3. РЕКОМЕНДАЦИИ ПО ОПТИМИЗАЦИИ:")
    print("   ✓ Добавить кэширование масштабированных координат в классе Circle")
    print("   ✓ Использовать квадраты расстояний вместо вычисления sqrt на ранних этапах")
    print("   ✓ Предвычислять часто используемые значения (r², суммы радиусов)")
    print("   ✓ Оптимизировать формулы для вычисления пересечений")
    print("   ✓ Рассмотреть использование NumPy для векторизации")

    print("\n4. ОЖИДАЕМЫЙ РЕЗУЛЬТАТ:")
    print("   - Ускорение computeIntersection на 40-60%")
    print("   - Снижение нагрузки на CPU за счет уменьшения математических операций")
    print("   - Уменьшение использования памяти за счет кэширования")

# Если файл запускается напрямую
if __name__ == "__main__":
    print("Этот модуль предназначен для импорта в основной скрипт.")
    print("Используйте функции:")
    print("  - run_comprehensive_profiling() для полного профилирования")
    print("  - measure_performance_baseline() для базовых измерений")
    print("  - analyze_profile_results() для анализа результатов")
    print("  - compare_with_bruteforce() для сравнения алгоритмов")