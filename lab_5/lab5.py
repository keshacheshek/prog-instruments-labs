"""
Statistical Language Processing tools (Chapter 22)

We define Unigram and Ngram text models, use them to generate random text,
and show the Viterbi algorithm for segmentation of letters into words.
Then we show a very simple Information Retrieval system, and an example
working on a tiny sample of Unix manual pages.
"""

import heapq
import os
import re
import logging
import logging.config
import logging.handlers
import json
import time
import traceback
import sys
import inspect
import threading
import queue
from datetime import datetime, timedelta
from collections import defaultdict, deque
from contextlib import contextmanager
from functools import wraps, lru_cache
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path

import numpy as np

import search
from probabilistic_learning import CountingProbDist
from utils import hashabledict


# =============================================================================
# ПРОДВИНУТАЯ КОНФИГУРАЦИЯ ЛОГИРОВАНИЯ ДЛЯ ШЕСТОГО КОММИТА
# =============================================================================

class EnhancedJSONFormatter(logging.Formatter):
    """Кастомный JSON форматтер для структурированного логирования"""

    def format(self, record: logging.LogRecord) -> str:
        """Форматирование записи лога в JSON"""
        log_object = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
            'process': record.process,
            'thread': record.thread,
            'thread_name': record.threadName,
        }

        # Добавляем дополнительную информацию, если есть
        if hasattr(record, 'extra'):
            log_object.update(record.extra)

        # Добавляем информацию об исключении, если есть
        if record.exc_info:
            log_object['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_object, ensure_ascii=False)


class ColoredConsoleFormatter(logging.Formatter):
    """Форматтер с цветами для консоли"""

    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[41m',   # Red background
        'RESET': '\033[0m',       # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Форматирование с цветами"""
        level_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']

        # Форматируем базовое сообщение
        message = super().format(record)

        # Добавляем цвета
        colored_message = f"{level_color}{message}{reset_color}"

        return colored_message


class LogBufferHandler(logging.Handler):
    """Буферизация логов для пакетной обработки"""

    def __init__(self, capacity: int = 1000, flush_interval: int = 60):
        super().__init__()
        self.buffer = deque(maxlen=capacity)
        self.flush_interval = flush_interval
        self.last_flush = time.time()
        self.lock = threading.RLock()

    def emit(self, record: logging.LogRecord):
        """Добавление записи в буфер"""
        with self.lock:
            self.buffer.append(record)

            # Автоматическая очистка по времени
            current_time = time.time()
            if current_time - self.last_flush >= self.flush_interval:
                self.flush()
                self.last_flush = current_time

    def flush(self):
        """Очистка буфера (может быть переопределена для отправки)"""
        with self.lock:
            if self.buffer:
                buffer_copy = list(self.buffer)
                self.buffer.clear()
                # Здесь можно отправить логи куда-то (например, в базу данных или API)
                # Для простоты просто очищаем
                return buffer_copy
        return []

    def get_recent_logs(self, count: int = 100) -> List[logging.LogRecord]:
        """Получение последних записей из буфера"""
        with self.lock:
            return list(self.buffer)[-count:]


class DynamicLogLevelManager:
    """Менеджер для динамического изменения уровней логирования"""

    def __init__(self):
        self.loggers: Dict[str, Dict[str, Any]] = {}
        self.default_levels = {}
        self.lock = threading.RLock()

    def register_logger(self, logger: logging.Logger,
                       default_level: str = 'INFO',
                       description: str = ''):
        """Регистрация логгера для управления"""
        with self.lock:
            self.loggers[logger.name] = {
                'logger': logger,
                'default_level': default_level,
                'current_level': logger.level,
                'description': description,
                'handlers': list(logger.handlers)
            }
            self.default_levels[logger.name] = default_level

    def set_level(self, logger_name: str, level: str):
        """Установка уровня логирования для логгера"""
        with self.lock:
            if logger_name in self.loggers:
                logger_info = self.loggers[logger_name]
                level_num = getattr(logging, level.upper(), logging.INFO)
                logger_info['logger'].setLevel(level_num)
                logger_info['current_level'] = level_num
                return True
            return False

    def reset_to_default(self, logger_name: str = None):
        """Сброс уровня логирования к умолчанию"""
        with self.lock:
            if logger_name:
                if logger_name in self.loggers:
                    default_level = self.default_levels[logger_name]
                    self.set_level(logger_name, default_level)
            else:
                for name in self.loggers:
                    default_level = self.default_levels[name]
                    self.set_level(name, default_level)

    def get_status(self) -> Dict[str, Any]:
        """Получение статуса всех логгеров"""
        with self.lock:
            status = {}
            for name, info in self.loggers.items():
                status[name] = {
                    'current_level': logging.getLevelName(info['current_level']),
                    'default_level': info['default_level'],
                    'description': info['description'],
                    'handlers_count': len(info['handlers'])
                }
            return status


class MetricsCollector:
    """Сборщик метрик производительности и использования"""

    def __init__(self):
        self.metrics = defaultdict(lambda: defaultdict(float))
        self.counters = defaultdict(int)
        self.timestamps = defaultdict(list)
        self.lock = threading.RLock()

    def record_metric(self, metric_name: str, value: float,
                     tags: Dict[str, str] = None):
        """Запись метрики"""
        with self.lock:
            key = metric_name
            if tags:
                key = f"{metric_name}_{'_'.join(f'{k}={v}' for k, v in tags.items())}"

            self.metrics[key]['sum'] += value
            self.metrics[key]['count'] += 1
            self.metrics[key]['min'] = min(self.metrics[key].get('min', float('inf')), value)
            self.metrics[key]['max'] = max(self.metrics[key].get('max', -float('inf')), value)
            self.metrics[key]['avg'] = self.metrics[key]['sum'] / self.metrics[key]['count']

            # Храним временные метки для последних значений
            self.timestamps[key].append((datetime.now(), value))
            if len(self.timestamps[key]) > 1000:  # Ограничиваем размер
                self.timestamps[key] = self.timestamps[key][-1000:]

    def increment_counter(self, counter_name: str, amount: int = 1,
                         tags: Dict[str, str] = None):
        """Увеличение счетчика"""
        with self.lock:
            key = counter_name
            if tags:
                key = f"{counter_name}_{'_'.join(f'{k}={v}' for k, v in tags.items())}"
            self.counters[key] += amount

    def get_metrics_report(self) -> Dict[str, Any]:
        """Получение отчета по метрикам"""
        with self.lock:
            report = {
                'metrics': dict(self.metrics),
                'counters': dict(self.counters),
                'timestamp': datetime.now().isoformat()
            }
            return report

    def clear(self):
        """Очистка всех метрик"""
        with self.lock:
            self.metrics.clear()
            self.counters.clear()
            self.timestamps.clear()


def setup_advanced_logging(config_file: str = 'logging_config.json'):
    """Настройка продвинутого логирования"""

    # Создаем директорию для логов, если ее нет
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    default_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'simple': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%H:%M:%S'
            },
            'json': {
                '()': '__main__.EnhancedJSONFormatter',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'colored': {
                '()': '__main__.ColoredConsoleFormatter',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'colored',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'filename': str(log_dir / 'text_processing.log'),
                'maxBytes': 10485760,  # 10MB
                'backupCount': 10,
                'encoding': 'utf-8'
            },
            'error_file': {
                'class': 'logging.FileHandler',
                'level': 'WARNING',
                'formatter': 'detailed',
                'filename': str(log_dir / 'errors.log'),
                'encoding': 'utf-8'
            },
            'json_file': {
                'class': 'logging.FileHandler',
                'level': 'INFO',
                'formatter': 'json',
                'filename': str(log_dir / 'metrics.json.log'),
                'encoding': 'utf-8'
            },
            'buffer_handler': {
                '()': '__main__.LogBufferHandler',
                'level': 'INFO',
                'capacity': 5000,
                'flush_interval': 30
            }
        },
        'loggers': {
            'text_processing': {
                'level': 'DEBUG',
                'handlers': ['console', 'file', 'error_file', 'json_file', 'buffer_handler'],
                'propagate': False
            },
            'text_processing.models': {
                'level': 'INFO',
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'text_processing.ir': {
                'level': 'INFO',
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'text_processing.decoders': {
                'level': 'INFO',
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'text_processing.metrics': {
                'level': 'INFO',
                'handlers': ['json_file'],
                'propagate': False
            }
        },
        'root': {
            'level': 'WARNING',
            'handlers': ['console']
        }
    }

    try:
        if Path(config_file).exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = default_config
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

        logging.config.dictConfig(config)

    except Exception as e:
        # Резервная базовая конфигурация
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # Создаем и настраиваем основные логгеры
    main_logger = logging.getLogger('text_processing')
    main_logger.info(f"Продвинутая система логирования инициализирована. Логи в директории: {log_dir}")

    # Инициализируем менеджер уровней логирования
    level_manager = DynamicLogLevelManager()

    # Инициализируем сборщик метрик
    metrics_collector = MetricsCollector()

    # Регистрируем метрики в отдельном логгере
    metrics_logger = logging.getLogger('text_processing.metrics')

    return main_logger, level_manager, metrics_collector, metrics_logger


# Глобальные объекты логирования
logger, level_manager, metrics, metrics_logger = setup_advanced_logging()


# =============================================================================
# ДЕКОРАТОРЫ И УТИЛИТЫ ДЛЯ ЛОГИРОВАНИЯ
# =============================================================================

def log_operation(name: str = None, level: str = 'INFO',
                  log_args: bool = True, log_result: bool = False,
                  measure_time: bool = True):
    """Декоратор для логирования операций"""
    def decorator(func):
        op_name = name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Определяем логгер для функции
            func_logger = logging.getLogger(f"text_processing.{func.__module__}.{func.__name__}")

            # Логируем начало операции
            start_time = time.time() if measure_time else None

            if log_args:
                arg_str = ', '.join([str(arg) for arg in args])
                kwarg_str = ', '.join([f'{k}={v}' for k, v in kwargs.items()])
                all_args = ', '.join(filter(None, [arg_str, kwarg_str]))
                func_logger.log(getattr(logging, level),
                               f"Начало операции '{op_name}' с аргументами: {all_args}")
            else:
                func_logger.log(getattr(logging, level), f"Начало операции '{op_name}'")

            try:
                result = func(*args, **kwargs)

                # Логируем результат
                if log_result:
                    result_str = str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
                    func_logger.log(getattr(logging, level),
                                   f"Операция '{op_name}' завершена. Результат: {result_str}")
                else:
                    func_logger.log(getattr(logging, level), f"Операция '{op_name}' завершена")

                # Логируем время выполнения
                if measure_time and start_time:
                    elapsed = time.time() - start_time
                    func_logger.debug(f"Операция '{op_name}' заняла {elapsed:.4f} секунд")
                    metrics.record_metric(f"operation.{op_name}.duration", elapsed)
                    metrics.increment_counter(f"operation.{op_name}.calls")

                    # Логируем метрики в отдельный логгер
                    if elapsed > 1.0:  # Долгие операции
                        metrics_logger.info(f"Долгая операция: {op_name} - {elapsed:.2f}с")

                return result

            except Exception as e:
                func_logger.error(f"Ошибка в операции '{op_name}': {e}", exc_info=True)
                metrics.increment_counter(f"operation.{op_name}.errors")
                raise

        return wrapper
    return decorator


@contextmanager
def log_context(name: str, level: str = 'INFO', log_enter: bool = True,
                log_exit: bool = True, measure_time: bool = True):
    """Контекстный менеджер для логирования"""
    context_logger = logging.getLogger(f"text_processing.context.{name}")

    start_time = time.time() if measure_time else None

    if log_enter:
        context_logger.log(getattr(logging, level), f"Вход в контекст: {name}")

    try:
        yield context_logger
    except Exception as e:
        context_logger.error(f"Исключение в контексте '{name}': {e}", exc_info=True)
        metrics.increment_counter(f"context.{name}.exceptions")
        raise
    finally:
        if log_exit:
            exit_message = f"Выход из контекста: {name}"

            if measure_time and start_time:
                elapsed = time.time() - start_time
                exit_message += f" (заняло {elapsed:.4f} секунд)"
                metrics.record_metric(f"context.{name}.duration", elapsed)

            context_logger.log(getattr(logging, level), exit_message)


class LoggingState:
    """Класс для управления состоянием логирования"""

    def __init__(self):
        self.original_levels = {}
        self.silenced_loggers = set()
        self.backup_handlers = {}

    def silence_logger(self, logger_name: str):
        """Временное отключение логгера"""
        logger_obj = logging.getLogger(logger_name)
        if logger_name not in self.original_levels:
            self.original_levels[logger_name] = logger_obj.level
        logger_obj.setLevel(logging.CRITICAL + 1)  # Уровень выше максимального
        self.silenced_loggers.add(logger_name)

    def restore_logger(self, logger_name: str):
        """Восстановление логгера"""
        if logger_name in self.original_levels:
            logger_obj = logging.getLogger(logger_name)
            logger_obj.setLevel(self.original_levels[logger_name])
            self.silenced_loggers.discard(logger_name)

    def disable_handlers(self, logger_name: str):
        """Отключение обработчиков логгера"""
        logger_obj = logging.getLogger(logger_name)
        if logger_name not in self.backup_handlers:
            self.backup_handlers[logger_name] = logger_obj.handlers.copy()
        logger_obj.handlers = []

    def enable_handlers(self, logger_name: str):
        """Включение обработчиков логгера"""
        if logger_name in self.backup_handlers:
            logger_obj = logging.getLogger(logger_name)
            logger_obj.handlers = self.backup_handlers[logger_name]
            del self.backup_handlers[logger_name]


# Глобальное состояние логирования
logging_state = LoggingState()


# =============================================================================
# МОДЕЛИ ЯЗЫКА С ПРОДВИНУТЫМ ЛОГИРОВАНИЕМ
# =============================================================================

class UnigramWordModel(CountingProbDist):
    """This is a discrete probability distribution over words, so you
    can add, sample, or get P[word], just like with CountingProbDist. You can
    also generate a random text, n words long, with P.samples(n)."""

    def __init__(self, observations, default=0):
        with log_context("UnigramWordModel.__init__", log_enter=True, log_exit=True):
            logger.info(f"Создание UnigramWordModel с {len(observations) if observations else 0} наблюдениями")
            logger.debug(f"Параметр default: {default}")

            # Call CountingProbDist constructor,
            # passing the observations and default parameters.
            super(UnigramWordModel, self).__init__(observations, default)

            unique_words = len(self) if hasattr(self, '__len__') else 'unknown'
            logger.debug(f"UnigramWordModel создан успешно. Уникальных слов: {unique_words}")

            # Регистрируем метрики
            metrics.increment_counter("models.unigram.created")
            if observations:
                metrics.record_metric("models.unigram.observations", len(observations))

    @log_operation("UnigramWordModel.samples", log_args=True, log_result=True)
    def samples(self, n):
        """Return a string of n words, random according to the model."""
        if n <= 0:
            logger.warning(f"Запрошена генерация {n} слов. Возвращаем пустую строку.")
            return ''

        words = []
        for i in range(n):
            word = self.sample()
            words.append(word)
            if i % 20 == 0 and i > 0:  # Логируем каждые 20 слов
                logger.debug(f"Сгенерировано {i+1}/{n} слов")

        result = ' '.join(words)
        logger.debug(f"Сгенерировано {n} слов. Первые 5: {words[:5]}")

        # Записываем метрики
        metrics.increment_counter("models.unigram.samples_generated", n)

        return result

    @log_operation("UnigramWordModel.__getitem__", level='DEBUG', log_args=True, log_result=True)
    def __getitem__(self, word):
        """Override to add logging for probability queries"""
        try:
            prob = super().__getitem__(word)
            logger.debug(f"Вероятность слова '{word}': {prob}")
            metrics.increment_counter("models.unigram.probability_queries")
            return prob
        except KeyError as e:
            logger.warning(f"Слово '{word}' не найдено в модели. Возвращаем вероятность по умолчанию.")
            metrics.increment_counter("models.unigram.misses")
            return self.default


class NgramWordModel(CountingProbDist):
    """This is a discrete probability distribution over n-tuples of words.
    You can add, sample or get P[(word1, ..., wordn)]. The method P.samples(n)
    builds up an n-word sequence; P.add_cond_prob and P.add_sequence add data."""

    def __init__(self, n, observation_sequence=None, default=0):
        with log_context("NgramWordModel.__init__"):
            if n <= 0:
                logger.error(f"Попытка создать NgramWordModel с n={n}. n должно быть > 0.")
                metrics.increment_counter("models.ngram.creation_errors")
                raise ValueError("n must be > 0")

            logger.info(f"Создание NgramWordModel с n={n}")
            if observation_sequence:
                logger.debug(f"Начальная последовательность: {len(observation_sequence)} элементов")

            # In addition to the dictionary of n-tuples, cond_prob is a
            # mapping from (w1, ..., wn-1) to P(wn | w1, ... wn-1)
            CountingProbDist.__init__(self, default=default)
            self.n = n
            self.cond_prob = defaultdict()
            self.add_sequence(observation_sequence or [])

            cond_prob_size = len(self.cond_prob)
            logger.debug(f"NgramWordModel создан успешно. Условных распределений: {cond_prob_size}")

            # Регистрируем метрики
            metrics.increment_counter("models.ngram.created", tags={'n': str(n)})

    @log_operation("NgramWordModel.__getitem__", level='DEBUG')
    def __getitem__(self, ngram):
        """Override to add logging for ngram probability queries"""
        try:
            prob = super().__getitem__(ngram)
            logger.debug(f"Вероятность n-граммы {ngram}: {prob}")
            metrics.increment_counter("models.ngram.probability_queries")
            return prob
        except KeyError as e:
            logger.debug(f"N-грамма {ngram} не найдена. Возвращаем вероятность по умолчанию: {self.default}")
            metrics.increment_counter("models.ngram.misses")
            return self.default

    @log_operation("NgramWordModel.add_cond_prob", level='DEBUG')
    def add_cond_prob(self, ngram):
        """Build the conditional probabilities P(wn | (w1, ..., wn-1)"""
        logger.debug(f"Добавление условной вероятности для n-граммы: {ngram}")

        if len(ngram) != self.n:
            logger.error(f"Некорректная длина n-граммы: {len(ngram)} (ожидается {self.n})")
            metrics.increment_counter("models.ngram.invalid_ngrams")
            raise ValueError(f"ngram must have length {self.n}")

        prefix = ngram[:-1]
        if prefix not in self.cond_prob:
            logger.debug(f"Создание нового условного распределения для префикса: {prefix}")
            self.cond_prob[prefix] = CountingProbDist()

        self.cond_prob[prefix].add(ngram[-1])
        logger.debug(f"Добавлено слово '{ngram[-1]}' для префикса {prefix}")

    @log_operation("NgramWordModel.add_sequence", log_args=True)
    def add_sequence(self, words):
        """Add each tuple words[i:i+n], using a sliding window."""
        n = self.n

        if len(words) < n:
            logger.warning(f"Последовательность слишком короткая ({len(words)} слов) для n={n}")
            metrics.increment_counter("models.ngram.short_sequences")
            return

        logger.info(f"Добавление последовательности из {len(words)} слов в NgramWordModel (n={n})")

        ngram_count = 0
        for i in range(len(words) - n + 1):
            t = tuple(words[i:i + n])
            self.add(t)
            self.add_cond_prob(t)
            ngram_count += 1

            if ngram_count % 100 == 0:  # Логируем каждые 100 n-грамм
                logger.debug(f"Обработано {ngram_count} n-грамм")

        logger.info(f"Добавлено {ngram_count} n-грамм из {len(words)} слов")
        metrics.increment_counter("models.ngram.ngrams_added", ngram_count)
        metrics.record_metric("models.ngram.sequence_length", len(words))

    @log_operation("NgramWordModel.samples", log_args=True, log_result=True)
    def samples(self, nwords):
        """Generate an n-word sentence by picking random samples
        according to the model. At first pick a random n-gram and
        from then on keep picking a character according to
        P(c|wl-1, wl-2, ..., wl-n+1) where wl-1 ... wl-n+1 are the
        last n - 1 words in the generated sentence so far."""
        logger.info(f"Генерация {nwords} слов с помощью NgramWordModel (n={self.n})")

        if nwords < self.n:
            logger.warning(f"Запрошено {nwords} слов, но n={self.n}. Будет сгенерировано {self.n} слов.")
            nwords = self.n

        n = self.n
        output = list(self.sample())
        logger.debug(f"Начальная n-грамма: {output}")

        for i in range(n, nwords):
            last = output[-n + 1:]
            prefix_tuple = tuple(last)

            if prefix_tuple not in self.cond_prob:
                logger.warning(f"Префикс {prefix_tuple} не найден в условных распределениях. Начинаем новое предложение.")
                metrics.increment_counter("models.ngram.missing_prefixes")
                output.extend(self.sample())
                continue

            next_word = self.cond_prob[prefix_tuple].sample()
            output.append(next_word)

            if (i + 1) % 10 == 0:  # Логируем каждые 10 слов
                logger.debug(f"Сгенерировано {i+1}/{nwords} слов. Текущее слово: '{next_word}'")

        result = ' '.join(output)
        logger.info(f"Генерация завершена. Получено {len(output)} слов")
        logger.debug(f"Пример сгенерированного текста (первые 50 символов): {result[:50]}...")

        # Записываем метрики
        metrics.increment_counter("models.ngram.samples_generated", len(output))
        metrics.record_metric("models.ngram.generated_length", len(output))

        return result


class NgramCharModel(NgramWordModel):
    @log_operation("NgramCharModel.add_sequence", log_args=True)
    def add_sequence(self, words):
        """Add an empty space to every word to catch the beginning of words."""
        logger.debug(f"Добавление последовательности в NgramCharModel: {len(words)} слов")

        if not words:
            logger.warning("Пустая последовательность слов передана в NgramCharModel")
            return

        total_chars = sum(len(word) + 1 for word in words)  # +1 для пробела
        logger.debug(f"Общее количество символов для обработки: {total_chars}")

        for word_idx, word in enumerate(words):
            processed_word = ' ' + word
            super().add_sequence(processed_word)

            if word_idx % 50 == 0 and word_idx > 0:
                logger.debug(f"Обработано {word_idx+1}/{len(words)} слов")


class UnigramCharModel(NgramCharModel):
    def __init__(self, observation_sequence=None, default=0):
        with log_context("UnigramCharModel.__init__"):
            logger.info("Создание UnigramCharModel")
            CountingProbDist.__init__(self, default=default)
            self.n = 1
            self.cond_prob = defaultdict()
            self.add_sequence(observation_sequence or [])

            total_chars = len(self) if hasattr(self, '__len__') else 'unknown'
            logger.debug(f"UnigramCharModel создан успешно. Уникальных символов: {total_chars}")

            metrics.increment_counter("models.unigram_char.created")

    @log_operation("UnigramCharModel.add_sequence", log_args=True)
    def add_sequence(self, words):
        logger.info(f"Добавление последовательности в UnigramCharModel: {len(words)} слов")

        char_count = 0
        for word_idx, word in enumerate(words):
            if not word:
                continue

            for char_idx, char in enumerate(word):
                self.add(char)
                char_count += 1

                if char_count % 1000 == 0:
                    logger.debug(f"Обработано {char_count} символов")

            if word_idx % 100 == 0 and word_idx > 0:
                logger.debug(f"Обработано {word_idx+1}/{len(words)} слов")

        logger.info(f"Добавлено {char_count} символов из {len(words)} слов")
        metrics.increment_counter("models.unigram_char.chars_processed", char_count)


# =============================================================================
# АЛГОРИТМ ВИТЕРБИ С ПРОДВИНУТЫМ ЛОГИРОВАНИЕМ
# =============================================================================

@log_operation("viterbi_segment", log_args=True, log_result=True)
def viterbi_segment(text, P):
    """Find the best segmentation of the string of characters, given the
    UnigramWordModel P."""
    logger.info(f"Запуск алгоритма Витерби для текста длиной {len(text)} символов")
    logger.debug(f"Текст: '{text[:50]}...' если длиннее 50 символов")

    if not text:
        logger.warning("Пустой текст передан в viterbi_segment")
        return [], 0.0

    # best[i] = best probability for text[0:i]
    # words[i] = best word ending at position i
    n = len(text)
    words = [''] + list(text)
    best = [1.0] + [0.0] * n

    logger.debug(f"Инициализирован массив best размером {len(best)}")
    logger.debug(f"Используется модель P: {type(P).__name__}")

    # Fill in the vectors best words via dynamic programming
    progress_interval = max(1, n // 10)  # Логируем каждые 10%
    for i in range(1, n + 1):
        if i % progress_interval == 0:
            logger.debug(f"Обработано {i}/{n} позиций ({i/n*100:.1f}%)")

        best_candidate = None
        best_prob = 0.0

        for j in range(0, i):
            w = text[j:i]
            if not w:
                continue

            try:
                word_prob = P[w]
                curr_score = word_prob * best[j]

                if curr_score > best_prob:
                    best_prob = curr_score
                    best_candidate = w

            except Exception as e:
                logger.error(f"Ошибка при обработке слова '{w}': {e}")
                continue

        if best_candidate:
            best[i] = best_prob
            words[i] = best_candidate
            logger.debug(f"Позиция {i}: лучшее слово '{best_candidate}' с вероятностью {best_prob:.6f}")
        else:
            logger.warning(f"Не найдено подходящее слово для позиции {i}")
            best[i] = 0.0
            words[i] = ''

    logger.debug("Динамическое программирование завершено")

    # Now recover the sequence of best words
    sequence = []
    i = len(words) - 1
    step_count = 0

    while i > 0:
        if step_count > n:  # Защита от бесконечного цикла
            logger.error(f"Превышено максимальное количество шагов ({n}) при восстановлении последовательности")
            break

        current_word = words[i]
        if not current_word:
            logger.error(f"Пустое слово на позиции {i}. Прерывание восстановления.")
            break

        sequence.insert(0, current_word)
        i = i - len(current_word)
        step_count += 1

        logger.debug(f"Шаг {step_count}: слово '{current_word}', переходим к позиции {i}")

    final_prob = best[-1]
    logger.info(f"Алгоритм Витерби завершен. Найдено {len(sequence)} слов с вероятностью {final_prob:.10f}")
    logger.debug(f"Результат сегментации: {' '.join(sequence)}")

    if final_prob == 0.0:
        logger.warning("Итоговая вероятность равна 0. Возможно, текст не может быть корректно сегментирован.")

    # Записываем метрики
    metrics.increment_counter("algorithms.viterbi.segmentations")
    metrics.increment_counter("algorithms.viterbi.words_segmented", len(sequence))
    metrics.record_metric("algorithms.viterbi.text_length", n)
    metrics.record_metric("algorithms.viterbi.final_probability", final_prob)

    # Return sequence of best words and overall probability
    return sequence, final_prob


# =============================================================================
# ИНФОРМАЦИОННО-ПОИСКОВАЯ СИСТЕМА С ПРОДВИНУТЫМ ЛОГИРОВАНИЕМ
# =============================================================================


# TODO(tmrts): Expose raw index
class IRSystem:
    """A very simple Information Retrieval System, as discussed in Sect. 23.2.
    The constructor s = IRSystem('the a') builds an empty system with two
    stopwords. Next, index several documents with s.index_document(text, url).
    Then ask queries with s.query('query words', n) to retrieve the top n
    matching documents. Queries are literal words from the document,
    except that stopwords are ignored, and there is one special syntax:
    The query "learn: man cat", for example, runs "man cat" and indexes it."""

    def __init__(self, stopwords='the a of'):
        """Create an IR System. Optionally specify stopwords."""
        with log_context("IRSystem.__init__"):
            logger.info(f"Создание IRSystem со стоп-словами: {stopwords}")

            # index is a map of {word: {docid: count}}, where docid is an int,
            # indicating the index into the documents list.
            self.index = defaultdict(lambda: defaultdict(int))
            self.stopwords = set(words(stopwords))
            self.documents = []

            logger.debug(f"Инициализирован индекс. Стоп-слов: {len(self.stopwords)}")
            logger.debug(f"Стоп-слова: {sorted(self.stopwords)}")

            metrics.increment_counter("ir.system.created")

    @log_operation("IRSystem.index_collection", log_args=True)
    def index_collection(self, filenames):
        """Index a whole collection of files."""
        logger.info(f"Индексация коллекции из {len(filenames)} файлов")

        prefix = os.path.dirname(__file__)
        for idx, filename in enumerate(filenames):
            try:
                logger.debug(f"Индексация файла {idx+1}/{len(filenames)}: {filename}")

                if not os.path.exists(filename):
                    logger.error(f"Файл не найден: {filename}")
                    continue

                with open(filename, 'r', encoding='utf-8') as file:
                    content = file.read()

                rel_path = os.path.relpath(filename, prefix)
                self.index_document(content, rel_path)

                logger.debug(f"Файл {filename} успешно проиндексирован")

            except Exception as e:
                logger.error(f"Ошибка при индексации файла {filename}: {e}", exc_info=True)

        logger.info(f"Коллекция проиндексирована. Всего документов: {len(self.documents)}")
        metrics.increment_counter("ir.collections.indexed", len(filenames))

    @log_operation("IRSystem.index_document", log_args=True)
    def index_document(self, text, url):
        """Index the text of a document."""
        logger.info(f"Индексация документа: {url}")

        try:
            # For now, use first line for title
            if '\n' in text:
                title = text[:text.index('\n')].strip()
            else:
                title = text[:100].strip() + "..." if len(text) > 100 else text.strip()
                logger.warning(f"Документ {url} не содержит символов новой строки")

            docwords = words(text)
            docid = len(self.documents)

            logger.debug(f"Документ {docid}: заголовок='{title}', количество слов={len(docwords)}")

            self.documents.append(Document(title, url, len(docwords)))

            indexed_words = 0
            stopwords_count = 0

            for word in docwords:
                if word not in self.stopwords:
                    self.index[word][docid] += 1
                    indexed_words += 1
                else:
                    stopwords_count += 1

            logger.info(f"Документ {docid} проиндексирован. "
                       f"Индексировано слов: {indexed_words}, "
                       f"стоп-слов: {stopwords_count}, "
                       f"всего слов: {len(docwords)}")

            # Логируем статистику по уникальным словам
            unique_indexed_words = sum(1 for word in docwords if word not in self.stopwords)
            logger.debug(f"Уникальных индексированных слов: {unique_indexed_words}")

            # Записываем метрики
            metrics.increment_counter("ir.documents.indexed")
            metrics.increment_counter("ir.words.indexed", indexed_words)
            metrics.record_metric("ir.document.words", len(docwords))

        except Exception as e:
            logger.error(f"Ошибка при индексации документа {url}: {e}", exc_info=True)
            metrics.increment_counter("ir.indexing.errors")
            raise

    @log_operation("IRSystem.query", log_args=True, log_result=True)
    def query(self, query_text, n=10):
        """Return a list of n (score, docid) pairs for the best matches.
        Also handle the special syntax for 'learn: command'."""
        logger.info(f"Выполнение запроса: '{query_text}' (n={n})")

        if query_text.startswith("learn:"):
            logger.info(f"Обнаружен специальный синтаксис 'learn:' в запросе")
            command = query_text[len("learn:"):].strip()
            logger.debug(f"Выполнение команды: {command}")

            try:
                doctext = os.popen(command, 'r').read()
                logger.info(f"Команда выполнена успешно, получено {len(doctext)} символов")
                self.index_document(doctext, query_text)
                return []
            except Exception as e:
                logger.error(f"Ошибка при выполнении команды '{command}': {e}")
                return []

        # Обработка обычного запроса
        qwords = [w for w in words(query_text) if w not in self.stopwords]
        logger.debug(f"Слова запроса после фильтрации стоп-слов: {qwords}")

        if not qwords:
            logger.warning("После фильтрации стоп-слов не осталось слов в запросе")
            return []

        # Находим слово с наименьшим количеством документов для оптимизации
        try:
            shortest = min(qwords, key=lambda w: len(self.index[w]))
            logger.debug(f"Слово с наименьшим количеством документов: '{shortest}' "
                        f"(документов: {len(self.index[shortest])})")
        except ValueError as e:
            logger.error(f"Ошибка при поиске самого короткого слова: {e}")
            return []

        docids = self.index[shortest]
        logger.debug(f"Найдено {len(docids)} документов для слова '{shortest}'")

        # Вычисляем скоринг для каждого документа
        scored_docs = []
        for docid in docids:
            try:
                score = self.total_score(qwords, docid)
                scored_docs.append((score, docid))
            except Exception as e:
                logger.error(f"Ошибка при подсчете очков для документа {docid}: {e}")
                continue

        logger.info(f"Вычислены очки для {len(scored_docs)} документов")

        # Возвращаем топ-n результатов
        results = heapq.nlargest(n, scored_docs)
        logger.info(f"Возвращено {len(results)} лучших результатов")

        if results:
            best_score, best_docid = results[0]
            logger.debug(f"Лучший результат: документ {best_docid} с очками {best_score:.4f}")

        # Записываем метрики
        metrics.increment_counter("ir.queries.executed")
        metrics.increment_counter("ir.documents.scored", len(scored_docs))
        metrics.record_metric("ir.query.words_count", len(qwords))

        return results

    @log_operation("IRSystem.score", level='DEBUG')
    def score(self, word, docid):
        """Compute a score for this word on the document with this docid."""
        try:
            word_count = self.index[word][docid]
            doc_nwords = self.documents[docid].nwords

            if doc_nwords == 0:
                logger.warning(f"Документ {docid} имеет 0 слов. Возвращаем 0.")
                return 0.0

            score = np.log(1 + word_count) / np.log(1 + doc_nwords)
            logger.debug(f"Очки для слова '{word}' в документе {docid}: "
                        f"count={word_count}, nwords={doc_nwords}, score={score:.4f}")
            return score

        except (KeyError, IndexError) as e:
            logger.warning(f"Слово '{word}' не найдено в документе {docid}: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Ошибка при вычислении очков для слова '{word}' в документе {docid}: {e}")
            return 0.0

    @log_operation("IRSystem.total_score", level='DEBUG')
    def total_score(self, words, docid):
        """Compute the sum of the scores of these words on the document with this docid."""
        logger.debug(f"Вычисление суммарных очков для документа {docid}, слова: {words}")

        total = 0.0
        for word in words:
            word_score = self.score(word, docid)
            total += word_score
            logger.debug(f"Слово '{word}': {word_score:.4f}, сумма: {total:.4f}")

        logger.debug(f"Итоговые очки для документа {docid}: {total:.4f}")
        return total

    @log_operation("IRSystem.present", log_args=True)
    def present(self, results):
        """Present the results as a list."""
        logger.info(f"Отображение {len(results)} результатов")

        if not results:
            logger.warning("Нет результатов для отображения")
            print("No results found.")
            return

        print(f"\n{'Score':<7} | {'URL':<25} | {'Title'}")
        print("-" * 70)

        for idx, (score, docid) in enumerate(results):
            try:
                doc = self.documents[docid]
                score_percent = 100 * score
                print(f"{score_percent:5.2f}% | {doc.url:<25} | {doc.title[:45].expandtabs()}")
                logger.debug(f"Результат {idx+1}: docid={docid}, score={score:.4f}, "
                           f"title='{doc.title[:30]}...'")
            except IndexError as e:
                logger.error(f"Ошибка при отображении результата {idx}: документ {docid} не найден")
                print(f"{'ERROR':<7} | {'N/A':<25} | Document {docid} not found")
            except Exception as e:
                logger.error(f"Ошибка при отображении результата {idx}: {e}")
                print(f"{'ERROR':<7} | {'N/A':<25} | Error displaying document")

    @log_operation("IRSystem.present_results", log_args=True)
    def present_results(self, query_text, n=10):
        """Get results for the query and present them."""
        logger.info(f"Выполнение и отображение результатов для запроса: '{query_text}'")

        try:
            results = self.query(query_text, n)
            self.present(results)
            logger.info("Результаты успешно отображены")
        except Exception as e:
            logger.error(f"Ошибка при выполнении запроса '{query_text}': {e}", exc_info=True)
            print(f"Error processing query: {e}")


class UnixConsultant(IRSystem):
    """A trivial IR system over a small collection of Unix man pages."""

    def __init__(self):
        with log_context("UnixConsultant.__init__"):
            logger.info("Создание UnixConsultant (специализированная IR система для man-страниц)")

            IRSystem.__init__(self, stopwords="how do i the a of")

            import os
            aima_root = os.path.dirname(__file__)
            mandir = os.path.join(aima_root, 'aima-data/MAN/')

            logger.debug(f"Поиск man-страниц в директории: {mandir}")

            try:
                if not os.path.exists(mandir):
                    logger.error(f"Директория с man-страницами не найдена: {mandir}")
                    raise FileNotFoundError(f"Directory not found: {mandir}")

                man_files = [os.path.join(mandir, f) for f in os.listdir(mandir) if f.endswith('.txt')]
                logger.info(f"Найдено {len(man_files)} man-страниц в директории {mandir}")

                if not man_files:
                    logger.warning("Не найдено ни одного файла с man-страницами")

                self.index_collection(man_files)
                logger.info(f"UnixConsultant создан успешно. Проиндексировано {len(self.documents)} документов")

                metrics.increment_counter("ir.unix_consultant.created")

            except Exception as e:
                logger.error(f"Ошибка при создании UnixConsultant: {e}", exc_info=True)
                raise


class Document:
    """Metadata for a document: title and url; maybe add others later."""

    def __init__(self, title, url, nwords):
        logger.debug(f"Создание документа: title='{title[:30]}...', url={url}, nwords={nwords}")
        self.title = title
        self.url = url
        self.nwords = nwords

    def __repr__(self):
        return f"Document(title='{self.title[:20]}...', url='{self.url}', nwords={self.nwords})"


# =============================================================================
# УТИЛИТЫ С ПРОДВИНУТЫМ ЛОГИРОВАНИЕМ
# =============================================================================

@lru_cache(maxsize=128)
@log_operation("words", level='DEBUG', log_args=True, log_result=True)
def words(text, reg=re.compile('[a-z0-9]+')):
    """Return a list of the words in text, ignoring punctuation and
    converting everything to lowercase (to canonicalize).
    >>> words("``EGAD!'' Edgar cried.")
    ['egad', 'edgar', 'cried']
    """
    try:
        result = reg.findall(text.lower())
        logger.debug(f"Извлечено {len(result)} слов из текста (первые 5: {result[:5]})")
        metrics.increment_counter("utils.words.extracted", len(result))
        return result
    except Exception as e:
        logger.error(f"Ошибка при извлечении слов из текста: {e}")
        return []


@log_operation("canonicalize", level='DEBUG', log_args=True, log_result=True)
def canonicalize(text):
    """Return a canonical text: only lowercase letters and blanks.
    >>> canonicalize("``EGAD!'' Edgar cried.")
    'egad edgar cried'
    """
    try:
        result = ' '.join(words(text))
        logger.debug(f"Канонизированный текст (первые 50 символов): '{result[:50]}...'")
        metrics.increment_counter("utils.canonicalize.calls")
        return result
    except Exception as e:
        logger.error(f"Ошибка при канонизации текста: {e}")
        return ""


# =============================================================================
# ШИФРЫ С ПРОДВИНУТЫМ ЛОГИРОВАНИЕМ
# =============================================================================

# Example application (not in book): decode a cipher.
# A cipher is a code that substitutes one character for another.
# A shift cipher is a rotation of the letters in the alphabet,
# such as the famous rot13, which maps A to N, B to M, etc.

alphabet = 'abcdefghijklmnopqrstuvwxyz'


# Encoding

@log_operation("shift_encode", level='DEBUG', log_args=True, log_result=True)
def shift_encode(plaintext, n):
    """Encode text with a shift cipher that moves each letter up by n letters.
    >>> shift_encode('abc z', 1)
    'bcd a'
    """
    logger.debug(f"Шифрование сдвигом (n={n}): '{plaintext[:20]}...'")
    result = encode(plaintext, alphabet[n:] + alphabet[:n])
    logger.debug(f"Результат шифрования: '{result[:20]}...'")
    metrics.increment_counter("ciphers.shift.encode")
    return result


@log_operation("rot13", level='DEBUG', log_args=True, log_result=True)
def rot13(plaintext):
    """Encode text by rotating letters by 13 spaces in the alphabet.
    >>> rot13('hello')
    'uryyb'
    >>> rot13(rot13('hello'))
    'hello'
    """
    logger.debug(f"ROT13 шифрование: '{plaintext[:20]}...'")
    result = shift_encode(plaintext, 13)
    logger.debug(f"Результат ROT13: '{result[:20]}...'")
    metrics.increment_counter("ciphers.rot13.encode")
    return result


@log_operation("translate", level='DEBUG', log_args=True, log_result=True)
def translate(plaintext, function):
    """Translate chars of a plaintext with the given function."""
    logger.debug(f"Перевод текста (длина={len(plaintext)})")
    result = ""
    for char in plaintext:
        result += function(char)
    logger.debug(f"Перевод завершен. Результат: '{result[:20]}...'")
    return result


@log_operation("maketrans", level='DEBUG', log_args=True, log_result=True)
def maketrans(from_, to_):
    """Create a translation table and return the proper function."""
    logger.debug(f"Создание таблицы перевода: from_='{from_}', to_='{to_}'")
    trans_table = {}
    for n, char in enumerate(from_):
        trans_table[char] = to_[n]

    logger.debug(f"Таблица перевода создана ({len(trans_table)} записей)")
    return lambda char: trans_table.get(char, char)


@log_operation("encode", level='DEBUG', log_args=True, log_result=True)
def encode(plaintext, code):
    """Encode text using a code which is a permutation of the alphabet."""
    logger.debug(f"Кодирование текста длиной {len(plaintext)} символов")
    trans = maketrans(alphabet + alphabet.upper(), code + code.upper())

    result = translate(plaintext, trans)
    logger.debug(f"Текст закодирован. Результат: '{result[:20]}...'")
    return result


@log_operation("bigrams", level='DEBUG', log_args=True, log_result=True)
def bigrams(text):
    """Return a list of pairs in text (a sequence of letters or words).
    >>> bigrams('this')
    ['th', 'hi', 'is']
    >>> bigrams(['this', 'is', 'a', 'test'])
    [['this', 'is'], ['is', 'a'], ['a', 'test']]
    """
    logger.debug(f"Создание биграмм из текста длиной {len(text)}")
    result = [text[i:i + 2] for i in range(len(text) - 1)]
    logger.debug(f"Создано {len(result)} биграмм. Первые 5: {result[:5]}")
    metrics.increment_counter("utils.bigrams.created", len(result))
    return result


# Decoding a Shift (or Caesar) Cipher


class ShiftDecoder:
    """There are only 26 possible encodings, so we can try all of them,
    and return the one with the highest probability, according to a
    bigram probability distribution."""

    def __init__(self, training_text):
        with log_context("ShiftDecoder.__init__"):
            logger.info("Создание ShiftDecoder")
            logger.debug(f"Длина тренировочного текста: {len(training_text)} символов")

            training_text = canonicalize(training_text)
            logger.debug(f"Длина канонизированного текста: {len(training_text)} символов")

            self.P2 = CountingProbDist(bigrams(training_text), default=1)
            logger.debug(f"ShiftDecoder создан. Размер модели биграмм: {len(self.P2)}")

            metrics.increment_counter("decoders.shift.created")

    @log_operation("ShiftDecoder.score", level='DEBUG', log_args=True, log_result=True)
    def score(self, plaintext):
        """Return a score for text based on how common letters pairs are."""
        logger.debug(f"Вычисление скоринга для текста: '{plaintext[:30]}...'")

        s = 1.0
        bigram_list = bigrams(plaintext)
        logger.debug(f"Анализ {len(bigram_list)} биграмм")

        for bi in bigram_list:
            try:
                prob = self.P2[bi]
                s = s * prob
                logger.debug(f"Биграмма '{bi}': вероятность={prob}, текущий score={s}")
            except Exception as e:
                logger.warning(f"Ошибка при обработке биграммы '{bi}': {e}")
                # Используем вероятность по умолчанию для продолжения
                s = s * self.P2.default

        logger.debug(f"Итоговый score: {s}")
        return s

    @log_operation("ShiftDecoder.decode", log_args=True, log_result=True)
    def decode(self, ciphertext):
        """Return the shift decoding of text with the best score."""
        logger.info(f"Декодирование шифротекста: '{ciphertext[:30]}...'")

        if not ciphertext:
            logger.warning("Пустой шифротекст передан для декодирования")
            return ciphertext

        try:
            all_decodings = list(all_shifts(ciphertext))
            logger.debug(f"Сгенерировано {len(all_decodings)} вариантов декодирования")

            # Вычисляем скоринг для каждого варианта
            scored_decodings = []
            for i, decoding in enumerate(all_decodings):
                score = self.score(decoding)
                scored_decodings.append((score, decoding))

                if i % 5 == 0:  # Логируем каждые 5 вариантов
                    logger.debug(f"Вариант {i}: score={score:.10f}, text='{decoding[:20]}...'")

            # Выбираем лучший вариант
            best_score, best_decoding = max(scored_decodings, key=lambda x: x[0])
            logger.info(f"Найдено лучшее декодирование со score={best_score:.10f}")
            logger.debug(f"Лучший результат: '{best_decoding[:50]}...'")

            # Записываем метрики
            metrics.increment_counter("decoders.shift.decodings")
            metrics.record_metric("decoders.shift.best_score", best_score)

            return best_decoding

        except Exception as e:
            logger.error(f"Ошибка при декодировании шифротекста: {e}", exc_info=True)
            # Возвращаем оригинальный текст в случае ошибки
            return ciphertext


@log_operation("all_shifts", level='DEBUG', log_args=True)
def all_shifts(text):
    """Return a list of all 26 possible encodings of text by a shift cipher."""
    logger.debug(f"Генерация всех 26 сдвигов для текста: '{text[:20]}...'")

    for i, _ in enumerate(alphabet):
        try:
            shifted = shift_encode(text, i)
            logger.debug(f"Сдвиг {i}: '{shifted[:20]}...'")
            yield shifted
        except Exception as e:
            logger.error(f"Ошибка при генерации сдвига {i} для текста: {e}")
            yield text  # Возвращаем оригинальный текст в случае ошибки


# Decoding a General Permutation Cipher


class PermutationDecoder:
    """This is a much harder problem than the shift decoder. There are 26!
    permutations, so we can't try them all. Instead we have to search.
    We want to search well, but there are many things to consider:
    Unigram probabilities (E is the most common letter); Bigram probabilities
    (TH is the most common bigram); word probabilities (I and A are the most
    common one-letter words, etc.); etc.
    We could represent a search state as a permutation of the 26 letters,
    and alter the solution through hill climbing. With an initial guess
    based on unigram probabilities, this would probably fare well. However,
    I chose instead to have an incremental representation. A state is
    represented as a letter-to-letter map; for example {'z': 'e'} to
    represent that 'z' will be translated to 'e'."""

    def __init__(self, training_text, ciphertext=None):
        with log_context("PermutationDecoder.__init__"):
            logger.info("Создание PermutationDecoder")
            logger.debug(f"Длина тренировочного текста: {len(training_text)} символов")

            try:
                self.Pwords = UnigramWordModel(words(training_text))
                self.P1 = UnigramWordModel(training_text)  # By letter
                self.P2 = NgramWordModel(2, words(training_text))  # By letter pair

                logger.debug(f"PermutationDecoder создан. "
                            f"Pwords размер: {len(self.Pwords)}, "
                            f"P1 размер: {len(self.P1)}, "
                            f"P2 размер: {len(self.P2)}")

                if ciphertext:
                    logger.debug(f"Начальный шифротекст: '{ciphertext[:30]}...'")

                metrics.increment_counter("decoders.permutation.created")

            except Exception as e:
                logger.error(f"Ошибка при создании PermutationDecoder: {e}", exc_info=True)
                raise

    @log_operation("PermutationDecoder.decode", log_args=True, log_result=True)
    def decode(self, ciphertext):
        """Search for a decoding of the ciphertext."""
        logger.info(f"Начало декодирования PermutationCipher: '{ciphertext[:30]}...'")

        try:
            self.ciphertext = canonicalize(ciphertext)
            logger.debug(f"Канонизированный шифротекст: '{self.ciphertext[:30]}...'")

            # reduce domain to speed up search
            self.chardomain = {c for c in self.ciphertext if c != ' '}
            logger.debug(f"Домен символов для декодирования: {sorted(self.chardomain)}")

            problem = PermutationDecoderProblem(decoder=self)
            logger.debug("PermutationDecoderProblem создан")

            logger.info("Начало поиска наилучшей перестановки...")
            solution = search.best_first_graph_search(
                problem, lambda node: self.score(node.state))

            if not solution:
                logger.error("Поиск не нашел решения!")
                return ciphertext

            logger.debug(f"Решение найдено: {solution.state}")

            solution.state[' '] = ' '
            decoded = translate(self.ciphertext, lambda c: solution.state[c])

            logger.info(f"Декодирование завершено. Результат: '{decoded[:50]}...'")
            logger.debug(f"Полная таблица декодирования: {solution.state}")

            # Записываем метрики
            metrics.increment_counter("decoders.permutation.decodings")
            metrics.record_metric("decoders.permutation.domain_size", len(self.chardomain))

            return decoded

        except Exception as e:
            logger.error(f"Ошибка при декодировании PermutationCipher: {e}", exc_info=True)
            # Возвращаем оригинальный текст в случае ошибки
            return ciphertext

    @log_operation("PermutationDecoder.score", level='DEBUG', log_args=True, log_result=True)
    def score(self, code):
        """Score is product of word scores, unigram scores, and bigram scores.
        This can get very small, so we use logs and exp."""
        logger.debug(f"Вычисление скоринга для кода: {code}")

        try:
            # remake code dictionary to contain translation for all characters
            full_code = code.copy()
            full_code.update({x: x for x in self.chardomain if x not in code})
            full_code[' '] = ' '

            logger.debug(f"Полный код перевода: {full_code}")

            text = translate(self.ciphertext, lambda c: full_code[c])
            logger.debug(f"Переведенный текст: '{text[:30]}...'")

            # add small positive value to prevent computing log(0)
            word_log_sum = sum(np.log(self.Pwords[word] + 1e-20) for word in words(text))
            char_log_sum = sum(np.log(self.P1[c] + 1e-5) for c in text)
            bigram_log_sum = sum(np.log(self.P2[b] + 1e-10) for b in bigrams(text))

            total_log = word_log_sum + char_log_sum + bigram_log_sum

            logger.debug(f"Суммы логарифмов: "
                        f"words={word_log_sum:.4f}, "
                        f"chars={char_log_sum:.4f}, "
                        f"bigrams={bigram_log_sum:.4f}, "
                        f"total={total_log:.4f}")

            score = -np.exp(total_log)
            logger.debug(f"Итоговый score: {score}")

            return score

        except Exception as e:
            logger.error(f"Ошибка при вычислении скоринга: {e}")
            # Возвращаем очень плохой score в случае ошибки
            return -np.inf


class PermutationDecoderProblem(search.Problem):
    """Problem for searching the best permutation decoding."""

    def __init__(self, initial=None, goal=None, decoder=None):
        logger.debug(f"Инициализация PermutationDecoderProblem, decoder: {type(decoder).__name__}")
        super().__init__(initial or hashabledict(), goal)
        self.decoder = decoder

        if decoder and decoder.chardomain:
            logger.debug(f"Размер домена символов: {len(decoder.chardomain)}")

    @log_operation("PermutationDecoderProblem.actions", level='DEBUG', log_args=True, log_result=True)
    def actions(self, state):
        """Return possible actions from this state."""
        logger.debug(f"Получение действий для состояния: {state}")

        try:
            search_list = [c for c in self.decoder.chardomain if c not in state]
            target_list = [c for c in alphabet if c not in state.values()]

            logger.debug(f"Доступные символы для замены: {search_list}")
            logger.debug(f"Доступные целевые символы: {target_list}")

            if not search_list or not target_list:
                logger.debug("Нет доступных действий")
                return []

            # Find the best character to replace
            plain_char = max(search_list, key=lambda c: self.decoder.P1[c])
            logger.debug(f"Выбран символ для замены: '{plain_char}' "
                        f"(вероятность: {self.decoder.P1[plain_char]})")

            actions = []
            for cipher_char in target_list:
                actions.append((plain_char, cipher_char))

            logger.debug(f"Сгенерировано {len(actions)} действий")
            metrics.increment_counter("search.permutation.actions_generated", len(actions))
            return actions

        except Exception as e:
            logger.error(f"Ошибка при генерации действий: {e}")
            return []

    @log_operation("PermutationDecoderProblem.result", level='DEBUG', log_args=True, log_result=True)
    def result(self, state, action):
        """Return the state that results from executing the given action."""
        logger.debug(f"Применение действия {action} к состоянию {state}")

        try:
            new_state = hashabledict(state)  # copy to prevent hash issues
            new_state[action[0]] = action[1]
            logger.debug(f"Новое состояние: {new_state}")
            return new_state
        except Exception as e:
            logger.error(f"Ошибка при применении действия: {e}")
            return state

    @log_operation("PermutationDecoderProblem.goal_test", level='DEBUG', log_args=True, log_result=True)
    def goal_test(self, state):
        """We're done when all letters in search domain are assigned."""
        is_goal = len(state) >= len(self.decoder.chardomain)
        logger.debug(f"Проверка цели: состояние имеет {len(state)} назначений, "
                    f"требуется {len(self.decoder.chardomain)}. Цель достигнута: {is_goal}")
        return is_goal


# =============================================================================
# ТЕСТИРОВАНИЕ, МОНИТОРИНГ И ТОЧКА ВХОДА
# =============================================================================

@log_operation("test_decoders", log_args=False)
def test_decoders():
    """Функция для тестирования декодеров с логированием."""
    logger.info("Начало тестирования декодеров")

    # Тестовый текст для тренировки
    training_text = "the quick brown fox jumps over the lazy dog. hello world this is a test."

    # Тестовый шифротекст
    ciphertext = "uryyb jbeyq"

    with log_context("test_decoders_execution"):
        try:
            # Тестируем ShiftDecoder
            logger.info("Тестирование ShiftDecoder...")
            shift_decoder = ShiftDecoder(training_text)
            shift_result = shift_decoder.decode(ciphertext)
            logger.info(f"ShiftDecoder результат: '{shift_result}'")

            # Тестируем PermutationDecoder
            logger.info("Тестирование PermutationDecoder...")
            perm_decoder = PermutationDecoder(training_text)
            perm_result = perm_decoder.decode(ciphertext)
            logger.info(f"PermutationDecoder результат: '{perm_result}'")

            logger.info("Тестирование декодеров завершено")

        except Exception as e:
            logger.error(f"Ошибка при тестировании декодеров: {e}", exc_info=True)


@log_operation("run_demo", log_args=False)
def run_demo():
    """Запуск демонстрации возможностей системы"""
    logger.info("Запуск демонстрации модуля text.py")

    with log_context("demo_execution"):
        try:
            # Простой тест моделей
            logger.info("Создание тестовой UnigramWordModel...")
            test_model = UnigramWordModel(["hello", "world", "hello", "test"])
            logger.info(f"Тестовая модель создана. Пример генерации: {test_model.samples(3)}")

            # Тестируем декодеры
            test_decoders()

            logger.info("Все тесты завершены успешно")

            # Логируем итоговые метрики
            metrics_report = metrics.get_metrics_report()
            logger.info("Метрики работы системы:")
            for counter_name, count in metrics_report['counters'].items():
                logger.info(f"  {counter_name}: {count}")

            # Также логируем в JSON формате для последующего анализа
            metrics_logger.info("Итоговые метрики", extra=metrics_report)

        except Exception as e:
            logger.critical(f"Критическая ошибка при выполнении тестов: {e}", exc_info=True)


def log_system_info():
    """Логирование информации о системе"""
    logger.info("=" * 60)
    logger.info("ИНФОРМАЦИЯ О СИСТЕМЕ")
    logger.info("=" * 60)

    # Информация о Python
    logger.info(f"Python версия: {sys.version}")
    logger.info(f"Python исполняемый файл: {sys.executable}")

    # Информация о платформе
    logger.info(f"Платформа: {sys.platform}")

    # Информация о путях
    logger.info(f"Рабочая директория: {os.getcwd()}")

    # Информация о NumPy
    logger.info(f"NumPy версия: {np.__version__}")

    # Информация о логировании
    logger.info("Конфигурация логирования:")
    for handler in logger.handlers:
        logger.info(f"  Обработчик: {type(handler).__name__}, уровень: {logging.getLevelName(handler.level)}")

    logger.info("=" * 60)


class HealthCheck:
    """Класс для проверки здоровья системы"""

    def __init__(self):
        self.checks = []
        self.last_check = None

    def add_check(self, name: str, check_func: Callable[[], bool],
                 critical: bool = False):
        """Добавление проверки"""
        self.checks.append({
            'name': name,
            'func': check_func,
            'critical': critical
        })

    def run_checks(self) -> Dict[str, Dict[str, Any]]:
        """Запуск всех проверок"""
        logger.info("Запуск проверок здоровья системы")

        results = {}
        all_passed = True

        for check in self.checks:
            try:
                passed = check['func']()
                results[check['name']] = {
                    'passed': passed,
                    'critical': check['critical']
                }

                if passed:
                    logger.info(f"Проверка '{check['name']}': ПРОЙДЕНА")
                else:
                    logger.warning(f"Проверка '{check['name']}': НЕ ПРОЙДЕНА")
                    if check['critical']:
                        all_passed = False

            except Exception as e:
                logger.error(f"Ошибка при выполнении проверки '{check['name']}': {e}")
                results[check['name']] = {
                    'passed': False,
                    'critical': check['critical'],
                    'error': str(e)
                }
                if check['critical']:
                    all_passed = False

        self.last_check = datetime.now()

        if all_passed:
            logger.info("Все проверки здоровья пройдены успешно")
        else:
            logger.error("Некоторые критические проверки здоровья не пройдены")

        return results


if __name__ == "__main__":
    try:
        # Логируем информацию о системе
        log_system_info()

        # Регистрируем основные логгеры в менеджере уровней
        level_manager.register_logger(logger, 'INFO', 'Основной логгер системы')
        level_manager.register_logger(logging.getLogger('text_processing.models'),
                                      'INFO', 'Логгер моделей языка')
        level_manager.register_logger(logging.getLogger('text_processing.ir'),
                                      'INFO', 'Логгер информационно-поисковой системы')
        level_manager.register_logger(logging.getLogger('text_processing.decoders'),
                                      'INFO', 'Логгер декодеров шифров')

        # Настраиваем HealthCheck
        health_check = HealthCheck()


        def check_logging():
            """Проверка работоспособности логирования"""
            test_logger = logging.getLogger('health_check')
            test_logger.debug("Тестовое сообщение DEBUG")
            test_logger.info("Тестовое сообщение INFO")
            test_logger.warning("Тестовое сообщение WARNING")
            return True


        def check_numpy():
            """Проверка работоспособности NumPy"""
            try:
                arr = np.array([1, 2, 3])
                return len(arr) == 3
            except:
                return False


        health_check.add_check('Логирование', check_logging, critical=True)
        health_check.add_check('NumPy', check_numpy, critical=True)

        # Запускаем проверки здоровья
        health_results = health_check.run_checks()

        logger.info(f"Старт работы системы в {datetime.now()}")
        logger.debug(f"Аргументы командной строки: {sys.argv}")

        # Запускаем демо
        run_demo()

        logger.info(f"Система завершила работу в {datetime.now()}")

        # Финализируем метрики
        final_metrics = metrics.get_metrics_report()
        logger.info("Финальные метрики системы:")
        for key, value in final_metrics['counters'].items():
            logger.info(f"  {key}: {value}")

        # Также убедимся, что все логи записаны в файлы
        logging.shutdown()

    except KeyboardInterrupt:
        logger.info("Работа прервана пользователем (Ctrl+C)")
    except Exception as e:
        logger.critical(f"Непредвиденная ошибка при запуске системы: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Сохраняем финальные метрики в файл
        try:
            metrics_file = Path('logs') / 'final_metrics.json'
            final_report = metrics.get_metrics_report()
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(final_report, f, indent=2, ensure_ascii=False)
            logger.info(f"Финальные метрики сохранены в {metrics_file}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении метрик: {e}")