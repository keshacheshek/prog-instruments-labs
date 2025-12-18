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
import json
import time
import traceback
import sys
from datetime import datetime
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps

import numpy as np

import search
from probabilistic_learning import CountingProbDist
from utils import hashabledict


# =============================================================================
# РАСШИРЕННАЯ КОНФИГУРАЦИЯ ЛОГИРОВАНИЯ ДЛЯ ПЯТОГО КОММИТА
# =============================================================================

def load_logging_config(config_file='logging_config.json'):
    """Загрузка конфигурации логирования из JSON файла"""
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
                'format': '{"timestamp": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "file": "%(filename)s", "line": %(lineno)d, "message": "%(message)s"}',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'simple',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'filename': 'text_processing.log',
                'maxBytes': 10485760,
                'backupCount': 5,
                'encoding': 'utf-8'
            },
            'error_file': {
                'class': 'logging.FileHandler',
                'level': 'WARNING',
                'formatter': 'detailed',
                'filename': 'text_processing_errors.log',
                'encoding': 'utf-8'
            },
            'json_file': {
                'class': 'logging.FileHandler',
                'level': 'INFO',
                'formatter': 'json',
                'filename': 'text_processing_json.log',
                'encoding': 'utf-8'
            }
        },
        'loggers': {
            'text_processing': {
                'level': 'DEBUG',
                'handlers': ['console', 'file', 'error_file', 'json_file'],
                'propagate': False
            }
        },
        'root': {
            'level': 'WARNING',
            'handlers': ['console']
        }
    }

    try:
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logging.config.dictConfig(config)
                logger = logging.getLogger('text_processing')
                logger.info(f"Конфигурация логирования загружена из файла: {config_file}")
                return logger
        else:
            # Создаем файл конфигурации
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2)
            logging.config.dictConfig(default_config)
            logger = logging.getLogger('text_processing')
            logger.info(f"Создан файл конфигурации по умолчанию: {config_file}")
            return logger
    except Exception as e:
        # Если что-то пошло не так, используем базовую конфигурацию
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logger = logging.getLogger('text_processing')
        logger.warning(f"Не удалось загрузить конфигурацию из {config_file}: {e}. Используется базовая конфигурация.")
        return logger


# Инициализируем логгер
logger = load_logging_config()

# Дополнительные утилиты логирования
class PerformanceLogger:
    """Класс для логирования производительности"""

    def __init__(self, name="performance"):
        self.logger = logging.getLogger(f"text_processing.{name}")
        self.timers = {}
        self.counters = {}

    def start_timer(self, operation_name):
        """Начать замер времени для операции"""
        self.timers[operation_name] = time.time()
        self.logger.debug(f"Таймер запущен для операции: {operation_name}")

    def stop_timer(self, operation_name):
        """Остановить таймер и залогировать время"""
        if operation_name in self.timers:
            elapsed = time.time() - self.timers[operation_name]
            self.logger.info(f"Операция '{operation_name}' заняла {elapsed:.4f} секунд")
            del self.timers[operation_name]
            return elapsed
        return 0.0

    def increment_counter(self, counter_name, amount=1):
        """Увеличить счетчик"""
        if counter_name not in self.counters:
            self.counters[counter_name] = 0
        self.counters[counter_name] += amount
        self.logger.debug(f"Счетчик '{counter_name}' увеличен на {amount}. Текущее значение: {self.counters[counter_name]}")

    def get_counter(self, counter_name):
        """Получить значение счетчика"""
        return self.counters.get(counter_name, 0)

    def log_metrics(self):
        """Залогировать все метрики"""
        if self.timers:
            self.logger.warning(f"Активные таймеры: {list(self.timers.keys())}")

        if self.counters:
            metrics_str = ", ".join(f"{k}: {v}" for k, v in self.counters.items())
            self.logger.info(f"Метрики: {metrics_str}")


# Глобальный логгер производительности
perf_logger = PerformanceLogger()


@contextmanager
def log_execution_time(operation_name):
    """Контекстный менеджер для логирования времени выполнения"""
    perf_logger.start_timer(operation_name)
    try:
        yield
    finally:
        perf_logger.stop_timer(operation_name)


def log_exceptions(func):
    """Декоратор для логирования исключений"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Исключение в функции {func.__name__}: {e}", exc_info=True)
            raise
    return wrapper


def setup_module_logger(module_name):
    """Настройка логгера для конкретного модуля"""
    module_logger = logging.getLogger(f"text_processing.{module_name}")
    return module_logger


# Логгеры для разных компонентов системы
models_logger = setup_module_logger("models")
ir_logger = setup_module_logger("ir")
decoders_logger = setup_module_logger("decoders")
utils_logger = setup_module_logger("utils")


# =============================================================================
# МОДЕЛИ ЯЗЫКА
# =============================================================================

class UnigramWordModel(CountingProbDist):
    """This is a discrete probability distribution over words, so you
    can add, sample, or get P[word], just like with CountingProbDist. You can
    also generate a random text, n words long, with P.samples(n)."""

    def __init__(self, observations, default=0):
        models_logger.info(f"Создание UnigramWordModel с {len(observations) if observations else 0} наблюдениями")
        models_logger.debug(f"Параметр default: {default}")

        with log_execution_time("UnigramWordModel.__init__"):
            # Call CountingProbDist constructor,
            # passing the observations and default parameters.
            super(UnigramWordModel, self).__init__(observations, default)

            unique_words = len(self) if hasattr(self, '__len__') else 'unknown'
            models_logger.debug(f"UnigramWordModel создан успешно. Уникальных слов: {unique_words}")

    @log_exceptions
    def samples(self, n):
        """Return a string of n words, random according to the model."""
        models_logger.info(f"Генерация {n} слов с помощью UnigramWordModel")

        if n <= 0:
            models_logger.warning(f"Запрошена генерация {n} слов. Возвращаем пустую строку.")
            return ''

        with log_execution_time(f"UnigramWordModel.samples({n})"):
            words = []
            for i in range(n):
                word = self.sample()
                words.append(word)
                if i % 20 == 0 and i > 0:  # Логируем каждые 20 слов
                    models_logger.debug(f"Сгенерировано {i+1}/{n} слов")

            result = ' '.join(words)
            models_logger.debug(f"Сгенерировано {n} слов. Первые 5: {words[:5]}")
            return result

    def __getitem__(self, word):
        """Override to add logging for probability queries"""
        try:
            prob = super().__getitem__(word)
            models_logger.debug(f"Вероятность слова '{word}': {prob}")
            return prob
        except KeyError as e:
            models_logger.warning(f"Слово '{word}' не найдено в модели. Возвращаем вероятность по умолчанию.")
            return self.default


class NgramWordModel(CountingProbDist):
    """This is a discrete probability distribution over n-tuples of words.
    You can add, sample or get P[(word1, ..., wordn)]. The method P.samples(n)
    builds up an n-word sequence; P.add_cond_prob and P.add_sequence add data."""

    def __init__(self, n, observation_sequence=None, default=0):
        if n <= 0:
            models_logger.error(f"Попытка создать NgramWordModel с n={n}. n должно быть > 0.")
            raise ValueError("n must be > 0")

        models_logger.info(f"Создание NgramWordModel с n={n}")
        if observation_sequence:
            models_logger.debug(f"Начальная последовательность: {len(observation_sequence)} элементов")

        with log_execution_time("NgramWordModel.__init__"):
            # In addition to the dictionary of n-tuples, cond_prob is a
            # mapping from (w1, ..., wn-1) to P(wn | w1, ... wn-1)
            CountingProbDist.__init__(self, default=default)
            self.n = n
            self.cond_prob = defaultdict()
            self.add_sequence(observation_sequence or [])

            cond_prob_size = len(self.cond_prob)
            models_logger.debug(f"NgramWordModel создан успешно. Условных распределений: {cond_prob_size}")

    def __getitem__(self, ngram):
        """Override to add logging for ngram probability queries"""
        try:
            prob = super().__getitem__(ngram)
            models_logger.debug(f"Вероятность n-граммы {ngram}: {prob}")
            return prob
        except KeyError as e:
            models_logger.debug(f"N-грамма {ngram} не найдена. Возвращаем вероятность по умолчанию: {self.default}")
            return self.default

    @log_exceptions
    def add_cond_prob(self, ngram):
        """Build the conditional probabilities P(wn | (w1, ..., wn-1)"""
        models_logger.debug(f"Добавление условной вероятности для n-граммы: {ngram}")

        if len(ngram) != self.n:
            models_logger.error(f"Некорректная длина n-граммы: {len(ngram)} (ожидается {self.n})")
            raise ValueError(f"ngram must have length {self.n}")

        prefix = ngram[:-1]
        if prefix not in self.cond_prob:
            models_logger.debug(f"Создание нового условного распределения для префикса: {prefix}")
            self.cond_prob[prefix] = CountingProbDist()

        self.cond_prob[prefix].add(ngram[-1])
        models_logger.debug(f"Добавлено слово '{ngram[-1]}' для префикса {prefix}")

    @log_exceptions
    def add_sequence(self, words):
        """Add each tuple words[i:i+n], using a sliding window."""
        n = self.n

        if len(words) < n:
            models_logger.warning(f"Последовательность слишком короткая ({len(words)} слов) для n={n}")
            return

        models_logger.info(f"Добавление последовательности из {len(words)} слов в NgramWordModel (n={n})")

        with log_execution_time(f"NgramWordModel.add_sequence({len(words)})"):
            ngram_count = 0
            for i in range(len(words) - n + 1):
                t = tuple(words[i:i + n])
                self.add(t)
                self.add_cond_prob(t)
                ngram_count += 1

                if ngram_count % 100 == 0:  # Логируем каждые 100 n-грамм
                    models_logger.debug(f"Обработано {ngram_count} n-грамм")

            models_logger.info(f"Добавлено {ngram_count} n-грамм из {len(words)} слов")
            perf_logger.increment_counter("ngrams_added", ngram_count)

    @log_exceptions
    def samples(self, nwords):
        """Generate an n-word sentence by picking random samples
        according to the model. At first pick a random n-gram and
        from then on keep picking a character according to
        P(c|wl-1, wl-2, ..., wl-n+1) where wl-1 ... wl-n+1 are the
        last n - 1 words in the generated sentence so far."""
        models_logger.info(f"Генерация {nwords} слов с помощью NgramWordModel (n={self.n})")

        if nwords < self.n:
            models_logger.warning(f"Запрошено {nwords} слов, но n={self.n}. Будет сгенерировано {self.n} слов.")
            nwords = self.n

        with log_execution_time(f"NgramWordModel.samples({nwords})"):
            n = self.n
            output = list(self.sample())
            models_logger.debug(f"Начальная n-грамма: {output}")

            for i in range(n, nwords):
                last = output[-n + 1:]
                prefix_tuple = tuple(last)

                if prefix_tuple not in self.cond_prob:
                    models_logger.warning(f"Префикс {prefix_tuple} не найден в условных распределениях. Начинаем новое предложение.")
                    output.extend(self.sample())
                    continue

                next_word = self.cond_prob[prefix_tuple].sample()
                output.append(next_word)

                if (i + 1) % 10 == 0:  # Логируем каждые 10 слов
                    models_logger.debug(f"Сгенерировано {i+1}/{nwords} слов. Текущее слово: '{next_word}'")

            result = ' '.join(output)
            models_logger.info(f"Генерация завершена. Получено {len(output)} слов")
            models_logger.debug(f"Пример сгенерированного текста (первые 50 символов): {result[:50]}...")
            perf_logger.increment_counter("words_generated", len(output))
            return result


class NgramCharModel(NgramWordModel):
    @log_exceptions
    def add_sequence(self, words):
        """Add an empty space to every word to catch the beginning of words."""
        models_logger.debug(f"Добавление последовательности в NgramCharModel: {len(words)} слов")

        if not words:
            models_logger.warning("Пустая последовательность слов передана в NgramCharModel")
            return

        total_chars = sum(len(word) + 1 for word in words)  # +1 для пробела
        models_logger.debug(f"Общее количество символов для обработки: {total_chars}")

        with log_execution_time(f"NgramCharModel.add_sequence({len(words)})"):
            for word_idx, word in enumerate(words):
                processed_word = ' ' + word
                super().add_sequence(processed_word)

                if word_idx % 50 == 0 and word_idx > 0:
                    models_logger.debug(f"Обработано {word_idx+1}/{len(words)} слов")


class UnigramCharModel(NgramCharModel):
    def __init__(self, observation_sequence=None, default=0):
        models_logger.info("Создание UnigramCharModel")
        with log_execution_time("UnigramCharModel.__init__"):
            CountingProbDist.__init__(self, default=default)
            self.n = 1
            self.cond_prob = defaultdict()
            self.add_sequence(observation_sequence or [])

            total_chars = len(self) if hasattr(self, '__len__') else 'unknown'
            models_logger.debug(f"UnigramCharModel создан успешно. Уникальных символов: {total_chars}")

    @log_exceptions
    def add_sequence(self, words):
        models_logger.info(f"Добавление последовательности в UnigramCharModel: {len(words)} слов")

        with log_execution_time(f"UnigramCharModel.add_sequence({len(words)})"):
            char_count = 0
            for word_idx, word in enumerate(words):
                if not word:
                    continue

                for char_idx, char in enumerate(word):
                    self.add(char)
                    char_count += 1

                    if char_count % 1000 == 0:
                        models_logger.debug(f"Обработано {char_count} символов")

                if word_idx % 100 == 0 and word_idx > 0:
                    models_logger.debug(f"Обработано {word_idx+1}/{len(words)} слов")

            models_logger.info(f"Добавлено {char_count} символов из {len(words)} слов")
            perf_logger.increment_counter("chars_processed", char_count)


# =============================================================================
# АЛГОРИТМ ВИТЕРБИ
# =============================================================================

@log_exceptions
def viterbi_segment(text, P):
    """Find the best segmentation of the string of characters, given the
    UnigramWordModel P."""
    models_logger.info(f"Запуск алгоритма Витерби для текста длиной {len(text)} символов")
    models_logger.debug(f"Текст: '{text[:50]}...' если длиннее 50 символов")

    if not text:
        models_logger.warning("Пустой текст передан в viterbi_segment")
        return [], 0.0

    with log_execution_time(f"viterbi_segment({len(text)})"):
        # best[i] = best probability for text[0:i]
        # words[i] = best word ending at position i
        n = len(text)
        words = [''] + list(text)
        best = [1.0] + [0.0] * n

        models_logger.debug(f"Инициализирован массив best размером {len(best)}")
        models_logger.debug(f"Используется модель P: {type(P).__name__}")

        # Fill in the vectors best words via dynamic programming
        progress_interval = max(1, n // 10)  # Логируем каждые 10%
        for i in range(1, n + 1):
            if i % progress_interval == 0:
                models_logger.debug(f"Обработано {i}/{n} позиций ({i/n*100:.1f}%)")

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
                    models_logger.error(f"Ошибка при обработке слова '{w}': {e}")
                    continue

            if best_candidate:
                best[i] = best_prob
                words[i] = best_candidate
                models_logger.debug(f"Позиция {i}: лучшее слово '{best_candidate}' с вероятностью {best_prob:.6f}")
            else:
                models_logger.warning(f"Не найдено подходящее слово для позиции {i}")
                best[i] = 0.0
                words[i] = ''

        models_logger.debug("Динамическое программирование завершено")

        # Now recover the sequence of best words
        sequence = []
        i = len(words) - 1
        step_count = 0

        while i > 0:
            if step_count > n:  # Защита от бесконечного цикла
                models_logger.error(f"Превышено максимальное количество шагов ({n}) при восстановлении последовательности")
                break

            current_word = words[i]
            if not current_word:
                models_logger.error(f"Пустое слово на позиции {i}. Прерывание восстановления.")
                break

            sequence.insert(0, current_word)
            i = i - len(current_word)
            step_count += 1

            models_logger.debug(f"Шаг {step_count}: слово '{current_word}', переходим к позиции {i}")

        final_prob = best[-1]
        models_logger.info(f"Алгоритм Витерби завершен. Найдено {len(sequence)} слов с вероятностью {final_prob:.10f}")
        models_logger.debug(f"Результат сегментации: {' '.join(sequence)}")

        if final_prob == 0.0:
            models_logger.warning("Итоговая вероятность равна 0. Возможно, текст не может быть корректно сегментирован.")

        perf_logger.increment_counter("viterbi_segmentations")
        perf_logger.increment_counter("viterbi_words_segmented", len(sequence))

        # Return sequence of best words and overall probability
        return sequence, final_prob


# =============================================================================
# ИНФОРМАЦИОННО-ПОИСКОВАЯ СИСТЕМА
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
        ir_logger.info(f"Создание IRSystem со стоп-словами: {stopwords}")

        with log_execution_time("IRSystem.__init__"):
            # index is a map of {word: {docid: count}}, where docid is an int,
            # indicating the index into the documents list.
            self.index = defaultdict(lambda: defaultdict(int))
            self.stopwords = set(words(stopwords))
            self.documents = []

            ir_logger.debug(f"Инициализирован индекс. Стоп-слов: {len(self.stopwords)}")
            ir_logger.debug(f"Стоп-слова: {sorted(self.stopwords)}")

    @log_exceptions
    def index_collection(self, filenames):
        """Index a whole collection of files."""
        ir_logger.info(f"Индексация коллекции из {len(filenames)} файлов")

        with log_execution_time(f"IRSystem.index_collection({len(filenames)})"):
            prefix = os.path.dirname(__file__)
            for idx, filename in enumerate(filenames):
                try:
                    ir_logger.debug(f"Индексация файла {idx+1}/{len(filenames)}: {filename}")

                    if not os.path.exists(filename):
                        ir_logger.error(f"Файл не найден: {filename}")
                        continue

                    with open(filename, 'r', encoding='utf-8') as file:
                        content = file.read()

                    rel_path = os.path.relpath(filename, prefix)
                    self.index_document(content, rel_path)

                    ir_logger.debug(f"Файл {filename} успешно проиндексирован")

                except Exception as e:
                    ir_logger.error(f"Ошибка при индексации файла {filename}: {e}", exc_info=True)

            ir_logger.info(f"Коллекция проиндексирована. Всего документов: {len(self.documents)}")
            perf_logger.increment_counter("documents_indexed", len(filenames))

    @log_exceptions
    def index_document(self, text, url):
        """Index the text of a document."""
        ir_logger.info(f"Индексация документа: {url}")

        with log_execution_time(f"IRSystem.index_document({len(text)} chars)"):
            try:
                # For now, use first line for title
                if '\n' in text:
                    title = text[:text.index('\n')].strip()
                else:
                    title = text[:100].strip() + "..." if len(text) > 100 else text.strip()
                    ir_logger.warning(f"Документ {url} не содержит символов новой строки")

                docwords = words(text)
                docid = len(self.documents)

                ir_logger.debug(f"Документ {docid}: заголовок='{title}', количество слов={len(docwords)}")

                self.documents.append(Document(title, url, len(docwords)))

                indexed_words = 0
                stopwords_count = 0

                for word in docwords:
                    if word not in self.stopwords:
                        self.index[word][docid] += 1
                        indexed_words += 1
                    else:
                        stopwords_count += 1

                ir_logger.info(f"Документ {docid} проиндексирован. "
                            f"Индексировано слов: {indexed_words}, "
                            f"стоп-слов: {stopwords_count}, "
                            f"всего слов: {len(docwords)}")

                # Логируем статистику по уникальным словам
                unique_indexed_words = sum(1 for word in docwords if word not in self.stopwords)
                ir_logger.debug(f"Уникальных индексированных слов: {unique_indexed_words}")

                perf_logger.increment_counter("words_indexed", indexed_words)

            except Exception as e:
                ir_logger.error(f"Ошибка при индексации документа {url}: {e}", exc_info=True)
                raise

    @log_exceptions
    def query(self, query_text, n=10):
        """Return a list of n (score, docid) pairs for the best matches.
        Also handle the special syntax for 'learn: command'."""
        ir_logger.info(f"Выполнение запроса: '{query_text}' (n={n})")

        with log_execution_time(f"IRSystem.query('{query_text[:20]}...')"):
            if query_text.startswith("learn:"):
                ir_logger.info(f"Обнаружен специальный синтаксис 'learn:' в запросе")
                command = query_text[len("learn:"):].strip()
                ir_logger.debug(f"Выполнение команды: {command}")

                try:
                    doctext = os.popen(command, 'r').read()
                    ir_logger.info(f"Команда выполнена успешно, получено {len(doctext)} символов")
                    self.index_document(doctext, query_text)
                    return []
                except Exception as e:
                    ir_logger.error(f"Ошибка при выполнении команды '{command}': {e}")
                    return []

            # Обработка обычного запроса
            qwords = [w for w in words(query_text) if w not in self.stopwords]
            ir_logger.debug(f"Слова запроса после фильтрации стоп-слов: {qwords}")

            if not qwords:
                ir_logger.warning("После фильтрации стоп-слов не осталось слов в запросе")
                return []

            # Находим слово с наименьшим количеством документов для оптимизации
            try:
                shortest = min(qwords, key=lambda w: len(self.index[w]))
                ir_logger.debug(f"Слово с наименьшим количеством документов: '{shortest}' "
                            f"(документов: {len(self.index[shortest])})")
            except ValueError as e:
                ir_logger.error(f"Ошибка при поиске самого короткого слова: {e}")
                return []

            docids = self.index[shortest]
            ir_logger.debug(f"Найдено {len(docids)} документов для слова '{shortest}'")

            # Вычисляем скоринг для каждого документа
            scored_docs = []
            for docid in docids:
                try:
                    score = self.total_score(qwords, docid)
                    scored_docs.append((score, docid))
                except Exception as e:
                    ir_logger.error(f"Ошибка при подсчете очков для документа {docid}: {e}")
                    continue

            ir_logger.info(f"Вычислены очки для {len(scored_docs)} документов")

            # Возвращаем топ-n результатов
            results = heapq.nlargest(n, scored_docs)
            ir_logger.info(f"Возвращено {len(results)} лучших результатов")

            if results:
                best_score, best_docid = results[0]
                ir_logger.debug(f"Лучший результат: документ {best_docid} с очками {best_score:.4f}")

            perf_logger.increment_counter("queries_executed")
            perf_logger.increment_counter("documents_scored", len(scored_docs))

            return results

    @log_exceptions
    def score(self, word, docid):
        """Compute a score for this word on the document with this docid."""
        try:
            word_count = self.index[word][docid]
            doc_nwords = self.documents[docid].nwords

            if doc_nwords == 0:
                ir_logger.warning(f"Документ {docid} имеет 0 слов. Возвращаем 0.")
                return 0.0

            score = np.log(1 + word_count) / np.log(1 + doc_nwords)
            ir_logger.debug(f"Очки для слова '{word}' в документе {docid}: "
                        f"count={word_count}, nwords={doc_nwords}, score={score:.4f}")
            return score

        except (KeyError, IndexError) as e:
            ir_logger.warning(f"Слово '{word}' не найдено в документе {docid}: {e}")
            return 0.0
        except Exception as e:
            ir_logger.error(f"Ошибка при вычислении очков для слова '{word}' в документе {docid}: {e}")
            return 0.0

    @log_exceptions
    def total_score(self, words, docid):
        """Compute the sum of the scores of these words on the document with this docid."""
        ir_logger.debug(f"Вычисление суммарных очков для документа {docid}, слова: {words}")

        total = 0.0
        for word in words:
            word_score = self.score(word, docid)
            total += word_score
            ir_logger.debug(f"Слово '{word}': {word_score:.4f}, сумма: {total:.4f}")

        ir_logger.debug(f"Итоговые очки для документа {docid}: {total:.4f}")
        return total

    @log_exceptions
    def present(self, results):
        """Present the results as a list."""
        ir_logger.info(f"Отображение {len(results)} результатов")

        if not results:
            ir_logger.warning("Нет результатов для отображения")
            print("No results found.")
            return

        print(f"\n{'Score':<7} | {'URL':<25} | {'Title'}")
        print("-" * 70)

        for idx, (score, docid) in enumerate(results):
            try:
                doc = self.documents[docid]
                score_percent = 100 * score
                print(f"{score_percent:5.2f}% | {doc.url:<25} | {doc.title[:45].expandtabs()}")
                ir_logger.debug(f"Результат {idx+1}: docid={docid}, score={score:.4f}, "
                            f"title='{doc.title[:30]}...'")
            except IndexError as e:
                ir_logger.error(f"Ошибка при отображении результата {idx}: документ {docid} не найден")
                print(f"{'ERROR':<7} | {'N/A':<25} | Document {docid} not found")
            except Exception as e:
                ir_logger.error(f"Ошибка при отображении результата {idx}: {e}")
                print(f"{'ERROR':<7} | {'N/A':<25} | Error displaying document")

    @log_exceptions
    def present_results(self, query_text, n=10):
        """Get results for the query and present them."""
        ir_logger.info(f"Выполнение и отображение результатов для запроса: '{query_text}'")

        try:
            results = self.query(query_text, n)
            self.present(results)
            ir_logger.info("Результаты успешно отображены")
        except Exception as e:
            ir_logger.error(f"Ошибка при выполнении запроса '{query_text}': {e}", exc_info=True)
            print(f"Error processing query: {e}")


class UnixConsultant(IRSystem):
    """A trivial IR system over a small collection of Unix man pages."""

    def __init__(self):
        ir_logger.info("Создание UnixConsultant (специализированная IR система для man-страниц)")

        with log_execution_time("UnixConsultant.__init__"):
            IRSystem.__init__(self, stopwords="how do i the a of")

            import os
            aima_root = os.path.dirname(__file__)
            mandir = os.path.join(aima_root, 'aima-data/MAN/')

            ir_logger.debug(f"Поиск man-страниц в директории: {mandir}")

            try:
                if not os.path.exists(mandir):
                    ir_logger.error(f"Директория с man-страницами не найдена: {mandir}")
                    raise FileNotFoundError(f"Directory not found: {mandir}")

                man_files = [os.path.join(mandir, f) for f in os.listdir(mandir) if f.endswith('.txt')]
                ir_logger.info(f"Найдено {len(man_files)} man-страниц в директории {mandir}")

                if not man_files:
                    ir_logger.warning("Не найдено ни одного файла с man-страницами")

                self.index_collection(man_files)
                ir_logger.info(f"UnixConsultant создан успешно. Проиндексировано {len(self.documents)} документов")

            except Exception as e:
                ir_logger.error(f"Ошибка при создании UnixConsultant: {e}", exc_info=True)
                raise


class Document:
    """Metadata for a document: title and url; maybe add others later."""

    def __init__(self, title, url, nwords):
        ir_logger.debug(f"Создание документа: title='{title[:30]}...', url={url}, nwords={nwords}")
        self.title = title
        self.url = url
        self.nwords = nwords

    def __repr__(self):
        return f"Document(title='{self.title[:20]}...', url='{self.url}', nwords={self.nwords})"


# =============================================================================
# УТИЛИТЫ
# =============================================================================

@log_exceptions
def words(text, reg=re.compile('[a-z0-9]+')):
    """Return a list of the words in text, ignoring punctuation and
    converting everything to lowercase (to canonicalize).
    >>> words("``EGAD!'' Edgar cried.")
    ['egad', 'edgar', 'cried']
    """
    try:
        result = reg.findall(text.lower())
        utils_logger.debug(f"Извлечено {len(result)} слов из текста (первые 5: {result[:5]})")
        perf_logger.increment_counter("words_extracted", len(result))
        return result
    except Exception as e:
        utils_logger.error(f"Ошибка при извлечении слов из текста: {e}")
        return []


@log_exceptions
def canonicalize(text):
    """Return a canonical text: only lowercase letters and blanks.
    >>> canonicalize("``EGAD!'' Edgar cried.")
    'egad edgar cried'
    """
    try:
        result = ' '.join(words(text))
        utils_logger.debug(f"Канонизированный текст (первые 50 символов): '{result[:50]}...'")
        return result
    except Exception as e:
        utils_logger.error(f"Ошибка при канонизации текста: {e}")
        return ""


# =============================================================================
# ШИФРЫ
# =============================================================================

# Example application (not in book): decode a cipher.
# A cipher is a code that substitutes one character for another.
# A shift cipher is a rotation of the letters in the alphabet,
# such as the famous rot13, which maps A to N, B to M, etc.

alphabet = 'abcdefghijklmnopqrstuvwxyz'


# Encoding

@log_exceptions
def shift_encode(plaintext, n):
    """Encode text with a shift cipher that moves each letter up by n letters.
    >>> shift_encode('abc z', 1)
    'bcd a'
    """
    utils_logger.debug(f"Шифрование сдвигом (n={n}): '{plaintext[:20]}...'")
    result = encode(plaintext, alphabet[n:] + alphabet[:n])
    utils_logger.debug(f"Результат шифрования: '{result[:20]}...'")
    return result


@log_exceptions
def rot13(plaintext):
    """Encode text by rotating letters by 13 spaces in the alphabet.
    >>> rot13('hello')
    'uryyb'
    >>> rot13(rot13('hello'))
    'hello'
    """
    utils_logger.debug(f"ROT13 шифрование: '{plaintext[:20]}...'")
    result = shift_encode(plaintext, 13)
    utils_logger.debug(f"Результат ROT13: '{result[:20]}...'")
    return result


@log_exceptions
def translate(plaintext, function):
    """Translate chars of a plaintext with the given function."""
    utils_logger.debug(f"Перевод текста (длина={len(plaintext)})")
    result = ""
    for char in plaintext:
        result += function(char)
    utils_logger.debug(f"Перевод завершен. Результат: '{result[:20]}...'")
    return result


@log_exceptions
def maketrans(from_, to_):
    """Create a translation table and return the proper function."""
    utils_logger.debug(f"Создание таблицы перевода: from_='{from_}', to_='{to_}'")
    trans_table = {}
    for n, char in enumerate(from_):
        trans_table[char] = to_[n]

    utils_logger.debug(f"Таблица перевода создана ({len(trans_table)} записей)")
    return lambda char: trans_table.get(char, char)


@log_exceptions
def encode(plaintext, code):
    """Encode text using a code which is a permutation of the alphabet."""
    utils_logger.debug(f"Кодирование текста длиной {len(plaintext)} символов")
    trans = maketrans(alphabet + alphabet.upper(), code + code.upper())

    result = translate(plaintext, trans)
    utils_logger.debug(f"Текст закодирован. Результат: '{result[:20]}...'")
    return result


@log_exceptions
def bigrams(text):
    """Return a list of pairs in text (a sequence of letters or words).
    >>> bigrams('this')
    ['th', 'hi', 'is']
    >>> bigrams(['this', 'is', 'a', 'test'])
    [['this', 'is'], ['is', 'a'], ['a', 'test']]
    """
    utils_logger.debug(f"Создание биграмм из текста длиной {len(text)}")
    result = [text[i:i + 2] for i in range(len(text) - 1)]
    utils_logger.debug(f"Создано {len(result)} биграмм. Первые 5: {result[:5]}")
    return result


# Decoding a Shift (or Caesar) Cipher


class ShiftDecoder:
    """There are only 26 possible encodings, so we can try all of them,
    and return the one with the highest probability, according to a
    bigram probability distribution."""

    def __init__(self, training_text):
        decoders_logger.info("Создание ShiftDecoder")
        decoders_logger.debug(f"Длина тренировочного текста: {len(training_text)} символов")

        with log_execution_time("ShiftDecoder.__init__"):
            training_text = canonicalize(training_text)
            decoders_logger.debug(f"Длина канонизированного текста: {len(training_text)} символов")

            self.P2 = CountingProbDist(bigrams(training_text), default=1)
            decoders_logger.debug(f"ShiftDecoder создан. Размер модели биграмм: {len(self.P2)}")

    @log_exceptions
    def score(self, plaintext):
        """Return a score for text based on how common letters pairs are."""
        decoders_logger.debug(f"Вычисление скоринга для текста: '{plaintext[:30]}...'")

        s = 1.0
        bigram_list = bigrams(plaintext)
        decoders_logger.debug(f"Анализ {len(bigram_list)} биграмм")

        for bi in bigram_list:
            try:
                prob = self.P2[bi]
                s = s * prob
                decoders_logger.debug(f"Биграмма '{bi}': вероятность={prob}, текущий score={s}")
            except Exception as e:
                decoders_logger.warning(f"Ошибка при обработке биграммы '{bi}': {e}")
                # Используем вероятность по умолчанию для продолжения
                s = s * self.P2.default

        decoders_logger.debug(f"Итоговый score: {s}")
        return s

    @log_exceptions
    def decode(self, ciphertext):
        """Return the shift decoding of text with the best score."""
        decoders_logger.info(f"Декодирование шифротекста: '{ciphertext[:30]}...'")

        if not ciphertext:
            decoders_logger.warning("Пустой шифротекст передан для декодирования")
            return ciphertext

        with log_execution_time(f"ShiftDecoder.decode({len(ciphertext)})"):
            try:
                all_decodings = list(all_shifts(ciphertext))
                decoders_logger.debug(f"Сгенерировано {len(all_decodings)} вариантов декодирования")

                # Вычисляем скоринг для каждого варианта
                scored_decodings = []
                for i, decoding in enumerate(all_decodings):
                    score = self.score(decoding)
                    scored_decodings.append((score, decoding))

                    if i % 5 == 0:  # Логируем каждые 5 вариантов
                        decoders_logger.debug(f"Вариант {i}: score={score:.10f}, text='{decoding[:20]}...'")

                # Выбираем лучший вариант
                best_score, best_decoding = max(scored_decodings, key=lambda x: x[0])
                decoders_logger.info(f"Найдено лучшее декодирование со score={best_score:.10f}")
                decoders_logger.debug(f"Лучший результат: '{best_decoding[:50]}...'")

                perf_logger.increment_counter("shift_decodings")

                return best_decoding

            except Exception as e:
                decoders_logger.error(f"Ошибка при декодировании шифротекста: {e}", exc_info=True)
                # Возвращаем оригинальный текст в случае ошибки
                return ciphertext


@log_exceptions
def all_shifts(text):
    """Return a list of all 26 possible encodings of text by a shift cipher."""
    decoders_logger.debug(f"Генерация всех 26 сдвигов для текста: '{text[:20]}...'")

    for i, _ in enumerate(alphabet):
        try:
            shifted = shift_encode(text, i)
            decoders_logger.debug(f"Сдвиг {i}: '{shifted[:20]}...'")
            yield shifted
        except Exception as e:
            decoders_logger.error(f"Ошибка при генерации сдвига {i} для текста: {e}")
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
        decoders_logger.info("Создание PermutationDecoder")
        decoders_logger.debug(f"Длина тренировочного текста: {len(training_text)} символов")

        with log_execution_time("PermutationDecoder.__init__"):
            try:
                self.Pwords = UnigramWordModel(words(training_text))
                self.P1 = UnigramWordModel(training_text)  # By letter
                self.P2 = NgramWordModel(2, words(training_text))  # By letter pair

                decoders_logger.debug(f"PermutationDecoder создан. "
                            f"Pwords размер: {len(self.Pwords)}, "
                            f"P1 размер: {len(self.P1)}, "
                            f"P2 размер: {len(self.P2)}")

                if ciphertext:
                    decoders_logger.debug(f"Начальный шифротекст: '{ciphertext[:30]}...'")

            except Exception as e:
                decoders_logger.error(f"Ошибка при создании PermutationDecoder: {e}", exc_info=True)
                raise

    @log_exceptions
    def decode(self, ciphertext):
        """Search for a decoding of the ciphertext."""
        decoders_logger.info(f"Начало декодирования PermutationCipher: '{ciphertext[:30]}...'")

        with log_execution_time(f"PermutationDecoder.decode({len(ciphertext)})"):
            try:
                self.ciphertext = canonicalize(ciphertext)
                decoders_logger.debug(f"Канонизированный шифротекст: '{self.ciphertext[:30]}...'")

                # reduce domain to speed up search
                self.chardomain = {c for c in self.ciphertext if c != ' '}
                decoders_logger.debug(f"Домен символов для декодирования: {sorted(self.chardomain)}")

                problem = PermutationDecoderProblem(decoder=self)
                decoders_logger.debug("PermutationDecoderProblem создан")

                decoders_logger.info("Начало поиска наилучшей перестановки...")
                solution = search.best_first_graph_search(
                    problem, lambda node: self.score(node.state))

                if not solution:
                    decoders_logger.error("Поиск не нашел решения!")
                    return ciphertext

                decoders_logger.debug(f"Решение найдено: {solution.state}")

                solution.state[' '] = ' '
                decoded = translate(self.ciphertext, lambda c: solution.state[c])

                decoders_logger.info(f"Декодирование завершено. Результат: '{decoded[:50]}...'")
                decoders_logger.debug(f"Полная таблица декодирования: {solution.state}")

                perf_logger.increment_counter("permutation_decodings")

                return decoded

            except Exception as e:
                decoders_logger.error(f"Ошибка при декодировании PermutationCipher: {e}", exc_info=True)
                # Возвращаем оригинальный текст в случае ошибки
                return ciphertext

    @log_exceptions
    def score(self, code):
        """Score is product of word scores, unigram scores, and bigram scores.
        This can get very small, so we use logs and exp."""
        decoders_logger.debug(f"Вычисление скоринга для кода: {code}")

        try:
            # remake code dictionary to contain translation for all characters
            full_code = code.copy()
            full_code.update({x: x for x in self.chardomain if x not in code})
            full_code[' '] = ' '

            decoders_logger.debug(f"Полный код перевода: {full_code}")

            text = translate(self.ciphertext, lambda c: full_code[c])
            decoders_logger.debug(f"Переведенный текст: '{text[:30]}...'")

            # add small positive value to prevent computing log(0)
            word_log_sum = sum(np.log(self.Pwords[word] + 1e-20) for word in words(text))
            char_log_sum = sum(np.log(self.P1[c] + 1e-5) for c in text)
            bigram_log_sum = sum(np.log(self.P2[b] + 1e-10) for b in bigrams(text))

            total_log = word_log_sum + char_log_sum + bigram_log_sum

            decoders_logger.debug(f"Суммы логарифмов: "
                        f"words={word_log_sum:.4f}, "
                        f"chars={char_log_sum:.4f}, "
                        f"bigrams={bigram_log_sum:.4f}, "
                        f"total={total_log:.4f}")

            score = -np.exp(total_log)
            decoders_logger.debug(f"Итоговый score: {score}")

            return score

        except Exception as e:
            decoders_logger.error(f"Ошибка при вычислении скоринга: {e}")
            # Возвращаем очень плохой score в случае ошибки
            return -np.inf


class PermutationDecoderProblem(search.Problem):
    """Problem for searching the best permutation decoding."""

    def __init__(self, initial=None, goal=None, decoder=None):
        decoders_logger.debug(f"Инициализация PermutationDecoderProblem, decoder: {type(decoder).__name__}")
        super().__init__(initial or hashabledict(), goal)
        self.decoder = decoder

        if decoder and decoder.chardomain:
            decoders_logger.debug(f"Размер домена символов: {len(decoder.chardomain)}")

    @log_exceptions
    def actions(self, state):
        """Return possible actions from this state."""
        decoders_logger.debug(f"Получение действий для состояния: {state}")

        try:
            search_list = [c for c in self.decoder.chardomain if c not in state]
            target_list = [c for c in alphabet if c not in state.values()]

            decoders_logger.debug(f"Доступные символы для замены: {search_list}")
            decoders_logger.debug(f"Доступные целевые символы: {target_list}")

            if not search_list or not target_list:
                decoders_logger.debug("Нет доступных действий")
                return []

            # Find the best character to replace
            plain_char = max(search_list, key=lambda c: self.decoder.P1[c])
            decoders_logger.debug(f"Выбран символ для замены: '{plain_char}' "
                        f"(вероятность: {self.decoder.P1[plain_char]})")

            actions = []
            for cipher_char in target_list:
                actions.append((plain_char, cipher_char))

            decoders_logger.debug(f"Сгенерировано {len(actions)} действий")
            return actions

        except Exception as e:
            decoders_logger.error(f"Ошибка при генерации действий: {e}")
            return []

    @log_exceptions
    def result(self, state, action):
        """Return the state that results from executing the given action."""
        decoders_logger.debug(f"Применение действия {action} к состоянию {state}")

        try:
            new_state = hashabledict(state)  # copy to prevent hash issues
            new_state[action[0]] = action[1]
            decoders_logger.debug(f"Новое состояние: {new_state}")
            return new_state
        except Exception as e:
            decoders_logger.error(f"Ошибка при применении действия: {e}")
            return state

    @log_exceptions
    def goal_test(self, state):
        """We're done when all letters in search domain are assigned."""
        is_goal = len(state) >= len(self.decoder.chardomain)
        decoders_logger.debug(f"Проверка цели: состояние имеет {len(state)} назначений, "
                    f"требуется {len(self.decoder.chardomain)}. Цель достигнута: {is_goal}")
        return is_goal


# =============================================================================
# ТЕСТИРОВАНИЕ И ТОЧКА ВХОДА
# =============================================================================

@log_exceptions
def test_decoders():
    """Функция для тестирования декодеров с логированием."""
    logger.info("Начало тестирования декодеров")

    # Тестовый текст для тренировки
    training_text = "the quick brown fox jumps over the lazy dog. hello world this is a test."

    # Тестовый шифротекст
    ciphertext = "uryyb jbeyq"

    with log_execution_time("test_decoders"):
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


@log_exceptions
def run_demo():
    """Запуск демонстрации возможностей системы"""
    logger.info("Запуск демонстрации модуля text.py")

    with log_execution_time("run_demo"):
        try:
            # Простой тест моделей
            logger.info("Создание тестовой UnigramWordModel...")
            test_model = UnigramWordModel(["hello", "world", "hello", "test"])
            logger.info(f"Тестовая модель создана. Пример генерации: {test_model.samples(3)}")

            # Тестируем декодеры
            test_decoders()

            logger.info("Все тесты завершены успешно")

            # Логируем итоговые метрики
            perf_logger.log_metrics()

        except Exception as e:
            logger.critical(f"Критическая ошибка при выполнении тестов: {e}", exc_info=True)


# Точка входа
if __name__ == "__main__":
    try:
        logger.info(f"Запуск модуля text.py в {datetime.now()}")
        logger.debug(f"Аргументы командной строки: {sys.argv}")

        run_demo()

        logger.info(f"Модуль text.py завершил работу в {datetime.now()}")

    except KeyboardInterrupt:
        logger.info("Работа прервана пользователем")
    except Exception as e:
        logger.critical(f"Непредвиденная ошибка: {e}", exc_info=True)
        sys.exit(1)