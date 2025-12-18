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
from collections import defaultdict

import numpy as np

import search
from probabilistic_learning import CountingProbDist
from utils import hashabledict


# Улучшенная настройка логирования для третьего коммита
def setup_logging():
    """Настройка логирования с несколькими обработчиками"""
    logger = logging.getLogger(__name__)

    # Очищаем существующие обработчики, если они есть
    logger.handlers.clear()

    # Устанавливаем уровень для нашего логгера
    logger.setLevel(logging.DEBUG)

    # Создаем обработчик для консоли
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Создаем обработчик для файла (только для отладочных сообщений)
    file_handler = logging.FileHandler('text_processing.log', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # Создаем форматтер
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler.setFormatter(console_formatter)
    file_handler.setFormatter(file_formatter)

    # Добавляем обработчики к логгеру
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info("Логирование инициализировано (консоль: INFO, файл: DEBUG)")
    logger.debug(f"Файл лога: text_processing.log")

    return logger


# Инициализируем логгер
logger = setup_logging()


class UnigramWordModel(CountingProbDist):
    """This is a discrete probability distribution over words, so you
    can add, sample, or get P[word], just like with CountingProbDist. You can
    also generate a random text, n words long, with P.samples(n)."""

    def __init__(self, observations, default=0):
        logger.info(f"Создание UnigramWordModel с {len(observations) if observations else 0} наблюдениями")
        logger.debug(f"Параметр default: {default}")

        # Call CountingProbDist constructor,
        # passing the observations and default parameters.
        super(UnigramWordModel, self).__init__(observations, default)

        unique_words = len(self) if hasattr(self, '__len__') else 'unknown'
        logger.debug(f"UnigramWordModel создан успешно. Уникальных слов: {unique_words}")

    def samples(self, n):
        """Return a string of n words, random according to the model."""
        logger.info(f"Генерация {n} слов с помощью UnigramWordModel")

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
        return result

    def __getitem__(self, word):
        """Override to add logging for probability queries"""
        try:
            prob = super().__getitem__(word)
            logger.debug(f"Вероятность слова '{word}': {prob}")
            return prob
        except KeyError as e:
            logger.warning(f"Слово '{word}' не найдено в модели. Возвращаем вероятность по умолчанию.")
            return self.default


class NgramWordModel(CountingProbDist):
    """This is a discrete probability distribution over n-tuples of words.
    You can add, sample or get P[(word1, ..., wordn)]. The method P.samples(n)
    builds up an n-word sequence; P.add_cond_prob and P.add_sequence add data."""

    def __init__(self, n, observation_sequence=None, default=0):
        if n <= 0:
            logger.error(f"Попытка создать NgramWordModel с n={n}. n должно быть > 0.")
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

    def __getitem__(self, ngram):
        """Override to add logging for ngram probability queries"""
        try:
            prob = super().__getitem__(ngram)
            logger.debug(f"Вероятность n-граммы {ngram}: {prob}")
            return prob
        except KeyError as e:
            logger.debug(f"N-грамма {ngram} не найдена. Возвращаем вероятность по умолчанию: {self.default}")
            return self.default

    def add_cond_prob(self, ngram):
        """Build the conditional probabilities P(wn | (w1, ..., wn-1)"""
        logger.debug(f"Добавление условной вероятности для n-граммы: {ngram}")

        if len(ngram) != self.n:
            logger.error(f"Некорректная длина n-граммы: {len(ngram)} (ожидается {self.n})")
            raise ValueError(f"ngram must have length {self.n}")

        prefix = ngram[:-1]
        if prefix not in self.cond_prob:
            logger.debug(f"Создание нового условного распределения для префикса: {prefix}")
            self.cond_prob[prefix] = CountingProbDist()

        self.cond_prob[prefix].add(ngram[-1])
        logger.debug(f"Добавлено слово '{ngram[-1]}' для префикса {prefix}")

    def add_sequence(self, words):
        """Add each tuple words[i:i+n], using a sliding window."""
        n = self.n

        if len(words) < n:
            logger.warning(f"Последовательность слишком короткая ({len(words)} слов) для n={n}")
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
                output.extend(self.sample())
                continue

            next_word = self.cond_prob[prefix_tuple].sample()
            output.append(next_word)

            if (i + 1) % 10 == 0:  # Логируем каждые 10 слов
                logger.debug(f"Сгенерировано {i+1}/{nwords} слов. Текущее слово: '{next_word}'")

        result = ' '.join(output)
        logger.info(f"Генерация завершена. Получено {len(output)} слов")
        logger.debug(f"Пример сгенерированного текста (первые 50 символов): {result[:50]}...")
        return result


class NgramCharModel(NgramWordModel):
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
        logger.info("Создание UnigramCharModel")
        CountingProbDist.__init__(self, default=default)
        self.n = 1
        self.cond_prob = defaultdict()
        self.add_sequence(observation_sequence or [])

        total_chars = len(self) if hasattr(self, '__len__') else 'unknown'
        logger.debug(f"UnigramCharModel создан успешно. Уникальных символов: {total_chars}")

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


# ______________________________________________________________________________


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

    # Return sequence of best words and overall probability
    return sequence, final_prob


# ______________________________________________________________________________


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
        logger.info(f"Создание IRSystem со стоп-словами: {stopwords}")

        # index is a map of {word: {docid: count}}, where docid is an int,
        # indicating the index into the documents list.
        self.index = defaultdict(lambda: defaultdict(int))
        self.stopwords = set(words(stopwords))
        self.documents = []

        logger.debug(f"Инициализирован индекс. Стоп-слов: {len(self.stopwords)}")
        logger.debug(f"Стоп-слова: {sorted(self.stopwords)}")

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

        except Exception as e:
            logger.error(f"Ошибка при индексации документа {url}: {e}", exc_info=True)
            raise

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

        return results

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


def words(text, reg=re.compile('[a-z0-9]+')):
    """Return a list of the words in text, ignoring punctuation and
    converting everything to lowercase (to canonicalize).
    >>> words("``EGAD!'' Edgar cried.")
    ['egad', 'edgar', 'cried']
    """
    try:
        result = reg.findall(text.lower())
        logger.debug(f"Извлечено {len(result)} слов из текста (первые 5: {result[:5]})")
        return result
    except Exception as e:
        logger.error(f"Ошибка при извлечении слов из текста: {e}")
        return []


def canonicalize(text):
    """Return a canonical text: only lowercase letters and blanks.
    >>> canonicalize("``EGAD!'' Edgar cried.")
    'egad edgar cried'
    """
    try:
        result = ' '.join(words(text))
        logger.debug(f"Канонизированный текст (первые 50 символов): '{result[:50]}...'")
        return result
    except Exception as e:
        logger.error(f"Ошибка при канонизации текста: {e}")
        return ""


# ______________________________________________________________________________

# Example application (not in book): decode a cipher.
# A cipher is a code that substitutes one character for another.
# A shift cipher is a rotation of the letters in the alphabet,
# such as the famous rot13, which maps A to N, B to M, etc.

alphabet = 'abcdefghijklmnopqrstuvwxyz'


# Encoding


def shift_encode(plaintext, n):
    """Encode text with a shift cipher that moves each letter up by n letters.
    >>> shift_encode('abc z', 1)
    'bcd a'
    """
    logger.debug(f"Шифрование сдвигом (n={n}): '{plaintext[:20]}...'")
    result = encode(plaintext, alphabet[n:] + alphabet[:n])
    logger.debug(f"Результат шифрования: '{result[:20]}...'")
    return result


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
    return result


def translate(plaintext, function):
    """Translate chars of a plaintext with the given function."""
    logger.debug(f"Перевод текста (длина={len(plaintext)})")
    result = ""
    for char in plaintext:
        result += function(char)
    logger.debug(f"Перевод завершен. Результат: '{result[:20]}...'")
    return result


def maketrans(from_, to_):
    """Create a translation table and return the proper function."""
    logger.debug(f"Создание таблицы перевода: from_='{from_}', to_='{to_}'")
    trans_table = {}
    for n, char in enumerate(from_):
        trans_table[char] = to_[n]

    logger.debug(f"Таблица перевода создана ({len(trans_table)} записей)")
    return lambda char: trans_table.get(char, char)


def encode(plaintext, code):
    """Encode text using a code which is a permutation of the alphabet."""
    logger.debug(f"Кодирование текста длиной {len(plaintext)} символов")
    trans = maketrans(alphabet + alphabet.upper(), code + code.upper())

    result = translate(plaintext, trans)
    logger.debug(f"Текст закодирован. Результат: '{result[:20]}...'")
    return result


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
    return result


# Decoding a Shift (or Caesar) Cipher


class ShiftDecoder:
    """There are only 26 possible encodings, so we can try all of them,
    and return the one with the highest probability, according to a
    bigram probability distribution."""

    def __init__(self, training_text):
        logger.info("Создание ShiftDecoder")
        logger.debug(f"Длина тренировочного текста: {len(training_text)} символов")

        training_text = canonicalize(training_text)
        logger.debug(f"Длина канонизированного текста: {len(training_text)} символов")

        self.P2 = CountingProbDist(bigrams(training_text), default=1)
        logger.debug(f"ShiftDecoder создан. Размер модели биграмм: {len(self.P2)}")

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

            return best_decoding