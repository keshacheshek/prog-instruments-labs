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


# Улучшенная настройка логирования для второго коммита
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
        prefix = os.path.dirname(__file__)
        for filename in filenames:
            self.index_document(open(filename).read(), os.path.relpath(filename, prefix))

    def index_document(self, text, url):
        """Index the text of a document."""
        # For now, use first line for title
        title = text[:text.index('\n')].strip()
        docwords = words(text)
        docid = len(self.documents)
        self.documents.append(Document(title, url, len(docwords)))
        for word in docwords:
            if word not in self.stopwords:
                self.index[word][docid] += 1

    def query(self, query_text, n=10):
        """Return a list of n (score, docid) pairs for the best matches.
        Also handle the special syntax for 'learn: command'."""
        if query_text.startswith("learn:"):
            doctext = os.popen(query_text[len("learn:"):], 'r').read()
            self.index_document(doctext, query_text)
            return []

        qwords = [w for w in words(query_text) if w not in self.stopwords]
        shortest = min(qwords, key=lambda w: len(self.index[w]))
        docids = self.index[shortest]
        return heapq.nlargest(n, ((self.total_score(qwords, docid), docid) for docid in docids))

    def score(self, word, docid):
        """Compute a score for this word on the document with this docid."""
        # There are many options; here we take a very simple approach
        return np.log(1 + self.index[word][docid]) / np.log(1 + self.documents[docid].nwords)

    def total_score(self, words, docid):
        """Compute the sum of the scores of these words on the document with this docid."""
        return sum(self.score(word, docid) for word in words)

    def present(self, results):
        """Present the results as a list."""
        for (score, docid) in results:
            doc = self.documents[docid]
            print("{:5.2}|{:25} | {}".format(100 * score, doc.url, doc.title[:45].expandtabs()))

    def present_results(self, query_text, n=10):
        """Get results for the query and present them."""
        self.present(self.query(query_text, n))


class UnixConsultant(IRSystem):
    """A trivial IR system over a small collection of Unix man pages."""

    def __init__(self):
        IRSystem.__init__(self, stopwords="how do i the a of")

        import os
        aima_root = os.path.dirname(__file__)
        mandir = os.path.join(aima_root, 'aima-data/MAN/')
        man_files = [mandir + f for f in os.listdir(mandir) if f.endswith('.txt')]

        self.index_collection(man_files)


class Document:
    """Metadata for a document: title and url; maybe add others later."""

    def __init__(self, title, url, nwords):
        self.title = title
        self.url = url
        self.nwords = nwords


def words(text, reg=re.compile('[a-z0-9]+')):
    """Return a list of the words in text, ignoring punctuation and
    converting everything to lowercase (to canonicalize).
    >>> words("``EGAD!'' Edgar cried.")
    ['egad', 'edgar', 'cried']
    """
    return reg.findall(text.lower())


def canonicalize(text):
    """Return a canonical text: only lowercase letters and blanks.
    >>> canonicalize("``EGAD!'' Edgar cried.")
    'egad edgar cried'
    """
    return ' '.join(words(text))


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
    return encode(plaintext, alphabet[n:] + alphabet[:n])


def rot13(plaintext):
    """Encode text by rotating letters by 13 spaces in the alphabet.
    >>> rot13('hello')
    'uryyb'
    >>> rot13(rot13('hello'))
    'hello'
    """
    return shift_encode(plaintext, 13)


def translate(plaintext, function):
    """Translate chars of a plaintext with the given function."""
    result = ""
    for char in plaintext:
        result += function(char)
    return result


def maketrans(from_, to_):
    """Create a translation table and return the proper function."""
    trans_table = {}
    for n, char in enumerate(from_):
        trans_table[char] = to_[n]

    return lambda char: trans_table.get(char, char)


def encode(plaintext, code):
    """Encode text using a code which is a permutation of the alphabet."""
    trans = maketrans(alphabet + alphabet.upper(), code + code.upper())

    return translate(plaintext, trans)


def bigrams(text):
    """Return a list of pairs in text (a sequence of letters or words).
    >>> bigrams('this')
    ['th', 'hi', 'is']
    >>> bigrams(['this', 'is', 'a', 'test'])
    [['this', 'is'], ['is', 'a'], ['a', 'test']]
    """
    return [text[i:i + 2] for i in range(len(text) - 1)]


# Decoding a Shift (or Caesar) Cipher


class ShiftDecoder:
    """There are only 26 possible encodings, so we can try all of them,
    and return the one with the highest probability, according to a
    bigram probability distribution."""

    def __init__(self, training_text):
        training_text = canonicalize(training_text)
        self.P2 = CountingProbDist(bigrams(training_text), default=1)

    def score(self, plaintext):
        """Return a score for text based on how common letters pairs are."""

        s = 1.0
        for bi in bigrams(plaintext):
            s = s * self.P2[bi]

        return s

    def decode(self, ciphertext):
        """Return the shift decoding of text with the best score."""

        return max(all_shifts(ciphertext), key=lambda shift: self.score(shift))


def all_shifts(text):
    """Return a list of all 26 possible encodings of text by a shift cipher."""

    yield from (shift_encode(text, i) for i, _ in enumerate(alphabet))


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
        self.Pwords = UnigramWordModel(words(training_text))
        self.P1 = UnigramWordModel(training_text)  # By letter
        self.P2 = NgramWordModel(2, words(training_text))  # By letter pair

    def decode(self, ciphertext):
        """Search for a decoding of the ciphertext."""
        self.ciphertext = canonicalize(ciphertext)
        # reduce domain to speed up search
        self.chardomain = {c for c in self.ciphertext if c != ' '}
        problem = PermutationDecoderProblem(decoder=self)
        solution = search.best_first_graph_search(
            problem, lambda node: self.score(node.state))

        solution.state[' '] = ' '
        return translate(self.ciphertext, lambda c: solution.state[c])

    def score(self, code):
        """Score is product of word scores, unigram scores, and bigram scores.
        This can get very small, so we use logs and exp."""

        # remake code dictionary to contain translation for all characters
        full_code = code.copy()
        full_code.update({x: x for x in self.chardomain if x not in code})
        full_code[' '] = ' '
        text = translate(self.ciphertext, lambda c: full_code[c])

        # add small positive value to prevent computing log(0)
        # TODO: Modify the values to make score more accurate
        logP = (sum(np.log(self.Pwords[word] + 1e-20) for word in words(text)) +
                sum(np.log(self.P1[c] + 1e-5) for c in text) +
                sum(np.log(self.P2[b] + 1e-10) for b in bigrams(text)))
        return -np.exp(logP)


class PermutationDecoderProblem(search.Problem):

    def __init__(self, initial=None, goal=None, decoder=None):
        super().__init__(initial or hashabledict(), goal)
        self.decoder = decoder

    def actions(self, state):
        search_list = [c for c in self.decoder.chardomain if c not in state]
        target_list = [c for c in alphabet if c not in state.values()]
        # Find the best character to replace
        plain_char = max(search_list, key=lambda c: self.decoder.P1[c])
        for cipher_char in target_list:
            yield (plain_char, cipher_char)

    def result(self, state, action):
        new_state = hashabledict(state)  # copy to prevent hash issues
        new_state[action[0]] = action[1]
        return new_state

    def goal_test(self, state):
        """We're done when all letters in search domain are assigned."""
        return len(state) >= len(self.decoder.chardomain)