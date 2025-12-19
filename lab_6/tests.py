import os
import sys
import tempfile
from unittest.mock import patch, mock_open, MagicMock
import math
import pytest
import scipy.special

# Добавляем текущую директорию в PYTHONPATH для импорта main
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from main import read_file, write_file, frequency_test, consecutive_bits_test, longest_sequence_test, main
    from constants import PI, PATH_CPP_SEQ, PATH_JAVA_SEQ, PATH_CPP_NIST_RES, PATH_JAVA_NIST_RES
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Убедитесь, что main.py и constants.py находятся в той же папке")
    raise


def test_read_file():
    """Тест чтения файла с использованием временного файла."""
    content = "0101010101"
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        assert read_file(tmp_path) == content
    finally:
        os.unlink(tmp_path)


def test_read_file_empty():
    """Тест чтения пустого файла."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
        tmp.write("")
        tmp_path = tmp.name

    try:
        assert read_file(tmp_path) == ""
    finally:
        os.unlink(tmp_path)


def test_read_file_with_special_characters():
    """Тест чтения файла со специальными символами."""
    content = "0101\n0101\t01"
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        assert read_file(tmp_path) == content
    finally:
        os.unlink(tmp_path)


def test_write_file():
    """Тест записи данных в файл."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
        tmp_path = tmp.name

    try:
        test1 = 0.123456
        test2 = 0.789012
        test3 = 0.345678

        write_file(tmp_path, test1, test2, test3)

        with open(tmp_path, 'r', encoding='utf-8') as f:
            content = f.read()

        expected_lines = [
            f"Результат частотного теста P-value: {test1}",
            f"Результат теста на одинаковые подряд идущие биты P-value: {test2}",
            f"Результат теста на самую длинную последовательность: {test3}"
        ]

        assert content == '\n'.join(expected_lines)
    finally:
        os.unlink(tmp_path)


def test_write_file_creates_file():
    """Тест, что write_file создает файл, если его нет."""
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "non_existent_file.txt")

        # Убедимся, что файла нет
        assert not os.path.exists(file_path)

        write_file(file_path, 0.1, 0.2, 0.3)

        # Проверяем, что файл создан
        assert os.path.exists(file_path)


def test_write_file_overwrites_content():
    """Тест, что write_file перезаписывает существующий файл."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
        tmp.write("Старое содержимое")
        tmp_path = tmp.name

    try:
        write_file(tmp_path, 0.5, 0.6, 0.7)

        with open(tmp_path, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "Старое содержимое" not in content
        assert "Результат частотного теста P-value: 0.5" in content
    finally:
        os.unlink(tmp_path)


def test_frequency_test_balanced():
    """Тест частотного теста для сбалансированной последовательности."""
    # Сбалансированная последовательность (5 нулей, 5 единиц)
    sequence = "0101010101"
    p_value = frequency_test(sequence)

    # Для сбалансированной последовательности p-value должен быть близок к 1
    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1
    # В сбалансированном случае p-value будет высоким (> 0.1)
    assert p_value > 0.1


def test_frequency_test_all_zeros():
    """Тест частотного теста для последовательности из всех нулей."""
    sequence = "0000000000"
    p_value = frequency_test(sequence)

    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1
    # Для несбалансированной последовательности p-value будет очень маленьким
    assert p_value < 0.05


def test_frequency_test_all_ones():
    """Тест частотного теста для последовательности из всех единиц."""
    sequence = "1111111111"
    p_value = frequency_test(sequence)

    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1
    # Для несбалансированной последовательности p-value будет очень маленьким
    assert p_value < 0.05


def test_frequency_test_single_bit():
    """Тест частотного теста для последовательности из одного бита."""
    sequence = "1"
    p_value = frequency_test(sequence)

    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1


def test_frequency_test_empty_sequence():
    """Тест частотного теста для пустой последовательности."""
    sequence = ""
    p_value = frequency_test(sequence)

    assert isinstance(p_value, float)
    assert p_value == 1.0


def test_frequency_test_large_sequence():
    """Тест частотного теста для большой последовательности."""
    sequence = "01" * 1000  # 2000 бит, сбалансированная
    p_value = frequency_test(sequence)

    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1
    # Для большой сбалансированной последовательности p-value должен быть высоким
    assert p_value > 0.1


def test_consecutive_bits_test_alternating():
    """Тест на одинаковые подряд идущие биты для чередующейся последовательности."""
    sequence = "0101010101"
    p_value = consecutive_bits_test(sequence)

    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1
    # Для чередующейся последовательности p-value должен быть высоким


def test_consecutive_bits_test_all_same():
    """Тест на одинаковые подряд идущие биты для последовательности из одинаковых битов."""
    sequence = "0000000000"
    p_value = consecutive_bits_test(sequence)

    assert isinstance(p_value, float)
    # Для последовательности из одинаковых битов возвращается 0.0
    assert p_value == 0.0


def test_consecutive_bits_test_all_ones():
    """Тест на одинаковые подряд идущие биты для последовательности из всех единиц."""
    sequence = "1111111111"
    p_value = consecutive_bits_test(sequence)

    assert isinstance(p_value, float)
    assert p_value == 0.0


def test_consecutive_bits_test_mixed():
    """Тест на одинаковые подряд идущие биты для смешанной последовательности."""
    sequence = "000111000111"
    p_value = consecutive_bits_test(sequence)

    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1


def test_consecutive_bits_test_edge_cases():
    """Тест граничных случаев для consecutive_bits_test."""
    # Короткая последовательность
    sequence = "01"
    p_value = consecutive_bits_test(sequence)
    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1

    # Последовательность длиной 1
    sequence = "1"
    p_value = consecutive_bits_test(sequence)
    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1


def test_consecutive_bits_test_empty_sequence():
    """Тест для пустой последовательности в consecutive_bits_test."""
    sequence = ""
    p_value = consecutive_bits_test(sequence)
    assert isinstance(p_value, float)
    assert p_value == 1.0


def test_consecutive_bits_test_single_zero():
    """Тест для последовательности из одного нуля."""
    sequence = "0"
    p_value = consecutive_bits_test(sequence)
    assert isinstance(p_value, float)
    assert p_value == 0.0


def test_consecutive_bits_test_single_one():
    """Тест для последовательности из одной единицы."""
    sequence = "1"
    p_value = consecutive_bits_test(sequence)
    assert isinstance(p_value, float)
    assert p_value == 0.0


def test_consecutive_bits_test_near_balanced():
    """Тест для почти сбалансированной последовательности."""
    sequence = "1111100000"  # 5 единиц, 5 нулей, но не чередующиеся
    p_value = consecutive_bits_test(sequence)

    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1


# Параметризованные тесты для longest_sequence_test
@pytest.mark.parametrize("sequence,block_size,expected_range", [
    # Короткая последовательность с маленькими блоками
    ("11110000", 4, (0, 1)),  # 2 блока по 4 бита
    ("10101010", 4, (0, 1)),  # чередующиеся биты
    # Длинная последовательность
    ("1111111100000000", 8, (0, 1)),  # 2 блока по 8 бит
    ("1" * 16, 8, (0, 1)),  # все единицы
    ("0" * 16, 8, (0, 1)),  # все нули
    # Крайние случаи
    ("1" * 8, 8, (0, 1)),  # ровно один блок из единиц
    ("0" * 8, 8, (0, 1)),  # ровно один блок из нулей
])
def test_longest_sequence_test_parametrized(sequence, block_size, expected_range):
    """Параметризованный тест для longest_sequence_test."""
    # Патчим print в longest_sequence_test, чтобы не загрязнять вывод
    with patch('builtins.print'):
        p_value = longest_sequence_test(sequence, block_size)

    assert isinstance(p_value, float)
    assert expected_range[0] <= p_value <= expected_range[1]


def test_longest_sequence_test_basic():
    """Базовый тест для longest_sequence_test."""
    sequence = "1100110011001100"  # 16 бит
    with patch('builtins.print'):
        p_value = longest_sequence_test(sequence, block_size=8)

    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1


def test_longest_sequence_test_single_block():
    """Тест для одного блока."""
    sequence = "11110000"  # 8 бит - ровно один блок
    with patch('builtins.print'):
        p_value = longest_sequence_test(sequence, block_size=8)

    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1


def test_longest_sequence_test_multiple_blocks():
    """Тест для нескольких блоков."""
    sequence = "11111111000000001111111100000000"  # 32 бита = 4 блока по 8
    with patch('builtins.print'):
        p_value = longest_sequence_test(sequence, block_size=8)

    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1


def test_longest_sequence_test_long_runs():
    """Тест с длинными последовательностями единиц."""
    sequence = "111111110000111111110000"  # 24 бита
    with patch('builtins.print'):
        p_value = longest_sequence_test(sequence, block_size=8)

    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1


def test_longest_sequence_test_different_block_sizes():
    """Тест с разными размерами блоков."""
    sequence = "1" * 32  # 32 единицы

    for block_size in [4, 8, 16]:
        with patch('builtins.print'):
            p_value = longest_sequence_test(sequence, block_size)

        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1


def test_longest_sequence_test_empty_sequence():
    """Тест с пустой последовательностью."""
    sequence = ""
    with patch('builtins.print'):
        p_value = longest_sequence_test(sequence, block_size=8)

    # После исправления функции это должно возвращать 1.0
    assert isinstance(p_value, float)
    assert p_value == 1.0


def test_longest_sequence_test_very_long_sequence():
    """Тест с очень длинной последовательностью."""
    sequence = "01" * 1000  # 2000 бит
    with patch('builtins.print'):
        p_value = longest_sequence_test(sequence, block_size=100)

    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1


def test_longest_sequence_test_block_size_zero():
    """Тест с размером блока 0."""
    sequence = "01010101"
    with patch('builtins.print'):
        p_value = longest_sequence_test(sequence, block_size=0)

    assert isinstance(p_value, float)
    assert p_value == 1.0


# Тесты для функции main с использованием моков
@patch('main.read_file')
@patch('main.frequency_test')
@patch('main.consecutive_bits_test')
@patch('main.longest_sequence_test')
@patch('main.write_file')
def test_main_normal_execution(mock_write_file, mock_longest_sequence_test,
                               mock_consecutive_bits_test, mock_frequency_test,
                               mock_read_file):
    """Тест основной функции main с моками всех зависимостей."""
    # Настраиваем моки
    mock_read_file.side_effect = ["cpp_sequence", "java_sequence"]
    mock_frequency_test.side_effect = [0.123, 0.456]
    mock_consecutive_bits_test.side_effect = [0.789, 0.012]
    mock_longest_sequence_test.side_effect = [0.345, 0.678]

    # Патчим константы
    with patch('main.PATH_CPP_SEQ', 'cpp_seq.txt'), \
            patch('main.PATH_JAVA_SEQ', 'java_seq.txt'), \
            patch('main.PATH_CPP_NIST_RES', 'cpp_res.txt'), \
            patch('main.PATH_JAVA_NIST_RES', 'java_res.txt'):
        # Вызываем main
        with patch('builtins.print'):
            main()

        # Проверяем вызовы read_file
        assert mock_read_file.call_count == 2
        mock_read_file.assert_any_call('cpp_seq.txt')
        mock_read_file.assert_any_call('java_seq.txt')

        # Проверяем вызовы тестовых функций для C++
        mock_frequency_test.assert_any_call("cpp_sequence")
        mock_consecutive_bits_test.assert_any_call("cpp_sequence")
        mock_longest_sequence_test.assert_any_call("cpp_sequence")

        # Проверяем вызовы тестовых функций для Java
        mock_frequency_test.assert_any_call("java_sequence")
        mock_consecutive_bits_test.assert_any_call("java_sequence")
        mock_longest_sequence_test.assert_any_call("java_sequence")

        # Проверяем вызовы write_file
        assert mock_write_file.call_count == 2
        mock_write_file.assert_any_call('cpp_res.txt', 0.123, 0.789, 0.345)
        mock_write_file.assert_any_call('java_res.txt', 0.456, 0.012, 0.678)


@patch('main.read_file')
@patch('main.frequency_test')
@patch('main.consecutive_bits_test')
@patch('main.longest_sequence_test')
@patch('main.write_file')
def test_main_with_io_error(mock_write_file, mock_longest_sequence_test,
                            mock_consecutive_bits_test, mock_frequency_test,
                            mock_read_file):
    """Тест main при возникновении IOError при записи."""
    # Настраиваем моки
    mock_read_file.side_effect = ["cpp_sequence", "java_sequence"]
    mock_frequency_test.side_effect = [0.123, 0.456]
    mock_consecutive_bits_test.side_effect = [0.789, 0.012]
    mock_longest_sequence_test.side_effect = [0.345, 0.678]

    # Настраиваем write_file для вызова исключения
    mock_write_file.side_effect = IOError("Не удалось записать файл")

    # Патчим константы
    with patch('main.PATH_CPP_SEQ', 'cpp_seq.txt'), \
            patch('main.PATH_JAVA_SEQ', 'java_seq.txt'), \
            patch('main.PATH_CPP_NIST_RES', 'cpp_res.txt'), \
            patch('main.PATH_JAVA_NIST_RES', 'java_res.txt'):
        # Захватываем вывод в консоль
        with patch('builtins.print') as mock_print:
            main()

            # Проверяем, что была попытка напечатать сообщение об ошибке
            mock_print.assert_any_call("Ошибка: Не удалось записать данные в файл")


@patch('main.read_file')
@patch('main.frequency_test')
@patch('main.consecutive_bits_test')
@patch('main.longest_sequence_test')
@patch('main.write_file')
def test_main_with_general_exception(mock_write_file, mock_longest_sequence_test,
                                     mock_consecutive_bits_test, mock_frequency_test,
                                     mock_read_file):
    """Тест main при возникновении общего исключения."""
    # Настраиваем моки
    mock_read_file.side_effect = ["cpp_sequence", "java_sequence"]
    mock_frequency_test.side_effect = [0.123, 0.456]
    mock_consecutive_bits_test.side_effect = [0.789, 0.012]
    mock_longest_sequence_test.side_effect = [0.345, 0.678]

    # Настраиваем write_file для вызова исключения
    mock_write_file.side_effect = Exception("Неизвестная ошибка")

    # Патчим константы
    with patch('main.PATH_CPP_SEQ', 'cpp_seq.txt'), \
            patch('main.PATH_JAVA_SEQ', 'java_seq.txt'), \
            patch('main.PATH_CPP_NIST_RES', 'cpp_res.txt'), \
            patch('main.PATH_JAVA_NIST_RES', 'java_res.txt'):
        # Захватываем вывод в консоль
        with patch('builtins.print') as mock_print:
            main()

            # Проверяем, что была попытка напечатать сообщение об ошибке
            mock_print.assert_any_call("Произошла ошибка: Неизвестная ошибка")


@patch('main.read_file')
@patch('main.frequency_test')
@patch('main.consecutive_bits_test')
@patch('main.longest_sequence_test')
@patch('main.write_file')
@patch('builtins.print')
def test_main_output_print_statements(mock_print, mock_write_file, mock_longest_sequence_test,
                                      mock_consecutive_bits_test, mock_frequency_test,
                                      mock_read_file):
    """Тест проверки вывода в консоль из функции main."""
    # Настраиваем моки
    mock_read_file.side_effect = ["cpp_sequence", "java_sequence"]
    mock_frequency_test.side_effect = [0.123456, 0.789012]
    mock_consecutive_bits_test.side_effect = [0.345678, 0.901234]
    mock_longest_sequence_test.side_effect = [0.567890, 0.123456]

    # Патчим константы
    with patch('main.PATH_CPP_SEQ', 'cpp_seq.txt'), \
            patch('main.PATH_JAVA_SEQ', 'java_seq.txt'), \
            patch('main.PATH_CPP_NIST_RES', 'cpp_res.txt'), \
            patch('main.PATH_JAVA_NIST_RES', 'java_res.txt'):

        # Вызываем main
        main()

        # Проверяем, что были напечатаны все ожидаемые сообщения
        printed_calls = [str(call) for call in mock_print.call_args_list]

        # Проверяем ключевые фразы
        cpp_printed = False
        java_printed = False
        for call in printed_calls:
            if "C++ Sequence:" in call:
                cpp_printed = True
            if "Java Sequence:" in call:
                java_printed = True

        assert cpp_printed, "Не напечатано 'C++ Sequence:'"
        assert java_printed, "Не напечатано 'Java Sequence:'"

        # Проверяем, что было напечатано достаточно информации
        assert mock_print.call_count >= 8  # 8 основных сообщений


def test_main_direct_call():
    """Тест прямого вызова main (интеграционный тест)."""
    # Здесь мы используем контекстный менеджер для патчинга констант
    # чтобы избежать зависимостей от реальных файлов
    with patch('main.PATH_CPP_SEQ', 'test_cpp.txt'), \
            patch('main.PATH_JAVA_SEQ', 'test_java.txt'), \
            patch('main.PATH_CPP_NIST_RES', 'test_cpp_res.txt'), \
            patch('main.PATH_JAVA_NIST_RES', 'test_java_res.txt'), \
            patch('main.read_file') as mock_read_file, \
            patch('main.write_file') as mock_write_file:
        # Настраиваем моки
        mock_read_file.side_effect = ["0101010101", "111000111000"]

        # Вызываем main
        with patch('builtins.print'):
            main()

        # Проверяем вызовы
        assert mock_read_file.call_count == 2
        assert mock_write_file.call_count == 2


# Интеграционные тесты с реальными файлами
def test_integration_with_real_files():
    """Интеграционный тест с созданием реальных файлов."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Создаем тестовые файлы
        cpp_seq_path = os.path.join(tmpdir, "cpp_seq.txt")
        java_seq_path = os.path.join(tmpdir, "java_seq.txt")
        cpp_res_path = os.path.join(tmpdir, "cpp_res.txt")
        java_res_path = os.path.join(tmpdir, "java_res.txt")

        # Записываем тестовые последовательности
        with open(cpp_seq_path, 'w', encoding='utf-8') as f:
            f.write("0101010101")

        with open(java_seq_path, 'w', encoding='utf-8') as f:
            f.write("111000111000")

        # Патчим константы для использования временных файлов
        with patch('main.PATH_CPP_SEQ', cpp_seq_path), \
                patch('main.PATH_JAVA_SEQ', java_seq_path), \
                patch('main.PATH_CPP_NIST_RES', cpp_res_path), \
                patch('main.PATH_JAVA_NIST_RES', java_res_path):
            # Вызываем main
            with patch('builtins.print'):
                main()

            # Проверяем, что результаты были записаны
            assert os.path.exists(cpp_res_path)
            assert os.path.exists(java_res_path)

            # Проверяем содержимое файлов результатов
            with open(cpp_res_path, 'r', encoding='utf-8') as f:
                cpp_content = f.read()
                assert "Результат частотного теста P-value:" in cpp_content

            with open(java_res_path, 'r', encoding='utf-8') as f:
                java_content = f.read()
                assert "Результат теста на одинаковые подряд идущие биты P-value:" in java_content


def test_integration_full_pipeline():
    """Полный интеграционный тест всего пайплайна."""
    # Создаем тестовую последовательность
    test_sequence = "0101010101" * 10  # 100 бит

    # Тестируем все функции по отдельности
    p1 = frequency_test(test_sequence)
    p2 = consecutive_bits_test(test_sequence)

    with patch('builtins.print'):
        p3 = longest_sequence_test(test_sequence, block_size=8)

    # Проверяем результаты
    assert isinstance(p1, float)
    assert isinstance(p2, float)
    assert isinstance(p3, float)
    assert 0 <= p1 <= 1
    assert 0 <= p2 <= 1
    assert 0 <= p3 <= 1


# Тесты для проверки математических свойств
def test_frequency_test_symmetry():
    """Проверка симметрии частотного теста для нулей и единиц."""
    seq_zeros = "0" * 100
    seq_ones = "1" * 100

    p_zeros = frequency_test(seq_zeros)
    p_ones = frequency_test(seq_ones)

    # Для последовательностей из всех нулей и всех единиц
    # p-value должны быть одинаковыми (симметрия)
    assert abs(p_zeros - p_ones) < 1e-10


def test_consecutive_bits_test_for_random_sequence():
    """Тест consecutive_bits_test для случайной последовательности."""
    import random
    random.seed(42)

    # Генерируем случайную последовательность
    sequence = ''.join(str(random.randint(0, 1)) for _ in range(100))

    p_value = consecutive_bits_test(sequence)

    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1


# Тест для проверки обработки ошибок в read_file
def test_read_file_nonexistent():
    """Тест чтения несуществующего файла."""
    with pytest.raises(FileNotFoundError):
        read_file("non_existent_file_12345.txt")


# Тест для проверки констант
def test_constants_import():
    """Тест импорта констант."""
    # Просто проверяем, что импорт работает
    assert PI is not None
    assert len(PI) == 4
    assert isinstance(PI[0], float)
    assert PATH_CPP_SEQ is not None
    assert PATH_JAVA_SEQ is not None
    assert PATH_CPP_NIST_RES is not None
    assert PATH_JAVA_NIST_RES is not None


# Дополнительные тесты для покрытия edge cases
def test_frequency_test_invalid_characters():
    """Тест частотного теста с недопустимыми символами."""
    # Тест должен корректно обрабатывать только '0' и '1'
    sequence = "0101a0101"  # содержит букву 'a'
    p_value = frequency_test(sequence)

    # Проверяем, что функция не падает и возвращает значение
    assert isinstance(p_value, float)


def test_write_file_with_none_values():
    """Тест записи None значений."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
        tmp_path = tmp.name

    try:
        write_file(tmp_path, None, None, None)

        with open(tmp_path, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "None" in content
    finally:
        os.unlink(tmp_path)


def test_main_with_empty_sequences():
    """Тест main с пустыми последовательностями."""
    with patch('main.PATH_CPP_SEQ', 'test_cpp.txt'), \
            patch('main.PATH_JAVA_SEQ', 'test_java.txt'), \
            patch('main.PATH_CPP_NIST_RES', 'test_cpp_res.txt'), \
            patch('main.PATH_JAVA_NIST_RES', 'test_java_res.txt'), \
            patch('main.read_file') as mock_read_file, \
            patch('main.write_file') as mock_write_file:
        # Настраиваем моки для пустых последовательностей
        mock_read_file.side_effect = ["", ""]

        # Вызываем main
        with patch('builtins.print'):
            main()

        # Проверяем вызовы
        assert mock_read_file.call_count == 2
        assert mock_write_file.call_count == 2