import os
import sys
import tempfile
from unittest.mock import patch, mock_open, MagicMock
import math
import pytest

# Добавляем путь к исходному коду в PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import read_file, write_file, frequency_test, consecutive_bits_test, longest_sequence_test, main


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
    # согласно условию в функции
    if abs(0.0 - 0.5) >= (2 / len(sequence) ** 0.5):
        assert p_value == 0.0
    else:
        assert 0 <= p_value <= 1


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

    # Последовательность длиной 1
    sequence = "1"
    p_value = consecutive_bits_test(sequence)
    assert isinstance(p_value, float)


# Параметризованные тесты для longest_sequence_test
@pytest.mark.parametrize("sequence,block_size,expected_range", [
    # Короткая последовательность с маленькими блоками
    ("11110000", 4, (0, 1)),  # 2 блока по 4 бита
    ("10101010", 4, (0, 1)),  # чередующиеся биты
    # Длинная последовательность
    ("1111111100000000", 8, (0, 1)),  # 2 блока по 8 бит
    ("1" * 16, 8, (0, 1)),  # все единицы
    ("0" * 16, 8, (0, 1)),  # все нули
])
def test_longest_sequence_test_parametrized(sequence, block_size, expected_range):
    """Параметризованный тест для longest_sequence_test."""
    p_value = longest_sequence_test(sequence, block_size)

    assert isinstance(p_value, float)
    assert expected_range[0] <= p_value <= expected_range[1]


def test_longest_sequence_test_basic():
    """Базовый тест для longest_sequence_test."""
    sequence = "1100110011001100"  # 16 бит
    p_value = longest_sequence_test(sequence, block_size=8)

    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1


def test_longest_sequence_test_single_block():
    """Тест для одного блока."""
    sequence = "11110000"  # 8 бит - ровно один блок
    p_value = longest_sequence_test(sequence, block_size=8)

    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1


def test_longest_sequence_test_multiple_blocks():
    """Тест для нескольких блоков."""
    sequence = "11111111000000001111111100000000"  # 32 бита = 4 блока по 8
    p_value = longest_sequence_test(sequence, block_size=8)

    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1


def test_longest_sequence_test_long_runs():
    """Тест с длинными последовательностями единиц."""
    sequence = "111111110000111111110000"  # 24 бита
    p_value = longest_sequence_test(sequence, block_size=8)

    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1


def test_longest_sequence_test_different_block_sizes():
    """Тест с разными размерами блоков."""
    sequence = "1" * 32  # 32 единицы

    for block_size in [4, 8, 16]:
        p_value = longest_sequence_test(sequence, block_size)

        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1


def test_longest_sequence_test_empty_sequence():
    """Тест с пустой последовательностью."""
    sequence = ""
    p_value = longest_sequence_test(sequence, block_size=8)

    # Функция должна вернуть число (возможно, NaN или Inf при делении на ноль)
    # Проверяем, что это float
    assert isinstance(p_value, float)


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

    # Вызываем main
    main()

    # Проверяем вызовы read_file
    assert mock_read_file.call_count == 2
    mock_read_file.assert_any_call('cpp_seq.txt')  # из constants.PATH_CPP_SEQ
    mock_read_file.assert_any_call('java_seq.txt')  # из constants.PATH_JAVA_SEQ

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
    mock_write_file.assert_any_call('cpp_res.txt', 0.123, 0.789, 0.345)  # из constants.PATH_CPP_NIST_RES
    mock_write_file.assert_any_call('java_res.txt', 0.456, 0.012, 0.678)  # из constants.PATH_JAVA_NIST_RES


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

    # Вызываем main
    main()

    # Проверяем, что были напечатаны все ожидаемые сообщения
    mock_print.assert_any_call("C++ Sequence:")
    mock_print.assert_any_call("Frequency test P-value: 0.123456")
    mock_print.assert_any_call("Identical consecutive bits P-value: 0.345678")
    mock_print.assert_any_call("Longest sequence of units in a block P-value: 0.567890")

    mock_print.assert_any_call("\nJava Sequence:")
    mock_print.assert_any_call("Frequency Test P-value: 0.789012")
    mock_print.assert_any_call("Identical consecutive bits P-value: 0.901234")
    mock_print.assert_any_call("Longest sequence of units in a block P-value: 0.123456")

    # Проверяем финальное сообщение
    mock_print.assert_any_call("Анализ завершен. Результаты сохранены в файлы 'cpp_res.txt' и 'java_res.txt'")


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
        main()

        # Проверяем вызовы
        assert mock_read_file.call_count == 2
        assert mock_write_file.call_count == 2