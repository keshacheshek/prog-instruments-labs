import os
import sys
import tempfile
import math
from unittest.mock import patch, MagicMock
import pytest

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
    """Тест чтения файла."""
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


def test_frequency_test_balanced():
    """Тест частотного теста для сбалансированной последовательности."""
    sequence = "0101010101"
    p_value = frequency_test(sequence)

    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1
    assert p_value > 0.1  # Для сбалансированной последовательности


def test_frequency_test_all_zeros():
    """Тест частотного теста для последовательности из всех нулей."""
    sequence = "0000000000"
    p_value = frequency_test(sequence)

    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1
    assert p_value < 0.05  # Для несбалансированной последовательности


def test_frequency_test_empty_sequence():
    """Тест частотного теста для пустой последовательности."""
    sequence = ""
    p_value = frequency_test(sequence)

    assert isinstance(p_value, float)
    assert p_value == 1.0


def test_consecutive_bits_test_alternating():
    """Тест на одинаковые подряд идущие биты для чередующейся последовательности."""
    sequence = "0101010101"
    p_value = consecutive_bits_test(sequence)

    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1


def test_consecutive_bits_test_all_same():
    """Тест на одинаковые подряд идущие биты для последовательности из одинаковых битов."""
    sequence = "0000000000"
    p_value = consecutive_bits_test(sequence)

    assert isinstance(p_value, float)
    assert p_value == 0.0


def test_consecutive_bits_test_empty_sequence():
    """Тест для пустой последовательности в consecutive_bits_test."""
    sequence = ""
    p_value = consecutive_bits_test(sequence)
    assert isinstance(p_value, float)
    assert p_value == 1.0


# Параметризованный тест для longest_sequence_test (сложный тест №1)
@pytest.mark.parametrize("sequence,block_size", [
    ("11110000", 4),
    ("10101010", 4),
    ("1111111100000000", 8),
    ("1" * 16, 8),
    ("0" * 16, 8),
])
def test_longest_sequence_test_parametrized(sequence, block_size):
    """Параметризованный тест для longest_sequence_test."""
    p_value = longest_sequence_test(sequence, block_size)

    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1


def test_longest_sequence_test_basic():
    """Базовый тест для longest_sequence_test."""
    sequence = "1100110011001100"
    p_value = longest_sequence_test(sequence, block_size=8)

    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1


def test_longest_sequence_test_empty_sequence():
    """Тест с пустой последовательностью."""
    sequence = ""
    p_value = longest_sequence_test(sequence, block_size=8)

    assert isinstance(p_value, float)
    assert p_value == 1.0


# Тест с моками (сложный тест №2)
@patch('main.read_file')
@patch('main.frequency_test')
@patch('main.consecutive_bits_test')
@patch('main.longest_sequence_test')
@patch('main.write_file')
def test_main_with_mocks(mock_write_file, mock_longest_sequence_test,
                         mock_consecutive_bits_test, mock_frequency_test,
                         mock_read_file):
    """Тест основной функции main с использованием моков."""
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
        # Вызываем main с заглушкой для print
        with patch('builtins.print'):
            main()

        # Проверяем вызовы read_file
        assert mock_read_file.call_count == 2
        mock_read_file.assert_any_call('cpp_seq.txt')
        mock_read_file.assert_any_call('java_seq.txt')

        # Проверяем вызовы тестовых функций
        assert mock_frequency_test.call_count == 2
        assert mock_consecutive_bits_test.call_count == 2
        assert mock_longest_sequence_test.call_count == 2

        # Проверяем вызовы write_file
        assert mock_write_file.call_count == 2


def test_constants_import():
    """Тест импорта констант."""
    assert PI is not None
    assert len(PI) == 4
    assert isinstance(PI[0], float)
    assert PATH_CPP_SEQ is not None
    assert PATH_JAVA_SEQ is not None
    assert PATH_CPP_NIST_RES is not None
    assert PATH_JAVA_NIST_RES is not None


# Дополнительные тесты для edge cases
def test_frequency_test_single_bit():
    """Тест частотного теста для последовательности из одного бита."""
    sequence = "1"
    p_value = frequency_test(sequence)

    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1


def test_consecutive_bits_test_single_bit():
    """Тест для последовательности из одного бита."""
    sequence = "1"
    p_value = consecutive_bits_test(sequence)
    assert isinstance(p_value, float)
    assert p_value == 0.0


def test_longest_sequence_test_single_block():
    """Тест для одного блока."""
    sequence = "11110000"
    p_value = longest_sequence_test(sequence, block_size=8)

    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1


# Тест для проверки симметрии частотного теста
def test_frequency_test_symmetry():
    """Проверка симметрии частотного теста для нулей и единиц."""
    seq_zeros = "0" * 100
    seq_ones = "1" * 100

    p_zeros = frequency_test(seq_zeros)
    p_ones = frequency_test(seq_ones)

    # Для последовательностей из всех нулей и всех единиц
    # p-value должны быть одинаковыми
    assert abs(p_zeros - p_ones) < 1e-10


# Тест для проверки обработки ошибок в read_file
def test_read_file_nonexistent():
    """Тест чтения несуществующего файла."""
    with pytest.raises(FileNotFoundError):
        read_file("non_existent_file_12345.txt")


# Интеграционный тест
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

        # Патчим константы
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


if __name__ == "__main__":
    # Запуск тестов напрямую (для отладки)
    pytest.main([__file__, "-v"])