import os
import sys
import tempfile
from unittest.mock import patch, mock_open

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