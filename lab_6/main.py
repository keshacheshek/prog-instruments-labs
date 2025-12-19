from constants import PI, PATH_CPP_SEQ, PATH_JAVA_SEQ, PATH_CPP_NIST_RES, PATH_JAVA_NIST_RES
import math


def read_file(file_name) -> str:
    """
    Чтение текста из файла
    :param file_name: путь до файла
    """
    with open(file_name, 'r', encoding='utf-8') as file:
        return file.read()


def write_file(file_path, test1, test2, test3):
    """
    Записываем данные в файл
    :param file_path: Путь до файла
    :param test1: Результат частотного побитового теста
    :param test2: Результат теста на одинаковые подряд идущие биты
    :param test3: Результат теста на самую длинную последовательность единиц в блоке
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(f"Результат частотного теста P-value: {test1}")
        file.write(f"\nРезультат теста на одинаковые подряд идущие биты P-value: {test2}")
        file.write(f"\nРезультат теста на самую длинную последовательность: {test3}")


def frequency_test(sequence: str) -> float:
    """
    Частотный побитовый анализ
    :param sequence: Исследуемая последовательность
    :return: P-значение
    """
    if len(sequence) == 0:
        return 1.0

    sn = 0
    for i in sequence:
        if i == '0':
            sn -= 1
        if i == '1':
            sn += 1

    sn = sn / (len(sequence) ** 0.5)
    return math.erfc(abs(sn / (2 ** 0.5)))


def consecutive_bits_test(sequence: str) -> float:
    """
    Тест на одинаковые подряд идущие биты
    :param sequence: Исследуемая последовательность
    :return: P-значение
    """
    if len(sequence) == 0:
        return 1.0

    sum1 = 0
    for i in sequence:
        if i == '1':
            sum1 += 1
    z = sum1 / len(sequence)

    if abs(z - 0.5) >= (2 / len(sequence) ** 0.5):
        return 0.0

    # Защита от деления на ноль
    if z == 0 or z == 1:
        return 0.0

    vn = 0
    for i in range(len(sequence) - 1):
        if sequence[i] != sequence[i + 1]:
            vn += 1

    denominator = 2 * (2 * len(sequence)) ** 0.5 * z * (1 - z)
    if denominator == 0:
        return 0.0

    return math.erfc(abs(vn - 2 * len(sequence) * z * (1 - z)) / denominator)


def longest_sequence_test(sequence, block_size=8) -> float:
    """
    Тест на самую длинную последовательность единиц в блоке
    :param sequence: Исследуемая последовательность
    :param block_size: Длина блока
    :return: P-значение
    """
    if len(sequence) == 0 or block_size == 0:
        return 1.0

    n = len(sequence)
    m = block_size

    # Разбиваем последовательность на блоки
    num_blocks = n // m
    if num_blocks == 0:
        return 1.0

    # Для каждого блока находим максимальную длину последовательности единиц
    max_runs = []
    for i in range(num_blocks):
        block = sequence[i * m:(i + 1) * m]
        max_run = 0
        current_run = 0

        for bit in block:
            if bit == '1':
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0

        max_runs.append(max_run)

    # Вычисляем статистику V по формуле NIST
    # Используем предопределенные значения PI для разных длин блока
    K = 3  # для блока 8 бит
    nu = [0, 0, 0, 0]  # частоты для 4 категорий

    for run in max_runs:
        if run <= 1:
            nu[0] += 1
        elif run == 2:
            nu[1] += 1
        elif run == 3:
            nu[2] += 1
        else:  # run >= 4
            nu[3] += 1

    # Вычисляем хи-квадрат статистику
    chi_square = 0
    for i in range(4):
        expected = PI[i] * num_blocks
        if expected > 0:  # избегаем деления на ноль
            chi_square += ((nu[i] - expected) ** 2) / expected

    # Вычисляем P-value через неполную гамма-функцию (упрощенный вариант)
    # Для K=3 степеней свободы
    # Используем аппроксимацию через математические функции Python
    p_value = math.exp(-chi_square / 2)

    # Ограничиваем значение от 0 до 1
    return max(0.0, min(1.0, p_value))


def main():
    sequence_cpp = read_file(PATH_CPP_SEQ)
    sequence_java = read_file(PATH_JAVA_SEQ)

    p_value_freq_cpp = frequency_test(sequence_cpp)
    p_value_cons_bit_cpp = consecutive_bits_test(sequence_cpp)
    p_value_long_seq_cpp = longest_sequence_test(sequence_cpp)

    p_value_freq_java = frequency_test(sequence_java)
    p_value_cons_bit_java = consecutive_bits_test(sequence_java)
    p_value_long_seq_java = longest_sequence_test(sequence_java)

    try:
        write_file(PATH_CPP_NIST_RES, p_value_freq_cpp, p_value_cons_bit_cpp, p_value_long_seq_cpp)
        write_file(PATH_JAVA_NIST_RES, p_value_freq_java, p_value_cons_bit_java, p_value_long_seq_java)

        print("C++ Sequence:")
        print(f"Frequency test P-value: {p_value_freq_cpp}")
        print(f"Identical consecutive bits P-value: {p_value_cons_bit_cpp}")
        print(f"Longest sequence of units in a block P-value: {p_value_long_seq_cpp}")

        print("\nJava Sequence:")
        print(f"Frequency Test P-value: {p_value_freq_java}")
        print(f"Identical consecutive bits P-value: {p_value_cons_bit_java}")
        print(f"Longest sequence of units in a block P-value: {p_value_long_seq_java}")

        print(f"Анализ завершен. Результаты сохранены в файлы '{PATH_CPP_NIST_RES}' и '{PATH_JAVA_NIST_RES}'")

    except IOError:
        print(f"Ошибка: Не удалось записать данные в файл")
    except Exception as e:
        print(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    main()