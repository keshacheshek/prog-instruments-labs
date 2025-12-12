# coding:utf-8

""" 利用 tkinter.Canvas 实现推箱子游戏

基本说明：
    1. 参考game_maps.py 中的现有游戏场景，自己设计新的游戏关卡（可借鉴其他推箱子游戏的场景，自定义模仿设置）
    2. 设置 push_box_game.BOX_SIZE, 设置箱子大小，缺省为64， 理论上，可按自己需要，任意设置， 一般大小为 32, 64, 96, 128
    3. 设置 push_box_game.start_index, 设置开始关卡数， 进行游戏
    4. 该程序仅用于演示参考，不是完整的商业游戏，大家可以自行扩展功能，如实时调整大小，回退功能等。

开发人员: Edwin.Zhang
开发时间: 2019-6-28
"""

# ==================== ИМПОРТЫ ====================
from PIL import Image, ImageTk
from tkinter import Tk, Canvas, Label
from tkinter.messagebox import showinfo
import numpy as np
from game_maps import basic_maps

# ==================== КОНСТАНТЫ ====================

# Конфигурация игры
TOTAL_GAMES = len(basic_maps)  # Общее количество уровней
DEFAULT_BOX_SIZE = 64  # Размер игрового блока по умолчанию
SUPPORTED_BOX_SIZES = (32, 64, 96, 128)  # Поддерживаемые размеры блоков

# Игровые константы
WALL = 0
WORKER = 1
BOX = 2
PASSAGEWAY = 3
DESTINATION = 4
WORKER_IN_DEST = 5
BOX_IN_DEST = 6

# Направления движения
DIRECTIONS = {
    "Up": (-1, 0, -2, 0),
    "Down": (1, 0, 2, 0),
    "Left": (0, -1, 0, -2),
    "Right": (0, 1, 0, 2),
}

# Пути к изображениям
IMAGE_PATHS = {
    'wall': 'images\\Wall.jpg',
    'worker': 'images\\Worker.jpg',
    'worker_in_dest': 'images\\WorkerInDest.jpg',
    'w_up': 'images\\w_up.jpg',
    'w_up_in': 'images\\w_up_in.jpg',
    'w_down': 'images\\w_down.jpg',
    'w_down_in': 'images\\w_down_in.jpg',
    'w_left': 'images\\w_left.jpg',
    'w_left_in': 'images\\w_left_in.jpg',
    'w_right': 'images\\w_right.jpg',
    'w_right_in': 'images\\w_right_in.jpg',
    'box': 'images\\Box.jpg',
    'passageway': 'images\\Passageway.jpg',
    'destination': 'images\\Destination.jpg',
    'redbox': 'images\\redbox.jpg',
    'restart': 'images\\restart.png'
}


# ==================== ФУНКЦИИ ====================

def resized_image(img_name, width=64, height=64):
    """Масштабирует изображение с сохранением пропорций"""
    img = Image.open(img_name)
    original_width, original_height = img.size

    # Сохраняем пропорции
    if original_width > original_height:
        new_width = width
        new_height = int(height * (original_height / original_width))
    else:
        new_height = height
        new_width = int(width * (original_width / original_height))

    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(resized_img)


def create_centered_window(root, width, height):
    """Создает окно, центрированное на экране"""
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    x_position = (screen_width - width) // 2
    y_position = (screen_height - height) // 3

    root.geometry(f'{width}x{height}+{x_position}+{y_position}')


# ==================== НАСТРОЙКА ПРИЛОЖЕНИЯ ====================

root = Tk()
BOX_SIZE = DEFAULT_BOX_SIZE

# Загрузка изображений
game_images = [
    resized_image(IMAGE_PATHS['wall'], BOX_SIZE, BOX_SIZE),  # Стена
    {
        'Stop': (
            resized_image(IMAGE_PATHS['worker'], BOX_SIZE, BOX_SIZE),
            resized_image(IMAGE_PATHS['worker_in_dest'], BOX_SIZE, BOX_SIZE)
        ),
        'Up': (
            resized_image(IMAGE_PATHS['w_up'], BOX_SIZE, BOX_SIZE),
            resized_image(IMAGE_PATHS['w_up_in'], BOX_SIZE, BOX_SIZE)
        ),
        'Down': (
            resized_image(IMAGE_PATHS['w_down'], BOX_SIZE, BOX_SIZE),
            resized_image(IMAGE_PATHS['w_down_in'], BOX_SIZE, BOX_SIZE)
        ),
        'Left': (
            resized_image(IMAGE_PATHS['w_left'], BOX_SIZE, BOX_SIZE),
            resized_image(IMAGE_PATHS['w_left_in'], BOX_SIZE, BOX_SIZE)
        ),
        'Right': (
            resized_image(IMAGE_PATHS['w_right'], BOX_SIZE, BOX_SIZE),
            resized_image(IMAGE_PATHS['w_right_in'], BOX_SIZE, BOX_SIZE)
        )
    },
    resized_image(IMAGE_PATHS['box'], BOX_SIZE, BOX_SIZE),  # Ящик
    resized_image(IMAGE_PATHS['passageway'], BOX_SIZE, BOX_SIZE),  # Проход
    resized_image(IMAGE_PATHS['destination'], BOX_SIZE, BOX_SIZE),  # Цель
    resized_image(IMAGE_PATHS['worker_in_dest'], BOX_SIZE, BOX_SIZE),  # Рабочий в цели
    resized_image(IMAGE_PATHS['redbox'], BOX_SIZE, BOX_SIZE)  # Ящик в цели
]


class BoxGame:
    def __init__(self, game_index=1):
        """
        Инициализация игры

        Args:
            game_index (int): Номер уровня, начинается с 1
        """
        self.game_index = max(1, game_index)
        self.game_steps = 0
        self.screen = None
        self.direction = "Stop"
        self.current_map = None
        self.rows = 0
        self.cols = 0
        self.worker_x = 0
        self.worker_y = 0

    def start_game(self, *args):
        """Запуск или перезапуск игры"""
        self._cleanup_previous_game()

        # Загрузка карты текущего уровня
        self.current_map = np.asarray(basic_maps[self.game_index - 1], dtype=np.int32)
        self.rows, self.cols = self.current_map.shape[0:2]
        self.worker_x, self.worker_y = 0, 0
        self.game_steps = 0
        self.direction = "Stop"

        # Создание игрового интерфейса
        self._create_game_interface()

        # Создание кнопки перезапуска
        self._create_restart_button()

    def _cleanup_previous_game(self):
        """Очистка предыдущей игровой сессии"""
        if self.screen is not None:
            self.screen.forget()
            self.screen = None

        if hasattr(self, 'btn_refresh'):
            self.btn_refresh.forget()
            delattr(self, 'btn_refresh')

    def _create_game_interface(self):
        """Создание игрового интерфейса"""
        window_width = self.cols * BOX_SIZE + BOX_SIZE // 2 + 40
        window_height = self.rows * BOX_SIZE + BOX_SIZE // 2 + 100

        create_centered_window(root, window_width, window_height)
        self._update_game_title()

        self.screen = Canvas(
            root,
            bg='white',
            width=BOX_SIZE * self.cols,
            height=BOX_SIZE * self.rows
        )
        self.screen.configure(highlightthickness=0)
        self._refresh_screen()
        self.screen.bind("<KeyPress>", self._handle_key_press)
        self.screen.pack(pady=20)
        self.screen.focus_set()

    def _create_restart_button(self):
        """Создание кнопки перезапуска игры"""
        self.btn_refresh = Label(root, width=150, height=50)
        restart_image = resized_image(IMAGE_PATHS['restart'], 120, 120)
        self.btn_refresh.image = restart_image
        self.btn_refresh.config(image=restart_image)
        self.btn_refresh.bind("<Button-1>", self.start_game)
        self.btn_refresh.pack()

    def _refresh_screen(self):
        """Обновление игрового экрана"""
        self.screen.delete('all')

        for i in range(self.rows):
            for j in range(self.cols):
                cell_value = self.current_map[i, j]
                image_to_draw = self._get_cell_image(cell_value, i, j)

                if image_to_draw:
                    x_position = j * BOX_SIZE + BOX_SIZE // 2
                    y_position = i * BOX_SIZE + BOX_SIZE // 2
                    self.screen.create_image((x_position, y_position), image=image_to_draw)

        root.update()
        self.screen.focus_set()

    def _get_cell_image(self, cell_value, i, j):
        """Получение изображения для клетки"""
        if cell_value == WORKER:
            self.worker_x, self.worker_y = i, j
            return game_images[WORKER][self.direction][0]
        elif cell_value == WORKER_IN_DEST:
            self.worker_x, self.worker_y = i, j
            return game_images[WORKER][self.direction][1]
        elif cell_value == -1:
            return None
        else:
            return game_images[cell_value]

    def _handle_key_press(self, event):
        """Обработка нажатий клавиш"""
        key = event.keysym

        if key in DIRECTIONS:
            self._move_worker(key)
        elif key == "space":
            self.start_game()

    def _move_worker(self, direction):
        """Перемещение рабочего в указанном направлении"""
        self.direction = direction
        move_data = DIRECTIONS[direction]

        next_x = self.worker_x + move_data[0]
        next_y = self.worker_y + move_data[1]
        after_next_x = self.worker_x + move_data[2]
        after_next_y = self.worker_y + move_data[3]

        next_cell = self._get_cell_value(next_x, next_y)
        after_next_cell = self._get_cell_value(after_next_x, after_next_y)

        if not self._is_valid_move(next_cell, after_next_cell, next_x, next_y, after_next_x, after_next_y):
            return

        self._process_movement(next_x, next_y, after_next_x, after_next_y, next_cell, after_next_cell)

        self.game_steps += 1
        self._update_game_title()
        self._refresh_screen()

        if self._is_level_completed():
            self._handle_level_completion()

    def _get_cell_value(self, x, y):
        """Получение значения клетки"""
        if self._is_within_bounds(x, y):
            return self.current_map[x, y]
        return None

    def _is_valid_move(self, next_cell, after_next_cell, next_x, next_y, after_next_x, after_next_y):
        """Проверка валидности хода"""
        if next_cell == WALL or not self._is_within_bounds(next_x, next_y):
            return False

        if next_cell == BOX:
            if (after_next_cell == WALL or
                    not self._is_within_bounds(after_next_x, after_next_y) or
                    after_next_cell == BOX):
                return False

        return True

    def _process_movement(self, next_x, next_y, after_next_x, after_next_y, next_cell, after_next_cell):
        """Обработка движения рабочего"""
        self._clear_worker_position()

        if next_cell == PASSAGEWAY:
            self.current_map[next_x, next_y] = WORKER
        elif next_cell == DESTINATION:
            self.current_map[next_x, next_y] = WORKER_IN_DEST
        elif next_cell == BOX:
            if after_next_cell == PASSAGEWAY:
                self.current_map[after_next_x, after_next_y] = BOX
                self.current_map[next_x, next_y] = WORKER
            elif after_next_cell == DESTINATION:
                self.current_map[after_next_x, after_next_y] = BOX_IN_DEST
                self.current_map[next_x, next_y] = WORKER
        elif next_cell == BOX_IN_DEST:
            if after_next_cell == PASSAGEWAY:
                self.current_map[after_next_x, after_next_y] = BOX
                self.current_map[next_x, next_y] = WORKER_IN_DEST
            elif after_next_cell == DESTINATION:
                self.current_map[after_next_x, after_next_y] = BOX_IN_DEST
                self.current_map[next_x, next_y] = WORKER_IN_DEST

        self.worker_x, self.worker_y = next_x, next_y

    def _clear_worker_position(self):
        """Очистка текущей позиции рабочего"""
        if self.current_map[self.worker_x, self.worker_y] == WORKER:
            self.current_map[self.worker_x, self.worker_y] = PASSAGEWAY
        elif self.current_map[self.worker_x, self.worker_y] == WORKER_IN_DEST:
            self.current_map[self.worker_x, self.worker_y] = DESTINATION

    def _is_within_bounds(self, row, col):
        """Проверка, находится ли позиция в пределах карты"""
        return 0 <= row < self.rows and 0 <= col < self.cols

    def _update_game_title(self):
        """Обновление заголовка окна"""
        title = f"推箱子 - 第({self.game_index}/{TOTAL_GAMES})关    总步数: {self.game_steps}"
        root.title(title)

    def _is_level_completed(self):
        """Проверка завершения уровня"""
        return not np.any([
            self.current_map == DESTINATION,
            self.current_map == WORKER_IN_DEST
        ])

    def _handle_level_completion(self):
        """Обработка завершения уровня"""
        message = f"恭喜你顺利通过第({self.game_index})关!\n\n一共用了({self.game_steps})步"
        showinfo(title="提示", message=message)

        self.game_index = self.game_index % TOTAL_GAMES + 1
        self.start_game()


if __name__ == "__main__":
    START_LEVEL = 1
    BoxGame(game_index=START_LEVEL).start_game()
    root.mainloop()