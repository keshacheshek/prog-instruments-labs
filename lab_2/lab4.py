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
from typing import Tuple, Optional, Dict

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

def resized_image(img_name: str, width: int = 64, height: int = 64) -> ImageTk.PhotoImage:
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


def create_centered_window(root: Tk, width: int, height: int) -> None:
    """Создает окно, центрированное на экране"""
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    x_position = (screen_width - width) // 2
    y_position = (screen_height - height) // 3

    root.geometry(f'{width}x{height}+{x_position}+{y_position}')


# ==================== КЛАСС ИГРОВОЙ КАРТЫ ====================

class GameMap:
    """Класс для управления игровой картой"""

    def __init__(self, map_data: np.ndarray):
        self.map_data = np.asarray(map_data, dtype=np.int32)
        self.rows, self.cols = self.map_data.shape
        self.worker_position = self._find_worker_position()

    def _find_worker_position(self) -> Tuple[int, int]:
        """Находит позицию рабочего на карте"""
        for i in range(self.rows):
            for j in range(self.cols):
                if self.map_data[i, j] in (WORKER, WORKER_IN_DEST):
                    return i, j
        return 0, 0

    def get_cell_value(self, row: int, col: int) -> Optional[int]:
        """Получает значение клетки карты"""
        if self.is_within_bounds(row, col):
            return self.map_data[row, col]
        return None

    def set_cell_value(self, row: int, col: int, value: int) -> None:
        """Устанавливает значение клетки карты"""
        if self.is_within_bounds(row, col):
            self.map_data[row, col] = value

    def is_within_bounds(self, row: int, col: int) -> bool:
        """Проверяет, находится ли позиция в пределах карты"""
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_level_completed(self) -> bool:
        """Проверяет, завершен ли уровень"""
        return not np.any([
            self.map_data == DESTINATION,
            self.map_data == WORKER_IN_DEST
        ])

    def clear_worker_position(self, position: Tuple[int, int]) -> None:
        """Очищает текущую позицию рабочего"""
        row, col = position
        if self.map_data[row, col] == WORKER:
            self.map_data[row, col] = PASSAGEWAY
        elif self.map_data[row, col] == WORKER_IN_DEST:
            self.map_data[row, col] = DESTINATION

    def update_worker_position(self, new_position: Tuple[int, int], old_position: Tuple[int, int]) -> None:
        """Обновляет позицию рабочего"""
        self.clear_worker_position(old_position)
        row, col = new_position
        current_value = self.map_data[row, col]

        if current_value == PASSAGEWAY:
            self.map_data[row, col] = WORKER
        elif current_value == DESTINATION:
            self.map_data[row, col] = WORKER_IN_DEST


# ==================== КЛАСС ОТРИСОВКИ ИГРЫ ====================

class GameRenderer:
    """Класс для отрисовки игрового состояния"""

    def __init__(self, canvas: Canvas, box_size: int):
        self.canvas = canvas
        self.box_size = box_size
        self.images = self._load_images()

    def _load_images(self) -> list:
        """Загружает изображения для игры"""
        return [
            resized_image(IMAGE_PATHS['wall'], self.box_size, self.box_size),  # Стена
            {
                'Stop': (
                    resized_image(IMAGE_PATHS['worker'], self.box_size, self.box_size),
                    resized_image(IMAGE_PATHS['worker_in_dest'], self.box_size, self.box_size)
                ),
                'Up': (
                    resized_image(IMAGE_PATHS['w_up'], self.box_size, self.box_size),
                    resized_image(IMAGE_PATHS['w_up_in'], self.box_size, self.box_size)
                ),
                'Down': (
                    resized_image(IMAGE_PATHS['w_down'], self.box_size, self.box_size),
                    resized_image(IMAGE_PATHS['w_down_in'], self.box_size, self.box_size)
                ),
                'Left': (
                    resized_image(IMAGE_PATHS['w_left'], self.box_size, self.box_size),
                    resized_image(IMAGE_PATHS['w_left_in'], self.box_size, self.box_size)
                ),
                'Right': (
                    resized_image(IMAGE_PATHS['w_right'], self.box_size, self.box_size),
                    resized_image(IMAGE_PATHS['w_right_in'], self.box_size, self.box_size)
                )
            },
            resized_image(IMAGE_PATHS['box'], self.box_size, self.box_size),  # Ящик
            resized_image(IMAGE_PATHS['passageway'], self.box_size, self.box_size),  # Проход
            resized_image(IMAGE_PATHS['destination'], self.box_size, self.box_size),  # Цель
            resized_image(IMAGE_PATHS['worker_in_dest'], self.box_size, self.box_size),  # Рабочий в цели
            resized_image(IMAGE_PATHS['redbox'], self.box_size, self.box_size)  # Ящик в цели
        ]

    def render(self, game_map: GameMap, worker_direction: str) -> None:
        """Отрисовывает игровую карту"""
        self.canvas.delete('all')

        for i in range(game_map.rows):
            for j in range(game_map.cols):
                cell_value = game_map.get_cell_value(i, j)
                image_to_draw = self._get_cell_image(cell_value, worker_direction)

                if image_to_draw:
                    x_position = j * self.box_size + self.box_size // 2
                    y_position = i * self.box_size + self.box_size // 2
                    self.canvas.create_image((x_position, y_position), image=image_to_draw)

    def _get_cell_image(self, cell_value: int, worker_direction: str) -> Optional[ImageTk.PhotoImage]:
        """Получает изображение для клетки"""
        if cell_value == WORKER:
            return self.images[WORKER][worker_direction][0]
        elif cell_value == WORKER_IN_DEST:
            return self.images[WORKER][worker_direction][1]
        elif cell_value == -1:
            return None
        elif cell_value in range(len(self.images)):
            return self.images[cell_value]
        return None


# ==================== КЛАСС ИГРОВОГО КОНТРОЛЛЕРА ====================

class GameController:
    """Класс для управления игровой логикой"""

    def __init__(self):
        self.game_map = None
        self.game_steps = 0
        self.current_level = 1

    def load_level(self, level_index: int) -> GameMap:
        """Загружает уровень по индексу"""
        if 1 <= level_index <= TOTAL_GAMES:
            map_data = basic_maps[level_index - 1]
            self.game_map = GameMap(map_data)
            self.current_level = level_index
            self.game_steps = 0
            return self.game_map
        raise ValueError(f"Уровень {level_index} не существует")

    def move_worker(self, direction: str) -> bool:
        """Пытается переместить рабочего в указанном направлении"""
        if self.game_map is None:
            return False

        move_data = DIRECTIONS[direction]
        current_row, current_col = self.game_map.worker_position

        next_row = current_row + move_data[0]
        next_col = current_col + move_data[1]
        after_next_row = current_row + move_data[2]
        after_next_col = current_col + move_data[3]

        next_cell = self.game_map.get_cell_value(next_row, next_col)
        after_next_cell = self.game_map.get_cell_value(after_next_row, after_next_col)

        if not self._is_valid_move(next_cell, after_next_cell, next_row, next_col, after_next_row, after_next_col):
            return False

        self._execute_move(current_row, current_col, next_row, next_col, after_next_row, after_next_col,
                           next_cell, after_next_cell)

        self.game_steps += 1
        return True

    def _is_valid_move(self, next_cell: Optional[int], after_next_cell: Optional[int],
                       next_row: int, next_col: int, after_next_row: int, after_next_col: int) -> bool:
        """Проверяет валидность хода"""
        if next_cell == WALL or not self.game_map.is_within_bounds(next_row, next_col):
            return False

        if next_cell == BOX:
            if (after_next_cell == WALL or
                    not self.game_map.is_within_bounds(after_next_row, after_next_col) or
                    after_next_cell == BOX):
                return False

        return True

    def _execute_move(self, current_row: int, current_col: int,
                      next_row: int, next_col: int, after_next_row: int, after_next_col: int,
                      next_cell: Optional[int], after_next_cell: Optional[int]) -> None:
        """Выполняет перемещение рабочего"""
        current_position = (current_row, current_col)
        next_position = (next_row, next_col)

        if next_cell in (PASSAGEWAY, DESTINATION):
            self.game_map.update_worker_position(next_position, current_position)

        elif next_cell == BOX:
            if after_next_cell == PASSAGEWAY:
                self.game_map.set_cell_value(after_next_row, after_next_col, BOX)
                self.game_map.update_worker_position(next_position, current_position)
            elif after_next_cell == DESTINATION:
                self.game_map.set_cell_value(after_next_row, after_next_col, BOX_IN_DEST)
                self.game_map.update_worker_position(next_position, current_position)

        elif next_cell == BOX_IN_DEST:
            if after_next_cell == PASSAGEWAY:
                self.game_map.set_cell_value(after_next_row, after_next_col, BOX)
                self.game_map.clear_worker_position(current_position)
                self.game_map.set_cell_value(next_row, next_col, WORKER_IN_DEST)
            elif after_next_cell == DESTINATION:
                self.game_map.set_cell_value(after_next_row, after_next_col, BOX_IN_DEST)
                self.game_map.clear_worker_position(current_position)
                self.game_map.set_cell_value(next_row, next_col, WORKER_IN_DEST)

        # Обновляем позицию рабочего в объекте карты
        self.game_map.worker_position = next_position

    def is_level_completed(self) -> bool:
        """Проверяет, завершен ли текущий уровень"""
        return self.game_map.is_level_completed() if self.game_map else False

    def next_level(self) -> bool:
        """Переходит к следующему уровню"""
        self.current_level = self.current_level % TOTAL_GAMES + 1
        return True


# ==================== КЛАСС ИГРОВОГО ИНТЕРФЕЙСА ====================

class GameUI:
    """Класс для управления пользовательским интерфейсом"""

    def __init__(self, root: Tk, box_size: int = DEFAULT_BOX_SIZE):
        self.root = root
        self.box_size = box_size
        self.canvas = None
        self.restart_button = None
        self.game_controller = GameController()
        self.renderer = None
        self.worker_direction = "Stop"

    def setup(self) -> None:
        """Настраивает пользовательский интерфейс"""
        self.root.title("推箱子")
        self._create_game_interface()
        self._create_restart_button()
        self._bind_events()

    def _create_game_interface(self) -> None:
        """Создает игровой интерфейс"""
        game_map = self.game_controller.load_level(self.game_controller.current_level)

        window_width = game_map.cols * self.box_size + self.box_size // 2 + 40
        window_height = game_map.rows * self.box_size + self.box_size // 2 + 100

        create_centered_window(self.root, window_width, window_height)
        self._update_title()

        self.canvas = Canvas(
            self.root,
            bg='white',
            width=self.box_size * game_map.cols,
            height=self.box_size * game_map.rows
        )
        self.canvas.configure(highlightthickness=0)
        self.canvas.pack(pady=20)

        self.renderer = GameRenderer(self.canvas, self.box_size)
        self._render_game()

    def _create_restart_button(self) -> None:
        """Создает кнопку перезапуска"""
        self.restart_button = Label(self.root, width=150, height=50)
        restart_image = resized_image(IMAGE_PATHS['restart'], 120, 120)
        self.restart_button.image = restart_image
        self.restart_button.config(image=restart_image)
        self.restart_button.bind("<Button-1>", self._restart_game)
        self.restart_button.pack()

    def _bind_events(self) -> None:
        """Привязывает обработчики событий"""
        self.canvas.bind("<KeyPress>", self._handle_key_press)
        self.canvas.focus_set()

    def _handle_key_press(self, event) -> None:
        """Обрабатывает нажатия клавиш"""
        key = event.keysym

        if key in DIRECTIONS:
            self._handle_movement(key)
        elif key == "space":
            self._restart_game()

    def _handle_movement(self, direction: str) -> None:
        """Обрабатывает движение рабочего"""
        self.worker_direction = direction

        if self.game_controller.move_worker(direction):
            self._render_game()
            self._update_title()

            if self.game_controller.is_level_completed():
                self._handle_level_completion()

    def _render_game(self) -> None:
        """Отрисовывает игровое состояние"""
        if self.renderer and self.game_controller.game_map:
            self.renderer.render(self.game_controller.game_map, self.worker_direction)
            self.root.update()
            self.canvas.focus_set()

    def _update_title(self) -> None:
        """Обновляет заголовок окна"""
        title = (f"推箱子 - 第({self.game_controller.current_level}/{TOTAL_GAMES})关    "
                 f"总步数: {self.game_controller.game_steps}")
        self.root.title(title)

    def _restart_game(self, event=None) -> None:
        """Перезапускает текущий уровень"""
        self.game_controller.load_level(self.game_controller.current_level)
        self.worker_direction = "Stop"
        self._render_game()
        self._update_title()

    def _handle_level_completion(self) -> None:
        """Обрабатывает завершение уровня"""
        message = (f"恭喜你顺利通过第({self.game_controller.current_level})关!\n\n"
                   f"一共用了({self.game_controller.game_steps})步")
        showinfo(title="提示", message=message)

        self.game_controller.next_level()
        self.worker_direction = "Stop"
        self._create_game_interface()
        self._update_title()


# ==================== ОСНОВНОЙ КОД ====================

if __name__ == "__main__":
    START_LEVEL = 1

    root = Tk()
    game_ui = GameUI(root)
    game_ui.setup()

    root.mainloop()