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
from typing import Tuple, Optional, Dict, List, Any
from dataclasses import dataclass
from enum import IntEnum
import os


# ==================== ПЕРЕЧИСЛЕНИЯ И СТРУКТУРЫ ====================

class CellType(IntEnum):
    """Типы клеток игрового поля"""
    WALL = 0
    WORKER = 1
    BOX = 2
    PASSAGEWAY = 3
    DESTINATION = 4
    WORKER_IN_DEST = 5
    BOX_IN_DEST = 6


class Direction(IntEnum):
    """Направления движения"""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STOP = 4


@dataclass
class ImageConfig:
    """Конфигурация изображения"""
    path: str
    width: int
    height: int
    keep_aspect_ratio: bool = True


@dataclass
class GameConfig:
    """Конфигурация игры"""
    box_size: int = 64
    supported_box_sizes: Tuple[int, ...] = (32, 64, 96, 128)
    images_directory: str = "images"
    enable_image_cache: bool = True
    default_start_level: int = 1


# ==================== КОНСТАНТЫ ====================

TOTAL_GAMES = len(basic_maps)  # Общее количество уровней

DIRECTION_MAPPING = {
    "Up": (-1, 0, -2, 0),
    "Down": (1, 0, 2, 0),
    "Left": (0, -1, 0, -2),
    "Right": (0, 1, 0, 2),
}

DIRECTION_TO_ENUM = {
    "Up": Direction.UP,
    "Down": Direction.DOWN,
    "Left": Direction.LEFT,
    "Right": Direction.RIGHT,
    "Stop": Direction.STOP
}

ENUM_TO_DIRECTION = {v: k for k, v in DIRECTION_TO_ENUM.items()}


# ==================== КЛАСС УПРАВЛЕНИЯ РЕСУРСАМИ ====================

class ResourceManager:
    """Менеджер ресурсов с кэшированием"""

    _instance = None
    _image_cache: Dict[str, Dict[Tuple[int, int], ImageTk.PhotoImage]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._config = GameConfig()
            self._image_configs = self._create_image_configs()

    def _create_image_configs(self) -> Dict[str, ImageConfig]:
        """Создает конфигурации изображений"""
        return {
            'wall': ImageConfig('Wall.jpg', self._config.box_size, self._config.box_size),
            'worker': ImageConfig('Worker.jpg', self._config.box_size, self._config.box_size),
            'worker_in_dest': ImageConfig('WorkerInDest.jpg', self._config.box_size, self._config.box_size),
            'w_up': ImageConfig('w_up.jpg', self._config.box_size, self._config.box_size),
            'w_up_in': ImageConfig('w_up_in.jpg', self._config.box_size, self._config.box_size),
            'w_down': ImageConfig('w_down.jpg', self._config.box_size, self._config.box_size),
            'w_down_in': ImageConfig('w_down_in.jpg', self._config.box_size, self._config.box_size),
            'w_left': ImageConfig('w_left.jpg', self._config.box_size, self._config.box_size),
            'w_left_in': ImageConfig('w_left_in.jpg', self._config.box_size, self._config.box_size),
            'w_right': ImageConfig('w_right.jpg', self._config.box_size, self._config.box_size),
            'w_right_in': ImageConfig('w_right_in.jpg', self._config.box_size, self._config.box_size),
            'box': ImageConfig('Box.jpg', self._config.box_size, self._config.box_size),
            'passageway': ImageConfig('Passageway.jpg', self._config.box_size, self._config.box_size),
            'destination': ImageConfig('Destination.jpg', self._config.box_size, self._config.box_size),
            'redbox': ImageConfig('redbox.jpg', self._config.box_size, self._config.box_size),
            'restart': ImageConfig('restart.png', 120, 120, keep_aspect_ratio=False)
        }

    def get_full_path(self, image_name: str) -> str:
        """Получает полный путь к изображению"""
        base_path = self._config.images_directory
        if not os.path.exists(base_path):
            os.makedirs(base_path, exist_ok=True)
        return os.path.join(base_path, image_name)

    def load_image(self, image_config: ImageConfig) -> ImageTk.PhotoImage:
        """Загружает и масштабирует изображение"""
        cache_key = (image_config.path, image_config.width, image_config.height)

        # Проверка кэша
        if self._config.enable_image_cache and image_config.path in self._image_cache:
            if cache_key in self._image_cache[image_config.path]:
                return self._image_cache[image_config.path][cache_key]

        # Загрузка изображения
        full_path = self.get_full_path(image_config.path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Изображение не найдено: {full_path}")

        img = Image.open(full_path)

        # Масштабирование
        if image_config.keep_aspect_ratio:
            scaled_img = self._resize_keep_aspect(img, image_config.width, image_config.height)
        else:
            scaled_img = img.resize((image_config.width, image_config.height), Image.Resampling.LANCZOS)

        tk_image = ImageTk.PhotoImage(scaled_img)

        # Кэширование
        if self._config.enable_image_cache:
            if image_config.path not in self._image_cache:
                self._image_cache[image_config.path] = {}
            self._image_cache[image_config.path][cache_key] = tk_image

        return tk_image

    def _resize_keep_aspect(self, img: Image.Image, target_width: int, target_height: int) -> Image.Image:
        """Масштабирует изображение с сохранением пропорций"""
        original_width, original_height = img.size

        if original_width > original_height:
            new_width = target_width
            new_height = int(target_height * (original_height / original_width))
        else:
            new_height = target_height
            new_width = int(target_width * (original_width / original_height))

        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def clear_cache(self) -> None:
        """Очищает кэш изображений"""
        self._image_cache.clear()

    def update_config(self, new_config: GameConfig) -> None:
        """Обновляет конфигурацию и очищает кэш"""
        self._config = new_config
        self._image_configs = self._create_image_configs()
        self.clear_cache()


# ==================== КЛАСС РЕСУРСОВ ИГРЫ ====================

class GameResources:
    """Класс для управления игровыми ресурсами"""

    def __init__(self, box_size: int = 64):
        self.box_size = box_size
        self.resource_manager = ResourceManager()
        self._images = None

        # Обновляем конфигурацию менеджера ресурсов
        config = GameConfig(box_size=box_size)
        self.resource_manager.update_config(config)

        self._load_all_images()

    def _load_all_images(self) -> None:
        """Загружает все изображения для игры"""
        configs = self.resource_manager._image_configs

        self._images = {
            CellType.WALL: self.resource_manager.load_image(configs['wall']),
            CellType.BOX: self.resource_manager.load_image(configs['box']),
            CellType.PASSAGEWAY: self.resource_manager.load_image(configs['passageway']),
            CellType.DESTINATION: self.resource_manager.load_image(configs['destination']),
            CellType.WORKER_IN_DEST: self.resource_manager.load_image(configs['worker_in_dest']),
            CellType.BOX_IN_DEST: self.resource_manager.load_image(configs['redbox']),
            'worker': {
                Direction.STOP: (
                    self.resource_manager.load_image(configs['worker']),
                    self.resource_manager.load_image(configs['worker_in_dest'])
                ),
                Direction.UP: (
                    self.resource_manager.load_image(configs['w_up']),
                    self.resource_manager.load_image(configs['w_up_in'])
                ),
                Direction.DOWN: (
                    self.resource_manager.load_image(configs['w_down']),
                    self.resource_manager.load_image(configs['w_down_in'])
                ),
                Direction.LEFT: (
                    self.resource_manager.load_image(configs['w_left']),
                    self.resource_manager.load_image(configs['w_left_in'])
                ),
                Direction.RIGHT: (
                    self.resource_manager.load_image(configs['w_right']),
                    self.resource_manager.load_image(configs['w_right_in'])
                )
            },
            'restart': self.resource_manager.load_image(configs['restart'])
        }

    def get_cell_image(self, cell_type: CellType, direction: Direction = Direction.STOP,
                       is_in_destination: bool = False) -> Optional[ImageTk.PhotoImage]:
        """Получает изображение для типа клетки"""
        if cell_type == CellType.WORKER:
            direction_key = direction if direction in self._images['worker'] else Direction.STOP
            return self._images['worker'][direction_key][1 if is_in_destination else 0]

        if cell_type in self._images:
            return self._images[cell_type]

        return None

    def get_restart_image(self) -> ImageTk.PhotoImage:
        """Получает изображение для кнопки перезапуска"""
        return self._images['restart']

    def update_box_size(self, new_box_size: int) -> None:
        """Обновляет размер блока и перезагружает изображения"""
        if new_box_size != self.box_size:
            self.box_size = new_box_size
            config = GameConfig(box_size=new_box_size)
            self.resource_manager.update_config(config)
            self._load_all_images()


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
                cell_value = self.map_data[i, j]
                if cell_value == CellType.WORKER or cell_value == CellType.WORKER_IN_DEST:
                    return i, j
        return 0, 0

    def get_cell_type(self, row: int, col: int) -> Optional[CellType]:
        """Получает тип клетки карты"""
        if self.is_within_bounds(row, col):
            value = self.map_data[row, col]
            try:
                return CellType(value)
            except ValueError:
                return None
        return None

    def set_cell_type(self, row: int, col: int, cell_type: CellType) -> None:
        """Устанавливает тип клетки карты"""
        if self.is_within_bounds(row, col):
            self.map_data[row, col] = cell_type.value

    def is_within_bounds(self, row: int, col: int) -> bool:
        """Проверяет, находится ли позиция в пределах карты"""
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_level_completed(self) -> bool:
        """Проверяет, завершен ли уровень"""
        return not np.any([
            self.map_data == CellType.DESTINATION.value,
            self.map_data == CellType.WORKER_IN_DEST.value
        ])

    def clear_worker_position(self, position: Tuple[int, int]) -> None:
        """Очищает текущую позицию рабочего"""
        row, col = position
        if self.map_data[row, col] == CellType.WORKER.value:
            self.map_data[row, col] = CellType.PASSAGEWAY.value
        elif self.map_data[row, col] == CellType.WORKER_IN_DEST.value:
            self.map_data[row, col] = CellType.DESTINATION.value

    def update_worker_position(self, new_position: Tuple[int, int],
                               old_position: Tuple[int, int]) -> None:
        """Обновляет позицию рабочего"""
        self.clear_worker_position(old_position)
        row, col = new_position
        current_type = self.get_cell_type(row, col)

        if current_type == CellType.PASSAGEWAY:
            self.set_cell_type(row, col, CellType.WORKER)
        elif current_type == CellType.DESTINATION:
            self.set_cell_type(row, col, CellType.WORKER_IN_DEST)

        self.worker_position = new_position


# ==================== КЛАСС ОТРИСОВКИ ИГРЫ ====================

class GameRenderer:
    """Класс для отрисовки игрового состояния"""

    def __init__(self, canvas: Canvas, resources: GameResources):
        self.canvas = canvas
        self.resources = resources
        self.box_size = resources.box_size

    def render(self, game_map: GameMap, worker_direction: Direction) -> None:
        """Отрисовывает игровую карту"""
        self.canvas.delete('all')

        for i in range(game_map.rows):
            for j in range(game_map.cols):
                cell_type = game_map.get_cell_type(i, j)

                if cell_type is not None:
                    image_to_draw = self._get_cell_image(cell_type, i, j, game_map, worker_direction)

                    if image_to_draw:
                        x_position = j * self.box_size + self.box_size // 2
                        y_position = i * self.box_size + self.box_size // 2
                        self.canvas.create_image((x_position, y_position), image=image_to_draw)

        self.canvas.update()

    def _get_cell_image(self, cell_type: CellType, row: int, col: int,
                        game_map: GameMap, worker_direction: Direction) -> Optional[ImageTk.PhotoImage]:
        """Получает изображение для клетки"""
        if cell_type == CellType.WORKER:
            is_in_destination = False
            return self.resources.get_cell_image(cell_type, worker_direction, is_in_destination)

        elif cell_type == CellType.WORKER_IN_DEST:
            is_in_destination = True
            return self.resources.get_cell_image(CellType.WORKER, worker_direction, is_in_destination)

        else:
            return self.resources.get_cell_image(cell_type)


# ==================== КЛАСС ИГРОВОГО КОНТРОЛЛЕРА ====================

class GameController:
    """Класс для управления игровой логикой"""

    def __init__(self):
        self.game_map = None
        self.game_steps = 0
        self.current_level = 1
        self._move_history: List[Tuple[GameMap, int]] = []

    def load_level(self, level_index: int) -> GameMap:
        """Загружает уровень по индексу"""
        if 1 <= level_index <= TOTAL_GAMES:
            map_data = basic_maps[level_index - 1]
            self.game_map = GameMap(map_data)
            self.current_level = level_index
            self.game_steps = 0
            self._move_history.clear()
            return self.game_map
        raise ValueError(f"Уровень {level_index} не существует")

    def move_worker(self, direction_str: str) -> bool:
        """Пытается переместить рабочего в указанном направлении"""
        if self.game_map is None:
            return False

        # Сохраняем текущее состояние для возможного отката
        self._save_state()

        direction = DIRECTION_TO_ENUM.get(direction_str, Direction.STOP)
        move_data = DIRECTION_MAPPING.get(direction_str, (0, 0, 0, 0))

        current_row, current_col = self.game_map.worker_position

        next_row = current_row + move_data[0]
        next_col = current_col + move_data[1]
        after_next_row = current_row + move_data[2]
        after_next_col = current_col + move_data[3]

        next_cell = self.game_map.get_cell_type(next_row, next_col)
        after_next_cell = self.game_map.get_cell_type(after_next_row, after_next_col)

        if not self._is_valid_move(next_cell, after_next_cell, next_row, next_col,
                                   after_next_row, after_next_col):
            return False

        self._execute_move(current_row, current_col, next_row, next_col,
                           after_next_row, after_next_col, next_cell, after_next_cell)

        self.game_steps += 1
        return True

    def _save_state(self) -> None:
        """Сохраняет текущее состояние игры"""
        if self.game_map:
            # Сохраняем копию карты и количество шагов
            map_copy = GameMap(self.game_map.map_data.copy())
            self._move_history.append((map_copy, self.game_steps))

            # Ограничиваем историю последними 50 ходами
            if len(self._move_history) > 50:
                self._move_history.pop(0)

    def undo_move(self) -> bool:
        """Отменяет последний ход"""
        if self._move_history:
            self.game_map, self.game_steps = self._move_history.pop()
            return True
        return False

    def _is_valid_move(self, next_cell: Optional[CellType], after_next_cell: Optional[CellType],
                       next_row: int, next_col: int, after_next_row: int, after_next_col: int) -> bool:
        """Проверяет валидность хода"""
        if next_cell == CellType.WALL or not self.game_map.is_within_bounds(next_row, next_col):
            return False

        if next_cell == CellType.BOX:
            if (after_next_cell == CellType.WALL or
                    not self.game_map.is_within_bounds(after_next_row, after_next_col) or
                    after_next_cell == CellType.BOX):
                return False

        return True

    def _execute_move(self, current_row: int, current_col: int,
                      next_row: int, next_col: int, after_next_row: int, after_next_col: int,
                      next_cell: Optional[CellType], after_next_cell: Optional[CellType]) -> None:
        """Выполняет перемещение рабочего"""
        current_position = (current_row, current_col)
        next_position = (next_row, next_col)

        if next_cell in (CellType.PASSAGEWAY, CellType.DESTINATION):
            self.game_map.update_worker_position(next_position, current_position)

        elif next_cell == CellType.BOX:
            if after_next_cell == CellType.PASSAGEWAY:
                self.game_map.set_cell_type(after_next_row, after_next_col, CellType.BOX)
                self.game_map.update_worker_position(next_position, current_position)
            elif after_next_cell == CellType.DESTINATION:
                self.game_map.set_cell_type(after_next_row, after_next_col, CellType.BOX_IN_DEST)
                self.game_map.update_worker_position(next_position, current_position)

        elif next_cell == CellType.BOX_IN_DEST:
            if after_next_cell == CellType.PASSAGEWAY:
                self.game_map.set_cell_type(after_next_row, after_next_col, CellType.BOX)
                self.game_map.clear_worker_position(current_position)
                self.game_map.set_cell_type(next_row, next_col, CellType.WORKER_IN_DEST)
            elif after_next_cell == CellType.DESTINATION:
                self.game_map.set_cell_type(after_next_row, after_next_col, CellType.BOX_IN_DEST)
                self.game_map.clear_worker_position(current_position)
                self.game_map.set_cell_type(next_row, next_col, CellType.WORKER_IN_DEST)

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

    def __init__(self, root: Tk, box_size: int = 64):
        self.root = root
        self.box_size = box_size
        self.canvas = None
        self.restart_button = None
        self.game_controller = GameController()
        self.resources = GameResources(box_size)
        self.renderer = None
        self.worker_direction = Direction.STOP

        self._setup_resources()

    def _setup_resources(self) -> None:
        """Настраивает ресурсы игры"""
        # Проверяем существование директории с изображениями
        if not os.path.exists(self.resources.resource_manager._config.images_directory):
            print(
                f"Предупреждение: Директория '{self.resources.resource_manager._config.images_directory}' не найдена.")
            print("Пожалуйста, убедитесь что изображения находятся в правильной директории.")

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

        self._center_window(window_width, window_height)
        self._update_title()

        self.canvas = Canvas(
            self.root,
            bg='white',
            width=self.box_size * game_map.cols,
            height=self.box_size * game_map.rows
        )
        self.canvas.configure(highlightthickness=0)
        self.canvas.pack(pady=20)

        self.renderer = GameRenderer(self.canvas, self.resources)
        self._render_game()

    def _center_window(self, width: int, height: int) -> None:
        """Центрирует окно на экране"""
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        x_position = (screen_width - width) // 2
        y_position = (screen_height - height) // 3

        self.root.geometry(f'{width}x{height}+{x_position}+{y_position}')

    def _create_restart_button(self) -> None:
        """Создает кнопку перезапуска"""
        self.restart_button = Label(self.root, width=150, height=50)
        restart_image = self.resources.get_restart_image()
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

        if key in DIRECTION_MAPPING:
            self._handle_movement(key)
        elif key == "space":
            self._restart_game()
        elif key == "Escape":
            self.root.quit()
        elif key == "z" and event.state & 0x0004:  # Ctrl+Z
            if self.game_controller.undo_move():
                self._render_game()
                self._update_title()

    def _handle_movement(self, direction_str: str) -> None:
        """Обрабатывает движение рабочего"""
        self.worker_direction = DIRECTION_TO_ENUM.get(direction_str, Direction.STOP)

        if self.game_controller.move_worker(direction_str):
            self._render_game()
            self._update_title()

            if self.game_controller.is_level_completed():
                self._handle_level_completion()

    def _render_game(self) -> None:
        """Отрисовывает игровое состояние"""
        if self.renderer and self.game_controller.game_map:
            self.renderer.render(self.game_controller.game_map, self.worker_direction)
            self.canvas.focus_set()

    def _update_title(self) -> None:
        """Обновляет заголовок окна"""
        title = (f"推箱子 - 第({self.game_controller.current_level}/{TOTAL_GAMES})关    "
                 f"总步数: {self.game_controller.game_steps}")
        self.root.title(title)

    def _restart_game(self, event=None) -> None:
        """Перезапускает текущий уровень"""
        self.game_controller.load_level(self.game_controller.current_level)
        self.worker_direction = Direction.STOP
        self._render_game()
        self._update_title()

    def _handle_level_completion(self) -> None:
        """Обрабатывает завершение уровня"""
        message = (f"恭喜你顺利通过第({self.game_controller.current_level})关!\n\n"
                   f"一共用了({self.game_controller.game_steps})步")
        showinfo(title="提示", message=message)

        self.game_controller.next_level()
        self.worker_direction = Direction.STOP
        self._create_game_interface()
        self._update_title()


# ==================== ОСНОВНОЙ КОД ====================

if __name__ == "__main__":
    import sys

    try:
        config = GameConfig()
        START_LEVEL = config.default_start_level

        root = Tk()
        game_ui = GameUI(root, box_size=config.box_size)
        game_ui.setup()

        root.mainloop()

    except FileNotFoundError as e:
        print(f"Ошибка загрузки ресурсов: {e}")
        print("Пожалуйста, убедитесь что все изображения находятся в папке 'images'")
        sys.exit(1)
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")
        sys.exit(1)