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
from typing import Tuple, Optional, Dict, List, Any, Set
from dataclasses import dataclass, field
from enum import IntEnum, Enum
from abc import ABC, abstractmethod
import os
import pickle


# ==================== ПЕРЕЧИСЛЕНИЯ И СТРУКТУРЫ ====================

class GameState(Enum):
    """Состояния игры"""
    MENU = "menu"
    PLAYING = "playing"
    PAUSED = "paused"
    LEVEL_COMPLETE = "level_complete"
    GAME_OVER = "game_over"


class CellType(IntEnum):
    """Типы клеток игрового поля"""
    WALL = 0
    WORKER = 1
    BOX = 2
    PASSAGEWAY = 3
    DESTINATION = 4
    WORKER_IN_DEST = 5
    BOX_IN_DEST = 6


class Direction(Enum):
    """Направления движения"""
    UP = "Up"
    DOWN = "Down"
    LEFT = "Left"
    RIGHT = "Right"
    STOP = "Stop"

    def get_offset(self) -> Tuple[int, int, int, int]:
        """Возвращает смещения для направления"""
        return DIRECTION_OFFSETS[self]


@dataclass
class Position:
    """Позиция на игровом поле"""
    row: int
    col: int

    def add(self, direction: Direction) -> 'Position':
        """Добавляет смещение направления к позиции"""
        offset = direction.get_offset()
        return Position(self.row + offset[0], self.col + offset[1])

    def double_add(self, direction: Direction) -> 'Position':
        """Добавляет двойное смещение направления к позиции"""
        offset = direction.get_offset()
        return Position(self.row + offset[2], self.col + offset[3])

    def __hash__(self):
        return hash((self.row, self.col))

    def __eq__(self, other):
        if not isinstance(other, Position):
            return False
        return self.row == other.row and self.col == other.col


@dataclass
class Move:
    """Информация о ходе"""
    direction: Direction
    positions: List[Tuple[Position, CellType]]
    step_count: int


@dataclass
class GameConfig:
    """Конфигурация игры"""
    box_size: int = 64
    supported_box_sizes: Tuple[int, ...] = (32, 64, 96, 128)
    images_directory: str = "images"
    enable_image_cache: bool = True
    default_start_level: int = 1
    max_undo_steps: int = 50
    enable_save_game: bool = True
    save_file: str = "game_save.pkl"


# ==================== КОНСТАНТЫ ====================

TOTAL_GAMES = len(basic_maps)  # Общее количество уровней

DIRECTION_OFFSETS = {
    Direction.UP: (-1, 0, -2, 0),
    Direction.DOWN: (1, 0, 2, 0),
    Direction.LEFT: (0, -1, 0, -2),
    Direction.RIGHT: (0, 1, 0, 2),
    Direction.STOP: (0, 0, 0, 0)
}


# ==================== КОМПОНЕНТЫ ДВИЖКА ====================

class CollisionDetector:
    """Детектор коллизий для игрового поля"""

    @staticmethod
    def can_move_to(cell_type: CellType) -> bool:
        """Может ли рабочий переместиться в клетку"""
        return cell_type in {CellType.PASSAGEWAY, CellType.DESTINATION}

    @staticmethod
    def can_push_box(current_cell: CellType, next_cell: CellType) -> bool:
        """Может ли рабочий толкнуть ящик"""
        if current_cell not in {CellType.BOX, CellType.BOX_IN_DEST}:
            return False

        return next_cell in {CellType.PASSAGEWAY, CellType.DESTINATION}

    @staticmethod
    def is_obstacle(cell_type: CellType) -> bool:
        """Является ли клетка препятствием"""
        return cell_type == CellType.WALL


class MovementEngine:
    """Движок перемещений для игровой логики"""

    def __init__(self, game_map: 'GameMap'):
        self.game_map = game_map

    def calculate_move(self, position: Position, direction: Direction) -> Optional[Move]:
        """Рассчитывает возможный ход"""
        next_pos = position.add(direction)
        next_cell = self.game_map.get_cell_type(next_pos.row, next_pos.col)

        if next_cell is None or CollisionDetector.is_obstacle(next_cell):
            return None

        if CollisionDetector.can_move_to(next_cell):
            return Move(
                direction=direction,
                positions=[(position, self.game_map.get_cell_type(position.row, position.col)),
                           (next_pos, next_cell)],
                step_count=1
            )

        if CollisionDetector.can_push_box(next_cell, self._get_cell_after(next_pos, direction)):
            after_next_pos = next_pos.add(direction)
            after_next_cell = self.game_map.get_cell_type(after_next_pos.row, after_next_pos.col)

            return Move(
                direction=direction,
                positions=[(position, self.game_map.get_cell_type(position.row, position.col)),
                           (next_pos, next_cell),
                           (after_next_pos, after_next_cell)],
                step_count=1
            )

        return None

    def _get_cell_after(self, position: Position, direction: Direction) -> Optional[CellType]:
        """Получает клетку после указанной позиции в заданном направлении"""
        next_pos = position.add(direction)
        return self.game_map.get_cell_type(next_pos.row, next_pos.col)

    def apply_move(self, move: Move) -> Dict[Position, CellType]:
        """Применяет ход к игровому полю и возвращает изменения"""
        changes = {}

        if len(move.positions) == 2:
            # Простое перемещение рабочего
            old_pos, old_type = move.positions[0]
            new_pos, new_type = move.positions[1]

            # Очищаем старую позицию
            if old_type == CellType.WORKER:
                changes[old_pos] = CellType.PASSAGEWAY
            elif old_type == CellType.WORKER_IN_DEST:
                changes[old_pos] = CellType.DESTINATION

            # Занимаем новую позицию
            if new_type == CellType.PASSAGEWAY:
                changes[new_pos] = CellType.WORKER
            elif new_type == CellType.DESTINATION:
                changes[new_pos] = CellType.WORKER_IN_DEST

        elif len(move.positions) == 3:
            # Перемещение с толканием ящика
            worker_pos, worker_type = move.positions[0]
            box_pos, box_type = move.positions[1]
            target_pos, target_type = move.positions[2]

            # Очищаем позицию рабочего
            if worker_type == CellType.WORKER:
                changes[worker_pos] = CellType.PASSAGEWAY
            elif worker_type == CellType.WORKER_IN_DEST:
                changes[worker_pos] = CellType.DESTINATION

            # Перемещаем рабочего на место ящика
            if box_type == CellType.BOX:
                changes[box_pos] = CellType.WORKER
            elif box_type == CellType.BOX_IN_DEST:
                changes[box_pos] = CellType.WORKER_IN_DEST

            # Перемещаем ящик на целевую позицию
            if target_type == CellType.PASSAGEWAY:
                changes[target_pos] = CellType.BOX
            elif target_type == CellType.DESTINATION:
                changes[target_pos] = CellType.BOX_IN_DEST

        # Применяем изменения к карте
        for pos, cell_type in changes.items():
            self.game_map.set_cell_type(pos.row, pos.col, cell_type)

        return changes


class GameMap:
    """Класс для управления игровой картой"""

    def __init__(self, map_data: np.ndarray):
        self.map_data = np.asarray(map_data, dtype=np.int32)
        self.rows, self.cols = self.map_data.shape
        self.worker_position = self._find_worker_position()
        self.movement_engine = MovementEngine(self)
        self._initial_state = self.map_data.copy()

    def _find_worker_position(self) -> Position:
        """Находит позицию рабочего на карте"""
        for i in range(self.rows):
            for j in range(self.cols):
                cell_value = self.map_data[i, j]
                if cell_value == CellType.WORKER or cell_value == CellType.WORKER_IN_DEST:
                    return Position(i, j)
        return Position(0, 0)

    def reset(self) -> None:
        """Сбрасывает карту к начальному состоянию"""
        self.map_data = self._initial_state.copy()
        self.worker_position = self._find_worker_position()

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
        # Все ящики должны быть на местах назначения
        boxes_on_dest = np.sum(self.map_data == CellType.BOX_IN_DEST.value)
        total_destinations = np.sum(self.map_data == CellType.DESTINATION.value)
        total_boxes = np.sum(self.map_data == CellType.BOX.value)

        # Нет пустых мест назначения и нет ящиков не на местах
        return total_destinations == 0 and total_boxes == 0 and boxes_on_dest > 0

    def get_valid_moves(self) -> List[Direction]:
        """Возвращает список возможных ходов из текущей позиции"""
        valid_moves = []

        for direction in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
            move = self.movement_engine.calculate_move(self.worker_position, direction)
            if move is not None:
                valid_moves.append(direction)

        return valid_moves

    def update_worker_position(self, new_position: Position) -> None:
        """Обновляет позицию рабочего"""
        self.worker_position = new_position

    def count_boxes_on_destinations(self) -> int:
        """Считает количество ящиков на местах назначения"""
        return np.sum(self.map_data == CellType.BOX_IN_DEST.value)

    def count_total_destinations(self) -> int:
        """Считает общее количество мест назначения (пустых и занятых)"""
        destinations = np.sum(self.map_data == CellType.DESTINATION.value)
        boxes_on_dest = np.sum(self.map_data == CellType.BOX_IN_DEST.value)
        worker_on_dest = np.sum(self.map_data == CellType.WORKER_IN_DEST.value)

        return destinations + boxes_on_dest + worker_on_dest


# ==================== КЛАСС ИГРОВОГО КОНТРОЛЛЕРА ====================

class GameController:
    """Класс для управления игровой логикой"""

    def __init__(self, config: GameConfig):
        self.config = config
        self.game_map = None
        self.game_steps = 0
        self.current_level = config.default_start_level
        self._move_history: List[Move] = []
        self._state_history: List[Tuple[np.ndarray, Position, int]] = []
        self.game_state = GameState.PLAYING
        self._save_enabled = config.enable_save_game

    def load_level(self, level_index: int) -> GameMap:
        """Загружает уровень по индексу"""
        if 1 <= level_index <= TOTAL_GAMES:
            map_data = basic_maps[level_index - 1]
            self.game_map = GameMap(map_data)
            self.current_level = level_index
            self.game_steps = 0
            self._move_history.clear()
            self._state_history.clear()
            self.game_state = GameState.PLAYING
            return self.game_map
        raise ValueError(f"Уровень {level_index} не существует")

    def move_worker(self, direction: Direction) -> bool:
        """Пытается переместить рабочего в указанном направлении"""
        if self.game_map is None or self.game_state != GameState.PLAYING:
            return False

        # Сохраняем текущее состояние
        self._save_current_state()

        # Рассчитываем ход
        move = self.game_map.movement_engine.calculate_move(
            self.game_map.worker_position, direction
        )

        if move is None:
            return False

        # Применяем ход
        changes = self.game_map.movement_engine.apply_move(move)

        # Обновляем позицию рабочего
        for pos in changes:
            cell_type = self.game_map.get_cell_type(pos.row, pos.col)
            if cell_type in {CellType.WORKER, CellType.WORKER_IN_DEST}:
                self.game_map.update_worker_position(pos)
                break

        # Сохраняем ход в историю
        self._move_history.append(move)
        self.game_steps += 1

        # Проверяем завершение уровня
        if self.game_map.is_level_completed():
            self.game_state = GameState.LEVEL_COMPLETE

        # Ограничиваем историю
        if len(self._move_history) > self.config.max_undo_steps:
            self._move_history.pop(0)
        if len(self._state_history) > self.config.max_undo_steps:
            self._state_history.pop(0)

        return True

    def _save_current_state(self) -> None:
        """Сохраняет текущее состояние игры"""
        if self.game_map:
            state = (
                self.game_map.map_data.copy(),
                Position(self.game_map.worker_position.row, self.game_map.worker_position.col),
                self.game_steps
            )
            self._state_history.append(state)

    def undo_move(self) -> bool:
        """Отменяет последний ход"""
        if not self._state_history or self.game_state != GameState.PLAYING:
            return False

        # Восстанавливаем предыдущее состояние
        map_data, worker_pos, steps = self._state_history.pop()

        if self.game_map:
            self.game_map.map_data = map_data
            self.game_map.worker_position = worker_pos
            self.game_steps = steps

        if self._move_history:
            self._move_history.pop()

        return True

    def reset_level(self) -> None:
        """Сбрасывает текущий уровень"""
        if self.game_map:
            self.game_map.reset()
            self.game_steps = 0
            self._move_history.clear()
            self._state_history.clear()
            self.game_state = GameState.PLAYING

    def next_level(self) -> bool:
        """Переходит к следующему уровню"""
        self.current_level = self.current_level % TOTAL_GAMES + 1
        return True

    def previous_level(self) -> bool:
        """Переходит к предыдущему уровню"""
        self.current_level = (self.current_level - 2) % TOTAL_GAMES + 1
        return True

    def save_game(self) -> bool:
        """Сохраняет игру в файл"""
        if not self._save_enabled or self.game_map is None:
            return False

        try:
            save_data = {
                'level': self.current_level,
                'steps': self.game_steps,
                'map_data': self.game_map.map_data,
                'worker_position': (self.game_map.worker_position.row, self.game_map.worker_position.col),
                'move_history': self._move_history,
                'state_history': [(arr, (pos.row, pos.col), steps)
                                  for arr, pos, steps in self._state_history]
            }

            with open(self.config.save_file, 'wb') as f:
                pickle.dump(save_data, f)

            return True
        except Exception as e:
            print(f"Ошибка сохранения игры: {e}")
            return False

    def load_game(self) -> bool:
        """Загружает игру из файла"""
        if not self._save_enabled:
            return False

        try:
            if not os.path.exists(self.config.save_file):
                return False

            with open(self.config.save_file, 'rb') as f:
                save_data = pickle.load(f)

            self.current_level = save_data['level']
            self.game_steps = save_data['steps']
            self.game_map = GameMap(save_data['map_data'])
            self.game_map.worker_position = Position(*save_data['worker_position'])

            # Восстанавливаем историю ходов
            self._move_history = save_data.get('move_history', [])

            # Восстанавливаем историю состояний
            self._state_history = []
            for arr, pos_tuple, steps in save_data.get('state_history', []):
                pos = Position(pos_tuple[0], pos_tuple[1])
                self._state_history.append((arr, pos, steps))

            self.game_state = GameState.PLAYING
            return True
        except Exception as e:
            print(f"Ошибка загрузки игры: {e}")
            return False

    def get_game_stats(self) -> Dict[str, Any]:
        """Возвращает статистику игры"""
        if self.game_map is None:
            return {}

        return {
            'current_level': self.current_level,
            'total_levels': TOTAL_GAMES,
            'steps': self.game_steps,
            'boxes_on_dest': self.game_map.count_boxes_on_destinations(),
            'total_destinations': self.game_map.count_total_destinations(),
            'valid_moves': len(self.game_map.get_valid_moves()),
            'game_state': self.game_state,
            'can_undo': len(self._state_history) > 0
        }

    def is_level_completed(self) -> bool:
        """Проверяет, завершен ли текущий уровень"""
        return self.game_state == GameState.LEVEL_COMPLETE


# ==================== РЕСУРСНЫЕ КЛАССЫ (с предыдущего коммита) ====================

@dataclass
class ImageConfig:
    """Конфигурация изображения"""
    path: str
    width: int
    height: int
    keep_aspect_ratio: bool = True


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

        if self._config.enable_image_cache and image_config.path in self._image_cache:
            if cache_key in self._image_cache[image_config.path]:
                return self._image_cache[image_config.path][cache_key]

        full_path = self.get_full_path(image_config.path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Изображение не найдено: {full_path}")

        img = Image.open(full_path)

        if image_config.keep_aspect_ratio:
            scaled_img = self._resize_keep_aspect(img, image_config.width, image_config.height)
        else:
            scaled_img = img.resize((image_config.width, image_config.height), Image.Resampling.LANCZOS)

        tk_image = ImageTk.PhotoImage(scaled_img)

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


class GameResources:
    """Класс для управления игровыми ресурсами"""

    def __init__(self, box_size: int = 64):
        self.box_size = box_size
        self.resource_manager = ResourceManager()
        self._images = None

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


# ==================== КЛАСС ИГРОВОГО ИНТЕРФЕЙСА ====================

class GameUI:
    """Класс для управления пользовательским интерфейсом"""

    def __init__(self, root: Tk, box_size: int = 64):
        self.root = root
        self.box_size = box_size
        self.canvas = None
        self.restart_button = None
        self.stats_label = None

        config = GameConfig(box_size=box_size)
        self.game_controller = GameController(config)
        self.resources = GameResources(box_size)
        self.renderer = None
        self.worker_direction = Direction.STOP

        self._setup_resources()

    def _setup_resources(self) -> None:
        """Настраивает ресурсы игры"""
        if not os.path.exists(self.resources.resource_manager._config.images_directory):
            print(
                f"Предупреждение: Директория '{self.resources.resource_manager._config.images_directory}' не найдена.")

    def setup(self) -> None:
        """Настраивает пользовательский интерфейс"""
        self.root.title("推箱子")
        self._create_game_interface()
        self._create_restart_button()
        self._create_stats_display()
        self._bind_events()

        # Пытаемся загрузить сохраненную игру
        if self.game_controller.config.enable_save_game:
            if self.game_controller.load_game():
                self._render_game()
                self._update_title()
                self._update_stats()

    def _create_game_interface(self) -> None:
        """Создает игровой интерфейс"""
        self.game_controller.load_level(self.game_controller.current_level)
        game_map = self.game_controller.game_map

        window_width = game_map.cols * self.box_size + self.box_size // 2 + 40
        window_height = game_map.rows * self.box_size + self.box_size // 2 + 150

        self._center_window(window_width, window_height)
        self._update_title()

        self.canvas = Canvas(
            self.root,
            bg='white',
            width=self.box_size * game_map.cols,
            height=self.box_size * game_map.rows
        )
        self.canvas.configure(highlightthickness=0)
        self.canvas.pack(pady=10)

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

    def _create_stats_display(self) -> None:
        """Создает отображение статистики"""
        self.stats_label = Label(self.root, text="", font=("Arial", 10))
        self.stats_label.pack(pady=5)

    def _bind_events(self) -> None:
        """Привязывает обработчики событий"""
        self.canvas.bind("<KeyPress>", self._handle_key_press)
        self.canvas.focus_set()

        # Привязываем обработчик закрытия окна для сохранения игры
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _handle_key_press(self, event) -> None:
        """Обрабатывает нажатия клавиш"""
        key = event.keysym

        # Обработка движения
        if key in [d.value for d in Direction if d != Direction.STOP]:
            direction = Direction(key)
            self._handle_movement(direction)

        # Специальные клавиши
        elif key == "space":
            self._restart_game()
        elif key == "Escape":
            self.root.quit()
        elif key == "z" and event.state & 0x0004:  # Ctrl+Z
            if self.game_controller.undo_move():
                self._render_game()
                self._update_title()
                self._update_stats()
        elif key == "r" and event.state & 0x0004:  # Ctrl+R
            self._restart_game()
        elif key == "n" and event.state & 0x0004:  # Ctrl+N
            self._next_level()
        elif key == "p" and event.state & 0x0004:  # Ctrl+P
            self._previous_level()
        elif key == "s" and event.state & 0x0004:  # Ctrl+S
            if self.game_controller.save_game():
                print("Игра сохранена")

    def _handle_movement(self, direction: Direction) -> None:
        """Обрабатывает движение рабочего"""
        self.worker_direction = direction

        if self.game_controller.move_worker(direction):
            self._render_game()
            self._update_title()
            self._update_stats()

            if self.game_controller.is_level_completed():
                self._handle_level_completion()

    def _render_game(self) -> None:
        """Отрисовывает игровое состояние"""
        if self.renderer and self.game_controller.game_map:
            self.renderer.render(self.game_controller.game_map, self.worker_direction)
            self.canvas.focus_set()

    def _update_title(self) -> None:
        """Обновляет заголовок окна"""
        stats = self.game_controller.get_game_stats()
        title = (f"推箱子 - 第({stats['current_level']}/{stats['total_levels']})关    "
                 f"总步数: {stats['steps']}")
        self.root.title(title)

    def _update_stats(self) -> None:
        """Обновляет отображение статистики"""
        stats = self.game_controller.get_game_stats()

        if stats:
            stats_text = (f"Ящиков на местах: {stats['boxes_on_dest']}/{stats['total_destinations']} | "
                          f"Возможных ходов: {stats['valid_moves']} | "
                          f"Состояние: {stats['game_state'].value}")
            self.stats_label.config(text=stats_text)

    def _restart_game(self, event=None) -> None:
        """Перезапускает текущий уровень"""
        self.game_controller.reset_level()
        self.worker_direction = Direction.STOP
        self._render_game()
        self._update_title()
        self._update_stats()

    def _next_level(self) -> None:
        """Переходит к следующему уровню"""
        self.game_controller.next_level()
        self.worker_direction = Direction.STOP
        self._create_game_interface()
        self._update_title()
        self._update_stats()

    def _previous_level(self) -> None:
        """Переходит к предыдущему уровню"""
        self.game_controller.previous_level()
        self.worker_direction = Direction.STOP
        self._create_game_interface()
        self._update_title()
        self._update_stats()

    def _handle_level_completion(self) -> None:
        """Обрабатывает завершение уровня"""
        stats = self.game_controller.get_game_stats()
        message = (f"恭喜你顺利通过第({stats['current_level']})关!\n\n"
                   f"一共用了({stats['steps']})步")
        showinfo(title="提示", message=message)

        # Сохраняем игру перед переходом на следующий уровень
        if self.game_controller.config.enable_save_game:
            self.game_controller.save_game()

        self._next_level()

    def _on_closing(self) -> None:
        """Обработчик закрытия окна"""
        # Сохраняем игру при закрытии
        if self.game_controller.config.enable_save_game:
            self.game_controller.save_game()

        self.root.destroy()


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
        import traceback

        traceback.print_exc()
        sys.exit(1)