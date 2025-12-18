# coding:utf-8

"""
推箱子游戏 - Sokoban Game
==============

基于 tkinter.Canvas 实现的推箱子游戏

功能特点：
1. 支持多个游戏关卡
2. 可自定义方块大小 (32, 64, 96, 128)
3. 支持保存/加载游戏进度
4. 多种界面主题
5. 动画效果
6. 历史记录和撤销功能
7. 详细的游戏统计

开发人员: Edwin.Zhang
初始开发时间: 2019-6-28
重构时间: 2024
版本: 2.0
"""

# ==================== ИМПОРТЫ ====================
from __future__ import annotations
import sys
import os
import pickle
import json
from datetime import datetime, timedelta
from typing import (
    Tuple, Optional, Dict, List, Any, Set, Callable,
    TypeVar, Generic, Union, Iterator
)
from dataclasses import dataclass, field, asdict
from enum import IntEnum, Enum, auto
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import weakref

# Третьи стороны библиотеки
from PIL import Image, ImageTk
import numpy as np

# tkinter импорты
import tkinter as tk
from tkinter import (
    Tk, Canvas, Label, Frame, Button, Menu, Toplevel,
    Scale, HORIZONTAL, VERTICAL, StringVar, IntVar,
    BooleanVar, Scrollbar, Text, END
)
from tkinter import ttk, messagebox, font
from tkinter.messagebox import showinfo, askyesno, showwarning

# Локальные импорты
try:
    from game_maps import basic_maps
except ImportError:
    print("错误: 找不到 game_maps.py 文件")
    print("请确保 game_maps.py 与当前脚本在同一目录下")
    sys.exit(1)


# ==================== НАСТРОЙКА ЛОГГИРОВАНИЯ ====================

class GameLogger:
    """Логгер для игры"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._setup_logger()
        return cls._instance

    def _setup_logger(self):
        """Настраивает логгер"""
        self.logger = logging.getLogger("SokobanGame")
        self.logger.setLevel(logging.INFO)

        # Создаем директорию для логов если её нет
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Файловый обработчик
        file_handler = logging.FileHandler(
            log_dir / f"game_{datetime.now():%Y%m%d_%H%M%S}.log",
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)

        # Консольный обработчик
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)

        # Форматтер
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def get_logger(self) -> logging.Logger:
        """Возвращает настроенный логгер"""
        return self.logger


# Инициализация логгера
logger = GameLogger().get_logger()

# ==================== ТИПЫ ДАННЫХ И УТИЛИТЫ ====================

T = TypeVar('T')
U = TypeVar('U')


class Singleton(type):
    """Метакласс для создания синглтонов"""
    _instances: Dict[Singleton, Any] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Observable:
    """Реализация паттерна Наблюдатель"""

    def __init__(self):
        self._observers: List[Callable] = []

    def attach(self, observer: Callable) -> None:
        """Добавляет наблюдателя"""
        if observer not in self._observers:
            self._observers.append(observer)

    def detach(self, observer: Callable) -> None:
        """Удаляет наблюдателя"""
        if observer in self._observers:
            self._observers.remove(observer)

    def notify(self, *args, **kwargs) -> None:
        """Уведомляет всех наблюдателей"""
        for observer in self._observers:
            try:
                observer(*args, **kwargs)
            except Exception as e:
                logger.error(f"Ошибка в наблюдателе: {e}")


class Timer:
    """Таймер для измерения времени"""

    def __init__(self):
        self._start_time: Optional[datetime] = None
        self._paused_time: Optional[datetime] = None
        self._total_paused = timedelta()

    def start(self) -> None:
        """Запускает таймер"""
        self._start_time = datetime.now()
        self._total_paused = timedelta()

    def pause(self) -> None:
        """Ставит таймер на паузу"""
        if self._start_time and not self._paused_time:
            self._paused_time = datetime.now()

    def resume(self) -> None:
        """Возобновляет таймер"""
        if self._paused_time:
            self._total_paused += datetime.now() - self._paused_time
            self._paused_time = None

    def get_elapsed(self) -> timedelta:
        """Возвращает прошедшее время"""
        if not self._start_time:
            return timedelta()

        if self._paused_time:
            current = self._paused_time
        else:
            current = datetime.now()

        return (current - self._start_time) - self._total_paused

    def reset(self) -> None:
        """Сбрасывает таймер"""
        self._start_time = None
        self._paused_time = None
        self._total_paused = timedelta()


# ==================== ПЕРЕЧИСЛЕНИЯ И СТРУКТУРЫ ====================

class GameState(Enum):
    """Состояния игры"""
    MENU = auto()
    PLAYING = auto()
    PAUSED = auto()
    LEVEL_COMPLETE = auto()
    GAME_OVER = auto()
    SETTINGS = auto()


class UIMode(Enum):
    """Режимы интерфейса"""
    CLASSIC = "classic"
    DARK = "dark"
    LIGHT = "light"
    COLORFUL = "colorful"


class Difficulty(Enum):
    """Уровни сложности"""
    EASY = "easy"
    NORMAL = "normal"
    HARD = "hard"
    EXPERT = "expert"


class CellType(IntEnum):
    """Типы клеток игрового поля"""
    EMPTY = -1
    WALL = 0
    WORKER = 1
    BOX = 2
    PASSAGEWAY = 3
    DESTINATION = 4
    WORKER_IN_DEST = 5
    BOX_IN_DEST = 6

    @classmethod
    def is_worker(cls, cell_type: int) -> bool:
        """Проверяет, является ли клетка рабочим"""
        return cell_type in (cls.WORKER, cls.WORKER_IN_DEST)

    @classmethod
    def is_box(cls, cell_type: int) -> bool:
        """Проверяет, является ли клетка ящиком"""
        return cell_type in (cls.BOX, cls.BOX_IN_DEST)

    @classmethod
    def is_destination(cls, cell_type: int) -> bool:
        """Проверяет, является ли клетка местом назначения"""
        return cell_type in (cls.DESTINATION, cls.WORKER_IN_DEST, cls.BOX_IN_DEST)

    @classmethod
    def is_walkable(cls, cell_type: int) -> bool:
        """Проверяет, может ли рабочий пройти через клетку"""
        return cell_type in (cls.PASSAGEWAY, cls.DESTINATION)


class Direction(Enum):
    """Направления движения"""
    UP = "Up"
    DOWN = "Down"
    LEFT = "Left"
    RIGHT = "Right"
    STOP = "Stop"

    @property
    def offset(self) -> Tuple[int, int, int, int]:
        """Возвращает смещения для направления"""
        return DIRECTION_OFFSETS[self]

    @classmethod
    def from_key(cls, key: str) -> Optional[Direction]:
        """Создает направление из клавиши"""
        key_mapping = {
            "Up": cls.UP,
            "Down": cls.DOWN,
            "Left": cls.LEFT,
            "Right": cls.RIGHT
        }
        return key_mapping.get(key)


@dataclass(frozen=True)
class Position:
    """Неизменяемая позиция на игровом поле"""
    row: int
    col: int

    def __post_init__(self):
        """Валидация позиции"""
        if self.row < 0 or self.col < 0:
            raise ValueError("Позиция не может быть отрицательной")

    def move(self, direction: Direction) -> Position:
        """Двигает позицию в заданном направлении"""
        dr, dc, _, _ = direction.offset
        return Position(self.row + dr, self.col + dc)

    def distance_to(self, other: Position) -> float:
        """Вычисляет расстояние до другой позиции"""
        return ((self.row - other.row) ** 2 + (self.col - other.col) ** 2) ** 0.5

    def __str__(self) -> str:
        return f"({self.row}, {self.col})"


@dataclass
class MoveRecord:
    """Запись о ходе для истории"""
    direction: Direction
    timestamp: datetime
    positions_changed: Dict[Position, CellType]
    step_number: int

    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует в словарь для сериализации"""
        return {
            'direction': self.direction.value,
            'timestamp': self.timestamp.isoformat(),
            'positions_changed': {
                str(pos): cell_type.value
                for pos, cell_type in self.positions_changed.items()
            },
            'step_number': self.step_number
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MoveRecord:
        """Создает из словаря"""
        return cls(
            direction=Direction(data['direction']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            positions_changed={
                Position(*map(int, pos.strip('()').split(','))): CellType(cell_type)
                for pos, cell_type in data['positions_changed'].items()
            },
            step_number=data['step_number']
        )


@dataclass
class UIStyle:
    """Стиль интерфейса"""
    name: str
    bg_color: str
    fg_color: str
    button_bg: str
    button_fg: str
    canvas_bg: str
    highlight_color: str
    font_family: str = "Arial"
    font_size: int = 10
    border_radius: int = 5

    def get_font(self, size: Optional[int] = None, bold: bool = False) -> tuple:
        """Возвращает кортеж настроек шрифта"""
        if size is None:
            size = self.font_size

        weight = "bold" if bold else "normal"
        return (self.font_family, size, weight)


@dataclass
class UIConfig:
    """Конфигурация интерфейса"""
    box_size: int = 64
    show_grid: bool = False
    grid_color: str = "#CCCCCC"
    show_coordinates: bool = False
    animate_moves: bool = True
    animation_speed: int = 20  # мс
    ui_mode: UIMode = UIMode.CLASSIC
    enable_sounds: bool = False
    volume: int = 50
    language: str = "zh"  # zh, en, ru


@dataclass
class GameConfig:
    """Конфигурация игры"""
    box_size: int = 64
    supported_box_sizes: Tuple[int, ...] = (32, 64, 96, 128)
    images_directory: str = "images"
    enable_image_cache: bool = True
    default_start_level: int = 1
    max_undo_steps: int = 100
    enable_save_game: bool = True
    save_file: str = "game_save.pkl"
    config_file: str = "game_config.json"
    max_save_slots: int = 5
    difficulty: Difficulty = Difficulty.NORMAL

    ui_config: UIConfig = field(default_factory=UIConfig)

    def validate(self) -> List[str]:
        """Валидирует конфигурацию и возвращает список ошибок"""
        errors = []

        if self.box_size not in self.supported_box_sizes:
            errors.append(f"Неподдерживаемый размер блоков: {self.box_size}")

        if self.max_undo_steps < 0:
            errors.append("Максимальное количество отмен не может быть отрицательным")

        if not 0 <= self.ui_config.volume <= 100:
            errors.append(f"Громкость должна быть от 0 до 100, получено: {self.ui_config.volume}")

        return errors


# ==================== КОНСТАНТЫ ====================

TOTAL_GAMES = len(basic_maps)  # Общее количество уровней

DIRECTION_OFFSETS = {
    Direction.UP: (-1, 0, -2, 0),
    Direction.DOWN: (1, 0, 2, 0),
    Direction.LEFT: (0, -1, 0, -2),
    Direction.RIGHT: (0, 1, 0, 2),
    Direction.STOP: (0, 0, 0, 0)
}

# Тексты на разных языках
LANGUAGE_TEXTS = {
    "zh": {
        "title": "推箱子",
        "level": "关卡",
        "steps": "步数",
        "time": "时间",
        "boxes": "箱子",
        "pause": "暂停",
        "continue": "继续",
        "restart": "重新开始",
        "undo": "撤销",
        "settings": "设置",
        "exit": "退出"
    },
    "en": {
        "title": "Sokoban",
        "level": "Level",
        "steps": "Steps",
        "time": "Time",
        "boxes": "Boxes",
        "pause": "Pause",
        "continue": "Continue",
        "restart": "Restart",
        "undo": "Undo",
        "settings": "Settings",
        "exit": "Exit"
    },
    "ru": {
        "title": "Сокобан",
        "level": "Уровень",
        "steps": "Шаги",
        "time": "Время",
        "boxes": "Ящики",
        "pause": "Пауза",
        "continue": "Продолжить",
        "restart": "Перезапуск",
        "undo": "Отмена",
        "settings": "Настройки",
        "exit": "Выход"
    }
}

UI_STYLES = {
    UIMode.CLASSIC: UIStyle(
        name="classic",
        bg_color="#F0F0F0",
        fg_color="#000000",
        button_bg="#4CAF50",
        button_fg="#FFFFFF",
        canvas_bg="#FFFFFF",
        highlight_color="#2196F3"
    ),
    UIMode.DARK: UIStyle(
        name="dark",
        bg_color="#1E1E1E",
        fg_color="#FFFFFF",
        button_bg="#333333",
        button_fg="#FFFFFF",
        canvas_bg="#2D2D2D",
        highlight_color="#BB86FC",
        font_family="Consolas"
    ),
    UIMode.LIGHT: UIStyle(
        name="light",
        bg_color="#FFFFFF",
        fg_color="#212121",
        button_bg="#E3F2FD",
        button_fg="#0D47A1",
        canvas_bg="#FAFAFA",
        highlight_color="#1976D2"
    ),
    UIMode.COLORFUL: UIStyle(
        name="colorful",
        bg_color="#FFF8E1",
        fg_color="#5D4037",
        button_bg="#FF9800",
        button_fg="#FFFFFF",
        canvas_bg="#FFF3E0",
        highlight_color="#F57C00"
    )
}


# ==================== КЛАСС КОНФИГУРАЦИИ ====================

class ConfigManager(metaclass=Singleton):
    """Менеджер конфигурации игры"""

    def __init__(self, config_file: str = "game_config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> GameConfig:
        """Загружает конфигурацию из файла или создает новую"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Восстанавливаем объекты Enum
                data['ui_config']['ui_mode'] = UIMode(data['ui_config']['ui_mode'])
                data['difficulty'] = Difficulty(data.get('difficulty', 'normal'))

                return GameConfig(**data)
        except Exception as e:
            logger.warning(f"Не удалось загрузить конфигурацию: {e}. Используются значения по умолчанию.")

        # Создаем конфигурацию по умолчанию
        return GameConfig()

    def _validate_config(self) -> None:
        """Валидирует конфигурацию"""
        errors = self.config.validate()
        if errors:
            logger.warning(f"Обнаружены ошибки в конфигурации: {errors}")
            # Исправляем очевидные ошибки
            if self.config.box_size not in self.config.supported_box_sizes:
                self.config.box_size = 64
            if self.config.max_undo_steps < 0:
                self.config.max_undo_steps = 100

    def save_config(self) -> bool:
        """Сохраняет конфигурацию в файл"""
        try:
            data = asdict(self.config)
            # Сериализуем объекты Enum
            data['ui_config']['ui_mode'] = data['ui_config']['ui_mode'].value
            data['difficulty'] = data['difficulty'].value

            # Создаем backup старой конфигурации
            if self.config_file.exists():
                backup_file = self.config_file.with_suffix('.json.bak')
                self.config_file.rename(backup_file)

            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Конфигурация сохранена в {self.config_file}")
            return True

        except Exception as e:
            logger.error(f"Ошибка сохранения конфигурации: {e}")
            return False

    def update_config(self, **kwargs) -> None:
        """Обновляет конфигурацию с валидацией"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        self._validate_config()

    def get_ui_style(self) -> UIStyle:
        """Возвращает текущий стиль интерфейса"""
        return UI_STYLES.get(self.config.ui_config.ui_mode, UI_STYLES[UIMode.CLASSIC])

    def get_text(self, key: str) -> str:
        """Возвращает текст на текущем языке"""
        lang = self.config.ui_config.language
        texts = LANGUAGE_TEXTS.get(lang, LANGUAGE_TEXTS["zh"])
        return texts.get(key, key)


# ==================== РЕСУРСНЫЕ КЛАССЫ ====================

class ImageLoader:
    """Загрузчик изображений с кэшированием"""

    def __init__(self):
        self._cache: Dict[Tuple[str, int, int], ImageTk.PhotoImage] = {}
        self._failed_images: Set[str] = set()

    def load_image(self, path: str, width: int, height: int) -> Optional[ImageTk.PhotoImage]:
        """Загружает и масштабирует изображение"""
        cache_key = (path, width, height)

        if cache_key in self._cache:
            return self._cache[cache_key]

        if path in self._failed_images:
            return None

        try:
            if not Path(path).exists():
                logger.warning(f"Изображение не найдено: {path}")
                self._failed_images.add(path)
                return None

            img = Image.open(path)

            # Сохраняем пропорции
            original_width, original_height = img.size
            scale = min(width / original_width, height / original_height)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)

            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            tk_image = ImageTk.PhotoImage(resized_img)

            self._cache[cache_key] = tk_image
            return tk_image

        except Exception as e:
            logger.error(f"Ошибка загрузки изображения {path}: {e}")
            self._failed_images.add(path)
            return None

    def clear_cache(self) -> None:
        """Очищает кэш изображений"""
        self._cache.clear()
        self._failed_images.clear()


class ResourceManager(metaclass=Singleton):
    """Менеджер ресурсов игры"""

    def __init__(self):
        self.config_manager = ConfigManager()
        self.image_loader = ImageLoader()
        self._images: Dict[CellType, ImageTk.PhotoImage] = {}
        self._worker_images: Dict[Direction, Tuple[ImageTk.PhotoImage, ImageTk.PhotoImage]] = {}
        self._loaded = False

    def load_resources(self) -> bool:
        """Загружает все ресурсы"""
        if self._loaded:
            return True

        try:
            images_dir = Path(self.config_manager.config.images_directory)
            if not images_dir.exists():
                logger.error(f"Директория с изображениями не найдена: {images_dir}")
                return False

            box_size = self.config_manager.config.box_size

            # Загружаем основные изображения
            self._images = {
                CellType.WALL: self._load_image(images_dir / "Wall.jpg", box_size, box_size),
                CellType.BOX: self._load_image(images_dir / "Box.jpg", box_size, box_size),
                CellType.PASSAGEWAY: self._load_image(images_dir / "Passageway.jpg", box_size, box_size),
                CellType.DESTINATION: self._load_image(images_dir / "Destination.jpg", box_size, box_size),
                CellType.BOX_IN_DEST: self._load_image(images_dir / "redbox.jpg", box_size, box_size),
            }

            # Загружаем изображения рабочего
            self._worker_images = {
                Direction.STOP: (
                    self._load_image(images_dir / "Worker.jpg", box_size, box_size),
                    self._load_image(images_dir / "WorkerInDest.jpg", box_size, box_size)
                ),
                Direction.UP: (
                    self._load_image(images_dir / "w_up.jpg", box_size, box_size),
                    self._load_image(images_dir / "w_up_in.jpg", box_size, box_size)
                ),
                Direction.DOWN: (
                    self._load_image(images_dir / "w_down.jpg", box_size, box_size),
                    self._load_image(images_dir / "w_down_in.jpg", box_size, box_size)
                ),
                Direction.LEFT: (
                    self._load_image(images_dir / "w_left.jpg", box_size, box_size),
                    self._load_image(images_dir / "w_left_in.jpg", box_size, box_size)
                ),
                Direction.RIGHT: (
                    self._load_image(images_dir / "w_right.jpg", box_size, box_size),
                    self._load_image(images_dir / "w_right_in.jpg", box_size, box_size)
                )
            }

            # Проверяем что все необходимые изображения загружены
            required_images = list(CellType) + list(Direction)
            for img_type in required_images:
                if img_type not in self._images and img_type not in self._worker_images:
                    logger.warning(f"Не загружено изображение для: {img_type}")

            self._loaded = True
            logger.info("Ресурсы успешно загружены")
            return True

        except Exception as e:
            logger.error(f"Ошибка загрузки ресурсов: {e}")
            return False

    def _load_image(self, path: Path, width: int, height: int) -> Optional[ImageTk.PhotoImage]:
        """Загружает одно изображение"""
        if not path.exists():
            logger.warning(f"Изображение не найдено: {path}")
            return None

        return self.image_loader.load_image(str(path), width, height)

    def get_image(self, cell_type: CellType) -> Optional[ImageTk.PhotoImage]:
        """Получает изображение для типа клетки"""
        return self._images.get(cell_type)

    def get_worker_image(self, direction: Direction, in_destination: bool = False) -> Optional[ImageTk.PhotoImage]:
        """Получает изображение рабочего"""
        if direction not in self._worker_images:
            direction = Direction.STOP

        images = self._worker_images[direction]
        return images[1] if in_destination else images[0]

    def reload_resources(self, new_box_size: Optional[int] = None) -> bool:
        """Перезагружает ресурсы с новым размером"""
        if new_box_size:
            self.config_manager.update_config(box_size=new_box_size)

        self.image_loader.clear_cache()
        self._loaded = False
        return self.load_resources()


# ==================== ЯДРО ИГРЫ ====================

class GameMap:
    """Игровая карта с расширенными возможностями"""

    def __init__(self, map_data: np.ndarray, level_number: int = 1):
        self.map_data = np.asarray(map_data, dtype=np.int32)
        self.level_number = level_number
        self.rows, self.cols = self.map_data.shape
        self.initial_state = self.map_data.copy()

        # Находим позицию рабочего
        self.worker_position = self._find_worker_position()

        # Статистика
        self.total_boxes = np.sum(self.map_data == CellType.BOX.value)
        self.total_destinations = np.sum(
            (self.map_data == CellType.DESTINATION.value) |
            (self.map_data == CellType.BOX_IN_DEST.value) |
            (self.map_data == CellType.WORKER_IN_DEST.value)
        )

        logger.info(f"Создана карта уровня {level_number}: {self.rows}x{self.cols}")

    def _find_worker_position(self) -> Position:
        """Находит позицию рабочего на карте"""
        for i in range(self.rows):
            for j in range(self.cols):
                if CellType.is_worker(self.map_data[i, j]):
                    return Position(i, j)
        return Position(0, 0)

    def get_cell(self, position: Position) -> Optional[CellType]:
        """Возвращает тип клетки в позиции"""
        if self.is_valid_position(position):
            try:
                return CellType(self.map_data[position.row, position.col])
            except ValueError:
                return None
        return None

    def set_cell(self, position: Position, cell_type: CellType) -> bool:
        """Устанавливает тип клетки в позиции"""
        if self.is_valid_position(position):
            self.map_data[position.row, position.col] = cell_type.value
            return True
        return False

    def is_valid_position(self, position: Position) -> bool:
        """Проверяет, находится ли позиция в пределах карты"""
        return (0 <= position.row < self.rows and
                0 <= position.col < self.cols)

    def is_walkable(self, position: Position) -> bool:
        """Проверяет, может ли рабочий пройти через клетку"""
        cell_type = self.get_cell(position)
        return cell_type is not None and CellType.is_walkable(cell_type.value)

    def is_box(self, position: Position) -> bool:
        """Проверяет, находится ли в позиции ящик"""
        cell_type = self.get_cell(position)
        return cell_type is not None and CellType.is_box(cell_type.value)

    def count_boxes_on_destinations(self) -> int:
        """Считает количество ящиков на местах назначения"""
        return np.sum(self.map_data == CellType.BOX_IN_DEST.value)

    def is_level_completed(self) -> bool:
        """Проверяет, завершен ли уровень"""
        return self.count_boxes_on_destinations() == self.total_destinations

    def reset(self) -> None:
        """Сбрасывает карту к начальному состоянию"""
        self.map_data = self.initial_state.copy()
        self.worker_position = self._find_worker_position()

    def to_string(self) -> str:
        """Конвертирует карту в строку"""
        symbols = {
            CellType.WALL.value: '#',
            CellType.WORKER.value: '@',
            CellType.BOX.value: '$',
            CellType.PASSAGEWAY.value: ' ',
            CellType.DESTINATION.value: '.',
            CellType.WORKER_IN_DEST.value: '+',
            CellType.BOX_IN_DEST.value: '*',
            CellType.EMPTY.value: '?'
        }

        lines = []
        for i in range(self.rows):
            line = ''.join(symbols.get(self.map_data[i, j], '?')
                           for j in range(self.cols))
            lines.append(line)

        return '\n'.join(lines)


class GameController(Observable):
    """Контроллер игры с расширенной функциональностью"""

    def __init__(self):
        super().__init__()
        self.config_manager = ConfigManager()
        self.current_level = self.config_manager.config.default_start_level
        self.game_map: Optional[GameMap] = None
        self.move_history: List[MoveRecord] = []
        self.step_count = 0
        self.timer = Timer()
        self.game_state = GameState.MENU

        # Загружаем уровень
        self.load_level(self.current_level)

    def load_level(self, level_number: int) -> bool:
        """Загружает уровень по номеру"""
        try:
            if 1 <= level_number <= TOTAL_GAMES:
                map_data = basic_maps[level_number - 1]
                self.game_map = GameMap(map_data, level_number)
                self.current_level = level_number
                self.move_history.clear()
                self.step_count = 0
                self.timer.start()
                self.game_state = GameState.PLAYING

                logger.info(f"Загружен уровень {level_number}")
                self.notify('level_loaded', level_number)
                return True

            logger.error(f"Некорректный номер уровня: {level_number}")
            return False

        except Exception as e:
            logger.error(f"Ошибка загрузки уровня {level_number}: {e}")
            return False

    def move(self, direction: Direction) -> bool:
        """Выполняет ход в заданном направлении"""
        if (self.game_state != GameState.PLAYING or
                not self.game_map or
                direction == Direction.STOP):
            return False

        worker_pos = self.game_map.worker_position
        next_pos = worker_pos.move(direction)
        next_next_pos = next_pos.move(direction)

        # Проверяем возможность хода
        if not self._can_move(worker_pos, direction):
            return False

        # Сохраняем состояние до хода
        old_state = self._capture_state()

        # Выполняем ход
        if self._perform_move(worker_pos, next_pos, next_next_pos, direction):
            # Сохраняем ход в историю
            move_record = MoveRecord(
                direction=direction,
                timestamp=datetime.now(),
                positions_changed=self._get_changes(old_state),
                step_number=self.step_count
            )
            self.move_history.append(move_record)

            # Ограничиваем размер истории
            max_history = self.config_manager.config.max_undo_steps
            if len(self.move_history) > max_history:
                self.move_history = self.move_history[-max_history:]

            self.step_count += 1

            # Проверяем завершение уровня
            if self.game_map.is_level_completed():
                self.game_state = GameState.LEVEL_COMPLETE
                self.timer.pause()
                logger.info(f"Уровень {self.current_level} пройден за {self.step_count} шагов")
                self.notify('level_completed', self.current_level, self.step_count)

            self.notify('move_made', direction, self.step_count)
            return True

        return False

    def _can_move(self, position: Position, direction: Direction) -> bool:
        """Проверяет возможность хода"""
        if not self.game_map:
            return False

        next_pos = position.move(direction)
        next_cell = self.game_map.get_cell(next_pos)

        if next_cell is None or next_cell == CellType.WALL:
            return False

        if next_cell in (CellType.PASSAGEWAY, CellType.DESTINATION):
            return True

        if CellType.is_box(next_cell.value):
            next_next_pos = next_pos.move(direction)
            next_next_cell = self.game_map.get_cell(next_next_pos)

            return (next_next_cell is not None and
                    next_next_cell in (CellType.PASSAGEWAY, CellType.DESTINATION))

        return False

    def _perform_move(self, worker_pos: Position, next_pos: Position,
                      next_next_pos: Position, direction: Direction) -> bool:
        """Выполняет перемещение"""
        if not self.game_map:
            return False

        next_cell = self.game_map.get_cell(next_pos)

        # Простое перемещение
        if next_cell in (CellType.PASSAGEWAY, CellType.DESTINATION):
            self._move_worker(worker_pos, next_pos)
            return True

        # Перемещение с толканием ящика
        if CellType.is_box(next_cell.value):
            next_next_cell = self.game_map.get_cell(next_next_pos)

            if next_next_cell in (CellType.PASSAGEWAY, CellType.DESTINATION):
                # Толкаем ящик
                if next_next_cell == CellType.PASSAGEWAY:
                    self.game_map.set_cell(next_next_pos, CellType.BOX)
                else:
                    self.game_map.set_cell(next_next_pos, CellType.BOX_IN_DEST)

                # Перемещаем рабочего
                self._move_worker(worker_pos, next_pos)
                return True

        return False

    def _move_worker(self, from_pos: Position, to_pos: Position) -> None:
        """Перемещает рабочего"""
        if not self.game_map:
            return

        # Очищаем старую позицию
        from_cell = self.game_map.get_cell(from_pos)
        if from_cell == CellType.WORKER:
            self.game_map.set_cell(from_pos, CellType.PASSAGEWAY)
        elif from_cell == CellType.WORKER_IN_DEST:
            self.game_map.set_cell(from_pos, CellType.DESTINATION)

        # Занимаем новую позицию
        to_cell = self.game_map.get_cell(to_pos)
        if to_cell == CellType.PASSAGEWAY:
            self.game_map.set_cell(to_pos, CellType.WORKER)
        elif to_cell == CellType.DESTINATION:
            self.game_map.set_cell(to_pos, CellType.WORKER_IN_DEST)

        # Обновляем позицию рабочего
        self.game_map.worker_position = to_pos

    def _capture_state(self) -> Dict[Position, CellType]:
        """Сохраняет текущее состояние игры"""
        if not self.game_map:
            return {}

        state = {}
        for i in range(self.game_map.rows):
            for j in range(self.game_map.cols):
                pos = Position(i, j)
                cell_type = self.game_map.get_cell(pos)
                if cell_type:
                    state[pos] = cell_type

        return state

    def _get_changes(self, old_state: Dict[Position, CellType]) -> Dict[Position, CellType]:
        """Возвращает изменения между состояниями"""
        if not self.game_map:
            return {}

        changes = {}
        for pos, old_cell in old_state.items():
            new_cell = self.game_map.get_cell(pos)
            if new_cell != old_cell:
                changes[pos] = old_cell

        return changes

    def undo(self) -> bool:
        """Отменяет последний ход"""
        if not self.move_history or self.game_state != GameState.PLAYING:
            return False

        move_record = self.move_history.pop()

        # Восстанавливаем предыдущее состояние
        for pos, cell_type in move_record.positions_changed.items():
            if self.game_map:
                self.game_map.set_cell(pos, cell_type)
                if cell_type in (CellType.WORKER, CellType.WORKER_IN_DEST):
                    self.game_map.worker_position = pos

        self.step_count -= 1
        self.notify('move_undone', self.step_count)
        return True

    def reset_level(self) -> None:
        """Сбрасывает текущий уровень"""
        if self.game_map:
            self.game_map.reset()
            self.move_history.clear()
            self.step_count = 0
            self.timer.start()
            self.game_state = GameState.PLAYING
            self.notify('level_reset')

    def next_level(self) -> bool:
        """Переходит к следующему уровню"""
        next_level = (self.current_level % TOTAL_GAMES) + 1
        return self.load_level(next_level)

    def previous_level(self) -> bool:
        """Переходит к предыдущему уровню"""
        prev_level = ((self.current_level - 2) % TOTAL_GAMES) + 1
        return self.load_level(prev_level)

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику игры"""
        if not self.game_map:
            return {}

        boxes_on_dest = self.game_map.count_boxes_on_destinations()

        return {
            'level': self.current_level,
            'total_levels': TOTAL_GAMES,
            'steps': self.step_count,
            'boxes_on_dest': boxes_on_dest,
            'total_destinations': self.game_map.total_destinations,
            'time': str(self.timer.get_elapsed()).split('.')[0],
            'can_undo': len(self.move_history) > 0,
            'game_state': self.game_state
        }

    def save_game(self, slot: int = 0) -> bool:
        """Сохраняет игру в слот"""
        try:
            save_data = {
                'version': '2.0',
                'timestamp': datetime.now().isoformat(),
                'level': self.current_level,
                'steps': self.step_count,
                'map_data': self.game_map.map_data.tolist() if self.game_map else [],
                'move_history': [record.to_dict() for record in self.move_history],
                'timer_state': str(self.timer.get_elapsed())
            }

            save_file = Path(f"save_{slot}.pkl")
            with open(save_file, 'wb') as f:
                pickle.dump(save_data, f)

            logger.info(f"Игра сохранена в слот {slot}")
            return True

        except Exception as e:
            logger.error(f"Ошибка сохранения игры: {e}")
            return False

    def load_game(self, slot: int = 0) -> bool:
        """Загружает игру из слота"""
        try:
            save_file = Path(f"save_{slot}.pkl")
            if not save_file.exists():
                return False

            with open(save_file, 'rb') as f:
                save_data = pickle.load(f)

            # Загружаем уровень
            if not self.load_level(save_data['level']):
                return False

            # Восстанавливаем состояние
            self.step_count = save_data['steps']
            self.move_history = [
                MoveRecord.from_dict(record)
                for record in save_data.get('move_history', [])
            ]

            logger.info(f"Игра загружена из слота {slot}")
            self.notify('game_loaded')
            return True

        except Exception as e:
            logger.error(f"Ошибка загрузки игры: {e}")
            return False


# ==================== ПОЛЬЗОВАТЕЛЬСКИЙ ИНТЕРФЕЙС ====================

class ModernButton(ttk.Button):
    """Современная кнопка с улучшенным внешним видом"""

    def __init__(self, parent, **kwargs):
        style = ttk.Style()
        style.configure('Modern.TButton', padding=6, relief='flat')

        super().__init__(parent, style='Modern.TButton', **kwargs)

        self.bind('<Enter>', self._on_enter)
        self.bind('<Leave>', self._on_leave)

    def _on_enter(self, event):
        self.config(cursor='hand2')

    def _on_leave(self, event):
        self.config(cursor='')


class GameCanvas(tk.Canvas):
    """Улучшенный игровой холст"""

    def __init__(self, parent, box_size: int, style: UIStyle, **kwargs):
        super().__init__(parent, **kwargs)
        self.box_size = box_size
        self.style = style
        self.animation_queue = []
        self.current_animation = None

        self.configure(
            bg=self.style.canvas_bg,
            highlightthickness=0,
            cursor='crosshair'
        )

    def draw_cell(self, position: Position, image: ImageTk.PhotoImage) -> int:
        """Рисует клетку с изображением"""
        x = position.col * self.box_size + self.box_size // 2
        y = position.row * self.box_size + self.box_size // 2

        return self.create_image(x, y, image=image, anchor='center')

    def animate_move(self, from_pos: Position, to_pos: Position,
                     image: ImageTk.PhotoImage, duration: int = 200) -> None:
        """Анимирует перемещение объекта"""
        if self.current_animation:
            self.after_cancel(self.current_animation)

        start_x = from_pos.col * self.box_size + self.box_size // 2
        start_y = from_pos.row * self.box_size + self.box_size // 2
        end_x = to_pos.col * self.box_size + self.box_size // 2
        end_y = to_pos.row * self.box_size + self.box_size // 2

        item_id = self.create_image(start_x, start_y, image=image, anchor='center')

        steps = duration // 20
        dx = (end_x - start_x) / steps
        dy = (end_y - start_y) / steps

        current_step = 0

        def animate():
            nonlocal current_step
            if current_step < steps:
                self.move(item_id, dx, dy)
                current_step += 1
                self.current_animation = self.after(20, animate)
            else:
                self.delete(item_id)
                self.current_animation = None

        self.current_animation = self.after(20, animate)

    def draw_grid(self, rows: int, cols: int) -> None:
        """Рисует сетку на холсте"""
        # Вертикальные линии
        for col in range(cols + 1):
            x = col * self.box_size
            self.create_line(
                x, 0, x, rows * self.box_size,
                fill=self.style.grid_color if hasattr(self.style, 'grid_color') else '#CCCCCC',
                width=1, dash=(2, 2)
            )

        # Горизонтальные линии
        for row in range(rows + 1):
            y = row * self.box_size
            self.create_line(
                0, y, cols * self.box_size, y,
                fill=self.style.grid_color if hasattr(self.style, 'grid_color') else '#CCCCCC',
                width=1, dash=(2, 2)
            )

    def clear_animations(self) -> None:
        """Очищает все анимации"""
        if self.current_animation:
            self.after_cancel(self.current_animation)
            self.current_animation = None


class StatusBar(tk.Frame):
    """Строка состояния"""

    def __init__(self, parent, style: UIStyle, **kwargs):
        super().__init__(parent, **kwargs)
        self.style = style

        self.configure(
            bg=self.style.bg_color,
            height=30,
            relief='sunken',
            borderwidth=1
        )

        self.message_var = tk.StringVar(value="Готов")
        self.time_var = tk.StringVar(value="00:00:00")

        # Сообщение
        self.message_label = tk.Label(
            self,
            textvariable=self.message_var,
            bg=self.style.bg_color,
            fg=self.style.fg_color,
            font=self.style.get_font(9)
        )
        self.message_label.pack(side='left', padx=10)

        # Время
        self.time_label = tk.Label(
            self,
            textvariable=self.time_var,
            bg=self.style.bg_color,
            fg=self.style.fg_color,
            font=self.style.get_font(9)
        )
        self.time_label.pack(side='right', padx=10)

    def set_message(self, message: str) -> None:
        """Устанавливает сообщение"""
        self.message_var.set(message)

    def set_time(self, time_str: str) -> None:
        """Устанавливает время"""
        self.time_var.set(time_str)


class AboutDialog:
    """Диалог "О программе" """

    def __init__(self, parent):
        self.parent = parent
        self.dialog = None

    def show(self):
        """Показывает диалог"""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("О программе")
        self.dialog.geometry("400x300")
        self.dialog.resizable(False, False)
        self.dialog.transient(self.parent)
        self.dialog.grab_set()

        # Центрирование
        x = self.parent.winfo_rootx() + 50
        y = self.parent.winfo_rooty() + 50
        self.dialog.geometry(f"+{x}+{y}")

        # Содержимое
        content = """
        推箱子游戏 (Sokoban)

        Версия: 2.0
        Разработчик: Edwin.Zhang

        Исходный код: 2019-6-28
        Рефакторинг: 2024

        Технологии:
        - Python 3.8+
        - tkinter
        - PIL (Pillow)
        - NumPy

        Лицензия: MIT
        """

        text = tk.Text(
            self.dialog,
            wrap='word',
            height=15,
            width=40,
            font=('Arial', 10)
        )
        text.insert('1.0', content)
        text.config(state='disabled')
        text.pack(padx=10, pady=10, fill='both', expand=True)

        # Кнопка закрытия
        tk.Button(
            self.dialog,
            text="Закрыть",
            command=self.dialog.destroy
        ).pack(pady=(0, 10))


# ==================== ГЛАВНОЕ ОКНО ====================

class MainWindow:
    """Главное окно игры"""

    def __init__(self):
        self.root = tk.Tk()
        self.config_manager = ConfigManager()
        self.resource_manager = ResourceManager()
        self.game_controller = GameController()

        # UI компоненты
        self.canvas: Optional[GameCanvas] = None
        self.status_bar: Optional[StatusBar] = None
        self.menu_bar: Optional[tk.Menu] = None

        # Состояние
        self.current_style = self.config_manager.get_ui_style()
        self.is_paused = False
        self.update_id = None

        # Настройка окна
        self._setup_window()

        # Загрузка ресурсов
        if not self.resource_manager.load_resources():
            messagebox.showerror("Ошибка", "Не удалось загрузить ресурсы игры")
            self.root.destroy()
            return

        # Создание интерфейса
        self._create_ui()

        # Подписка на события
        self.game_controller.attach(self._on_game_event)

    def _setup_window(self):
        """Настраивает главное окно"""
        self.root.title(self.config_manager.get_text('title'))
        self.root.configure(bg=self.current_style.bg_color)

        # Минимальный размер
        self.root.minsize(400, 300)

        # Обработчик закрытия
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Иконка (если есть)
        icon_path = Path("icon.ico")
        if icon_path.exists():
            try:
                self.root.iconbitmap(str(icon_path))
            except:
                pass

    def _create_ui(self):
        """Создает пользовательский интерфейс"""
        # Главный контейнер
        main_container = tk.Frame(self.root, bg=self.current_style.bg_color)
        main_container.pack(fill='both', expand=True, padx=10, pady=10)

        # Холст
        self.canvas = GameCanvas(
            main_container,
            box_size=self.config_manager.config.box_size,
            style=self.current_style,
            width=400,
            height=400
        )
        self.canvas.pack(pady=(0, 10))

        # Панель управления
        control_frame = self._create_control_panel(main_container)
        control_frame.pack(fill='x', pady=(0, 10))

        # Строка состояния
        self.status_bar = StatusBar(main_container, self.current_style)
        self.status_bar.pack(fill='x', side='bottom')

        # Меню
        self._create_menu()

        # Привязка клавиш
        self._bind_keys()

        # Обновление интерфейса
        self._update_display()

        # Запуск игрового цикла
        self._start_game_loop()

    def _create_control_panel(self, parent) -> tk.Frame:
        """Создает панель управления"""
        frame = tk.Frame(parent, bg=self.current_style.bg_color)

        # Кнопки
        buttons = [
            ("⏮️", self._prev_level, "Предыдущий уровень"),
            ("⏭️", self._next_level, "Следующий уровень"),
            ("↺", self._restart_level, "Перезапустить уровень"),
            ("↶", self._undo_move, "Отменить ход"),
            ("⏸️", self._toggle_pause, "Пауза"),
            ("⚙️", self._show_settings, "Настройки"),
            ("ℹ️", self._show_about, "О программе")
        ]

        for icon, command, tooltip in buttons:
            btn = ModernButton(
                frame,
                text=icon,
                command=command
            )
            btn.pack(side='left', padx=2)

            # Простой тултип
            self._create_tooltip(btn, tooltip)

        return frame

    def _create_tooltip(self, widget, text):
        """Создает подсказку"""

        def show_tooltip(event):
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 25

            tooltip = tk.Toplevel(widget)
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{x}+{y}")

            label = tk.Label(
                tooltip,
                text=text,
                bg="#FFFFE0",
                fg="black",
                relief="solid",
                borderwidth=1
            )
            label.pack()

            widget.tooltip = tooltip

        def hide_tooltip(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                delattr(widget, 'tooltip')

        widget.bind("<Enter>", show_tooltip)
        widget.bind("<Leave>", hide_tooltip)

    def _create_menu(self):
        """Создает меню"""
        self.menu_bar = tk.Menu(self.root, bg=self.current_style.bg_color, fg=self.current_style.fg_color)
        self.root.config(menu=self.menu_bar)

        # Меню "Игра"
        game_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Игра", menu=game_menu)

        game_menu.add_command(label="Новая игра", command=self._new_game)
        game_menu.add_command(label="Сохранить", command=lambda: self.game_controller.save_game())
        game_menu.add_command(label="Загрузить", command=lambda: self.game_controller.load_game())
        game_menu.add_separator()
        game_menu.add_command(label="Выход", command=self._on_closing)

        # Меню "Уровень"
        level_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Уровень", menu=level_menu)

        level_menu.add_command(label="Предыдущий", command=self._prev_level)
        level_menu.add_command(label="Следующий", command=self._next_level)
        level_menu.add_separator()

        # Динамические пункты меню для уровней
        for i in range(min(10, TOTAL_GAMES)):
            level_menu.add_command(
                label=f"Уровень {i + 1}",
                command=lambda l=i + 1: self._load_specific_level(l)
            )

        # Меню "Справка"
        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Справка", menu=help_menu)

        help_menu.add_command(label="Управление", command=self._show_controls)
        help_menu.add_command(label="О программе", command=self._show_about)

    def _bind_keys(self):
        """Привязывает горячие клавиши"""
        # Управление
        self.root.bind('<Up>', lambda e: self._move(Direction.UP))
        self.root.bind('<Down>', lambda e: self._move(Direction.DOWN))
        self.root.bind('<Left>', lambda e: self._move(Direction.LEFT))
        self.root.bind('<Right>', lambda e: self._move(Direction.RIGHT))

        # Горячие клавиши
        self.root.bind('<Control-n>', lambda e: self._new_game())
        self.root.bind('<Control-s>', lambda e: self.game_controller.save_game())
        self.root.bind('<Control-l>', lambda e: self.game_controller.load_game())
        self.root.bind('<Control-z>', lambda e: self._undo_move())
        self.root.bind('<Control-r>', lambda e: self._restart_level())
        self.root.bind('<Escape>', lambda e: self._toggle_pause())
        self.root.bind('<space>', lambda e: self._restart_level())

        # Фокус на холсте
        self.canvas.bind('<Button-1>', lambda e: self.canvas.focus_set())
        self.canvas.focus_set()

    def _start_game_loop(self):
        """Запускает игровой цикл"""
        self._update_game_loop()

    def _update_game_loop(self):
        """Обновляет игровой цикл"""
        if not self.is_paused:
            self._update_display()

        self.update_id = self.root.after(100, self._update_game_loop)

    def _update_display(self):
        """Обновляет отображение игры"""
        # Очищаем холст
        self.canvas.delete('all')

        # Отрисовываем карту
        if self.game_controller.game_map:
            game_map = self.game_controller.game_map

            # Рисуем сетку если нужно
            if self.config_manager.config.ui_config.show_grid:
                self.canvas.draw_grid(game_map.rows, game_map.cols)

            # Рисуем все клетки
            for i in range(game_map.rows):
                for j in range(game_map.cols):
                    pos = Position(i, j)
                    cell_type = game_map.get_cell(pos)

                    if cell_type:
                        image = self._get_image_for_cell(pos, cell_type)
                        if image:
                            self.canvas.draw_cell(pos, image)

            # Обновляем статистику
            stats = self.game_controller.get_stats()
            self.status_bar.set_message(
                f"Уровень: {stats['level']} | "
                f"Шаги: {stats['steps']} | "
                f"Ящики: {stats['boxes_on_dest']}/{stats['total_destinations']}"
            )
            self.status_bar.set_time(stats['time'])

    def _get_image_for_cell(self, position: Position, cell_type: CellType) -> Optional[ImageTk.PhotoImage]:
        """Получает изображение для клетки"""
        game_map = self.game_controller.game_map
        if not game_map:
            return None

        # Для рабочего учитываем направление и положение
        if cell_type in (CellType.WORKER, CellType.WORKER_IN_DEST):
            # В реальной игре здесь нужно определить направление
            direction = Direction.STOP
            in_destination = (cell_type == CellType.WORKER_IN_DEST)
            return self.resource_manager.get_worker_image(direction, in_destination)

        # Для остальных клеток
        return self.resource_manager.get_image(cell_type)

    def _on_game_event(self, event_type: str, *args, **kwargs):
        """Обрабатывает игровые события"""
        if event_type == 'level_completed':
            level, steps = args
            self._show_level_complete_dialog(level, steps)
        elif event_type == 'move_made':
            direction, steps = args
            logger.info(f"Ход {direction.value}, всего шагов: {steps}")
        elif event_type == 'level_loaded':
            level = args[0]
            logger.info(f"Загружен уровень {level}")

    def _move(self, direction: Direction):
        """Обрабатывает движение"""
        if self.is_paused:
            return

        if self.game_controller.move(direction):
            self._update_display()

    def _new_game(self):
        """Начинает новую игру"""
        if messagebox.askyesno("Новая игра", "Начать новую игру?"):
            self.game_controller.load_level(1)
            self._update_display()

    def _prev_level(self):
        """Предыдущий уровень"""
        if self.game_controller.previous_level():
            self._update_display()

    def _next_level(self):
        """Следующий уровень"""
        if self.game_controller.next_level():
            self._update_display()

    def _load_specific_level(self, level: int):
        """Загружает конкретный уровень"""
        if self.game_controller.load_level(level):
            self._update_display()

    def _restart_level(self):
        """Перезапускает уровень"""
        self.game_controller.reset_level()
        self._update_display()

    def _undo_move(self):
        """Отменяет ход"""
        if self.game_controller.undo():
            self._update_display()

    def _toggle_pause(self):
        """Переключает паузу"""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.status_bar.set_message("Игра приостановлена")
        else:
            self.status_bar.set_message("Игра продолжается")

    def _show_settings(self):
        """Показывает настройки"""
        # Здесь должна быть реализация диалога настроек
        messagebox.showinfo("Настройки", "Настройки игры (реализация в процессе)")

    def _show_about(self):
        """Показывает информацию о программе"""
        AboutDialog(self.root).show()

    def _show_controls(self):
        """Показывает управление"""
        controls = """
        Управление игрой:

        Стрелки - перемещение
        Space - перезапуск уровня
        Ctrl+Z - отмена хода
        Ctrl+R - перезапуск уровня
        Ctrl+N - новая игра
        Ctrl+S - сохранить игру
        Ctrl+L - загрузить игру
        Escape - пауза/продолжить

        Мышь:
        ЛКМ на холсте - установить фокус
        """

        messagebox.showinfo("Управление", controls)

    def _show_level_complete_dialog(self, level: int, steps: int):
        """Показывает диалог завершения уровня"""
        message = (
            f"Поздравляем! Уровень {level} пройден!\n\n"
            f"Шагов: {steps}\n\n"
            f"Перейти к следующему уровню?"
        )

        if messagebox.askyesno("Уровень пройден!", message):
            self._next_level()

    def _on_closing(self):
        """Обработчик закрытия окна"""
        if messagebox.askokcancel("Выход", "Вы действительно хотите выйти?"):
            # Сохраняем конфигурацию
            self.config_manager.save_config()

            # Останавливаем игровой цикл
            if self.update_id:
                self.root.after_cancel(self.update_id)

            # Закрываем окно
            self.root.destroy()

    def run(self):
        """Запускает главный цикл"""
        self.root.mainloop()


# ==================== ТОЧКА ВХОДА ====================

def main():
    """Главная функция"""
    try:
        # Проверяем наличие необходимых файлов
        required_files = ['game_maps.py', 'images']
        missing_files = []

        for file in required_files:
            if not Path(file).exists():
                missing_files.append(file)

        if missing_files:
            error_msg = f"Отсутствуют необходимые файлы:\n" + "\n".join(missing_files)
            print(error_msg)
            messagebox.showerror("Ошибка", error_msg)
            return

        # Запускаем игру
        app = MainWindow()
        app.run()

    except Exception as e:
        logger.critical(f"Критическая ошибка: {e}", exc_info=True)
        messagebox.showerror(
            "Критическая ошибка",
            f"Произошла критическая ошибка:\n{str(e)}\n\n"
            "Подробности в файле лога."
        )
    finally:
        logger.info("Игра завершена")


if __name__ == "__main__":
    # Инициализация
    print("=" * 50)
    print("推箱子游戏 (Sokoban)")
    print("Версия 2.0")
    print("=" * 50)

    # Запуск
    main()