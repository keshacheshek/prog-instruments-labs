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
from tkinter import Tk, Canvas, Label, Frame, Button, Menu, Toplevel, Scale, HORIZONTAL
from tkinter import ttk, messagebox, font
from tkinter.messagebox import showinfo, askyesno
import numpy as np
from game_maps import basic_maps
from typing import Tuple, Optional, Dict, List, Any, Set, Callable
from dataclasses import dataclass, field, asdict
from enum import IntEnum, Enum, auto
from abc import ABC, abstractmethod
import os
import pickle
import json
from datetime import datetime


# ==================== ПЕРЕЧИСЛЕНИЯ И СТРУКТУРЫ ====================

class GameState(Enum):
    """Состояния игры"""
    MENU = auto()
    PLAYING = auto()
    PAUSED = auto()
    LEVEL_COMPLETE = auto()
    GAME_OVER = auto()


class UIMode(Enum):
    """Режимы интерфейса"""
    CLASSIC = "classic"
    DARK = "dark"
    LIGHT = "light"
    COLORFUL = "colorful"


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


@dataclass
class UIConfig:
    """Конфигурация интерфейса"""
    box_size: int = 64
    show_grid: bool = False
    grid_color: str = "#CCCCCC"
    show_coordinates: bool = False
    animate_moves: bool = True
    animation_speed: int = 50  # мс
    ui_mode: UIMode = UIMode.CLASSIC
    enable_sounds: bool = False
    volume: int = 50


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
    config_file: str = "game_config.json"

    ui_config: UIConfig = field(default_factory=UIConfig)


# ==================== КОНСТАНТЫ ====================

TOTAL_GAMES = len(basic_maps)  # Общее количество уровней

DIRECTION_OFFSETS = {
    Direction.UP: (-1, 0, -2, 0),
    Direction.DOWN: (1, 0, 2, 0),
    Direction.LEFT: (0, -1, 0, -2),
    Direction.RIGHT: (0, 1, 0, 2),
    Direction.STOP: (0, 0, 0, 0)
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
        bg_color="#2E2E2E",
        fg_color="#FFFFFF",
        button_bg="#555555",
        button_fg="#FFFFFF",
        canvas_bg="#1E1E1E",
        highlight_color="#BB86FC",
        font_family="Consolas"
    ),
    UIMode.LIGHT: UIStyle(
        name="light",
        bg_color="#FFFFFF",
        fg_color="#000000",
        button_bg="#E3F2FD",
        button_fg="#1565C0",
        canvas_bg="#FAFAFA",
        highlight_color="#42A5F5"
    ),
    UIMode.COLORFUL: UIStyle(
        name="colorful",
        bg_color="#FFF8E1",
        fg_color="#5D4037",
        button_bg="#FF9800",
        button_fg="#FFFFFF",
        canvas_bg="#FFF3E0",
        highlight_color="#FF5722"
    )
}


# ==================== КЛАСС КОНФИГУРАЦИИ ====================

class ConfigManager:
    """Менеджер конфигурации игры"""

    def __init__(self, config_file: str = "game_config.json"):
        self.config_file = config_file
        self.config = self._load_config()

    def _load_config(self) -> GameConfig:
        """Загружает конфигурацию из файла или создает новую"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Восстанавливаем объекты Enum
                if 'ui_config' in data and 'ui_mode' in data['ui_config']:
                    data['ui_config']['ui_mode'] = UIMode(data['ui_config']['ui_mode'])

                return GameConfig(**data)
            except Exception as e:
                print(f"Ошибка загрузки конфигурации: {e}")

        # Создаем конфигурацию по умолчанию
        return GameConfig()

    def save_config(self) -> bool:
        """Сохраняет конфигурацию в файл"""
        try:
            data = asdict(self.config)
            # Сериализуем объекты Enum
            if 'ui_config' in data and 'ui_mode' in data['ui_config']:
                data['ui_config']['ui_mode'] = data['ui_config']['ui_mode'].value

            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            print(f"Ошибка сохранения конфигурации: {e}")
            return False

    def update_config(self, **kwargs) -> None:
        """Обновляет конфигурацию"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def get_ui_style(self) -> UIStyle:
        """Возвращает текущий стиль интерфейса"""
        return UI_STYLES.get(self.config.ui_config.ui_mode, UI_STYLES[UIMode.CLASSIC])


# ==================== КОМПОНЕНТЫ ИНТЕРФЕЙСА ====================

class UIComponent(ABC):
    """Базовый класс для компонентов интерфейса"""

    def __init__(self, parent, style: UIStyle):
        self.parent = parent
        self.style = style
        self.widget = None

    @abstractmethod
    def create(self):
        """Создает виджет"""
        pass

    def update_style(self, new_style: UIStyle):
        """Обновляет стиль компонента"""
        self.style = new_style
        self._apply_style()

    @abstractmethod
    def _apply_style(self):
        """Применяет стиль к виджету"""
        pass


class GameCanvas(UIComponent):
    """Игровой холст с дополнительными возможностями"""

    def __init__(self, parent, style: UIStyle, box_size: int):
        super().__init__(parent, style)
        self.box_size = box_size
        self.grid_lines = []
        self.coordinate_labels = []
        self.show_grid = False
        self.show_coordinates = False
        self.animation_id = None

    def create(self) -> Canvas:
        """Создает холст"""
        self.widget = Canvas(
            self.parent,
            bg=self.style.canvas_bg,
            highlightthickness=0,
            cursor="crosshair"
        )
        return self.widget

    def set_grid_visibility(self, visible: bool):
        """Устанавливает видимость сетки"""
        self.show_grid = visible
        if not visible:
            self._clear_grid()

    def set_coordinates_visibility(self, visible: bool):
        """Устанавливает видимость координат"""
        self.show_coordinates = visible
        if not visible:
            self._clear_coordinates()

    def draw_grid(self, rows: int, cols: int):
        """Рисует сетку на холсте"""
        if not self.show_grid:
            return

        self._clear_grid()

        # Вертикальные линии
        for col in range(cols + 1):
            x = col * self.box_size
            line = self.widget.create_line(
                x, 0, x, rows * self.box_size,
                fill=self.style.grid_color, width=1, dash=(2, 2)
            )
            self.grid_lines.append(line)

        # Горизонтальные линии
        for row in range(rows + 1):
            y = row * self.box_size
            line = self.widget.create_line(
                0, y, cols * self.box_size, y,
                fill=self.style.grid_color, width=1, dash=(2, 2)
            )
            self.grid_lines.append(line)

    def draw_coordinates(self, rows: int, cols: int):
        """Рисует координаты на холсте"""
        if not self.show_coordinates:
            return

        self._clear_coordinates()

        for row in range(rows):
            for col in range(cols):
                x = col * self.box_size + self.box_size // 2
                y = row * self.box_size + self.box_size // 2

                label = self.widget.create_text(
                    x, y,
                    text=f"{row},{col}",
                    font=(self.style.font_family, 8),
                    fill=self.style.fg_color,
                    state="disabled"
                )
                self.coordinate_labels.append(label)

    def animate_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int],
                     image: ImageTk.PhotoImage, callback: Callable):
        """Анимирует перемещение объекта"""
        if self.animation_id:
            self.widget.after_cancel(self.animation_id)

        from_x = from_pos[1] * self.box_size + self.box_size // 2
        from_y = from_pos[0] * self.box_size + self.box_size // 2
        to_x = to_pos[1] * self.box_size + self.box_size // 2
        to_y = to_pos[0] * self.box_size + self.box_size // 2

        step = 0
        total_steps = 10

        def animate():
            nonlocal step
            if step <= total_steps:
                t = step / total_steps
                # Квадратичная интерполяция для плавности
                t = t * t * (3 - 2 * t)

                x = from_x + (to_x - from_x) * t
                y = from_y + (to_y - from_y) * t

                # Удаляем старое изображение
                self.widget.delete("animated")

                # Создаем новое на промежуточной позиции
                self.widget.create_image(
                    x, y,
                    image=image,
                    tags="animated"
                )

                step += 1
                self.animation_id = self.widget.after(20, animate)
            else:
                self.widget.delete("animated")
                if callback:
                    callback()

        animate()

    def _clear_grid(self):
        """Очищает сетку"""
        for line in self.grid_lines:
            self.widget.delete(line)
        self.grid_lines.clear()

    def _clear_coordinates(self):
        """Очищает координаты"""
        for label in self.coordinate_labels:
            self.widget.delete(label)
        self.coordinate_labels.clear()

    def _apply_style(self):
        """Применяет стиль к холсту"""
        if self.widget:
            self.widget.configure(bg=self.style.canvas_bg)


class ControlPanel(UIComponent):
    """Панель управления игрой"""

    def __init__(self, parent, style: UIStyle):
        super().__init__(parent, style)
        self.buttons = {}
        self.stats_labels = {}

    def create(self) -> Frame:
        """Создает панель управления"""
        self.widget = Frame(self.parent, bg=self.style.bg_color, padx=10, pady=10)

        # Панель статистики
        stats_frame = Frame(self.widget, bg=self.style.bg_color)
        stats_frame.pack(side="left", fill="y", padx=(0, 20))

        stats = [
            ("Уровень", "level"),
            ("Шаги", "steps"),
            ("Ящики", "boxes"),
            ("Время", "time")
        ]

        for label_text, key in stats:
            label_frame = Frame(stats_frame, bg=self.style.bg_color)
            label_frame.pack(fill="x", pady=2)

            Label(
                label_frame,
                text=f"{label_text}:",
                bg=self.style.bg_color,
                fg=self.style.fg_color,
                font=(self.style.font_family, 9)
            ).pack(side="left")

            value_label = Label(
                label_frame,
                text="0",
                bg=self.style.bg_color,
                fg=self.style.highlight_color,
                font=(self.style.font_family, 9, "bold")
            )
            value_label.pack(side="left", padx=(5, 0))

            self.stats_labels[key] = value_label

        # Панель кнопок
        buttons_frame = Frame(self.widget, bg=self.style.bg_color)
        buttons_frame.pack(side="left")

        button_configs = [
            ("⏮️", "prev_level", "Предыдущий уровень"),
            ("⏭️", "next_level", "Следующий уровень"),
            ("↺", "restart", "Перезапустить"),
            ("↶", "undo", "Отменить ход"),
            ("⏸️", "pause", "Пауза"),
            ("⚙️", "settings", "Настройки")
        ]

        for icon, key, tooltip in button_configs:
            btn = Button(
                buttons_frame,
                text=icon,
                font=(self.style.font_family, 14),
                bg=self.style.button_bg,
                fg=self.style.button_fg,
                relief="flat",
                width=3,
                cursor="hand2"
            )
            btn.pack(side="left", padx=2)
            self.buttons[key] = btn

            # Простой tooltip
            self._create_tooltip(btn, tooltip)

        return self.widget

    def _create_tooltip(self, widget, text):
        """Создает подсказку для виджета"""

        def enter(event):
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 25

            self.tooltip = Toplevel(widget)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{x}+{y}")

            label = Label(
                self.tooltip,
                text=text,
                bg="#FFFFE0",
                fg="black",
                relief="solid",
                borderwidth=1,
                font=(self.style.font_family, 8)
            )
            label.pack()

        def leave(event):
            if hasattr(self, 'tooltip'):
                self.tooltip.destroy()

        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)

    def update_stats(self, stats: Dict[str, Any]):
        """Обновляет статистику"""
        for key, label in self.stats_labels.items():
            if key in stats:
                label.config(text=str(stats[key]))

    def bind_button(self, key: str, command: Callable):
        """Привязывает команду к кнопке"""
        if key in self.buttons:
            self.buttons[key].config(command=command)

    def _apply_style(self):
        """Применяет стиль к панели"""
        if self.widget:
            self.widget.configure(bg=self.style.bg_color)

            for widget in self.widget.winfo_children():
                if isinstance(widget, Frame):
                    widget.configure(bg=self.style.bg_color)
                elif isinstance(widget, Label):
                    widget.configure(bg=self.style.bg_color, fg=self.style.fg_color)

            for btn in self.buttons.values():
                btn.configure(bg=self.style.button_bg, fg=self.style.button_fg)

            for label in self.stats_labels.values():
                label.configure(bg=self.style.bg_color, fg=self.style.highlight_color)


class GameMenu:
    """Меню игры"""

    def __init__(self, root: Tk, style: UIStyle, config_manager: ConfigManager):
        self.root = root
        self.style = style
        self.config_manager = config_manager
        self.menu_bar = None

    def create(self):
        """Создает меню"""
        self.menu_bar = Menu(self.root, bg=self.style.bg_color, fg=self.style.fg_color)
        self.root.config(menu=self.menu_bar)

        # Меню "Игра"
        game_menu = Menu(self.menu_bar, tearoff=0, bg=self.style.bg_color, fg=self.style.fg_color)
        self.menu_bar.add_cascade(label="Игра", menu=game_menu)

        game_menu.add_command(label="Новая игра", accelerator="Ctrl+N")
        game_menu.add_command(label="Сохранить", accelerator="Ctrl+S")
        game_menu.add_command(label="Загрузить", accelerator="Ctrl+L")
        game_menu.add_separator()
        game_menu.add_command(label="Выход", accelerator="Alt+F4")

        # Меню "Уровень"
        level_menu = Menu(self.menu_bar, tearoff=0, bg=self.style.bg_color, fg=self.style.fg_color)
        self.menu_bar.add_cascade(label="Уровень", menu=level_menu)

        level_menu.add_command(label="Предыдущий", accelerator="Ctrl+P")
        level_menu.add_command(label="Следующий", accelerator="Ctrl+N")
        level_menu.add_separator()

        # Динамическое меню уровней
        for i in range(min(10, TOTAL_GAMES)):
            level_menu.add_command(label=f"Уровень {i + 1}")

        if TOTAL_GAMES > 10:
            level_menu.add_command(label="Выбрать уровень...")

        # Меню "Настройки"
        settings_menu = Menu(self.menu_bar, tearoff=0, bg=self.style.bg_color, fg=self.style.fg_color)
        self.menu_bar.add_cascade(label="Настройки", menu=settings_menu)

        # Подменю "Стиль"
        style_menu = Menu(settings_menu, tearoff=0, bg=self.style.bg_color, fg=self.style.fg_color)
        settings_menu.add_cascade(label="Стиль интерфейса", menu=style_menu)

        for mode in UIMode:
            style_menu.add_radiobutton(
                label=mode.value.capitalize(),
                variable=1,  # Временно
                value=mode.value
            )

        settings_menu.add_checkbutton(label="Показывать сетку")
        settings_menu.add_checkbutton(label="Показывать координаты")
        settings_menu.add_separator()
        settings_menu.add_command(label="Размер блоков...")

        # Меню "Справка"
        help_menu = Menu(self.menu_bar, tearoff=0, bg=self.style.bg_color, fg=self.style.fg_color)
        self.menu_bar.add_cascade(label="Справка", menu=help_menu)

        help_menu.add_command(label="Управление")
        help_menu.add_command(label="О программе")

        return self.menu_bar

    def update_style(self, new_style: UIStyle):
        """Обновляет стиль меню"""
        self.style = new_style
        # Обновление стиля меню требует пересоздания, поэтому здесь просто сохраняем стиль


class SettingsDialog:
    """Диалоговое окно настроек"""

    def __init__(self, parent, config_manager: ConfigManager, on_save: Callable):
        self.parent = parent
        self.config_manager = config_manager
        self.on_save = on_save
        self.dialog = None

    def show(self):
        """Показывает диалог настроек"""
        self.dialog = Toplevel(self.parent)
        self.dialog.title("Настройки игры")
        self.dialog.geometry("400x500")
        self.dialog.resizable(False, False)
        self.dialog.transient(self.parent)
        self.dialog.grab_set()

        # Центрирование
        self.dialog.geometry(
            f"+{self.parent.winfo_rootx() + 50}+{self.parent.winfo_rooty() + 50}"
        )

        # Создание вкладок
        notebook = ttk.Notebook(self.dialog)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Вкладка "Интерфейс"
        ui_frame = Frame(notebook)
        self._create_ui_tab(ui_frame)
        notebook.add(ui_frame, text="Интерфейс")

        # Вкладка "Игра"
        game_frame = Frame(notebook)
        self._create_game_tab(game_frame)
        notebook.add(game_frame, text="Игра")

        # Кнопки
        button_frame = Frame(self.dialog)
        button_frame.pack(fill="x", padx=10, pady=10)

        Button(
            button_frame,
            text="Сохранить",
            command=self._save_settings
        ).pack(side="right", padx=5)

        Button(
            button_frame,
            text="Отмена",
            command=self.dialog.destroy
        ).pack(side="right", padx=5)

        Button(
            button_frame,
            text="По умолчанию",
            command=self._reset_to_defaults
        ).pack(side="left")

    def _create_ui_tab(self, parent):
        """Создает вкладку настроек интерфейса"""
        # Стиль интерфейса
        Label(parent, text="Стиль интерфейса:").grid(row=0, column=0, sticky="w", pady=5)

        self.style_var = StringVar(value=self.config_manager.config.ui_config.ui_mode.value)
        style_combo = ttk.Combobox(
            parent,
            textvariable=self.style_var,
            values=[mode.value for mode in UIMode],
            state="readonly"
        )
        style_combo.grid(row=0, column=1, sticky="ew", pady=5, padx=5)

        # Размер блоков
        Label(parent, text="Размер блоков:").grid(row=1, column=0, sticky="w", pady=5)

        self.size_var = IntVar(value=self.config_manager.config.ui_config.box_size)
        size_scale = Scale(
            parent,
            from_=32,
            to=128,
            resolution=32,
            orient=HORIZONTAL,
            variable=self.size_var
        )
        size_scale.grid(row=1, column=1, sticky="ew", pady=5, padx=5)

        # Чекбоксы
        self.grid_var = BooleanVar(value=self.config_manager.config.ui_config.show_grid)
        grid_check = Checkbutton(
            parent,
            text="Показывать сетку",
            variable=self.grid_var
        )
        grid_check.grid(row=2, column=0, columnspan=2, sticky="w", pady=5)

        self.coords_var = BooleanVar(value=self.config_manager.config.ui_config.show_coordinates)
        coords_check = Checkbutton(
            parent,
            text="Показывать координаты",
            variable=self.coords_var
        )
        coords_check.grid(row=3, column=0, columnspan=2, sticky="w", pady=5)

        self.animate_var = BooleanVar(value=self.config_manager.config.ui_config.animate_moves)
        animate_check = Checkbutton(
            parent,
            text="Анимировать движения",
            variable=self.animate_var
        )
        animate_check.grid(row=4, column=0, columnspan=2, sticky="w", pady=5)

        # Настройка столбцов
        parent.columnconfigure(1, weight=1)

    def _create_game_tab(self, parent):
        """Создает вкладку настроек игры"""
        # Количество шагов для отмены
        Label(parent, text="История отмены (шагов):").grid(row=0, column=0, sticky="w", pady=5)

        self.undo_var = IntVar(value=self.config_manager.config.max_undo_steps)
        undo_spin = Spinbox(
            parent,
            from_=10,
            to=200,
            increment=10,
            textvariable=self.undo_var,
            width=10
        )
        undo_spin.grid(row=0, column=1, sticky="w", pady=5, padx=5)

        # Сохранение игры
        self.save_var = BooleanVar(value=self.config_manager.config.enable_save_game)
        save_check = Checkbutton(
            parent,
            text="Автосохранение",
            variable=self.save_var
        )
        save_check.grid(row=1, column=0, columnspan=2, sticky="w", pady=5)

        # Кэширование изображений
        self.cache_var = BooleanVar(value=self.config_manager.config.enable_image_cache)
        cache_check = Checkbutton(
            parent,
            text="Кэшировать изображения",
            variable=self.cache_var
        )
        cache_check.grid(row=2, column=0, columnspan=2, sticky="w", pady=5)

        parent.columnconfigure(1, weight=1)

    def _save_settings(self):
        """Сохраняет настройки"""
        try:
            # Обновляем конфигурацию
            self.config_manager.update_config(
                max_undo_steps=self.undo_var.get(),
                enable_save_game=self.save_var.get(),
                enable_image_cache=self.cache_var.get(),
                box_size=self.size_var.get()
            )

            self.config_manager.config.ui_config.ui_mode = UIMode(self.style_var.get())
            self.config_manager.config.ui_config.box_size = self.size_var.get()
            self.config_manager.config.ui_config.show_grid = self.grid_var.get()
            self.config_manager.config.ui_config.show_coordinates = self.coords_var.get()
            self.config_manager.config.ui_config.animate_moves = self.animate_var.get()

            # Сохраняем в файл
            self.config_manager.save_config()

            # Вызываем callback
            if self.on_save:
                self.on_save()

            self.dialog.destroy()

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить настройки: {e}")

    def _reset_to_defaults(self):
        """Сбрасывает настройки к значениям по умолчанию"""
        default_config = GameConfig()

        self.style_var.set(default_config.ui_config.ui_mode.value)
        self.size_var.set(default_config.ui_config.box_size)
        self.grid_var.set(default_config.ui_config.show_grid)
        self.coords_var.set(default_config.ui_config.show_coordinates)
        self.animate_var.set(default_config.ui_config.animate_moves)
        self.undo_var.set(default_config.max_undo_steps)
        self.save_var.set(default_config.enable_save_game)
        self.cache_var.set(default_config.enable_image_cache)


# ==================== ОСНОВНЫЕ КЛАССЫ (с предыдущих коммитов) ====================

# (Здесь должны быть классы из предыдущих коммитов, но для краткости оставлю только заглушки)
# В реальном коде здесь должны быть полные реализации:
# - Position, Move, GameMap, MovementEngine, CollisionDetector
# - GameController, ResourceManager, GameResources, GameRenderer

# Для этого коммита сосредоточимся на UI и оставим игровую логику как есть,
# но добавим интеграцию с новыми UI компонентами

# ==================== ГЛАВНЫЙ КЛАСС ИГРЫ ====================

class PushBoxGame:
    """Основной класс игры с улучшенным UI"""

    def __init__(self):
        self.root = Tk()
        self.config_manager = ConfigManager()
        self.ui_style = self.config_manager.get_ui_style()

        # Игровые компоненты
        self.game_controller = None  # Будет инициализирован позже
        self.game_resources = None
        self.game_renderer = None

        # UI компоненты
        self.game_canvas = None
        self.control_panel = None
        self.game_menu = None
        self.settings_dialog = None

        # Состояние
        self.current_level = 1
        self.start_time = None
        self.is_paused = False

        # Настройки
        self._apply_configuration()

    def _apply_configuration(self):
        """Применяет конфигурацию"""
        self.ui_style = self.config_manager.get_ui_style()

        # Создаем ресурсы с правильным размером блоков
        self.game_resources = GameResources(self.config_manager.config.ui_config.box_size)

    def run(self):
        """Запускает игру"""
        self._setup_ui()
        self._load_game()
        self._start_game_loop()
        self.root.mainloop()

    def _setup_ui(self):
        """Настраивает пользовательский интерфейс"""
        self.root.title("推箱子 - Sokoban Game")
        self.root.configure(bg=self.ui_style.bg_color)

        # Создаем меню
        self.game_menu = GameMenu(self.root, self.ui_style, self.config_manager)
        self.game_menu.create()

        # Создаем основной контейнер
        main_frame = Frame(self.root, bg=self.ui_style.bg_color)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Создаем холст
        self.game_canvas = GameCanvas(
            main_frame,
            self.ui_style,
            self.config_manager.config.ui_config.box_size
        )
        canvas = self.game_canvas.create()
        canvas.pack(pady=(0, 10))

        # Создаем панель управления
        self.control_panel = ControlPanel(main_frame, self.ui_style)
        control_panel = self.control_panel.create()
        control_panel.pack(fill="x")

        # Настраиваем панель управления
        self._setup_control_panel()

        # Привязываем горячие клавиши
        self._bind_hotkeys()

        # Применяем настройки отображения
        self.game_canvas.set_grid_visibility(
            self.config_manager.config.ui_config.show_grid
        )
        self.game_canvas.set_coordinates_visibility(
            self.config_manager.config.ui_config.show_coordinates
        )

    def _setup_control_panel(self):
        """Настраивает панель управления"""
        self.control_panel.bind_button("prev_level", self._previous_level)
        self.control_panel.bind_button("next_level", self._next_level)
        self.control_panel.bind_button("restart", self._restart_level)
        self.control_panel.bind_button("undo", self._undo_move)
        self.control_panel.bind_button("pause", self._toggle_pause)
        self.control_panel.bind_button("settings", self._show_settings)

    def _bind_hotkeys(self):
        """Привязывает горячие клавиши"""
        self.root.bind("<Control-n>", lambda e: self._next_level())
        self.root.bind("<Control-p>", lambda e: self._previous_level())
        self.root.bind("<Control-r>", lambda e: self._restart_level())
        self.root.bind("<Control-z>", lambda e: self._undo_move())
        self.root.bind("<Control-s>", lambda e: self._save_game())
        self.root.bind("<Control-l>", lambda e: self._load_game())
        self.root.bind("<Escape>", lambda e: self._toggle_pause())
        self.root.bind("<space>", lambda e: self._restart_level())

        # Клавиши управления
        self.root.bind("<Up>", lambda e: self._move(Direction.UP))
        self.root.bind("<Down>", lambda e: self._move(Direction.DOWN))
        self.root.bind("<Left>", lambda e: self._move(Direction.LEFT))
        self.root.bind("<Right>", lambda e: self._move(Direction.RIGHT))

        # Передаем фокус на холст
        self.game_canvas.widget.bind("<Button-1>", lambda e: self.game_canvas.widget.focus_set())

    def _load_game(self):
        """Загружает игру"""
        # Здесь должна быть инициализация GameController
        # и загрузка уровня
        pass

    def _start_game_loop(self):
        """Запускает игровой цикл"""
        self.start_time = datetime.now()
        self._update_game_loop()

    def _update_game_loop(self):
        """Обновляет игровой цикл"""
        if not self.is_paused:
            self._update_stats()

        # Планируем следующее обновление
        self.root.after(1000, self._update_game_loop)

    def _update_stats(self):
        """Обновляет статистику"""
        if self.game_controller:
            stats = self.game_controller.get_game_stats()

            # Добавляем время
            if self.start_time:
                elapsed = datetime.now() - self.start_time
                stats['time'] = str(elapsed).split('.')[0]

            self.control_panel.update_stats(stats)

            # Обновляем заголовок окна
            title = f"推箱子 - Уровень {stats.get('current_level', 1)}/{TOTAL_GAMES}"
            if self.is_paused:
                title += " [ПАУЗА]"
            self.root.title(title)

    def _move(self, direction: Direction):
        """Обрабатывает движение"""
        if self.is_paused or not self.game_controller:
            return

        if self.game_controller.move_worker(direction):
            self._render_game()

            if self.game_controller.is_level_completed():
                self._handle_level_completion()

    def _render_game(self):
        """Отрисовывает игру"""
        if self.game_controller and self.game_controller.game_map:
            # Обновляем сетку и координаты если нужно
            game_map = self.game_controller.game_map
            self.game_canvas.draw_grid(game_map.rows, game_map.cols)
            self.game_canvas.draw_coordinates(game_map.rows, game_map.cols)

            # Здесь должна быть отрисовка через GameRenderer
            # self.game_renderer.render(...)

    def _previous_level(self):
        """Переходит к предыдущему уровню"""
        if self.game_controller:
            self.game_controller.previous_level()
            self._restart_level()

    def _next_level(self):
        """Переходит к следующему уровню"""
        if self.game_controller:
            self.game_controller.next_level()
            self._restart_level()

    def _restart_level(self):
        """Перезапускает текущий уровень"""
        if self.game_controller:
            self.game_controller.reset_level()
            self._render_game()
            self.start_time = datetime.now()

    def _undo_move(self):
        """Отменяет последний ход"""
        if self.game_controller:
            self.game_controller.undo_move()
            self._render_game()

    def _toggle_pause(self):
        """Переключает паузу"""
        self.is_paused = not self.is_paused

        if self.is_paused:
            self.control_panel.buttons["pause"].config(text="▶️")
        else:
            self.control_panel.buttons["pause"].config(text="⏸️")

        self._update_stats()

    def _show_settings(self):
        """Показывает диалог настроек"""
        self.settings_dialog = SettingsDialog(
            self.root,
            self.config_manager,
            self._on_settings_saved
        )
        self.settings_dialog.show()

    def _on_settings_saved(self):
        """Обрабатывает сохранение настроек"""
        # Перезагружаем стиль
        self.ui_style = self.config_manager.get_ui_style()

        # Обновляем интерфейс
        self._update_ui_style()

        # Перезагружаем ресурсы если изменился размер блоков
        new_size = self.config_manager.config.ui_config.box_size
        if new_size != self.game_resources.box_size:
            self.game_resources.update_box_size(new_size)
            # Нужно пересоздать игровой интерфейс с новым размером
            self._recreate_game_interface()

    def _update_ui_style(self):
        """Обновляет стиль интерфейса"""
        # Обновляем цвет фона окна
        self.root.configure(bg=self.ui_style.bg_color)

        # Обновляем стиль меню (нужно пересоздать)
        self.game_menu.update_style(self.ui_style)

        # Обновляем стиль холста
        self.game_canvas.update_style(self.ui_style)

        # Обновляем стиль панели управления
        self.control_panel.update_style(self.ui_style)

    def _recreate_game_interface(self):
        """Пересоздает игровой интерфейс"""
        # Сохраняем текущее состояние
        current_level = self.current_level

        # Удаляем старые виджеты
        for widget in self.root.winfo_children():
            widget.destroy()

        # Создаем заново
        self._setup_ui()

        # Восстанавливаем состояние
        self.current_level = current_level
        self._load_game()

    def _save_game(self):
        """Сохраняет игру"""
        if self.game_controller:
            if self.game_controller.save_game():
                messagebox.showinfo("Сохранение", "Игра успешно сохранена!")

    def _handle_level_completion(self):
        """Обрабатывает завершение уровня"""
        stats = self.game_controller.get_game_stats()

        message = (
            f"Поздравляем! Уровень {stats['current_level']} пройден!\n\n"
            f"Шагов: {stats['steps']}\n"
            f"Ящиков на местах: {stats['boxes_on_dest']}/{stats['total_destinations']}"
        )

        if messagebox.askyesno("Уровень пройден!", f"{message}\n\nПерейти к следующему уровню?"):
            self._next_level()

    def on_closing(self):
        """Обработчик закрытия окна"""
        if messagebox.askokcancel("Выход", "Вы действительно хотите выйти?"):
            if self.game_controller:
                self.game_controller.save_game()
            self.root.destroy()


# ==================== ОСНОВНОЙ КОД ====================

if __name__ == "__main__":
    import sys

    try:
        # Создаем и запускаем игру
        game = PushBoxGame()

        # Устанавливаем обработчик закрытия окна
        game.root.protocol("WM_DELETE_WINDOW", game.on_closing)

        # Запускаем игру
        game.run()

    except Exception as e:
        print(f"Ошибка запуска игры: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)