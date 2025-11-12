import pygame
import sys
import os
from typing import Optional

# ---------  импортируем ваш оригинальный код ------------
# (если у вас всё в одном файле – просто оставьте тот же импорт)
from interactor3000 import (DynamicMapGenerator, Engine,  # замените на нужное
                                Agent, Enemy, Coat, Gollum, MountainDoom)

# ----------------- настройки окна -------------------------
CELL = 40                # размер клетки (px)
MARGIN = 2               # рамка между клетками
COLS = ROWS = 13         # размер карты
WIDTH  = ROWS * (CELL + MARGIN) + MARGIN
HEIGHT = COLS * (CELL + MARGIN) + MARGIN + 120  + 60  # место под консоль + кнопки

FPS = 1
# ---------------------------------------------------------

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("A* / Backtracking visual")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 14)

# ----------------- цвета ---------------------------------
BG         = (30, 30, 40)
GRID       = (50, 50, 60)
FREE       = (90, 90, 100)
PERCEPT_A  = (50, 180, 50)
PERCEPT_E  = (220, 50, 50)
PERCEPT_X  = (220, 220, 50)   # пересечение зон
TEXT       = (220, 220, 220)

TOKEN_COLORS = {
    "F": (50,  220, 50),   # агент
    "O": (220, 50,  50),   # орки
    "U": (200, 70,  70),
    "N": (180, 40, 180),
    "W": (160, 100, 60),
    "C": (255, 140, 0),    # плащ
    "G": (30,  150, 255),  # Голлум
    "M": (180, 40, 180),   # гора
}

# ---------------------------------------------------------

def draw_cell(x: int, y: int, color, symbol: Optional[str] = None):
    """Рисует клетку (x,y) цветом color, при необходимости – символ."""
    rect = pygame.Rect(
        MARGIN + y * (CELL + MARGIN),
        MARGIN + x * (CELL + MARGIN),
        CELL, CELL
    )
    pygame.draw.rect(screen, color, rect, border_radius=4)
    if symbol:
        txt = font.render(symbol, True, (0, 0, 0))
        screen.blit(txt, txt.get_rect(center=rect.center))

def render_env(env, engine):
    """Полностью перерисовывает карту."""
    screen.fill(BG)

    # зоны восприятия
    agent_percepts = set()
    enemy_percepts = set()
    for e in env.entities:
        if isinstance(e, Enemy):
            for p in e.perceptions:
                enemy_percepts.add(tuple(p))
        elif isinstance(e, Agent):
            for p in e.perceptions:
                agent_percepts.add(tuple(p))
    intersect = agent_percepts & enemy_percepts

    # рисуем фон
    for i in range(ROWS):
        for j in range(COLS):
            draw_cell(i, j, FREE)

    # рисуем зоны
    for (i, j) in intersect:
        draw_cell(i, j, PERCEPT_X)
    for (i, j) in enemy_percepts - intersect:
        draw_cell(i, j, PERCEPT_E)
    for (i, j) in agent_percepts - intersect:
        draw_cell(i, j, PERCEPT_A)

    # сущности
    for e in env.entities:
        i, j = e.position
        color = TOKEN_COLORS.get(e.token, (255, 255, 255))
        draw_cell(i, j, color, e.token)

    # панель статуса
    y0 = HEIGHT - 120
    pygame.draw.rect(screen, (40, 40, 50), (0, y0, WIDTH, 120))
    texts = [
        f"Steps: {engine.stats.steps_taken}",
        f"Ring: {'ON' if engine.agent.ring else 'off'}",
        f"Coat: {'yes' if engine.coat_found else 'no'}",
        f"Gollum: {'found' if engine.gollum_found else 'unknown'}",
        f"Victory: {engine.stats.victory}",
    ]
    for k, t in enumerate(texts):
        screen.blit(font.render(t, True, TEXT), (10, y0 + 5 + k * 16))

def make_manual_move(engine, x, y):
    """Отправляет команду 'm x y' в движок."""
    engine.process_command(f"m {x} {y}")

def main():
    # --------- инициализация движка (берём ваши настройки) ----------
    generator = DynamicMapGenerator()
    env = generator.generate_map()
    engine = Engine(env, use_solver=True, perception_variant=1,
                    use_process_solver=True, silent_mode=False,
                    enable_detailed_logging=False,
                    enable_failure_logging=False,
                    solver_script="python3 Astar/fullAstaralg.py")  # поменяйте при необходимости
    engine.update()

    auto_mode = True          # стартуем в авто-режиме
    running = True
    while running:
        clock.tick(FPS)
        # ---------- события ----------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_RETURN and auto_mode:
                    # один шаг auto
                    engine.process_command("")
                elif event.key == pygame.K_SPACE:
                    # полностью автоматическое прохождение
                    engine.auto_solve()
                elif event.key == pygame.K_r:
                    engine.process_command("ring")
                elif event.key == pygame.K_m:
                    auto_mode = not auto_mode

            elif event.type == pygame.MOUSEBUTTONDOWN and not auto_mode:
                mx, my = event.pos
                if my > HEIGHT - 120:
                    continue
                j = mx // (CELL + MARGIN)
                i = my // (CELL + MARGIN)
                if 0 <= i < ROWS and 0 <= j < COLS:
                    make_manual_move(engine, i, j)

        # ---------- отрисовка ----------
        render_env(engine.env, engine)
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()