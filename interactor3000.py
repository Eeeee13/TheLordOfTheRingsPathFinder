# ==========================================================================
# ║                      ❤︎ Doshq's Interactor3000 ❤︎                       ║
# ==========================================================================

import os
import time
import random
import datetime
import statistics
import subprocess
from abc import ABC, abstractmethod
from collections import Counter
from typing import Optional, Union

import pygame

CELL_SIZE = 32   # размер клетки в пикселях

# pygame цвета
COLORS = {
    "empty": (200, 200, 200),
    "agent": (0, 180, 0),
    "enemy": (200, 0, 0),
    "coat": (255, 140, 0),
    "gollum": (40, 80, 200),
    "mountain": (160, 0, 160),
    "unknown": (120, 120, 120),
    "agent_vision": (120, 255, 120),
    "enemy_vision": (255, 120, 120),
    "intersection": (255, 255, 0),
}




class Command(ABC):
    @abstractmethod
    def execute(self, engine):
        pass


class MoveCommand(Command):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def execute(self, engine):
        engine.move_agent([self.x, self.y])


class AutoCommand(Command):
    def execute(self, engine):
        if hasattr(engine, 'use_solver') and engine.use_solver:
            engine.auto_solve()
            exit(0)
        else:
            print("Solver not enabled.")


class TestCommand(Command):
    def __init__(self, num_tests):
        self.num_tests = num_tests

    def execute(self, engine):
        print(f"Starting {self.num_tests} automated tests...")
        test_runner = TestRunner(num_tests=self.num_tests,
                                 perception_variant=engine.perception_variant,
                                 solver_script=default_solver)
        test_runner.run_tests()


class CompareCommand(Command):
    def __init__(self, num_tests):
        self.num_tests = num_tests

    def execute(self, engine):
        print(f"Starting comparative analysis with {self.num_tests} tests per algorithm...")
        comparative_runner = ComparativeTestRunner(num_tests=self.num_tests,
                                                   astar_solver_script=astar_script,
                                                   backtracking_solver_script=backtracking_script)
        comparative_runner.run_comparative_analysis()


class RingCommand(Command):
    def execute(self, engine):
        engine.activate_ring()


class UnringCommand(Command):
    def execute(self, engine):
        engine.deactivate_ring()


class PathCommand(Command):
    def execute(self, engine):
        engine.calculate_path()


class ExitCommand(Command):
    def execute(self, engine):
        exit()


class CommandFactory:
    @staticmethod
    def create_command(command_str):
        parts = command_str.strip().split()
        if not parts:
            return AutoCommand()  # Default command

        cmd_type = parts[0].lower()

        if cmd_type == "m" and len(parts) == 3:
            try:
                x, y = int(parts[1]), int(parts[2])
                return MoveCommand(x, y)
            except ValueError:
                return None
        elif cmd_type == "ring":
            return RingCommand()
        elif cmd_type == "unring":
            return UnringCommand()
        elif cmd_type == "path":
            return PathCommand()
        elif cmd_type == "auto":
            return AutoCommand()
        elif cmd_type == "test" and len(parts) == 2:
            num_tests = 100
            try:
                num_tests = int(parts[1])
                return TestCommand(num_tests)
            except ValueError:
                print("Using default number of tests (100).")
                return TestCommand(num_tests)
        elif cmd_type == "compare" and len(parts) == 2:
            num_tests = 100
            try:
                num_tests = int(parts[1])
                return CompareCommand(num_tests)
            except ValueError:
                print("Using default number of tests (100).")
                return CompareCommand(num_tests)
        elif cmd_type == "exit":
            return ExitCommand()

        return None


class DynamicMapGenerator:
    def __init__(self):
        self.size = 13

    @staticmethod
    def _create_temp_enemy(enemy_type, position):
        """Creates a temporary enemy for position checking"""
        if enemy_type == "O":
            return OrcPatrol(position)
        elif enemy_type == "U":
            return UrukHai(position)
        elif enemy_type == "N":
            return Nazgul(position)
        elif enemy_type == "W":
            return Watchtower(position)
        return None

    def generate_map(self, agent_perception_radius=1):
        """!! The right order entity spawning: from the biggest perception to smallest !!"""

        env = Environment()

        frodo = Agent("F", agent_perception_radius, [0, 0])
        env.add_entity(frodo)

        entities_to_place = [
            ("W", 1),  # Watchtower
            ("U", 1),  # Uruk-hai
        ]

        # Optional entity (Nazgul 0-1)
        if random.random() > 0.5:  # Nazgûl 0-1
            entities_to_place.append(("N", 1))

        # Orc Patrol 1-2
        orc_count = random.randint(1, 2)
        entities_to_place.append(("O", orc_count))

        # Friendly entities
        entities_to_place.extend([("C", 1),  # Coat
                                  ("G", 1),  # Gollum
                                  ("M", 1)])  # Mountain Doom

        for entity_type, count in entities_to_place:
            for _ in range(count):
                if entity_type in ["W", "U", "N", "O"]:  # Enemies
                    position = self._find_safe_enemy_position(env, entity_type)
                else:  # Friendly entities
                    position = self._find_safe_position(env)

                if entity_type == "C":
                    entity = Coat(position)
                elif entity_type == "G":
                    entity = Gollum(position)
                elif entity_type == "M":
                    entity = MountainDoom(position)
                elif entity_type == "W":
                    entity = Watchtower(position)
                elif entity_type == "U":
                    entity = UrukHai(position)
                elif entity_type == "N":
                    entity = Nazgul(position)
                elif entity_type == "O":
                    entity = OrcPatrol(position)
                else:
                    print(f"Unknown entity type: {entity_type}")
                    continue

                env.add_entity(entity)

        return env

    def _is_position_safe_for_frodo(self, enemy_position, enemy_type):
        # Create a temporary enemy for position checking
        temp_enemy = self._create_temp_enemy(enemy_type, enemy_position)
        if not temp_enemy:
            return True

        frodo_position = [0, 0]
        return frodo_position not in temp_enemy.perceptions and frodo_position != enemy_position

    def _find_safe_enemy_position(self, env, enemy_type, max_attempts=200):
        for attempt in range(max_attempts):
            position = self._find_safe_position(env, max_attempts=50)

            if position and self._is_position_safe_for_frodo(position, enemy_type):
                return position

        for x in range(self.size):
            for y in range(self.size):
                position = [x, y]
                if (env.map[x][y] is None and
                        [x, y] != [0, 0] and
                        self._is_position_safe_for_frodo(position, enemy_type)):
                    return position

        # In case of failure, return a random safe position
        return self._find_safe_position(env)

    def _find_safe_position(self, env, max_attempts=100):
        """Finding a safe position for the entity"""
        for _ in range(max_attempts):
            x, y = random.randint(0, self.size - 1), random.randint(0, self.size - 1)

            if env.map[x][y] is None and [x, y] != [0, 0]:
                return [x, y]

        # If it does not find the optimal position, return a random free position
        for x in range(self.size):
            for y in range(self.size):
                if env.map[x][y] is None and [x, y] != [0, 0]:
                    return [x, y]

        return [random.randint(0, self.size - 1), random.randint(0, self.size - 1)]


class Entity:
    def __init__(self, token: str, position: list):
        self.token = token
        self.position = position

    @staticmethod
    def gen_perception(center, radius=1, moore=False, ears=False):
        x, y = center
        coords = []
        for i in range(x - radius, x + radius + 1):
            for j in range(y - radius, y + radius + 1):
                if 0 <= i < 13 and 0 <= j < 13 and not (i == x and j == y):
                    if moore or abs(i - x) + abs(j - y) <= radius:
                        coords.append([i, j])
        if ears:
            ears_list = ((x - radius - 1, y - radius - 1), (x - radius - 1, y + radius + 1),
                         (x + radius + 1, y - radius - 1), (x + radius + 1, y + radius + 1))
            for i, j in ears_list:
                if 0 <= i < 13 and 0 <= j < 13 and not (i == x and j == y):
                    coords.append([i, j])
        return coords


class Enemy(Entity):
    def __init__(self, token: str, position: list,
                 base_perceptions: list[list],
                 on_ring_perceptions: list[list],
                 on_coat_perceptions: list[list]):
        super().__init__(token, position)
        self.base_perceptions = base_perceptions
        self.on_ring_perceptions = on_ring_perceptions
        self.on_coat_perceptions = on_coat_perceptions
        self.perceptions = list(base_perceptions)

    def on_ring_effect(self):
        self.perceptions = self.on_ring_perceptions

    def on_coat_effect(self):
        self.perceptions = self.on_coat_perceptions
        self.base_perceptions = self.on_coat_perceptions

    def clear_ring_effect(self):
        self.perceptions = self.base_perceptions


class Agent(Entity):
    def __init__(self, token: str, perception_radius: int, position=None):
        if position is None:
            position = [0, 0]
        if perception_radius not in (1, 2):
            raise ValueError("Agent's move_radius must be 1 or 2")
        super().__init__(token, position)
        self.radius = perception_radius
        self.perceptions = []
        self.ring = False
        self.coat = False
        self.update_perceptions()

    def recalculate_perceptions(self):
        x, y = self.position
        return [[i, j]
                for i in range(x - self.radius, x + self.radius + 1)
                for j in range(y - self.radius, y + self.radius + 1)
                if 0 <= i < 13 and 0 <= j < 13 and not (i == x and j == y)]

    def update_perceptions(self):
        self.perceptions = self.recalculate_perceptions()

    def move(self, new_position: list):
        if abs(self.position[0] - new_position[0]) > 1 or \
                abs(self.position[1] - new_position[1]) > 1:
            raise ValueError("Move exceeds manhattan movement")
        self.position = new_position
        self.update_perceptions()


class Coat(Entity):
    def __init__(self, position: list):
        super().__init__("C", position)


class Gollum(Entity):
    def __init__(self, position: list):
        super().__init__("G", position)


class MountainDoom(Entity):
    def __init__(self, position: list):
        super().__init__("M", position)


class OrcPatrol(Enemy):
    def __init__(self, position):
        base = Entity.gen_perception(position, radius=1, moore=False, ears=False)
        ring_and_coat = Entity.gen_perception(position, radius=0, moore=False, ears=False)
        super().__init__("O", position, base, on_ring_perceptions=ring_and_coat, on_coat_perceptions=ring_and_coat)


class UrukHai(Enemy):
    def __init__(self, position):
        base = Entity.gen_perception(position, radius=2, moore=False, ears=False)
        ring_and_coat = Entity.gen_perception(position, radius=1, moore=False, ears=False)
        super().__init__("U", position, base, on_ring_perceptions=ring_and_coat, on_coat_perceptions=ring_and_coat)


class Nazgul(Enemy):
    def __init__(self, position):
        base = Entity.gen_perception(position, radius=1, moore=True, ears=True)
        ring = Entity.gen_perception(position, radius=2, moore=True, ears=False)
        coat = Entity.gen_perception(position, radius=1, moore=True, ears=False)
        super().__init__("N", position, base, on_ring_perceptions=ring, on_coat_perceptions=coat)


class Watchtower(Enemy):
    def __init__(self, position):
        base = Entity.gen_perception(position, radius=2, moore=True, ears=False)
        ring = Entity.gen_perception(position, radius=2, moore=True, ears=True)
        super().__init__("W", position, base, on_ring_perceptions=ring, on_coat_perceptions=base)


class Environment:
    def __init__(self):
        self.size = 13
        self.map: list[list[Optional[Union["Entity", str]]]] = [
            [None for _ in range(self.size)] for _ in range(self.size)
        ]
        self.entities = []

    def add_entity(self, entity: Entity):
        x, y = entity.position
        if not (0 <= x < self.size and 0 <= y < self.size):
            raise ValueError("Entity position out of bounds")
        self.map[x][y] = entity
        if isinstance(entity, Enemy) or isinstance(entity, Agent):
            for p in entity.perceptions:
                if 0 <= p[0] < self.size and 0 <= p[1] < self.size:
                    self.map[p[0]][p[1]] = "p"
        self.entities.append(entity)

    def get_surrounding_dangers(self, position, perceptions_radius=1):
        """Returns a list of danger tokens in the surrounding area"""
        x, y = position
        surrounding = []

        if perceptions_radius == 1:
            x_directions = [-1, 0, 1]
            y_directions = [-1, 0, 1]
        elif perceptions_radius == 2:
            x_directions = [-2, -1, 0, 1, 2]
            y_directions = [-2, -1, 0, 1, 2]
        else:
            raise ValueError("perceptions_radius must be 1 or 2")

        for dx in x_directions:
            for dy in y_directions:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    entity = self.map[nx][ny]
                    if entity:
                        if isinstance(entity, Enemy):
                            surrounding.append((nx, ny, entity.token))
                        elif isinstance(entity, (Gollum, MountainDoom, Coat)):
                            surrounding.append((nx, ny, entity.token))
                    else:
                        for enemy in [e for e in self.entities if isinstance(e, Enemy)]:
                            if [nx, ny] in enemy.perceptions:
                                surrounding.append((nx, ny, "P"))
                                break

        return surrounding

    def clear_map(self):
        self.map = [[None for _ in range(self.size)] for _ in range(self.size)]

    def update_map(self):
        self.clear_map()

        for entity in self.entities:
            x, y = entity.position
            self.map[x][y] = entity

    def visualize_map(self, show_hidden=False):
        RESET = "\033[0m"
        RED = "\033[41m"
        GREEN = "\033[42m"
        BLUE = "\033[44m"
        YELLOW = "\033[43m"
        PURPLE = "\033[45m"
        ORANGE = "\033[48;5;208m"

        enemy_perception_zones = set()
        agent_perception_zones = set()

        for e in self.entities:
            if isinstance(e, Enemy):
                for p in e.perceptions:
                    if 0 <= p[0] < self.size and 0 <= p[1] < self.size:
                        enemy_perception_zones.add(tuple(p))
            elif isinstance(e, Agent):
                for p in e.perceptions:
                    if 0 <= p[0] < self.size and 0 <= p[1] < self.size:
                        agent_perception_zones.add(tuple(p))

        intersection_zones = agent_perception_zones.intersection(enemy_perception_zones)

        print("\n" + "═" * (self.size * 3 + 2))
        for i in range(self.size):
            print("║", end="")
            for j in range(self.size):
                entity = self.map[i][j]
                cell_color = ""
                symbol = "   "

                if entity:
                    if isinstance(entity, Agent):
                        cell_color = GREEN
                        symbol = f" {entity.token} "
                    elif isinstance(entity, Enemy):
                        cell_color = RED
                        symbol = f" {entity.token} "
                    elif isinstance(entity, Coat):
                        cell_color = ORANGE
                        symbol = f" {entity.token} "
                    elif isinstance(entity, Gollum):
                        cell_color = BLUE
                        symbol = f" {entity.token} "
                    elif isinstance(entity, MountainDoom):
                        if show_hidden:
                            cell_color = PURPLE
                            symbol = f" {entity.token} "
                        else:
                            symbol = " • "
                elif (i, j) in intersection_zones:
                    cell_color = YELLOW
                    symbol = " • "
                elif (i, j) in enemy_perception_zones:
                    cell_color = RED
                    symbol = " • "
                elif (i, j) in agent_perception_zones:
                    cell_color = GREEN
                    symbol = " • "
                else:
                    symbol = " • "

                print(f"{cell_color}{symbol}{RESET}", end="")
            print("║")
        print("═" * (self.size * 3 + 2))



    def draw_pygame(self, surface, cell_size=32, show_hidden=False):
        enemy_perception_zones = set()
        agent_perception_zones = set()

        # --- Сбор восприятия ---
        for e in self.entities:
            if isinstance(e, Enemy):
                for p in e.perceptions:
                    if 0 <= p[0] < self.size and 0 <= p[1] < self.size:
                        enemy_perception_zones.add(tuple(p))
            elif isinstance(e, Agent):
                for p in e.perceptions:
                    if 0 <= p[0] < self.size and 0 <= p[1] < self.size:
                        agent_perception_zones.add(tuple(p))

        intersection_zones = agent_perception_zones & enemy_perception_zones

        # --- Цвета ---
        COLOR_BG         = (220, 220, 220)
        COLOR_AGENT      = (50, 200, 50)
        COLOR_ENEMY      = (200, 50, 50)
        COLOR_COAT       = (255, 140, 0)
        COLOR_GOLLUM     = (50, 100, 255)
        COLOR_MOUNTAIN   = (180, 0, 180)
        COLOR_INTERSECT  = (255, 255, 0)
        COLOR_A_PERC     = (120, 255, 120)
        COLOR_E_PERC     = (255, 120, 120)

        # --- Рисование ---
        for i in range(self.size):
            for j in range(self.size):

                rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
                entity = self.map[i][j]

                # Определяем цвет клетки
                if entity:
                    if isinstance(entity, Agent):
                        color = COLOR_AGENT
                    elif isinstance(entity, Enemy):
                        color = COLOR_ENEMY
                    elif isinstance(entity, Coat):
                        color = COLOR_COAT
                    elif isinstance(entity, Gollum):
                        color = COLOR_GOLLUM
                    elif isinstance(entity, MountainDoom):
                        color = COLOR_MOUNTAIN if show_hidden else COLOR_BG
                    else:
                        color = COLOR_BG
                else:
                    if (i, j) in intersection_zones:
                        color = COLOR_INTERSECT
                    elif (i, j) in enemy_perception_zones:
                        color = COLOR_E_PERC
                    elif (i, j) in agent_perception_zones:
                        color = COLOR_A_PERC
                    else:
                        color = COLOR_BG

                pygame.draw.rect(surface, color, rect)

                # Рисуем символ сущности (token)
                if entity:
                    if not pygame.font.get_init():
                        pygame.font.init()

                    font = pygame.font.SysFont("Arial", int(cell_size * 0.6))
                    text = font.render(entity.token, True, (0, 0, 0))
                    text_rect = text.get_rect(center=rect.center)
                    surface.blit(text, text_rect)

                # Сетка
                pygame.draw.rect(surface, (0, 0, 0), rect, 1)


class GameStatistics:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.steps_taken = 0
        self.moves = []
        self.full_path = 0
        self.path_to_gollum = 0
        self.path_to_mount_doom = 0
        self.victory = False
        self.execution_time = 0

    def start(self):
        self.start_time = time.time()

    def finish(self, victory=False):
        self.end_time = time.time()
        self.victory = victory
        self.execution_time = self.end_time - self.start_time

    def increment_steps(self):
        self.steps_taken += 1

    def set_path_lengths(self, to_gollum, to_mount_doom):
        self.path_to_gollum = to_gollum
        self.path_to_mount_doom = to_mount_doom

    def set_full_path(self, full_path):
        self.full_path = full_path

    def add_move(self, move):
        self.moves.append(move)

    def print_statistics(self):
        print(f"\n{'-' * 5} GAME STATISTICS: {'-' * 5}")
        print(f"\tResult: {'VICTORY' if self.victory else 'DEFEAT'}")
        print(f"\tExecution Time: {self.execution_time:.2f} seconds")
        print(f"\tMoves Taken: {len(self.moves)}")

        if self.victory:
            print(f"\tTotal Path Length: {self.full_path}")


class ProcessInteractor:
    """Interactor that works with solver as a process via stdin/stdout"""

    def __init__(self, engine, solver_script='python3 astar.py'):
        self.mount_pos_sent = False
        self.engine = engine
        self.process = None
        self.solver_script = solver_script
        self.start_solver_process()

    def start_solver_process(self):
        """Start the solver as a separate process"""
        try:
            self.process = subprocess.Popen(
                self.solver_script.split(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            # Test if process started successfully
            if self.process.poll() is not None:
                raise Exception("Solver process failed to start")
            print(f"Solver process started successfully for {self.solver_script}")
        except Exception as e:
            print(f"❌ Failed to start solver process: {e}")
            self.process = None

    def send_state_to_solver(self):
        """Send current game state to solver via stdin"""
        if not self.process or self.process.poll() is not None:
            return False

        try:
            dangers = self.engine.env.get_surrounding_dangers(self.engine.agent.position,
                                                              self.engine.perception_variant)
            state_data = f"{len(dangers)}\n"

            # Send dangers
            for danger in dangers:
                state_data += f"{danger[0]} {danger[1]} {danger[2]}\n"

            # Send mount position
            if self.engine.mount_doom_revealed and not self.mount_pos_sent:
                mount_x, mount_y = self.engine.mount_doom_pos
                state_data += f"{mount_x} {mount_y}\n"
                self.mount_pos_sent = True

            # Send to solver
            if self.process.poll() is not None:
                print("❌ Solver process terminated")
                return False

            self.process.stdin.write(state_data)
            self.process.stdin.flush()
            return True

        except Exception as e:
            print(f"❌ Error sending state to solver: {e}")
            return False

    def get_command_from_solver(self):
        """Get command from solver via stdout"""
        try:
            command = self.process.stdout.readline().strip()
            return command if command else None
        except Exception as e:
            print(f"❌ Error reading from solver: {e}")
            return None

    def send_init_state_to_solver(self):
        try:
            # goal = self.engine._get_current_goal()

            state_data = f"{self.engine.agent.radius}\n"
            state_data += f"{self.engine.gollum.position[0]} {self.engine.gollum.position[1]}\n"

            # Send to solver
            self.process.stdin.write(state_data)
            self.process.stdin.flush()
            return True

        except Exception as e:
            print(f"❌ Error sending state to solver: {e}")
            return False

    def execute_astar(self):
        """Execute one step of A* via process communication"""

        if self.send_state_to_solver():
            return self.get_command_from_solver()
        return None

    def close(self):
        """Close the solver process"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except:         # noqa
                try:
                    self.process.kill()
                    self.process.wait()
                except:     # noqa
                    pass


class Engine:
    def __init__(self,
                 environment: Environment,
                 use_solver=False,
                 perception_variant=1,
                 use_process_solver=True,
                 silent_mode=False,
                 enable_detailed_logging=False,
                 enable_failure_logging=True,
                 solver_script="python3 astar.py",):

        self.env = environment
        self.perception_variant = perception_variant
        self.agent = next((e for e in environment.entities if isinstance(e, Agent)), None)
        self.enemies = [e for e in environment.entities if isinstance(e, Enemy)]
        self.gollum_found = False
        self.gollum = self._find_gollum()
        self.coat_found = False
        self.mount_doom_revealed = False
        self.mount_doom = self._find_mount_doom()
        self.stats = GameStatistics()
        self.stats.start()
        self.known_dangers = set()
        self.explored_cells = {tuple(self.agent.position)}
        self.visited_positions = {tuple(self.agent.position)}
        self.game_over = False
        self.silent_mode = silent_mode
        self.solver_script = solver_script

        self.logger = GameLogger()
        self.enable_logging = enable_failure_logging
        self.logger.enabled = enable_detailed_logging
        if enable_failure_logging:
            log_file = self.logger.start_new_log()
            if not silent_mode:
                print(f"Logging enabled: {log_file}")

        CELL = 40                # размер клетки (px)
        MARGIN = 2               # рамка между клетками
        COLS = ROWS = 13         # размер карты
        WIDTH  = ROWS * (CELL + MARGIN) + MARGIN
        HEIGHT = COLS * (CELL + MARGIN) + MARGIN + 120  + 60  # место под консоль + кнопки

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))

        self.use_solver = use_solver
        self.use_process_solver = use_process_solver
        self.process_interactor = None
        self.update()

        

        if use_solver:
            if use_process_solver:
                # Use process-based interactor
                self.process_interactor = ProcessInteractor(self, self.solver_script)

    def update_known_dangers(self):
        """Updates the set of known dangers based on current agent position"""
        dangers = self.env.get_surrounding_dangers(self.agent.position, self.perception_variant)
        for x, y, danger_type in dangers:
            if danger_type in ['O', 'U', 'N', 'W', 'P']:
                self.known_dangers.add((x, y))

    def update_explored_cells(self):
        """Updates the set of explored cells"""
        x, y = self.agent.position
        self.explored_cells.add((x, y))
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.env.size and 0 <= ny < self.env.size:
                    self.explored_cells.add((nx, ny))

    def update(self):
        self.env.update_map()
        # self.env.map = [[None for _ in range(self.env.size)] for _ in range(self.env.size)]
        # for e in self.env.entities:
        #     x, y = e.position
        #     self.env.map[x][y] = e

        self.update_known_dangers()
        self.update_explored_cells()
        self.visited_positions.add(tuple(self.agent.position))

        if not self.silent_mode:
            # Обработка событий (ВАЖНО!)
            pygame.time.delay(100)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return

            # Отрисовка
            self.screen.fill((0, 0, 0))  # Очистка экрана
            self.env.draw_pygame(self.screen, show_hidden=self.gollum_found)
            pygame.display.update()

            self._print_surrounding_info()

    def _print_surrounding_info(self):
        dangers = self.env.get_surrounding_dangers(self.agent.position, self.perception_variant)
        print(f"{len(dangers)}")
        for x, y, danger_type in dangers:
            print(f"{x} {y} {danger_type}")

    def _detect_collisions(self, x, y):
        for enemy in self.enemies:
            if [x, y] == enemy.position or [x, y] in enemy.perceptions:
                if self.enable_logging:
                    self.logger.log_collision([x, y], enemy.token)
                    self.logger.log_map(self.env, show_hidden=self.gollum_found)
                    self.logger.log_game_state(self)
                print(f"Collision with enemy in ({x}, {y})!")
                self._finish_game(False)
                return True

        # Meet with Coat
        coat = next((e for e in self.env.entities if isinstance(e, Coat)), None)
        if coat and [x, y] == coat.position and not self.coat_found:
            self.coat_found = True
            self._put_coat()
            if self.enable_logging:
                self.logger.log("Coat found and equipped!", include_timestamp=True)
            return False

        return False

    def _finish_game(self, victory):
        self.game_over = True
        if self.process_interactor:
            self.process_interactor.close()
        self.stats.finish(victory=victory)

        if self.enable_logging:
            if victory:
                self.logger.log_victory()
            self.logger.save_statistics(self.stats)
            self.logger.log(f"=== GAME LOG COMPLETED ===", include_timestamp=True)

        if not self.silent_mode:
            self.stats.print_statistics()

    def move_agent(self, new_pos):
        x, y = new_pos

        if not (0 <= x < self.env.size and 0 <= y < self.env.size):
            if self.enable_logging:
                self.logger.log(
                    f"Illegal move: out of bound in old: [{self.agent.position[0]}, {self.agent.position[1]}], "
                    f"new: [{x}, {y}]!", include_timestamp=True)
            print("❌ Out of bounds")
            return

        if abs(self.agent.position[0] - x) > self.agent.radius or \
                abs(self.agent.position[1] - y) > self.agent.radius:
            if self.enable_logging:
                self.logger.log(
                    f"Illegal move: exceeds allowed agent radius in old: [{self.agent.position[0]}, {self.agent.position[1]}], "
                    f"new: [{x}, {y}]!", include_timestamp=True)
            print("❌ Move exceeds allowed radius")
            return

        if self._detect_collisions(x, y):
            return

        self.agent.move([x, y])
        self.stats.increment_steps()
        self.update()

    def activate_ring(self):
        if not self.agent.ring:
            self.agent.ring = True
            for e in self.enemies:
                e.on_ring_effect()
            if self.enable_logging:
                self.logger.log(f"💍 Ring activated in [{self.agent.position[0]}, {self.agent.position[1]}]",
                                include_timestamp=True)
            if not self.silent_mode:
                print("💍 Ring activated!")
            self.update()

    def deactivate_ring(self):
        if self.agent.ring:
            self.agent.ring = False
            for e in self.enemies:
                e.clear_ring_effect()
            if self.enable_logging:
                self.logger.log(f"💍 Ring deactivated in [{self.agent.position[0]}, {self.agent.position[1]}]",
                                include_timestamp=True)
            if not self.silent_mode:
                print("💍 Ring deactivated")
            self.update()

    def _put_coat(self):
        for e in self.enemies:
            e.on_coat_effect()

    def _get_current_goal(self):
        """Returns the current goal based on the solver's state"""
        if not self.gollum_found:
            if self.gollum:
                return self.gollum.position
        else:
            if self.mount_doom:
                return self.mount_doom.position
        return None

    def auto_solve(self):
        if not self.use_solver:
            print("Solver not enabled!")
            return

        max_steps = 1000
        step_count = 0
        pygame.init()

        self.stats.start()

        self.process_interactor.send_init_state_to_solver()

        while step_count < max_steps and not self.game_over:
            command : str = self.process_interactor.execute_astar()
            step_count += 1

            if not command:
                continue

            if self.enable_logging:
                self.logger.log(f"Solver command: {command}", include_timestamp=True)
            # print("Command:", command)

            if command.startswith('e'):
                if command == 'e -1':
                    if self.enable_logging:
                        self.logger.log("❌ Solver cannot find solution", include_timestamp=True)
                        self.logger.log_map(self.env, show_hidden=self.gollum_found)
                    print("❌ Solver cannot find solution")
                    self._finish_game(False)
                else:
                    try:
                        steps = int(command.split()[1])
                        self.stats.set_full_path(steps)
                        self._finish_game(True)
                    except:
                        self._finish_game(False)

                if self.enable_logging:
                    self.logger.log(f"e {command.split()[1]}")
                if not self.silent_mode:
                    print(f"e {command.split()[1]}")
                break

            elif command.startswith('m'):
                self.stats.add_move(command)
                parts = command.split()
                x, y = int(parts[1]), int(parts[2])
                self.move_agent([x, y])
                self._check_goal_achievement()

            elif command == 'r':
                self.activate_ring()

            elif command == 'rr':
                self.deactivate_ring()

            else:
                if self.enable_logging:
                    self.logger.log(f"❌ Unknown command from solver: {command}", include_timestamp=True)
                    self.logger.log_map(self.env, show_hidden=self.gollum_found)
                    self.logger.log_game_state(self)

                if not self.silent_mode:
                    print(f"Unknown command received from solver: {command}")

                break   # DEBUGGING: COMMENT BREAK FOR ALLOWING ANY OUTPUTS

        if step_count >= max_steps and not self.game_over:
            if self.enable_logging:
                self.logger.log("❌ Auto-solve stopped: maximum steps reached", include_timestamp=True)
            if not self.silent_mode:
                print("❌ Auto-solve stopped: maximum steps reached")
            self._finish_game(False)

    def _find_gollum(self):
        gollum = next((e for e in self.env.entities if isinstance(e, Gollum)), None)
        return gollum

    def _find_mount_doom(self):
        mount_doom = next((e for e in self.env.entities if isinstance(e, MountainDoom)), None)
        return mount_doom

    def _check_goal_achievement(self):
        """Checks if the goal is achieved and performs necessary actions"""
        if self.gollum and self.agent.position == self.gollum.position and not self.gollum_found:
            self.gollum_found = True
            if self.mount_doom:
                if self.enable_logging:
                    self.logger.log(f"My precious! Mount Doom is at {self.mount_doom.position[0]} {self.mount_doom.position[1]}",
                                    include_timestamp=True)
                if not self.silent_mode:
                    print(f"My precious! Mount Doom is at {self.mount_doom.position[0]} {self.mount_doom.position[1]}")
                self.mount_doom_revealed = True
                self.mount_doom_pos = self.mount_doom.position
            return False

        if self.mount_doom and self.agent.position == self.mount_doom.position and self.gollum_found:
            if self.enable_logging:
                self.logger.log("🎉 The Ring is destroyed! You win!", include_timestamp=True)
            if not self.silent_mode:
                print("🎉 The Ring is destroyed! You win!")
            return True

        return False

    def process_command(self, command_str):
        command = CommandFactory.create_command(command_str)
        if command:
            command.execute(self)
        else:
            print("❌ Unknown command. Available: m x y, ring, unring, path, auto, test N, compare N, exit")


class GameLogger:
    def __init__(self, log_dir="game_logs"):
        self.log_dir = log_dir
        self.current_log_file = None
        self.log_buffer = []
        self.enabled = True

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def start_new_log(self, test_id=None):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if test_id is not None:
            filename = f"test_{test_id}_{timestamp}.log"
        else:
            filename = f"test_unknown_{timestamp}.log"

        self.current_log_file = os.path.join(self.log_dir, filename)
        self.log_buffer = []

        self.log(f"=== GAME LOG STARTED AT {datetime.datetime.now()} ===")
        return self.current_log_file

    def log(self, message, include_timestamp=True):
        if not self.enabled:
            return

        if include_timestamp:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            formatted_message = f"[{timestamp}] {message}"
        else:
            formatted_message = message

        self.log_buffer.append(formatted_message)

        if self.current_log_file:
            with open(self.current_log_file, 'a', encoding='utf-8') as f:
                f.write(formatted_message + '\n')

    def log_command(self, command):
        self.log(f"COMMAND: {command}")

    def log_map(self, environment, show_hidden=False):
        if not self.enabled:
            return

        map_str = self._get_map_string(environment, show_hidden)
        self.log("CURRENT MAP STATE:", include_timestamp=False)
        self.log(map_str, include_timestamp=False)

    def log_collision(self, position, enemy_type):
        self.log(f"❌ SERIOUS FAILURE: Collision with {enemy_type} at position {position}", include_timestamp=True)
        self.log("=== GAME OVER ===", include_timestamp=False)

    def log_victory(self):
        self.log("🎉 VICTORY: The Ring is destroyed!", include_timestamp=True)

    def log_game_state(self, engine):
        self.log("=== GAME STATE SNAPSHOT ===", include_timestamp=False)
        self.log(f"Agent position: {engine.agent.position}", include_timestamp=False)
        self.log(f"Ring active: {engine.agent.ring}", include_timestamp=False)
        self.log(f"Coat found: {engine.coat_found}", include_timestamp=False)
        self.log(f"Gollum found: {engine.gollum_found}", include_timestamp=False)
        self.log(f"Mount Doom revealed: {engine.mount_doom_revealed}", include_timestamp=False)
        self.log(f"Steps taken: {engine.stats.steps_taken}", include_timestamp=False)

    @staticmethod
    def _get_map_string(environment, show_hidden=False):
        env = environment
        enemy_perception_zones = set()
        agent_perception_zones = set()

        for e in env.entities:
            if isinstance(e, Enemy):
                for p in e.perceptions:
                    if 0 <= p[0] < env.size and 0 <= p[1] < env.size:
                        enemy_perception_zones.add(tuple(p))
            elif isinstance(e, Agent):
                for p in e.perceptions:
                    if 0 <= p[0] < env.size and 0 <= p[1] < env.size:
                        agent_perception_zones.add(tuple(p))

        intersection_zones = agent_perception_zones.intersection(enemy_perception_zones)

        lines = ["═" * (env.size * 3 + 2)]
        for i in range(env.size):
            line = "║"
            for j in range(env.size):
                entity = env.map[i][j]
                symbol = " * "

                if entity:
                    if isinstance(entity, Agent):
                        symbol = f" {entity.token} "
                    elif isinstance(entity, Enemy):
                        symbol = f" {entity.token} "
                    elif isinstance(entity, Coat):
                        symbol = f" {entity.token} "
                    elif isinstance(entity, Gollum):
                        symbol = f" {entity.token} "
                    elif isinstance(entity, MountainDoom):
                        if show_hidden:
                            symbol = f" {entity.token} "
                        else:
                            symbol = " • "
                elif (i, j) in intersection_zones:
                    symbol = " X "
                elif (i, j) in enemy_perception_zones:
                    symbol = " E "
                elif (i, j) in agent_perception_zones:
                    symbol = " A "
                else:
                    symbol = " • "

                line += symbol
            line += "║"
            lines.append(line)
        lines.append("═" * (env.size * 3 + 2))
        return "\n".join(lines)

    def save_statistics(self, stats, test_id=None):
        self.log(f"=== GAME STATISTICS [{test_id}] ===", include_timestamp=False)
        self.log(f"Result: {'VICTORY' if stats.victory else 'DEFEAT'}", include_timestamp=False)
        self.log(f"Execution Time: {stats.execution_time:.2f} seconds", include_timestamp=False)
        self.log(f"Moves Taken: {len(stats.moves)}", include_timestamp=False)
        if stats.victory:
            self.log(f"Total Path Length: {stats.full_path}", include_timestamp=False)
        self.log(f"Full path: {stats.full_path}", include_timestamp=False)

    def get_log_content(self):
        return "\n".join(self.log_buffer)


class TestRunner:
    def __init__(self, num_tests=100, perception_variant=1, solver_script="python3 astar.py", log_normal=True, log_failures=True):
        self.num_tests = num_tests
        self.perception_variant = perception_variant
        self.results = []
        self.solver_script = solver_script
        self.log_normal = log_normal
        self.log_failures = log_failures
        self.failure_logs = []

    def run_single_test(self, test_id):
        """Launches a single test and collects statistics"""
        generator = DynamicMapGenerator()
        env = generator.generate_map(self.perception_variant)
        enable_logging = self.log_failures

        print(f"[{test_id}]", end=" ")
        # silent_mode=True for suppressing output
        engine = Engine(env, use_solver=True, perception_variant=self.perception_variant,
                        use_process_solver=True, silent_mode=True, enable_detailed_logging=self.log_normal,
                        enable_failure_logging=self.log_failures, solver_script=self.solver_script)

        if enable_logging:
            engine.logger.start_new_log(test_id=test_id)

        engine.auto_solve()

        if enable_logging and not engine.stats.victory:
            self.failure_logs.append({
                'test_id': test_id,
                'log_content': engine.logger.get_log_content(),
                'stats': engine.stats
            })

        return engine.stats

    def run_tests(self):
        """Launches the specified number of tests"""
        print(f"Running {self.num_tests} tests...")

        for i in range(self.num_tests):
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{self.num_tests} tests...")

            stats = self.run_single_test(i + 1)
            self.results.append(stats)

        self._print_summary()
        self._save_failure_logs()

    def _save_failure_logs(self):
        """Saves failure logs to separate files"""
        if not self.failure_logs:
            return

        failure_dir = "failure_logs"
        if not os.path.exists(failure_dir):
            os.makedirs(failure_dir)

        for failure in self.failure_logs:
            filename = f"failure_test_{failure['test_id']}.log"
            filepath = os.path.join(failure_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"=== FAILURE LOG - Test {failure['test_id']} ===\n")
                f.write(failure['log_content'])

            print(f"Saved failure log: {filepath}")

    def _print_summary(self):
        """Prints a summary of the test results"""
        execution_times = [r.execution_time for r in self.results]
        victories = [r.victory for r in self.results]
        num_wins = sum(victories)
        num_losses = len(victories) - num_wins
        win_rate = (num_wins / len(victories)) * 100

        print(f"\n{'=' * 50}")
        print(f"TEST SUMMARY - {self.num_tests} Tests (Variant {self.perception_variant})")
        print(f"{'=' * 50}")

        print(f"\n--- RESULTS ---")
        print(f"Wins: {num_wins} ({win_rate:.1f}%)")
        print(f"Losses: {num_losses} ({100 - win_rate:.1f}%)")

        if execution_times:
            print(f"\n--- EXECUTION TIME (seconds) ---")
            print(f"Mean: {statistics.mean(execution_times):.3f}")
            print(f"Median: {statistics.median(execution_times):.3f}")
            if len(execution_times) > 1:
                print(f"Std Dev: {statistics.stdev(execution_times):.3f}")
            else:
                print(f"Std Dev: 0.000")

            # Calculate mode for execution time
            try:
                mode_time = statistics.mode(execution_times)
                print(f"Mode: {mode_time:.3f}")
            except statistics.StatisticsError:
                # If there's no unique mode, find the most common value
                time_counts = Counter(execution_times)
                mode_time = time_counts.most_common(1)[0][0]
                print(f"Mode: {mode_time:.3f} (most common)")

            print(f"Min: {min(execution_times):.3f}")
            print(f"Max: {max(execution_times):.3f}")

        if self.log_failures and self.failure_logs:
            print(f"Failure logs saved: {len(self.failure_logs)} in ./failure_logs/")


class ComparativeTestRunner:
    def __init__(self, num_tests=1000,
                 astar_solver_script="python3 astar.py",
                 backtracking_solver_script="python3 backtracking.py",
                 log_normal=False,
                 log_failures=True):

        self.num_tests = num_tests
        self.log_normal = log_normal
        self.log_failures = log_failures

        self.astar_solver_script = astar_solver_script
        self.backtracking_solver_script = backtracking_solver_script

        # Store results for each algorithm and variant
        self.astar_results_v1 = []
        self.astar_results_v2 = []
        self.backtracking_results_v1 = []
        self.backtracking_results_v2 = []

    def run_single_test(self, test_id, algorithm, perception_variant):
        """Run a single test with specified algorithm and perception variant"""
        generator = DynamicMapGenerator()
        env = generator.generate_map(perception_variant)

        solver_script = self.astar_solver_script if algorithm == 'astar' else self.backtracking_solver_script

        print(f"[{test_id}]", end=" ")
        engine = Engine(env, use_solver=True, perception_variant=perception_variant,
                        use_process_solver=True, silent_mode=True,
                        enable_detailed_logging=self.log_normal,
                        enable_failure_logging=self.log_failures,
                        solver_script=solver_script)

        if self.log_failures:
            engine.logger.start_new_log(test_id=f"{algorithm}_v{perception_variant}_{test_id}")

        engine.auto_solve()
        return engine.stats

    def run_comparative_analysis(self):
        """Run comparative analysis between algorithms"""
        print(f"Starting comparative analysis with {self.num_tests} tests per algorithm...")
        print(f"Total tests: {self.num_tests * 2} ({self.num_tests} per algorithm)")

        # Run A* tests (half with variant 1, half with variant 2)
        print("\n--- Running A* Tests ---")
        for i in range(self.num_tests):
            if (i + 1) % 50 == 0:
                print(f"Completed {i + 1}/{self.num_tests} A* tests...")

            # Alternate between variants
            variant = 1 if i < self.num_tests // 2 else 2
            stats = self.run_single_test(i + 1, 'astar', variant)

            if variant == 1:
                self.astar_results_v1.append(stats)
            else:
                self.astar_results_v2.append(stats)

        # Run Backtracking tests (half with variant 1, half with variant 2)
        print("\n--- Running Backtracking Tests ---")
        for i in range(self.num_tests):
            if (i + 1) % 50 == 0:
                print(f"Completed {i + 1}/{self.num_tests} Backtracking tests...")

            # Alternate between variants
            variant = 1 if i < self.num_tests // 2 else 2
            stats = self.run_single_test(i + 1, 'backtracking', variant)

            if variant == 1:
                self.backtracking_results_v1.append(stats)
            else:
                self.backtracking_results_v2.append(stats)

        self._print_comparative_summary()

    @staticmethod
    def _calculate_statistics(results):
        """Calculate statistics for a set of results"""
        if not results:
            return None

        execution_times = [r.execution_time for r in results]
        victories = [r.victory for r in results]
        num_wins = sum(victories)
        num_losses = len(victories) - num_wins
        win_rate = (num_wins / len(victories)) * 100

        stats = {
            'execution_times': execution_times,
            'victories': victories,
            'num_wins': num_wins,
            'num_losses': num_losses,
            'win_rate': win_rate,
            'mean_time': statistics.mean(execution_times) if execution_times else 0,
            'median_time': statistics.median(execution_times) if execution_times else 0,
        }

        # Calculate mode for execution time
        if execution_times:
            try:
                stats['mode_time'] = statistics.mode(execution_times)
            except statistics.StatisticsError:
                time_counts = Counter(execution_times)
                stats['mode_time'] = time_counts.most_common(1)[0][0]
        else:
            stats['mode_time'] = 0

        # Calculate standard deviation
        if len(execution_times) > 1:
            stats['std_dev_time'] = statistics.stdev(execution_times)
        else:
            stats['std_dev_time'] = 0

        return stats

    def _print_comparative_summary(self):
        """Print comprehensive comparative analysis"""
        print(f"\n{'=' * 80}")
        print(f"COMPARATIVE ANALYSIS SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total tests per algorithm: {self.num_tests}")
        print(f"Tests per variant: {self.num_tests // 2}")
        print(f"Total tests executed: {self.num_tests * 2}")

        # Calculate statistics for each group
        astar_v1_stats = self._calculate_statistics(self.astar_results_v1)
        astar_v2_stats = self._calculate_statistics(self.astar_results_v2)
        backtracking_v1_stats = self._calculate_statistics(self.backtracking_results_v1)
        backtracking_v2_stats = self._calculate_statistics(self.backtracking_results_v2)

        # Combined statistics
        astar_all = self.astar_results_v1 + self.astar_results_v2
        backtracking_all = self.backtracking_results_v1 + self.backtracking_results_v2
        astar_all_stats = self._calculate_statistics(astar_all)
        backtracking_all_stats = self._calculate_statistics(backtracking_all)

        # Print comparison tables
        self._print_detailed_comparison(astar_all_stats, backtracking_all_stats, "OVERALL COMPARISON")
        self._print_detailed_comparison(astar_v1_stats, backtracking_v1_stats, "VARIANT 1 COMPARISON")
        self._print_detailed_comparison(astar_v2_stats, backtracking_v2_stats, "VARIANT 2 COMPARISON")

        # Print algorithm performance across variants
        self._print_variant_comparison(astar_v1_stats, astar_v2_stats, "A*")
        self._print_variant_comparison(backtracking_v1_stats, backtracking_v2_stats, "Backtracking")

    @staticmethod
    def _print_detailed_comparison(stats1, stats2, title):
        """Print detailed comparison between two algorithms"""
        print(f"\n{'-' * 60}")
        print(f"{title}")
        print(f"{'-' * 60}")
        print(f"{'Metric':<25} {'A*':<15} {'Backtracking':<15} {'Difference':<15}")
        print(f"{'-' * 70}")

        if not stats1 or not stats2:
            print("Insufficient data for comparison")
            return

        # Wins and Losses
        print(
            f"{'Wins':<25} {stats1['num_wins']:<15} {stats2['num_wins']:<15} {stats1['num_wins'] - stats2['num_wins']:<15}")
        print(
            f"{'Losses':<25} {stats1['num_losses']:<15} {stats2['num_losses']:<15} {stats1['num_losses'] - stats2['num_losses']:<15}")
        print(
            f"{'Win Rate (%)':<25} {stats1['win_rate']:<15.1f} {stats2['win_rate']:<15.1f} {stats1['win_rate'] - stats2['win_rate']:<15.1f}")
        print(f"{'-' * 70}")

        # Execution Time Statistics
        print(
            f"{'Mean Time (s)':<25} {stats1['mean_time']:<15.3f} {stats2['mean_time']:<15.3f} {stats1['mean_time'] - stats2['mean_time']:<15.3f}")
        print(
            f"{'Median Time (s)':<25} {stats1['median_time']:<15.3f} {stats2['median_time']:<15.3f} {stats1['median_time'] - stats2['median_time']:<15.3f}")
        print(
            f"{'Mode Time (s)':<25} {stats1['mode_time']:<15.3f} {stats2['mode_time']:<15.3f} {stats1['mode_time'] - stats2['mode_time']:<15.3f}")
        print(
            f"{'Std Dev Time (s)':<25} {stats1['std_dev_time']:<15.3f} {stats2['std_dev_time']:<15.3f} {stats1['std_dev_time'] - stats2['std_dev_time']:<15.3f}")

    @staticmethod
    def _print_variant_comparison(stats_v1, stats_v2, algorithm_name):
        """Print comparison of algorithm performance across variants"""
        print(f"\n{'-' * 50}")
        print(f"{algorithm_name} - Variant Comparison")
        print(f"{'-' * 50}")
        print(f"{'Metric':<25} {'Variant 1':<15} {'Variant 2':<15} {'Difference':<15}")
        print(f"{'-' * 70}")

        if not stats_v1 or not stats_v2:
            print("Insufficient data for comparison")
            return

        print(
            f"{'Wins':<25} {stats_v1['num_wins']:<15} {stats_v2['num_wins']:<15} {stats_v1['num_wins'] - stats_v2['num_wins']:<15}")
        print(
            f"{'Win Rate (%)':<25} {stats_v1['win_rate']:<15.1f} {stats_v2['win_rate']:<15.1f} {stats_v1['win_rate'] - stats_v2['win_rate']:<15.1f}")
        print(
            f"{'Mean Time (s)':<25} {stats_v1['mean_time']:<15.3f} {stats_v2['mean_time']:<15.3f} {stats_v1['mean_time'] - stats_v2['mean_time']:<15.3f}")


if __name__ == '__main__':
    print("Generating map with random entities...")

    # Initial settings
    perception_variant = 1
    astar_script = "python3 Astar/fullAstaralg.py"
    backtracking_script = "python3 backTraking/fullBackTrackingAlg.py"
    default_solver = astar_script

    generator = DynamicMapGenerator()
    environment = generator.generate_map(perception_variant)

    eng = Engine(environment, use_solver=True,
                    perception_variant=perception_variant,
                    enable_detailed_logging=False,
                    enable_failure_logging=False,
                    solver_script=default_solver)
    eng.update()

    print("\nAvailable commands:")
    print("auto or <enter> — A* solver auto-solve")
    print("test <N> — run N automated tests")
    print("compare <N> — run comparative analysis with N tests per algorithm")
    print("m x y — move to position (x,y)")
    print("ring — put on the Ring")
    print("unring — take off the Ring")
    print("path — calculate path to current goal")
    print("exit — exit game")

    while True:
        try:
            cmd = input("\n>>> ").strip()
            eng.process_command(cmd)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            eng.stats.finish(victory=False)
            break
        except Exception as err:
            print(f"Error: {err}")