import sys
from typing import Set, Tuple, List, Dict, Optional
import heapq


from enum import Enum, auto

class ItemType(Enum):
    F = auto()  # Frodo
    O = auto()  # OrcPatrol
    U = auto()  # UrukHai  
    N = auto()  # Nazgul
    W = auto()  # MordorWatchtower
    G = auto()  # Gollum
    R = auto()  # The One Ring
    C = auto()  # Mithril Mail-coat
    M = auto()  # Mount Doom
    P = auto()  # Perception zone


class Cell:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.cell_type: Optional[ItemType] = None
        self.is_safe_no_ring: Optional[bool] = None  # None - неизвестно
        self.is_safe_with_ring: Optional[bool] = None  # None - неизвестно
        self.visited_no_ring: bool = False
        self.visited_with_ring: bool = False
        self.enemy_type: Optional[ItemType] = None
        
    def update_from_perception(self, item_type: ItemType, ring_on: bool, has_mithril: bool):
        """Обновляем информацию о клетке на основе перцепции"""
        if ring_on:
            self.visited_with_ring = True
        else:
            self.visited_no_ring = True
            
        # Обрабатываем разные типы объектов
        if item_type in [ItemType.O, ItemType.U, ItemType.N, ItemType.W, ItemType.P]:
            # Враг или зона поражения - клетка опасна
            if ring_on:
                self.is_safe_with_ring = False
            else:
                self.is_safe_no_ring = False
                
            if item_type != ItemType.P:  # Запоминаем тип врага
                self.enemy_type = item_type
                self.cell_type = item_type
                
        elif item_type in [ItemType.G, ItemType.C, ItemType.M]:
            # Безопасные объекты
            if ring_on:
                self.is_safe_with_ring = True
            else:
                self.is_safe_no_ring = True
            self.cell_type = item_type
            
        elif item_type == ItemType.R:
            # Кольцо - безопасно, но особый случай
            if ring_on:
                self.is_safe_with_ring = True
            else:
                self.is_safe_no_ring = True
            self.cell_type = item_type


class AStarNode:
    def __init__(self, x: int, y: int, ring_on: bool, g: int, h: int, 
                 parent: Optional['AStarNode'] = None):
        self.x = x
        self.y = y
        self.ring_on = ring_on  # состояние кольца в этом узле
        self.g = g  # стоимость от начала
        self.h = h  # эвристическая стоимость до цели
        self.parent = parent
        
    @property
    def f(self) -> int:
        return self.g + self.h
        
    def __lt__(self, other):
        return self.f < other.f
        
    def __eq__(self, other):
        return (self.x, self.y, self.ring_on) == (other.x, other.y, other.ring_on)
        
    def __hash__(self):
        return hash((self.x, self.y, self.ring_on))
        
    def __repr__(self):
        return f"Node({self.x},{self.y},{'ring' if self.ring_on else 'no_ring'}) f={self.f}"

def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

class Enemy:
    def __init__(self, enemy_type: ItemType, x: int, y: int):
        self.enemy_type = enemy_type
        self.position = (x, y)
        
    def calculate_lethal_zone(self, ring_on: bool, has_mithril: bool) -> Set[Tuple[int, int]]:
        """Вычисляем зону поражения в зависимости от состояния колец и кольчуги"""
        x, y = self.position
        lethal_zone = set()
        
        if self.enemy_type == ItemType.O:  # Orc Patrol
            if ring_on or has_mithril:
                # Только собственная клетка опасна
                lethal_zone.add((x, y))
            else:
                # Von Neumann radius 1
                for dx, dy in [(0,0), (1,0), (-1,0), (0,1), (0,-1)]:
                    lethal_zone.add((x + dx, y + dy))
                    
        elif self.enemy_type == ItemType.U:  # Uruk-hai
            if ring_on or has_mithril:
                # Von Neumann radius 1
                for dx, dy in [(0,0), (1,0), (-1,0), (0,1), (0,-1)]:
                    lethal_zone.add((x + dx, y + dy))
            else:
                # Von Neumann radius 2
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if abs(dx) + abs(dy) <= 2:  # Von Neumann condition
                            lethal_zone.add((x + dx, y + dy))
                            
        elif self.enemy_type == ItemType.N:  # Nazgul
            if ring_on:
                # Moore radius 2 with ears
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if abs(dx) <= 2 and abs(dy) <= 2:  # Moore
                            lethal_zone.add((x + dx, y + dy))
            else:
                # Moore radius 1 with ears
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        lethal_zone.add((x + dx, y + dy))
                        
        elif self.enemy_type == ItemType.W:  # Watchtower
            if ring_on:
                # Moore radius 2 with ears
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if abs(dx) <= 2 and abs(dy) <= 2:
                            lethal_zone.add((x + dx, y + dy))
            else:
                # Moore radius 2
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if abs(dx) <= 2 and abs(dy) <= 2:
                            lethal_zone.add((x + dx, y + dy))
        
        # Фильтруем клетки за пределами карты
        filtered_zone = set()
        for cell_x, cell_y in lethal_zone:
            if 0 <= cell_x <= 12 and 0 <= cell_y <= 12:
                filtered_zone.add((cell_x, cell_y))
                
        return filtered_zone
    
class RingDestroyerGame:
    def __init__(self):
        self.grid_size = 13
        self.initialize_maps()
        self.current_pos = (0, 0)
        self.ring_on = False
        self.has_mithril = False
        self.gollum_pos: Optional[Tuple[int, int]] = None
        self.doom_pos: Optional[Tuple[int, int]] = None
        self.found_gollum = False
        self.perception_radius = 1
        self.known_enemies: List[Enemy] = []
        self.steps_count = 0
        self.L1 = 0  # Длина пути до Голлума
        self.L2 = 0  # Длина пути от Голлума до Горы
        
    def initialize_maps(self):
        """Инициализируем обе карты (с кольцом и без)"""
        self.map_no_ring: Dict[Tuple[int, int], Cell] = {}
        self.map_with_ring: Dict[Tuple[int, int], Cell] = {}
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                self.map_no_ring[(x, y)] = Cell(x, y)
                self.map_with_ring[(x, y)] = Cell(x, y)
        
        # Начальная позиция безопасна и посещена без кольца
        start_cell_no_ring = self.map_no_ring[(0, 0)]
        start_cell_no_ring.is_safe_no_ring = True
        start_cell_no_ring.visited_no_ring = True
        
        start_cell_with_ring = self.map_with_ring[(0, 0)]
        start_cell_with_ring.is_safe_with_ring = None  # Пока неизвестно с кольцом
        
    def process_initial_input(self):
        """Обрабатываем начальный ввод от интерактора"""
        # Читаем вариант перцепции
        variant = int(sys.stdin.readline().strip())
        self.perception_radius = variant
        
        # Читаем позицию Голлума
        gollum_line = sys.stdin.readline().split()
        self.gollum_pos = (int(gollum_line[0]), int(gollum_line[1]))
        
        # Читаем начальную перцепцию для (0,0)
        self.read_and_update_perception()
        
    def read_and_update_perception(self):
        """Читаем данные перцепции от интерактора и обновляем карту"""
        n_line = sys.stdin.readline().strip()
        if not n_line:
            return
            
        n = int(n_line)
        perception_data = []
        
        for _ in range(n):
            line = sys.stdin.readline().split()
            if len(line) < 3:
                continue
                
            x = int(line[0])
            y = int(line[1])
            item_str = line[2]
            
            # Конвертируем строку в ItemType
            try:
                item_type = ItemType[item_str]
                perception_data.append((x, y, item_type))
            except KeyError:
                # Игнорируем неизвестные типы
                continue
                
        self.update_from_perception(perception_data)
        
    def update_from_perception(self, perception_data: List[Tuple[int, int, ItemType]]):
        """Обновляем карты на основе полученных данных перцепции"""
        current_map = self.map_with_ring if self.ring_on else self.map_no_ring
        
        # Сначала помечаем все воспринятые клетки как безопасные
        # (если они не содержат врагов, о которых узнаем ниже)
        for x, y, item_type in perception_data:
            cell = current_map.get((x, y))
            if cell:
                cell.update_from_perception(item_type, self.ring_on, self.has_mithril)
                
                # Если обнаружили врага - добавляем в known_enemies
                if item_type in [ItemType.O, ItemType.U, ItemType.N, ItemType.W]:
                    enemy = Enemy(item_type, x, y)
                    if not any(e.position == (x, y) for e in self.known_enemies):
                        self.known_enemies.append(enemy)
                
                # Если нашли Голлума - запоминаем позицию
                elif item_type == ItemType.G:
                    self.gollum_pos = (x, y)
                
                # Если нашли кольчугу - активируем
                elif item_type == ItemType.C:
                    self.has_mithril = True

    def get_current_map(self) -> Dict[Tuple[int, int], Cell]:
        """Получаем текущую карту в зависимости от состояния кольца"""
        return self.map_with_ring if self.ring_on else self.map_no_ring

    def is_fully_safe_cell(self, x: int, y: int) -> bool:
        """Проверяем, полностью ли безопасна клетка (и с кольцом, и без)"""
        cell_no_ring = self.map_no_ring.get((x, y))
        cell_with_ring = self.map_with_ring.get((x, y))
        
        return (cell_no_ring and cell_no_ring.is_safe_no_ring is True and
                cell_with_ring and cell_with_ring.is_safe_with_ring is True)
                
    def send_command(self, command: str):
        """Отправляем команду интерактору и обрабатываем ответ"""
        print(command)
        sys.stdout.flush()
        
        if command.startswith('e'):
            return True  # Завершаем выполнение
            
        # Обрабатываем ответ интерактора
        if command.startswith('m'):
            # Обновляем позицию после движения
            parts = command.split()
            new_x, new_y = int(parts[1]), int(parts[2])
            self.update_after_move(new_x, new_y)
            self.steps_count += 1
            
        elif command == 'r':
            self.update_after_ring_toggle(True)
        elif command == 'rr':
            self.update_after_ring_toggle(False)
            
        # Читаем обновленную перцепцию
        self.read_and_update_perception()
        
        # Проверяем специальные случаи
        self.check_special_events()
        
        return False
        
    def check_special_events(self):
        """Проверяем специальные события (нахождение Голлума и т.д.)"""
        # Проверяем, нашли ли Голлума
        if (not self.found_gollum and self.current_pos == self.gollum_pos):
            self.handle_gollum_found()
            
        # Проверяем, подобрали ли кольчугу
        current_map = self.get_current_map()
        cell = current_map.get(self.current_pos)
        if cell and cell.cell_type == ItemType.C:
            self.has_mithril = True
            # Помечаем, что кольчуга подобрана (убираем с карты)
            cell.cell_type = None
            
    def handle_gollum_found(self):
        """Обрабатываем нахождение Голлума"""
        self.found_gollum = True
        
        # Читаем сообщение о позиции Горы Огненной
        line = sys.stdin.readline().strip()
        if line.startswith("My precious!"):
            parts = line.split()
            if len(parts) >= 6:
                doom_x = int(parts[4])
                doom_y = int(parts[5])
                self.doom_pos = (doom_x, doom_y)
                
        # Вычисляем L1 - длину пути до Голлума
        path_to_gollum = self.a_star_search((0, 0), self.gollum_pos, False)
        if path_to_gollum:
            self.L1 = self.get_movement_count(path_to_gollum)
            
    def get_movement_count(self, path: List[Tuple[int, int, bool]]) -> int:
        """Подсчитываем количество движений в пути (исключая переключения кольца)"""
        count = 0
        for i in range(1, len(path)):
            # Считаем только смены позиций (движения)
            if path[i][0] != path[i-1][0] or path[i][1] != path[i-1][1]:
                count += 1
        return count
        
    def should_stop(self) -> bool:
        """Проверяем, достигли ли мы конечной цели"""
        if self.found_gollum and self.current_pos == self.doom_pos:
            # Вычисляем L2 - длину пути от Голлума до Горы
            if self.gollum_pos and self.doom_pos:
                path_to_doom = self.a_star_search(self.gollum_pos, self.doom_pos, False)
                if path_to_doom:
                    self.L2 = self.get_movement_count(path_to_doom)
            return True
        return False
        
    def get_final_command(self) -> str:
        """Получаем финальную команду для завершения"""
        if self.found_gollum and self.doom_pos and self.current_pos == self.doom_pos:
            total_length = self.L1 + self.L2
            return f"e {total_length}"
        else:
            return "e -1"
            
    def a_star_search(self, start: Tuple[int, int], goal: Tuple[int, int], 
                     start_ring: bool) -> Optional[List[Tuple[int, int, bool]]]:
        """
        Поиск пути от start до goal с начальным состоянием кольца start_ring
        Возвращает список (x, y, ring_state) или None если путь не найден
        """
        open_set = []
        closed_set = set()
        
        # Начальный узел
        start_node = AStarNode(start[0], start[1], start_ring, 0, 
                              manhattan_distance(start, goal))
        heapq.heappush(open_set, start_node)
        
        while open_set:
            current = heapq.heappop(open_set)
            
            # Проверяем, достигли ли цели
            if (current.x, current.y) == goal:
                return self._reconstruct_path(current)
                
            closed_set.add((current.x, current.y, current.ring_on))
            
            # Генерируем соседние узлы
            neighbors = self._get_neighbors(current)
            
            for neighbor in neighbors:
                if (neighbor.x, neighbor.y, neighbor.ring_on) in closed_set:
                    continue
                    
                # Проверяем, есть ли узел в open_set с меньшей стоимостью
                found_better = False
                for node in open_set:
                    if (node.x, node.y, node.ring_on) == (neighbor.x, neighbor.y, neighbor.ring_on):
                        if node.g <= neighbor.g:
                            found_better = True
                            break
                        # Заменяем узел на более дешевый
                        open_set.remove(node)
                        heapq.heapify(open_set)
                        break
                
                if not found_better:
                    heapq.heappush(open_set, neighbor)
                    
        return None  # Путь не найден
        
    def _reconstruct_path(self, node: AStarNode) -> List[Tuple[int, int, bool]]:
        """Восстанавливаем путь от конечного узла до начала"""
        path = []
        current = node
        while current:
            path.append((current.x, current.y, current.ring_on))
            current = current.parent
        return path[::-1]  # Разворачиваем путь (от начала к концу)
        
    def _get_neighbors(self, node: AStarNode) -> List[AStarNode]:
        """Генерируем все возможные соседние узлы"""
        neighbors = []
        x, y, ring_on = node.x, node.y, node.ring_on
        
        # 1. Движение в соседние клетки (ортогонально)
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            
            # Проверяем границы карты
            if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                continue
                
            # Проверяем безопасность клетки в текущем состоянии кольца
            current_map = self.map_with_ring if ring_on else self.map_no_ring
            cell = current_map.get((nx, ny))
            
            if cell and self._is_cell_safe(cell, ring_on):
                # Клетка безопасна - можем двигаться
                g_new = node.g + 1  # Стоимость движения = 1
                h_new = manhattan_distance((nx, ny), self._get_current_goal())
                neighbor = AStarNode(nx, ny, ring_on, g_new, h_new, node)
                neighbors.append(neighbor)
        
        # 2. Переключение кольца (только в полностью безопасных клетках)
        if self.is_fully_safe_cell(x, y):
            new_ring_state = not ring_on
            g_new = node.g  # Переключение кольца не имеет стоимости движения
            h_new = manhattan_distance((x, y), self._get_current_goal())
            switch_node = AStarNode(x, y, new_ring_state, g_new, h_new, node)
            neighbors.append(switch_node)
            
        return neighbors
        
    def _is_cell_safe(self, cell: Cell, ring_on: bool) -> bool:
        """Проверяем, безопасна ли клетка в данном состоянии кольца"""
        if ring_on:
            return cell.is_safe_with_ring is True
        else:
            return cell.is_safe_no_ring is True
            
    def _get_current_goal(self) -> Tuple[int, int]:
        """Получаем текущую цель (Голлум или Гора Огненная)"""
        if self.found_gollum and self.doom_pos:
            return self.doom_pos
        else:
            return self.gollum_pos
            
    def update_after_move(self, new_x: int, new_y: int):
        """Обновляем состояние после движения"""
        self.current_pos = (new_x, new_y)
        
        # Помечаем клетку как посещенную
        current_map = self.get_current_map()
        cell = current_map.get(self.current_pos)
        if cell:
            if self.ring_on:
                cell.visited_with_ring = True
            else:
                cell.visited_no_ring = True
                
    def update_after_ring_toggle(self, new_ring_state: bool):
        """Обновляем состояние после переключения кольца"""
        self.ring_on = new_ring_state
        
        # Помечаем текущую клетку как посещенную с новым состоянием кольца
        current_map = self.get_current_map()
        cell = current_map.get(self.current_pos)
        if cell:
            if self.ring_on:
                cell.visited_with_ring = True
            else:
                cell.visited_no_ring = True
                
    def execute_astar_move(self) -> Optional[str]:
        """
        Выполняет один ход на основе A* поиска
        Возвращает команду для интерактора или None если путь завершен
        """
        # Определяем текущую цель
        if not self.found_gollum:
            goal = self.gollum_pos
        else:
            goal = self.doom_pos
            
        if goal is None:
            return "e -1"
            
        # Если мы уже в целевой клетке
        if self.current_pos == goal:
            if not self.found_gollum:
                # Должны найти Голлума в этом ходу
                return None
            else:
                # Достигли Горы Огненной
                return None
                
        # Ищем путь до текущей цели
        path = self.a_star_search(self.current_pos, goal, self.ring_on)
        
        if not path or len(path) < 2:
            # Путь не найден
            return "e -1"
            
        # Берем следующий шаг из пути
        next_x, next_y, next_ring_state = path[1]
        
        # Проверяем, нужно ли переключить кольцо
        if next_ring_state != self.ring_on:
            if next_ring_state:
                return "r"  # Надеть кольцо
            else:
                return "rr"  # Снять кольцо
        else:
            # Двигаемся в следующую клетку
            return f"m {next_x} {next_y}"



def main():
    game = RingDestroyerGame()
    
    # Обрабатываем начальный ввод
    game.process_initial_input()
    
    # Основной игровой цикл
    try:
        while True:
            # Проверяем, не достигли ли мы цели
            if game.should_stop():
                final_command = game.get_final_command()
                print(final_command)
                sys.stdout.flush()
                break
                
            # Получаем следующую команду с помощью A*
            command = game.execute_astar_move()
            
            if not command:
                # Если путь не найден
                print("e -1")
                sys.stdout.flush()
                break
                
            # Отправляем команду и проверяем завершение
            should_exit = game.send_command(command)
            if should_exit:
                break
                
    except Exception as e:
        # В случае ошибки отправляем команду завершения
        print(f"e -1")
        sys.stdout.flush()
        
if __name__ == "__main__":
    main()