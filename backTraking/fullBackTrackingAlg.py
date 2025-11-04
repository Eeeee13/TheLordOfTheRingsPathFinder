import sys
import heapq
from enum import Enum, auto
from typing import Set, Tuple, List, Dict, Optional

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
        self.is_safe_no_ring: Optional[bool] = None
        self.is_safe_with_ring: Optional[bool] = None
        self.visited_no_ring: bool = False
        self.visited_with_ring: bool = False
        self.enemy_type: Optional[ItemType] = None
        
    def __repr__(self):
        return f"Cell({self.x},{self.y})"
    
    def update_from_perception(self, item_type: ItemType, ring_on: bool, has_mithril: bool):
        if ring_on:
            self.visited_with_ring = True
        else:
            self.visited_no_ring = True
            
        if item_type in [ItemType.O, ItemType.U, ItemType.N, ItemType.W, ItemType.P]:
            if ring_on:
                self.is_safe_with_ring = False
            else:
                self.is_safe_no_ring = False
                
            if item_type != ItemType.P:
                self.enemy_type = item_type
                self.cell_type = item_type
                
        elif item_type in [ItemType.G, ItemType.C, ItemType.M]:
            if ring_on:
                self.is_safe_with_ring = True
            else:
                self.is_safe_no_ring = True
            self.cell_type = item_type
            
        elif item_type == ItemType.R:
            if ring_on:
                self.is_safe_with_ring = True
            else:
                self.is_safe_no_ring = True
            self.cell_type = item_type


class BacktrackNode:
    def __init__(self, x: int, y: int, ring_on: bool, mithril_on: bool, 
                 path: List[Tuple[int, int, bool]], depth: int):
        self.x = x
        self.y = y
        self.ring_on = ring_on
        self.mithril_on = mithril_on
        self.path = path
        self.depth = depth
        
    def __repr__(self):
        return f"BacktrackNode({self.x},{self.y},ring={self.ring_on},mithril={self.mithril_on},depth={self.depth})"

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
        self.L1 = 0
        self.L2 = 0
        
    def initialize_maps(self):
        self.map_no_ring: Dict[Tuple[int, int], Cell] = {}
        self.map_with_ring: Dict[Tuple[int, int], Cell] = {}
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                self.map_no_ring[(x, y)] = Cell(x, y)
                self.map_with_ring[(x, y)] = Cell(x, y)
        
        start_cell_no_ring = self.map_no_ring[(0, 0)]
        start_cell_no_ring.is_safe_no_ring = True
        start_cell_no_ring.visited_no_ring = True
        
        start_cell_with_ring = self.map_with_ring[(0, 0)]
        start_cell_with_ring.is_safe_with_ring = None

    def process_initial_input(self):
        variant = int(sys.stdin.readline().strip())
        self.perception_radius = variant
        
        gollum_line = sys.stdin.readline().split()
        self.gollum_pos = (int(gollum_line[0]), int(gollum_line[1]))
        
        self.read_and_update_perception()
        
    def read_and_update_perception(self):
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
            
            try:
                item_type = ItemType[item_str]
                perception_data.append((x, y, item_type))
            except KeyError:
                continue
                
        self.update_from_perception(perception_data)
        
    def update_from_perception(self, perception_data: List[Tuple[int, int, ItemType]]):
        current_map = self.map_with_ring if self.ring_on else self.map_no_ring
        
        for x, y, item_type in perception_data:
            cell = current_map.get((x, y))
            if cell:
                cell.update_from_perception(item_type, self.ring_on, self.has_mithril)
                
                if item_type in [ItemType.O, ItemType.U, ItemType.N, ItemType.W]:
                    enemy = Enemy(item_type, x, y)
                    if not any(e.position == (x, y) for e in self.known_enemies):
                        self.known_enemies.append(enemy)
                
                elif item_type == ItemType.G:
                    self.gollum_pos = (x, y)
                
                elif item_type == ItemType.C:
                    self.has_mithril = True

    def get_current_map(self) -> Dict[Tuple[int, int], Cell]:
        return self.map_with_ring if self.ring_on else self.map_no_ring

    def is_fully_safe_cell(self, x: int, y: int) -> bool:
        cell_no_ring = self.map_no_ring.get((x, y))
        cell_with_ring = self.map_with_ring.get((x, y))
        
        return (cell_no_ring and cell_no_ring.is_safe_no_ring is True and
                cell_with_ring and cell_with_ring.is_safe_with_ring is True)
                
    def send_command(self, command: str):
        print(command)
        sys.stdout.flush()
        
        if command.startswith('e'):
            return True
            
        if command.startswith('m'):
            parts = command.split()
            new_x, new_y = int(parts[1]), int(parts[2])
            self.update_after_move(new_x, new_y)
            self.steps_count += 1
            
        elif command == 'r':
            self.update_after_ring_toggle(True)
        elif command == 'rr':
            self.update_after_ring_toggle(False)
            
        self.read_and_update_perception()
        self.check_special_events()
        
        return False
        
    def check_special_events(self):
        if (not self.found_gollum and self.current_pos == self.gollum_pos):
            self.handle_gollum_found()
            
        current_map = self.get_current_map()
        cell = current_map.get(self.current_pos)
        if cell and cell.cell_type == ItemType.C:
            self.has_mithril = True
            cell.cell_type = None
            
    def handle_gollum_found(self):
        self.found_gollum = True
        
        line = sys.stdin.readline().strip()
        if line.startswith("My precious!"):
            parts = line.split()
            if len(parts) >= 6:
                doom_x = int(parts[4])
                doom_y = int(parts[5])
                self.doom_pos = (doom_x, doom_y)
                
        path_to_gollum = self.backtracking_search((0, 0), self.gollum_pos, False, self.has_mithril)
        if path_to_gollum:
            self.L1 = self.get_movement_count(path_to_gollum)
            
    def get_movement_count(self, path: List[Tuple[int, int, bool]]) -> int:
        count = 0
        for i in range(1, len(path)):
            if path[i][0] != path[i-1][0] or path[i][1] != path[i-1][1]:
                count += 1
        return count
        
    def should_stop(self) -> bool:
        if self.found_gollum and self.current_pos == self.doom_pos:
            if self.gollum_pos and self.doom_pos:
                path_to_doom = self.backtracking_search(self.gollum_pos, self.doom_pos, False, self.has_mithril)
                if path_to_doom:
                    self.L2 = self.get_movement_count(path_to_doom)
            return True
        return False
        
    def get_final_command(self) -> str:
        if self.found_gollum and self.doom_pos and self.current_pos == self.doom_pos:
            total_length = self.L1 + self.L2
            return f"e {total_length}"
        else:
            return "e -1"
            
    def backtracking_search(self, start: Tuple[int, int], goal: Tuple[int, int], 
                          start_ring: bool, start_mithril: bool, 
                          max_depth: int = 10) -> Optional[List[Tuple[int, int, bool]]]:
        """Backtracking поиск пути с мемоизацией и эвристикой"""
        start_node = BacktrackNode(
            start[0], start[1], start_ring, start_mithril,
            [(start[0], start[1], start_ring)], 0
        )
        
        # Мемоизация: visited states для всего поиска
        visited = set()
        # Стек для DFS (будем использовать как LIFO)
        stack = [start_node]
        
        while stack:
            node = stack.pop()
            
            # Проверяем достижение цели
            if (node.x, node.y) == goal:
                return node.path
                
            # Проверяем глубину
            if node.depth >= max_depth:
                continue
                
            # Проверяем мемоизацию
            state_key = (node.x, node.y, node.ring_on, node.mithril_on)
            if state_key in visited:
                continue
            visited.add(state_key)
            
            # Генерируем преемников
            successors = []
            
            # Движения в соседние клетки
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = node.x + dx, node.y + dy
                
                if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                    continue
                    
                # Проверяем безопасность в текущем состоянии кольца
                current_map = self.map_with_ring if node.ring_on else self.map_no_ring
                cell = current_map.get((nx, ny))
                
                if cell and self._is_cell_safe(cell, node.ring_on):
                    new_path = node.path + [(nx, ny, node.ring_on)]
                    new_node = BacktrackNode(nx, ny, node.ring_on, node.mithril_on, new_path, node.depth + 1)
                    successors.append(new_node)
            
            # Переключение кольца (только в полностью безопасных клетках)
            if self.is_fully_safe_cell(node.x, node.y):
                new_ring_state = not node.ring_on
                new_path = node.path + [(node.x, node.y, new_ring_state)]
                new_node = BacktrackNode(node.x, node.y, new_ring_state, node.mithril_on, new_path, node.depth + 1)
                successors.append(new_node)
            
            # Упорядочиваем преемников по эвристике (манхэттенское расстояние до цели)
            # Сортируем в обратном порядке для стека (чтобы лучшие шли первыми)
            successors.sort(key=lambda n: manhattan_distance((n.x, n.y), goal), reverse=True)
            
            # Добавляем в стек
            stack.extend(successors)
            
        return None
        
    def _is_cell_safe(self, cell: Cell, ring_on: bool) -> bool:
        if ring_on:
            return cell.is_safe_with_ring is True
        else:
            return cell.is_safe_no_ring is True
            
    def _get_current_goal(self) -> Tuple[int, int]:
        if self.found_gollum and self.doom_pos:
            return self.doom_pos
        else:
            return self.gollum_pos
            
    def update_after_move(self, new_x: int, new_y: int):
        self.current_pos = (new_x, new_y)
        
        current_map = self.get_current_map()
        cell = current_map.get(self.current_pos)
        if cell:
            if self.ring_on:
                cell.visited_with_ring = True
            else:
                cell.visited_no_ring = True
                
    def update_after_ring_toggle(self, new_ring_state: bool):
        self.ring_on = new_ring_state
        
        current_map = self.get_current_map()
        cell = current_map.get(self.current_pos)
        if cell:
            if self.ring_on:
                cell.visited_with_ring = True
            else:
                cell.visited_no_ring = True
                
    def execute_backtracking_move(self) -> Optional[str]:
        goal = self._get_current_goal()
        if goal is None:
            return "e -1"
            
        if self.current_pos == goal:
            if not self.found_gollum:
                return None
            else:
                return None
                
        path = self.backtracking_search(self.current_pos, goal, self.ring_on, self.has_mithril)
        
        if not path or len(path) < 2:
            return "e -1"
            
        next_x, next_y, next_ring_state = path[1]
        
        if next_ring_state != self.ring_on:
            return "r" if next_ring_state else "rr"
        else:
            return f"m {next_x} {next_y}"

def main():
    game = RingDestroyerGame()
    
    game.process_initial_input()
    
    try:
        while True:
            if game.should_stop():
                final_command = game.get_final_command()
                print(final_command)
                sys.stdout.flush()
                break
                
            command = game.execute_backtracking_move()
            
            if not command:
                print("e -1")
                sys.stdout.flush()
                break
                
            should_exit = game.send_command(command)
            if should_exit:
                break
                
    except Exception as e:
        print("e -1")
        sys.stdout.flush()
        

main()