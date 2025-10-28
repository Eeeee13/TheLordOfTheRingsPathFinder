from typing import Set, Tuple, List, Dict, Optional
from items import ItemType
from enemies import Enemy
from cell import Cell
from aStar import AStarNode, heapq
from helpers import *


class RingDestroyerGame:
    def __init__(self):
        self.grid: List[List[Cell]] = []
        self.current_pos = (0, 0)
        self.ring_on = False
        self.has_mithril = False
        self.gollum_pos: Optional[Tuple[int, int]] = None
        self.doom_pos: Optional[Tuple[int, int]] = None
        self.found_gollum = False
        self.perception_radius = 1
        self.known_enemies: List[Enemy] = []
        
        # Две карты: с кольцом и без
        self.map_no_ring: Dict[Tuple[int, int], Cell] = {}
        self.map_with_ring: Dict[Tuple[int, int], Cell] = {}
        
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
        
        return (cell_no_ring and cell_no_ring.is_safe_no_ring and
                cell_with_ring and cell_with_ring.is_safe_with_ring)
        
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
        
    def execute_astar_move(self) -> Optional[str]:
        """
        Выполняет один ход на основе A* поиска
        Возвращает команду для интерактора или None если путь завершен
        """
        if not self.found_gollum:
            goal = self.gollum_pos
        else:
            goal = self.doom_pos
            
        if goal is None:
            return None
            
        # Ищем путь до текущей цели
        path = self.a_star_search(self.current_pos, goal, self.ring_on)
        
        if not path:
            # Путь не найден - карта нерешаема
            return "e -1"
            
        if len(path) == 1:
            # Мы уже в целевой клетке
            if not self.found_gollum and (self.current_pos == self.gollum_pos):
                self.found_gollum = True
                # Получаем позицию Горы Огненной от интерактора
                return None  # Нужно обработать специальный случай
            else:
                return "e 0"  # Завершаем выполнение
                
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
        
    def execute_astar_move(self) -> Optional[str]:
        """
        Выполняет один ход на основе A* поиска
        Возвращает команду для интерактора или None если путь завершен
        """
        if not self.found_gollum:
            goal = self.gollum_pos
        else:
            goal = self.doom_pos
            
        if goal is None:
            return None
            
        # Ищем путь до текущей цели
        path = self.a_star_search(self.current_pos, goal, self.ring_on)
        
        if not path:
            # Путь не найден - карта нерешаема
            return "e -1"
            
        if len(path) == 1:
            # Мы уже в целевой клетке
            if not self.found_gollum and (self.current_pos == self.gollum_pos):
                self.found_gollum = True
                # Получаем позицию Горы Огненной от интерактора
                return None  # Нужно обработать специальный случай
            else:
                return "e 0"  # Завершаем выполнение
                
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

    
    def explore_optimal_path(self):
        # Основная логика исследования и движения
        # Будем переключать кольцо в безопасных клетках для изучения местности
        pass