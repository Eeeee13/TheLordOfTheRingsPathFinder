import numpy as np
from enemies import Enemy
from items import ItemType
import random
from typing import List, Tuple, Set, Optional
from enum import Enum
import random
from typing import List, Tuple, Set, Optional, Dict
from enum import Enum
import colorama
from colorama import Fore, Back, Style

# Инициализация colorama
colorama.init()

import random
from typing import List, Tuple, Set, Optional, Dict
from enum import Enum
import colorama
from colorama import Fore, Back, Style

# Инициализация colorama
colorama.init()

class InteractiveSimulator:
    def __init__(self, map_generator, variant=1, seed=None):
        self.variant = variant
        self.map_generator = map_generator
        self.grid, self.enemies, self.gollum_pos, self.mount_doom_pos, self.mithril_pos = map_generator.generate_map()
        self.visualizer = MapVisualizer()
        
        # Состояние агента
        self.frodo_pos = (0, 0)
        self.ring_on = False
        self.has_mithril = False
        self.found_gollum = False
        self.knows_mount_doom = False
        self.game_over = False
        self.path_length = 0
        
        # История для отладки
        self.move_history = []
        
    def get_perception(self) -> List[str]:
        """
        Возвращает восприятие агента в текущей клетке согласно варианту
        Формат: [количество_объектов, "x y тип", ...]
        """
        perception_radius = 1 if self.variant == 1 else 2
        perceived_cells = []
        
        # Собираем все клетки в зоне восприятия (исключая текущую позицию Фродо)
        for dx in range(-perception_radius, perception_radius + 1):
            for dy in range(-perception_radius, perception_radius + 1):
                if dx == 0 and dy == 0:
                    continue  # Пропускаем клетку Фродо
                    
                x, y = self.frodo_pos[0] + dx, self.frodo_pos[1] + dy
                
                # Проверяем границы карты
                if 0 <= x < 13 and 0 <= y < 13:
                    cell_type = self.get_cell_type_for_perception(x, y)
                    if cell_type:
                        perceived_cells.append(f"{x} {y} {cell_type}")
        
        return perceived_cells
    
    def get_cell_type_for_perception(self, x: int, y: int) -> Optional[str]:
        """
        Определяет тип клетки для восприятия с учетом текущего состояния
        """
        # Если в клетке враг - показываем его тип
        if self.grid[x][y] in [ItemType.O, ItemType.U, ItemType.N, ItemType.W]:
            return self.grid[x][y].name
        
        # Если в клетке Голлум и он еще не найден
        if self.grid[x][y] == ItemType.G and not self.found_gollum:
            return 'G'
            
        # Если в клетке кольчуга и она еще не подобрана
        if self.grid[x][y] == ItemType.C and not self.has_mithril:
            return 'C'
            
        # Если в клетке Гора Огня и она уже известна
        if self.grid[x][y] == ItemType.M and self.knows_mount_doom:
            return 'M'
            
        # Проверяем, является ли клетка опасной зоной
        if self.is_cell_dangerous(x, y):
            return 'P'
            
        return None
    
    def is_cell_dangerous(self, x: int, y: int) -> bool:
        """
        Проверяет, является ли клетка опасной (в зоне поражения врага)
        """
        for enemy in self.enemies:
            lethal_zone = enemy.calculate_lethal_zone(
                ring_on=self.ring_on, 
                has_mithril=self.has_mithril
            )
            if (x, y) in lethal_zone:
                return True
        return False
    
    def process_move(self, target_x: int, target_y: int) -> Dict:
        """
        Обрабатывает движение агента в указанную клетку
        """
        # Проверяем валидность хода
        current_x, current_y = self.frodo_pos
        if abs(target_x - current_x) + abs(target_y - current_y) != 1:
            return {
                'success': False,
                'message': 'Invalid move: can only move to adjacent cells'
            }
        
        # Проверяем границы
        if not (0 <= target_x < 13 and 0 <= target_y < 13):
            return {
                'success': False, 
                'message': 'Invalid move: coordinates out of bounds'
            }
        
        # Обновляем позицию и историю
        self.frodo_pos = (target_x, target_y)
        self.path_length += 1
        self.move_history.append(self.frodo_pos)
        
        # Проверяем, не в опасной ли клетке
        if self.is_cell_dangerous(target_x, target_y):
            self.game_over = True
            return {
                'success': False,
                'message': 'Game over: Frodo died!',
                'perception': []
            }
        
        # Проверяем специальные клетки
        result = {'success': True, 'special_events': []}
        
        # Клетка с Голлумом
        if self.grid[target_x][target_y] == ItemType.G and not self.found_gollum:
            self.found_gollum = True
            self.knows_mount_doom = True
            result['special_events'].append(
                f"My precious! Mount Doom is {self.mount_doom_pos[0]} {self.mount_doom_pos[1]}"
            )
        
        # Клетка с кольчугой
        if self.grid[target_x][target_y] == ItemType.C and not self.has_mithril:
            self.has_mithril = True
            # По условию - не информируем агента
        
        # Клетка с Горой Огня
        if (self.grid[target_x][target_y] == ItemType.M and 
            self.found_gollum and not self.game_over):
            self.game_over = True
            result['special_events'].append("Ring destroyed! You win!")
        
        # Получаем восприятие для новой позиции
        result['perception'] = self.get_perception()
        
        return result
    
    def process_ring_command(self, command: str) -> Dict:
        """
        Обрабатывает команды кольца (r - надеть, rr - снять)
        """
        if command == 'r' and self.ring_on:
            return {'success': False, 'message': 'Ring already on'}
        if command == 'rr' and not self.ring_on:
            return {'success': False, 'message': 'Ring already off'}
            
        self.ring_on = (command == 'r')
        
        return {
            'success': True,
            'perception': self.get_perception()
        }
    
    def process_end_command(self, result_code: int) -> Dict:
        """
        Обрабатывает команду завершения
        """
        self.game_over = True
        return {
            'success': True,
            'message': f'Game ended with code: {result_code}'
        }
    
    def run_interactive_session(self):
        """
        Запускает интерактивную сессию с агентом через консоль
        """
        print(f"=== Ring Destroyer Simulator (Variant {self.variant}) ===")
        print(f"Gollum position: {self.gollum_pos}")
        print(f"Mount Doom position: {self.mount_doom_pos}")
        print(f"Mithril position: {self.mithril_pos}")
        print("\nInitial perception at (0, 0):")
        
        # Начальное восприятие
        initial_perception = self.get_perception()
        print(f"{len(initial_perception)}")
        for obj in initial_perception:
            print(obj)
        
        print("\nAvailable commands:")
        print("  m x y - move to cell (x,y)")
        print("  r - put ring on") 
        print("  rr - take ring off")
        print("  e n - end game (n = path length or -1)")
        print("  debug - show current map")
        print("  quit - exit simulator")
        
        # Основной игровой цикл
        while not self.game_over:
            try:
                command = input("\nCommand: ").strip()
                
                if command == 'quit':
                    break
                elif command == 'debug':
                    self.print_debug_info()
                    continue
                
                # Обработка команд движения
                if command.startswith('m '):
                    parts = command.split()
                    if len(parts) != 3:
                        print("Invalid move command. Use: m x y")
                        continue
                    
                    try:
                        x, y = int(parts[1]), int(parts[2])
                        result = self.process_move(x, y)
                        
                        if result['success']:
                            print(f"Moved to ({x}, {y})")
                            print(f"Perception: {len(result['perception'])} objects")
                            for obj in result['perception']:
                                print(obj)
                            
                            # Специальные события после восприятия
                            for event in result.get('special_events', []):
                                print(event)
                                
                            if self.game_over and "win" in result.get('special_events', [''])[0].lower():
                                print(f"Congratulations! Path length: {self.path_length}")
                        else:
                            print(f"Move failed: {result['message']}")
                            if self.game_over:
                                break
                                
                    except ValueError:
                        print("Invalid coordinates")
                
                # Обработка команд кольца
                elif command in ['r', 'rr']:
                    result = self.process_ring_command(command)
                    if result['success']:
                        ring_status = "on" if self.ring_on else "off"
                        print(f"Ring is now {ring_status}")
                        print(f"Perception: {len(result['perception'])} objects")
                        for obj in result['perception']:
                            print(obj)
                    else:
                        print(f"Ring command failed: {result['message']}")
                
                # Обработка команды завершения
                elif command.startswith('e '):
                    parts = command.split()
                    if len(parts) != 2:
                        print("Invalid end command. Use: e n")
                        continue
                    
                    try:
                        result_code = int(parts[1])
                        result = self.process_end_command(result_code)
                        print(result['message'])
                        break
                    except ValueError:
                        print("Invalid result code")
                
                else:
                    print("Unknown command")
                    
            except KeyboardInterrupt:
                print("\nGame interrupted by user")
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\n=== Game Over ===")
        self.print_final_stats()
    
    def print_debug_info(self):
        """Печатает отладочную информацию о текущем состоянии"""
        print("\n" + "="*50)
        print("DEBUG INFORMATION:")
        print(f"Frodo position: {self.frodo_pos}")
        print(f"Ring on: {self.ring_on}")
        print(f"Has mithril: {self.has_mithril}")
        print(f"Found Gollum: {self.found_gollum}")
        print(f"Knows Mount Doom: {self.knows_mount_doom}")
        print(f"Path length: {self.path_length}")
        
        # Визуализация карты с текущей позицией
        debug_grid = [row[:] for row in self.grid]  # Копируем сетку
        debug_grid[self.frodo_pos[0]][self.frodo_pos[1]] = ItemType.F  # Добавляем Фродо
        
        print("\nCurrent map view:")
        map_visual = self.visualizer.visualize_with_legend(debug_grid, self.enemies)
        print(map_visual)
        
        # Опасные клетки вокруг
        print(f"\nDangerous cells in perception zone:")
        perception = self.get_perception()
        danger_cells = [obj for obj in perception if obj.endswith('P')]
        print(f"Count: {len(danger_cells)}")
        for obj in danger_cells:
            print(f"  {obj}")
    
    def print_final_stats(self):
        """Печатает финальную статистику"""
        print(f"Final position: {self.frodo_pos}")
        print(f"Total moves: {self.path_length}")
        print(f"Gollum found: {self.found_gollum}")
        print(f"Ring destroyed: {self.game_over and self.frodo_pos == self.mount_doom_pos}")
        print(f"Mithril acquired: {self.has_mithril}")
        print(f"Move history: {self.move_history}")

# Обновленный MapVisualizer для работы с симулятором
class MapVisualizer:
    def __init__(self):
        self.color_scheme = {
            ItemType.O: Fore.RED + 'O' + Style.RESET_ALL,
            ItemType.U: Fore.LIGHTRED_EX + 'U' + Style.RESET_ALL,
            ItemType.N: Fore.MAGENTA + 'N' + Style.RESET_ALL,
            ItemType.W: Fore.YELLOW + 'W' + Style.RESET_ALL,
            ItemType.G: Fore.GREEN + 'G' + Style.RESET_ALL,
            ItemType.M: Fore.LIGHTRED_EX + 'M' + Style.RESET_ALL,
            ItemType.C: Fore.CYAN + 'C' + Style.RESET_ALL,
            ItemType.R: Fore.YELLOW + 'R' + Style.RESET_ALL,
            ItemType.F: Fore.WHITE + 'F' + Style.RESET_ALL,
            ItemType.P: Back.RED + ' ' + Style.RESET_ALL,
        }
        
        self.empty_safe = Fore.WHITE + '.' + Style.RESET_ALL
        self.empty_danger = Back.RED + ' ' + Style.RESET_ALL
    
    def visualize(self, grid, enemies, show_danger_zones=True, show_coordinates=True):
        """Визуализация карты (без изменений)"""
        # ... (предыдущая реализация)
        vis_grid = [[None for _ in range(13)] for _ in range(13)]
        
        for x in range(13):
            for y in range(13):
                if grid[x][y] is not None:
                    vis_grid[x][y] = self.color_scheme.get(grid[x][y], '?')
        
        if show_danger_zones:
            danger_zones = self._calculate_danger_zones(enemies, ring_on=False, has_mithril=False)
            for (x, y) in danger_zones:
                if grid[x][y] is None:
                    vis_grid[x][y] = self.color_scheme[ItemType.P]
        
        result = []
        
        if show_coordinates:
            header = "   " + " ".join(f"{i:2}" for i in range(13))
            result.append(header)
            result.append("  " + "─" * 39)
        
        for x in range(13):
            row_parts = []
            if show_coordinates:
                row_parts.append(f"{x:2}│")
            
            for y in range(13):
                if vis_grid[x][y] is not None:
                    row_parts.append(vis_grid[x][y])
                else:
                    row_parts.append(self.empty_safe)
            
            result.append(" ".join(row_parts))
        
        return "\n".join(result)
    
    def visualize_with_legend(self, grid, enemies):
        """Визуализация с легендой"""
        map_visual = self.visualize(grid, enemies, show_danger_zones=True, show_coordinates=True)
        
        legend = [
            "\n" + "="*50 + " ЛЕГЕНДА " + "="*50,
            f"{self.color_scheme[ItemType.F]} - Фродо",
            f"{self.color_scheme[ItemType.O]} - Орк",
            f"{self.color_scheme[ItemType.U]} - Урук-хай", 
            f"{self.color_scheme[ItemType.N]} - Назгул",
            f"{self.color_scheme[ItemType.W]} - Дозорная башня",
            f"{self.color_scheme[ItemType.G]} - Голлум",
            f"{self.color_scheme[ItemType.M]} - Гора Огня",
            f"{self.color_scheme[ItemType.C]} - Мифриловая кольчуга",
            f"{self.color_scheme[ItemType.P]} - Зона поражения врага",
            f"{self.empty_safe} - Безопасная клетка",
        ]
        
        return map_visual + "\n" + "\n".join(legend)
    
    def _calculate_danger_zones(self, enemies, ring_on, has_mithril):
        """Вычисляет опасные зоны"""
        danger_zones = set()
        for enemy in enemies:
            zone = enemy.calculate_lethal_zone(ring_on=ring_on, has_mithril=has_mithril)
            danger_zones.update(zone)
        return danger_zones


class MapGenerator:
    def __init__(self, seed=None):
        self.size = 13
        self.visualizer = MapVisualizer()
        if seed:
            random.seed(seed)
    
    def generate_map(self) -> Tuple[List[List[Optional[ItemType]]], List[Enemy], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """Генерирует карту (реализация без изменений)"""
        # ... (предыдущая реализация generate_map)
        grid = [[None for _ in range(self.size)] for _ in range(self.size)]
        enemies = []
        occupied_cells = set()
        danger_zones = set()
        
        occupied_cells.add((0, 0))
        
        enemy_types = [
            (ItemType.W, 1),
            (ItemType.U, 1),
            (ItemType.N, random.randint(0, 1)),
            (ItemType.O, random.randint(1, 2))
        ]
        
        for enemy_type, count in enemy_types:
            for _ in range(count):
                enemy_pos = self._find_safe_position(occupied_cells, danger_zones, enemy_type)
                if enemy_pos is None:
                    continue
                    
                enemy = Enemy(enemy_type, enemy_pos[0], enemy_pos[1])
                enemies.append(enemy)
                occupied_cells.add(enemy_pos)
                base_zone = enemy.calculate_lethal_zone(ring_on=False, has_mithril=False)
                danger_zones.update(base_zone)
                grid[enemy_pos[0]][enemy_pos[1]] = enemy_type
        
        gollum_pos = self._find_safe_position(occupied_cells, danger_zones)
        if gollum_pos:
            occupied_cells.add(gollum_pos)
            grid[gollum_pos[0]][gollum_pos[1]] = ItemType.G
        
        mount_doom_pos = self._find_safe_position(occupied_cells, danger_zones)
        if mount_doom_pos:
            occupied_cells.add(mount_doom_pos)
            grid[mount_doom_pos[0]][mount_doom_pos[1]] = ItemType.M
        
        mithril_pos = self._find_safe_position(occupied_cells, danger_zones)
        if mithril_pos:
            occupied_cells.add(mithril_pos)
            grid[mithril_pos[0]][mithril_pos[1]] = ItemType.C
        
        return grid, enemies, gollum_pos, mount_doom_pos, mithril_pos
    
    def _find_safe_position(self, occupied_cells, danger_zones, enemy_type=None):
        """Находит безопасную позицию (реализация без изменений)"""
        all_cells = [(x, y) for x in range(self.size) for y in range(self.size)]
        random.shuffle(all_cells)
        
        for cell in all_cells:
            if cell in occupied_cells or cell in danger_zones:
                continue
                
            if enemy_type:
                temp_enemy = Enemy(enemy_type, cell[0], cell[1])
                enemy_zone = temp_enemy.calculate_lethal_zone(ring_on=False, has_mithril=False)
                if (0, 0) in enemy_zone:
                    continue
            
            return cell
        
        return None
    
    def generate_and_visualize(self, seed=None) -> Dict:
        """Генерирует карту и возвращает полную информацию с визуализацией"""
        if seed:
            random.seed(seed)
            
        grid, enemies, gollum_pos, mount_doom_pos, mithril_pos = self.generate_map()
        
        # Визуализация
        map_visual = self.visualizer.visualize(grid, enemies)
        
        # Детальная информация
        # self.visualizer.print_detailed_map_info(grid, enemies, gollum_pos, mount_doom_pos, mithril_pos)
        print(map_visual)
        
        return {
            'grid': grid,
            'enemies': enemies,
            'gollum_pos': gollum_pos,
            'mount_doom_pos': mount_doom_pos,
            'mithril_pos': mithril_pos,
            'visualization': map_visual
        }

# # Пример использования
# if __name__ == "__main__":
#     print("ГЕНЕРАТОР КАРТ 'ВЛАСТЕЛИН КОЛЕЦ'")
#     print("=" * 60)
    
#     generator = MapGenerator(seed=42)
    
#     # Генерация и визуализация одной карты
#     map_data = generator.generate_and_visualize()
    
#     print("\n" + "="*60)
#     print("Генерация завершена!")
    
#     # Генерация 5 карт для демонстрации
#     print("\nГенерация 5 случайных карт:")
#     print("=" * 60)
    
#     for i in range(5):
#         print(f"\nКАРТА #{i+1}:")
#         generator.generate_and_visualize(seed=i*100)
#         print("\n" + "-"*60)



# Пример использования
if __name__ == "__main__":
    print("ИНТЕРАКТИВНЫЙ СИМУЛЯТОР 'ВЛАСТЕЛИН КОЛЕЦ'")
    print("=" * 60)
    
    # Выбор варианта восприятия
    variant = int(input("Выберите вариант восприятия (1 или 2): ") or "1")
    seed = input("Введите seed для генерации карты (или оставьте пустым): ")
    seed = int(seed) if seed else None
    
    # Создаем генератор и симулятор
    generator = MapGenerator(seed=seed)
    simulator = InteractiveSimulator(generator, variant=variant)
    map_data = generator.generate_and_visualize(seed=seed)
    
    # Запускаем интерактивную сессию
    simulator.run_interactive_session()
