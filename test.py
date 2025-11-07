import sys
import time
import statistics
from collections import defaultdict
from typing import List, Tuple, Dict, Any
import numpy as np

# Импортируем ваши алгоритмы
from Astar.fullAstaralg import RingDestroyerGame as AStarGame
from backTraking.fullBackTrackingAlg import RingDestroyerGame as BacktrackingGame

class MapGenerator:
    def __init__(self, seed=None):
        self.size = 13
        if seed:
            np.random.seed(seed)
    
    def generate_map(self) -> Tuple[List[List[str]], List[Tuple], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """Генерирует случайную карту"""
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        
        # Начальная позиция Фродо
        grid[0][0] = 'F'
        
        # Размещаем Голлума в случайной позиции (но не в начале)
        gollum_pos = self._get_random_position(exclude_positions=[(0, 0)])
        grid[gollum_pos[0]][gollum_pos[1]] = 'G'
        
        # Размещаем Гору Огненную
        doom_pos = self._get_random_position(exclude_positions=[(0, 0), gollum_pos])
        grid[doom_pos[0]][doom_pos[1]] = 'M'
        
        # Размещаем кольчугу
        mithril_pos = self._get_random_position(exclude_positions=[(0, 0), gollum_pos, doom_pos])
        grid[mithril_pos[0]][mithril_pos[1]] = 'C'
        
        # Размещаем врагов
        enemies = []
        enemy_types = ['O', 'U', 'N', 'W']
        
        for enemy_type in enemy_types:
            enemy_pos = self._get_random_position(exclude_positions=[(0, 0), gollum_pos, doom_pos, mithril_pos] + enemies)
            if enemy_pos:
                grid[enemy_pos[0]][enemy_pos[1]] = enemy_type
                enemies.append(enemy_pos)
        
        return grid, enemies, gollum_pos, doom_pos, mithril_pos
    
    def _get_random_position(self, exclude_positions=None):
        """Генерирует случайную позицию, исключая указанные"""
        if exclude_positions is None:
            exclude_positions = []
            
        all_positions = [(x, y) for x in range(self.size) for y in range(self.size)]
        available_positions = [pos for pos in all_positions if pos not in exclude_positions]
        
        if not available_positions:
            return None
            
        return available_positions[np.random.randint(0, len(available_positions))]

class GameSimulator:
    """Симулятор игры для тестирования алгоритмов"""
    
    def __init__(self, grid, enemies, gollum_pos, doom_pos, mithril_pos):
        self.grid = grid
        self.enemies = enemies
        self.gollum_pos = gollum_pos
        self.doom_pos = doom_pos
        self.mithril_pos = mithril_pos
        self.size = 13

    def _update_game_perception(self, game, perception: List[Tuple[int, int, str]]):
        """Обновляет перцепцию в игровом объекте"""
        # Преобразуем данные перцепции в формат, ожидаемый игрой
        perception_data = []
        for x, y, obj_type in perception:
            # Преобразуем строковый тип в ItemType
            from fullAstaralg import ItemType  # Импортируем из любого из алгоритмов
            try:
                item_type = ItemType[obj_type]
                perception_data.append((x, y, item_type))
            except KeyError:
                continue
        
        # Обновляем карту игры
        game.update_from_perception(perception_data)

    def get_perception(self, x: int, y: int, radius: int) -> List[Tuple[int, int, str]]:
        """Генерирует данные перцепции для заданной позиции и радиуса"""
        perception = []
        
        # Проверяем все клетки в радиусе
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                
                # Проверяем границы
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    cell_value = self.grid[nx][ny]
                    if cell_value != '.' and cell_value != 'F':  # Игнорируем пустые клетки и Фродо
                        perception.append((nx, ny, cell_value))
        
        return perception
    
    def _add_perception_to_stdin(self, mock_stdin, perception: List[Tuple[int, int, str]]):
        """Добавляет данные перцепции в mock stdin"""
        # Преобразуем в формат, который ожидает алгоритм
        lines = []
        lines.append(str(len(perception)))
        for x, y, obj_type in perception:
            lines.append(f"{x} {y} {obj_type}")
        
        # Добавляем в mock stdin
        mock_stdin.input_data.extend(lines)
        
    def simulate_game(self, algorithm_class, variant=1, max_steps=1000):
        """Запускает симуляцию игры с заданным алгоритмом"""
        start_time = time.time()
        
        # Создаем экземпляр алгоритма
        game = algorithm_class()
        died = False
        steps = 0
        found_gollum = False
        reached_doom = False
        died = False

        game.get_initial_input(variant, self.gollum_pos)

        try:
            while True:
                if game.should_stop():
                    final_command = game.get_final_command()
                    print(final_command)
                    sys.stdout.flush()
                    break
                    
                command = game.execute_move()

                 # Проверяем смерть (упрощенная проверка)
                if self._is_position_deadly(game.current_pos, game.ring_on, game.has_mithril):
                    died = True
                    break

                # Обрабатываем команду
                if command.startswith('m'):
                    parts = command.split()
                    if len(parts) >= 3:
                        new_x, new_y = int(parts[1]), int(parts[2])
                        game.current_pos = (new_x, new_y)
                        
                        # Обновляем перцепцию после движения
                        perception = self.get_perception(new_x, new_y, variant)
                        self._update_game_perception(game, perception)
                        
                elif command == 'r':
                    game.ring_on = True
                    # Обновляем перцепцию после смены кольца
                    perception = self.get_perception(game.current_pos[0], game.current_pos[1], variant)
                    self._update_game_perception(game, perception)
                    
                elif command == 'rr':
                    game.ring_on = False
                    # Обновляем перцепцию после смены кольца
                    perception = self.get_perception(game.current_pos[0], game.current_pos[1], variant)
                    self._update_game_perception(game, perception)
                    
                elif command.startswith('e'):
                    break
                
                if not command:
                    print("e -1")
                    break
                    
                should_exit = game.send_command(command)
                if should_exit:
                    break
                    
        except Exception as e:
            print("e -1")

        end_time = time.time()
        execution_time = end_time - start_time


        # Определяем результат
        if reached_doom:
            result = "win"
            # Вычисляем длину пути (упрощенно)
            path_length = steps
        elif died:
            result = "loss"
            path_length = 0
        else:
            result = "timeout"
            path_length = 0
            
        return {
            'result': result,
            'execution_time': execution_time,
            'steps': steps,
            'path_length': path_length,
            'found_gollum': found_gollum,
            'reached_doom': reached_doom
        }
    
    def _is_position_deadly(self, pos, ring_on, has_mithril):
        """Упрощенная проверка на смертельную позицию"""
        x, y = pos
        
        # Проверяем врагов в соседних клетках
        for enemy_pos in self.enemies:
            enemy_x, enemy_y = enemy_pos
            distance = abs(x - enemy_x) + abs(y - enemy_y)
            
            # Упрощенные правила смертельности
            if distance == 0:  # На той же клетке
                return True
            elif distance == 1 and not (ring_on or has_mithril):
                return True
                
        return False

class StatisticalAnalyzer:
    """Анализатор статистики"""
    
    def __init__(self):
        self.results = defaultdict(list)
        
    def add_result(self, algorithm: str, variant: int, result: Dict[str, Any]):
        """Добавляет результат теста"""
        key = f"{algorithm}_v{variant}"
        self.results[key].append(result)
    
    def calculate_statistics(self):
        """Вычисляет статистику по всем результатам"""
        stats = {}
        
        for key, results in self.results.items():
            execution_times = [r['execution_time'] for r in results]
            path_lengths = [r['path_length'] for r in results if r['path_length'] > 0]
            wins = sum(1 for r in results if r['result'] == 'win')
            losses = sum(1 for r in results if r['result'] == 'loss')
            timeouts = sum(1 for r in results if r['result'] == 'timeout')
            total = len(results)
            
            stats[key] = {
                'execution_time': {
                    'mean': statistics.mean(execution_times) if execution_times else 0,
                    'median': statistics.median(execution_times) if execution_times else 0,
                    'mode': statistics.mode(execution_times) if execution_times else 0,
                    'stdev': statistics.stdev(execution_times) if len(execution_times) > 1 else 0
                },
                'path_length': {
                    'mean': statistics.mean(path_lengths) if path_lengths else 0,
                    'median': statistics.median(path_lengths) if path_lengths else 0,
                    'mode': statistics.mode(path_lengths) if path_lengths else 0,
                    'stdev': statistics.stdev(path_lengths) if len(path_lengths) > 1 else 0
                },
                'wins': wins,
                'losses': losses,
                'timeouts': timeouts,
                'win_rate': (wins / total) * 100 if total > 0 else 0,
                'loss_rate': (losses / total) * 100 if total > 0 else 0,
                'timeout_rate': (timeouts / total) * 100 if total > 0 else 0,
                'total_tests': total
            }
            
        return stats
    
    def print_statistics(self):
        """Выводит статистику в читаемом формате"""
        stats = self.calculate_statistics()
        
        print("=" * 80)
        print("СТАТИСТИЧЕСКИЙ АНАЛИЗ АЛГОРИТМОВ")
        print("=" * 80)
        
        for key, data in stats.items():
            print(f"\n{key.upper()}")
            print("-" * 40)
            
            print(f"Всего тестов: {data['total_tests']}")
            print(f"Побед: {data['wins']} ({data['win_rate']:.2f}%)")
            print(f"Поражений: {data['losses']} ({data['loss_rate']:.2f}%)")
            print(f"Таймаутов: {data['timeouts']} ({data['timeout_rate']:.2f}%)")
            
            print(f"\nВремя выполнения (секунды):")
            print(f"  Среднее: {data['execution_time']['mean']:.4f}")
            print(f"  Медиана: {data['execution_time']['median']:.4f}")
            print(f"  Мода: {data['execution_time']['mode']:.4f}")
            print(f"  Стандартное отклонение: {data['execution_time']['stdev']:.4f}")
            
            if data['path_length']['mean'] > 0:
                print(f"\nДлина пути (шаги):")
                print(f"  Среднее: {data['path_length']['mean']:.2f}")
                print(f"  Медиана: {data['path_length']['median']:.2f}")
                print(f"  Мода: {data['path_length']['mode']:.2f}")
                print(f"  Стандартное отклонение: {data['path_length']['stdev']:.2f}")
        
        # Сравнительная статистика
        self._print_comparison(stats)
    
    def _print_comparison(self, stats):
        """Выводит сравнительную статистику"""
        print("\n" + "=" * 80)
        print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ")
        print("=" * 80)
        
        algorithms = ['A*', 'Backtracking']
        variants = [1, 2]
        
        for algo in algorithms:
            for var in variants:
                key = f"{algo}_v{var}"
                if key in stats:
                    data = stats[key]
                    print(f"\n{algo} (Вариант {var}):")
                    print(f"  Процент побед: {data['win_rate']:.2f}%")
                    print(f"  Среднее время: {data['execution_time']['mean']:.4f}с")
                    if data['path_length']['mean'] > 0:
                        print(f"  Средняя длина пути: {data['path_length']['mean']:.2f} шагов")

def main():
    """Основная функция для запуска статистического анализа"""
    num_tests = 10
    map_generator = MapGenerator(seed=42)  # Фиксируем seed для воспроизводимости
    analyzer = StatisticalAnalyzer()
    
    print(f"Запуск статистического анализа на {num_tests} тестовых картах...")
    print("Это может занять некоторое время...")
    
    for test_num in range(num_tests):
        if test_num % 100 == 0:
            print(f"Выполнено тестов: {test_num}/{num_tests}")
        
        # Генерируем карту
        grid, enemies, gollum_pos, doom_pos, mithril_pos = map_generator.generate_map()
        simulator = GameSimulator(grid, enemies, gollum_pos, doom_pos, mithril_pos)
        
        # Тестируем все комбинации алгоритмов и вариантов
        algorithms = [
            ('A*', AStarGame),
            ('Backtracking', BacktrackingGame)
        ]
        
        variants = [1, 2]
        
        for algo_name, algo_class in algorithms:
            for variant in variants:
                result = simulator.simulate_game(algo_class, variant)
                analyzer.add_result(algo_name, variant, result)
    
    # Выводим результаты
    analyzer.print_statistics()
    
    # Сохраняем сырые данные для дальнейшего анализа
    print(f"\nСырые данные сохранены для {num_tests} тестов.")

if __name__ == "__main__":
    main()