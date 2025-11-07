import numpy as np
from objects.enemies import Enemy
from objects.items import ItemType
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

class MapGenerator:
    def __init__(self, seed=None):
        self.size = 13
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