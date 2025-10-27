from typing import Set, Tuple, List, Dict, Optional
from items import ItemType

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