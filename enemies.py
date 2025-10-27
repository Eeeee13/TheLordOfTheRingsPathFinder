from typing import Set, Tuple, List, Dict, Optional
from items import Items

class Enemy:
    def __init__(self, enemy_type: Items, x: int, y: int):
        self.enemy_type = enemy_type
        self.position = (x, y)
        self.lethal_zone_no_ring: Set[Tuple[int, int]] = set()
        self.lethal_zone_with_ring: Set[Tuple[int, int]] = set()
        
    def calculate_zones(self, has_mithril: bool):
        # Вычисляем зоны поражения в зависимости от типа врага и наличия колечка/кольчуги
        # Реализуем логику из задания
        pass