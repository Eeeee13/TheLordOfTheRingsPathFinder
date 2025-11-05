from typing import Set, Tuple, List, Dict, Optional
from objects.items import ItemType
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