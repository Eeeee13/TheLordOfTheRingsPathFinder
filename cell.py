from typing import Set, Tuple, List, Dict, Optional
from items import ItemType

class Cell:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.cell_type: Optional[ItemType] = None  # Основной тип объекта в клетке
        self.is_safe_no_ring: Optional[bool] = None  # None - неизвестно, True/False - известно
        self.is_safe_with_ring: Optional[bool] = None  # None - неизвестно, True/False - известно
        self.visited_no_ring: bool = False  # Посещена без кольца
        self.visited_with_ring: bool = False  # Посещена с кольцом
        self.enemy_type: Optional[ItemType] = None  # Если есть враг (O, U, N, W)
        
    def __repr__(self):
        return f"Cell({self.x},{self.y})"
    
    def update_from_perception(self, item_type: ItemType, ring_on: bool, has_mithril: bool):
        """Обновляем информацию о клетке на основе перцепции"""
        if ring_on:
            self.visited_with_ring = True
        else:
            self.visited_no_ring = True
            
        # Если видим врага или зону поражения - клетка опасна
        if item_type in [ItemType.O, ItemType.U, ItemType.N, ItemType.W, ItemType.P]:
            if ring_on:
                self.is_safe_with_ring = False
            else:
                self.is_safe_no_ring = False
                
            # Запоминаем тип врага, если это не зона восприятия
            if item_type != ItemType.P:
                self.enemy_type = item_type
                self.cell_type = item_type
        else:
            # Если видим безопасный объект
            if ring_on:
                self.is_safe_with_ring = True
            else:
                self.is_safe_no_ring = True
            self.cell_type = item_type