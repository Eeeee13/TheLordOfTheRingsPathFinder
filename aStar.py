from typing import Set, Tuple, List, Dict, Optional

class AStarNode:
    def __init__(self, position: Tuple[int, int], g: int, h: int, 
                 parent: Optional['AStarNode'] = None):
        self.position = position
        self.g = g  # стоимость от начала
        self.h = h  # эвристическая стоимость до цели
        self.parent = parent
        
    @property
    def f(self) -> int:
        return self.g + self.h