import heapq
from typing import List, Tuple, Optional, Dict

class AStarNode:
    def __init__(self, x: int, y: int, ring_on: bool, g: int, h: int, 
                 parent: Optional['AStarNode'] = None):
        self.x = x
        self.y = y
        self.ring_on = ring_on  # состояние кольца в этом узле
        self.g = g  # стоимость от начала
        self.h = h  # эвристическая стоимость до цели
        self.parent = parent
        
    @property
    def f(self) -> int:
        return self.g + self.h
        
    def __lt__(self, other):
        return self.f < other.f
        
    def __eq__(self, other):
        return (self.x, self.y, self.ring_on) == (other.x, other.y, other.ring_on)
        
    def __hash__(self):
        return hash((self.x, self.y, self.ring_on))
        
    def __repr__(self):
        return f"Node({self.x},{self.y},{'ring' if self.ring_on else 'no_ring'}) f={self.f}"