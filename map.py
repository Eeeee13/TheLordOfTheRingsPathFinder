import numpy as np
from items import ItemType

class Map:
    def __init__(self):
        self.mapa = np.zeros((13,13))
        self.mapa[0][0] = ItemType.F.value

    def get_free_cell(self):
        while True:
            x = np.random.randint()
            y = np.random.randint()
            if self.mapa[x][y] == 0:
                return (x, y)
        
    def random_place_Person(self, persona: ItemType.value):
        x, y = self.get_free_cell()
        self.mapa[x][y] = persona

    def place_Person(self, cell:tuple, persona: ItemType.value):
        x, y = cell
        self.mapa[x][y] = persona

    def look(self, sell:tuple, Person):
        pass