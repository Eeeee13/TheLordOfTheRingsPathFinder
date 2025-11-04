# Генерация одной карты
from map import MapGenerator


generator = MapGenerator(seed=42)
grid, enemies, gollum_pos, mount_doom_pos, mithril_pos = generator.generate_map()

# Генерация 1000 карт для статистики
maps = []
for i in range(1000):
    map_data = generator.generate_map()
    maps.append(map_data)

print(maps[0])