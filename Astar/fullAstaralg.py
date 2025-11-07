import sys
from typing import Set, Tuple, List, Dict, Optional
import heapq

from enum import Enum, auto

class ItemType(Enum):
    F = auto()  # Frodo
    O = auto()  # OrcPatrol
    U = auto()  # UrukHai  
    N = auto()  # Nazgul
    W = auto()  # MordorWatchtower
    G = auto()  # Gollum
    R = auto()  # The One Ring
    C = auto()  # Mithril Mail-coat
    M = auto()  # Mount Doom
    P = auto()  # Perception zone

class Cell:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.cell_type: Optional[ItemType] = None
        self.is_safe_no_ring: Optional[bool] = None  
        self.is_safe_with_ring: Optional[bool] = None  
        self.visited_no_ring: bool = False
        self.visited_with_ring: bool = False
        self.enemy_type: Optional[ItemType] = None
        
    def update_from_perception(self, item_type: ItemType, ring_on: bool, has_mithril: bool):
        """Update cell info based on what we see"""
        if ring_on:
            self.visited_with_ring = True
        else:
            self.visited_no_ring = True
            
        # Handle different object types
        if item_type in [ItemType.O, ItemType.U, ItemType.N, ItemType.W, ItemType.P]:
            # Enemy or danger zone - cell is dangerous
            if ring_on:
                self.is_safe_with_ring = False
            else:
                self.is_safe_no_ring = False
                
            if item_type != ItemType.P:  # Remember enemy type
                self.enemy_type = item_type
                self.cell_type = item_type
                
        elif item_type in [ItemType.G, ItemType.C, ItemType.M]:
            # Safe objects
            if ring_on:
                self.is_safe_with_ring = True
            else:
                self.is_safe_no_ring = True
            self.cell_type = item_type
            
        elif item_type == ItemType.R:
            # Ring - safe but special case
            if ring_on:
                self.is_safe_with_ring = True
            else:
                self.is_safe_no_ring = True
            self.cell_type = item_type


class AStarNode:
    def __init__(self, x: int, y: int, ring_on: bool, g: int, h: int, 
                 parent: Optional['AStarNode'] = None):
        self.x = x
        self.y = y
        self.ring_on = ring_on  # is ring on at this node
        self.g = g  # cost from start
        self.h = h  # estimated cost to goal
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

def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

class Enemy:
    def __init__(self, enemy_type: ItemType, x: int, y: int):
        self.enemy_type = enemy_type
        self.position = (x, y)
        
    def calculate_lethal_zone(self, ring_on: bool, has_mithril: bool) -> Set[Tuple[int, int]]:
        """Calculate danger zone based on ring and armor state"""
        x, y = self.position
        lethal_zone = set()
        
        if self.enemy_type == ItemType.O:  # Orc Patrol
            if ring_on or has_mithril:
                # Only their own cell is dangerous
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
        
        # Filter out cells outside the map
        filtered_zone = set()
        for cell_x, cell_y in lethal_zone:
            if 0 <= cell_x <= 12 and 0 <= cell_y <= 12:
                filtered_zone.add((cell_x, cell_y))
                
        return filtered_zone
    


class RingDestroyerGame:
    def __init__(self):
        self.grid_size = 13
        self.initialize_maps()
        self.current_pos = (0, 0)
        self.ring_on = False
        self.has_mithril = False
        self.gollum_pos: Optional[Tuple[int, int]] = None
        self.doom_pos: Optional[Tuple[int, int]] = None
        self.found_gollum = False
        self.perception_radius = 1
        self.known_enemies: List[Enemy] = []
        self.steps_count = 0
        self.L1 = 0  # path length to Gollum
        self.L2 = 0  # path length from Gollum to Mount Doom
        
    def initialize_maps(self):
        """Set up both maps (with ring and without)"""
        self.map_no_ring: Dict[Tuple[int, int], Cell] = {}
        self.map_with_ring: Dict[Tuple[int, int], Cell] = {}
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                self.map_no_ring[(x, y)] = Cell(x, y)
                self.map_with_ring[(x, y)] = Cell(x, y)
        
        # Starting position is safe and visited without ring
        start_cell_no_ring = self.map_no_ring[(0, 0)]
        start_cell_no_ring.is_safe_no_ring = True
        start_cell_no_ring.visited_no_ring = True
        
        start_cell_with_ring = self.map_with_ring[(0, 0)]
        start_cell_with_ring.is_safe_with_ring = None  
        
    def process_initial_input(self):
        """Process initial input from the game"""
        # Read perception variant
        variant = int(sys.stdin.readline().strip())
        self.perception_radius = variant
        
        # Read Gollum position
        gollum_line = sys.stdin.readline().split()
        self.gollum_pos = (int(gollum_line[0]), int(gollum_line[1]))
        
        # Read initial perception for (0,0)
        self.read_and_update_perception()

    def get_initial_input(self, variant, gollum_line):
        self.perception_radius = variant
        self.gollum_pos = (int(gollum_line[0]), int(gollum_line[1]))
        self.read_and_update_perception()

    def read_and_update_perception(self):
        """Read perception data from game and update map"""
        n_line = sys.stdin.readline().strip()
        if not n_line:
            return
            
        n = int(n_line)
        perception_data = []
        
        for _ in range(n):
            line = sys.stdin.readline().split()
            if len(line) < 3:
                continue
                
            x = int(line[0])
            y = int(line[1])
            item_str = line[2]
            
            # Convert string to ItemType
            try:
                item_type = ItemType[item_str]
                perception_data.append((x, y, item_type))
            except KeyError:
                # Skip unknown types
                continue
                
        self.update_from_perception(perception_data)

      
        
    def update_from_perception(self, perception_data: List[Tuple[int, int, ItemType]]):
        """Update maps based on what we see"""
        current_map = self.map_with_ring if self.ring_on else self.map_no_ring
        
        # First mark all perceived cells
        for x, y, item_type in perception_data:
            cell = current_map.get((x, y))
            if cell:
                cell.update_from_perception(item_type, self.ring_on, self.has_mithril)
                
                # If we found an enemy - add to known_enemies
                if item_type in [ItemType.O, ItemType.U, ItemType.N, ItemType.W]:
                    enemy = Enemy(item_type, x, y)
                    if not any(e.position == (x, y) for e in self.known_enemies):
                        self.known_enemies.append(enemy)
                
                # If we found Gollum - remember position
                elif item_type == ItemType.G:
                    self.gollum_pos = (x, y)
                
                # If we found armor - activate it
                elif item_type == ItemType.C:
                    self.has_mithril = True

    def get_current_map(self) -> Dict[Tuple[int, int], Cell]:
        """Get current map based on ring state"""
        return self.map_with_ring if self.ring_on else self.map_no_ring

    def is_fully_safe_cell(self, x: int, y: int) -> bool:
        """Check if cell is safe both with and without ring"""
        cell_no_ring = self.map_no_ring.get((x, y))
        cell_with_ring = self.map_with_ring.get((x, y))
        
        return (cell_no_ring and cell_no_ring.is_safe_no_ring is True and
                cell_with_ring and cell_with_ring.is_safe_with_ring is True)
                
    def send_command(self, command: str):
        """Send command to game and process response"""
        print(command)
        sys.stdout.flush()
        
        if command.startswith('e'):
            return True  # End execution
            
        # Process game response
        if command.startswith('m'):
            # Update position after move
            parts = command.split()
            new_x, new_y = int(parts[1]), int(parts[2])
            self.update_after_move(new_x, new_y)
            self.steps_count += 1
            
        elif command == 'r':
            self.update_after_ring_toggle(True)
        elif command == 'rr':
            self.update_after_ring_toggle(False)
            
        # Read updated perception
        self.read_and_update_perception()
        
        # Check special events
        self.check_special_events()
        
        return False
        
    def check_special_events(self):
        # Check if we found Gollum
        if (not self.found_gollum and self.current_pos == self.gollum_pos):
            self.handle_gollum_found()
            
        # Check if we picked up armor
        current_map = self.get_current_map()
        cell = current_map.get(self.current_pos)
        if cell and cell.cell_type == ItemType.C:
            self.has_mithril = True
            # Mark armor as picked up (remove from map)
            cell.cell_type = None
            
    def handle_gollum_found(self):
        self.found_gollum = True
        
        # Read message about Mount Doom position
        line = sys.stdin.readline().strip()
        parts = line.split()
        doom_x = int(parts[0])
        doom_y = int(parts[1])
        self.doom_pos = (doom_x, doom_y)
                
        # Calculate L1 - path length to Gollum
        path_to_gollum = self.a_star_search((0, 0), self.gollum_pos, False)
        if path_to_gollum:
            self.L1 = self.get_movement_count(path_to_gollum)
            
    def get_movement_count(self, path: List[Tuple[int, int, bool]]) -> int:
        """Count number of moves in path (excluding ring toggles)"""
        count = 0
        for i in range(1, len(path)):
            # Count only position changes (moves)
            if path[i][0] != path[i-1][0] or path[i][1] != path[i-1][1]:
                count += 1
        return count
        
    def should_stop(self) -> bool:
        """Check if we reached the final goal"""
        if self.found_gollum and self.current_pos == self.doom_pos:
            # Calculate L2 - path length from Gollum to Mount Doom
            if self.gollum_pos and self.doom_pos:
                path_to_doom = self.a_star_search(self.gollum_pos, self.doom_pos, False)
                if path_to_doom:
                    self.L2 = self.get_movement_count(path_to_doom)
            return True
        return False
        
    def get_final_command(self) -> str:
        """Get final command to finish the game"""
        if self.found_gollum and self.doom_pos and self.current_pos == self.doom_pos:
            total_length = self.L1 + self.L2
            return f"e {total_length}"
        else:
            return "e -1"
            
    def a_star_search(self, start: Tuple[int, int], goal: Tuple[int, int], 
                     start_ring: bool) -> Optional[List[Tuple[int, int, bool]]]:
        """
        A* pathfinding from start to goal with initial ring state start_ring
        Returns list of (x, y, ring_state) or None if no path found
        """
        open_set = []
        closed_set = set()
        
        # Starting node
        start_node = AStarNode(start[0], start[1], start_ring, 0, 
                              manhattan_distance(start, goal))
        heapq.heappush(open_set, start_node)
        
        while open_set:
            current = heapq.heappop(open_set)
            
            # Check if we reached the goal
            if (current.x, current.y) == goal:
                return self._reconstruct_path(current)
                
            closed_set.add((current.x, current.y, current.ring_on))
            
            # Generate neighbor nodes
            neighbors = self._get_neighbors(current)
            
            for neighbor in neighbors:
                if (neighbor.x, neighbor.y, neighbor.ring_on) in closed_set:
                    continue
                    
                # Check if there's a node in open_set with lower cost
                found_better = False
                for node in open_set:
                    if (node.x, node.y, node.ring_on) == (neighbor.x, neighbor.y, neighbor.ring_on):
                        if node.g <= neighbor.g:
                            found_better = True
                            break
                        # Replace node with cheaper one
                        open_set.remove(node)
                        heapq.heapify(open_set)
                        break
                
                if not found_better:
                    heapq.heappush(open_set, neighbor)
                    
        return None  # No path found
        
    def _reconstruct_path(self, node: AStarNode) -> List[Tuple[int, int, bool]]:
        """Reconstruct path from end node to start"""
        path = []
        current = node
        while current:
            path.append((current.x, current.y, current.ring_on))
            current = current.parent
        return path[::-1]  # Reverse path (from start to end)
        
    def _get_neighbors(self, node: AStarNode) -> List[AStarNode]:
        """Generate all possible neighbor nodes"""
        neighbors = []
        x, y, ring_on = node.x, node.y, node.ring_on
        
        # Move to neighboring cells (orthogonally)
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            
            # Check map boundaries
            if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                continue
                
            # Check if cell is safe in current ring state
            current_map = self.map_with_ring if ring_on else self.map_no_ring
            cell = current_map.get((nx, ny))
            
            if cell and self._is_cell_safe(cell.x, cell.y, ring_on):
                # Cell is safe - can move
                g_new = node.g + 1  # Move cost = 1
                h_new = manhattan_distance((nx, ny), self._get_current_goal())
                neighbor = AStarNode(nx, ny, ring_on, g_new, h_new, node)
                neighbors.append(neighbor)
        
        # Toggle ring (only in fully safe cells)
        if self.is_fully_safe_cell(x, y):
            new_ring_state = not ring_on
            g_new = node.g  # Ring toggle has no move cost
            h_new = manhattan_distance((x, y), self._get_current_goal())
            switch_node = AStarNode(x, y, new_ring_state, g_new, h_new, node)
            neighbors.append(switch_node)
            
        return neighbors
        
    def _is_cell_safe(self, x: int, y: int, ring_on: bool) -> bool:
        """Check if cell is safe considering known enemies"""
        # Check explicit safety flags
        current_map = self.map_with_ring if ring_on else self.map_no_ring
        cell = current_map.get((x, y))
        
        if cell:
            if ring_on and cell.is_safe_with_ring is not None:
                return cell.is_safe_with_ring
            elif not ring_on and cell.is_safe_no_ring is not None:
                return cell.is_safe_no_ring
        
        # If no explicit info, check against known enemies
        for enemy in self.known_enemies:
            lethal_zone = enemy.calculate_lethal_zone(ring_on, self.has_mithril)
            if (x, y) in lethal_zone:
                return False
        
        # If no enemy info and no explicit flags, consider safe
        return True
            
    def _get_current_goal(self) -> Tuple[int, int]:
        """Get current goal (Gollum or Mount Doom)"""
        if self.found_gollum and self.doom_pos:
            return self.doom_pos
        else:
            return self.gollum_pos
            
    def update_after_move(self, new_x: int, new_y: int):
        """Update state after moving"""
        self.current_pos = (new_x, new_y)
        
        # Mark cell as visited
        current_map = self.get_current_map()
        cell = current_map.get(self.current_pos)
        if cell:
            if self.ring_on:
                cell.visited_with_ring = True
            else:
                cell.visited_no_ring = True
                
    def update_after_ring_toggle(self, new_ring_state: bool):
        """Update state after toggling ring"""
        self.ring_on = new_ring_state
        
        # Mark current cell as visited with new ring state
        current_map = self.get_current_map()
        cell = current_map.get(self.current_pos)
        if cell:
            if self.ring_on:
                cell.visited_with_ring = True
            else:
                cell.visited_no_ring = True
                
    def execute_move(self) -> Optional[str]:
        """Execute one move based on A* search"""
        if self.should_stop():
            return None
            
        goal = self._get_current_goal()
        if goal is None:
            return "e -1"
        
        # If we're already at goal cell
        if self.current_pos == goal:
            if not self.found_gollum:
                # Activate Gollum search
                return None
            else:
                # Reached Mount Doom
                return None
                
        # Find path to current goal
        path = self.a_star_search(self.current_pos, goal, self.ring_on)
        
        if not path or len(path) < 2:
            # print(f"DEBUG: No path found from {self.current_pos} to {goal}", file=sys.stderr)
            return "e -1"
            
        # Take next step from path
        next_x, next_y, next_ring_state = path[1]
        
        # Check if we need to toggle ring
        if next_ring_state != self.ring_on:
            return "r" if next_ring_state else "rr"
        else:
            return f"m {next_x} {next_y}"


def main():
    game = RingDestroyerGame()
    
    # Process initial input
    game.process_initial_input()
    
    # Main game loop
    try:
        while True:
            # Check if we reached the goal
            if game.should_stop():
                final_command = game.get_final_command()
                print(final_command)
                sys.stdout.flush()
                break
                
            # Get next command using A*
            command = game.execute_move()
            
            if not command:
                # If no path found
                print("e -1")
                sys.stdout.flush()
                break
                
            # Send command and check for exit
            should_exit = game.send_command(command)
            if should_exit:
                break
                
    except Exception as e:
        # In case of error send exit command
        print(f"{e}: e -1")
        sys.stdout.flush()
        
if __name__ == "__main__":
    main()