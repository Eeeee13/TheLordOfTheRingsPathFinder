import sys
import heapq
from enum import Enum, auto
from typing import Set, Tuple, List, Dict, Optional

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
        
    def __repr__(self):
        return f"Cell({self.x},{self.y})"
        
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


class BacktrackNode:
    def __init__(self, x: int, y: int, ring_on: bool, mithril_on: bool, 
                 path: List[Tuple[int, int, bool]], depth: int):
        self.x = x
        self.y = y
        self.ring_on = ring_on
        self.mithril_on = mithril_on
        self.path = path  # list of (x, y, ring_state)
        self.depth = depth  # how many steps from start
        
    def __repr__(self):
        return f"BacktrackNode({self.x},{self.y},ring={self.ring_on},mithril={self.mithril_on},depth={self.depth})"
    
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

        self.action_history: List[Tuple[int, int, bool]] = []  # (x, y, ring_on)
        self.max_history = 10

        self.position_visit_count: Dict[Tuple[int, int], int] = {}
        self.stuck_counter = 0  # Count how long we've been stuck
        self.last_progress_step = 0  # Last step where we made progress
        self.forbidden_cells: Set[Tuple[int, int]] = set()  # Temporarily forbidden
        
    def initialize_maps(self):
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
        return self.map_with_ring if self.ring_on else self.map_no_ring

    def is_fully_safe_cell(self, x: int, y: int) -> bool:
        cell_no_ring = self.map_no_ring.get((x, y))
        cell_with_ring = self.map_with_ring.get((x, y))
        
        return (cell_no_ring and cell_no_ring.is_safe_no_ring is True and
                cell_with_ring and cell_with_ring.is_safe_with_ring is True)
                
    def send_command(self, command: str):
        """Send command to game and process response"""
        print(command)
        sys.stdout.flush()
        
        # Track action BEFORE executing
        self.action_history.append((self.current_pos[0], self.current_pos[1], self.ring_on))
        if len(self.action_history) > self.max_history:
            self.action_history.pop(0)
        
        if command.startswith('e'):
            return True
            
        if command.startswith('m'):
            parts = command.split()
            new_x, new_y = int(parts[1]), int(parts[2])
            self.update_after_move(new_x, new_y)
            self.steps_count += 1
            
        elif command == 'r':
            self.update_after_ring_toggle(True)
        elif command == 'rr':
            self.update_after_ring_toggle(False)
            
        self.read_and_update_perception()
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
        path_to_gollum = self.backtracking_search((0, 0), self.gollum_pos, False, self.has_mithril)
        if path_to_gollum:
            self.L1 = self.get_movement_count(path_to_gollum)
            
    def get_movement_count(self, path: List[Tuple[int, int, bool]]) -> int:
        count = 0
        for i in range(1, len(path)):
            # Count only position changes (moves)
            if path[i][0] != path[i-1][0] or path[i][1] != path[i-1][1]:
                count += 1
        return count
        
    def should_stop(self) -> bool:
        if self.found_gollum and self.current_pos == self.doom_pos:
            # Calculate L2 - path length from Gollum to Mount Doom
            if self.gollum_pos and self.doom_pos:
                path_to_doom = self.backtracking_search(self.gollum_pos, self.doom_pos, False, self.has_mithril)
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
            
    def backtracking_search(self, start: Tuple[int, int], goal: Tuple[int, int], 
                  start_ring: bool, start_mithril: bool, 
                  max_depth: int = 100) -> Optional[List[Tuple[int, int, bool]]]:
        """Find path from start to goal using backtracking with cycle prevention"""
        start_node = BacktrackNode(
            start[0], start[1], start_ring, start_mithril,
            [(start[0], start[1], start_ring)], 0
        )
        
        visited = set()
        stack = [start_node]
        
        while stack:
            node = stack.pop()
            
            # State key for detecting cycles
            state_key = (node.x, node.y, node.ring_on, node.mithril_on)
            
            # Already visited this state? Skip
            if state_key in visited:
                continue
            visited.add(state_key)
            
            # Reached goal? Return the path
            if (node.x, node.y) == goal:
                return node.path
            
            # Too deep? Skip    
            if node.depth >= max_depth:
                continue
            
            successors = []
            
            # Get last few positions to detect position cycles
            recent_positions = []
            if len(node.path) >= 4:
                recent_positions = [(x, y) for x, y, _ in node.path[-4:]]
            
            # Try moving to neighbors
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = node.x + dx, node.y + dy
                
                # Out of bounds? Skip
                if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                    continue
                
                # Check if this creates a position cycle (A->B->A pattern)
                if len(node.path) >= 2:
                    prev_x, prev_y, _ = node.path[-1]
                    if len(node.path) >= 3:
                        prev_prev_x, prev_prev_y, _ = node.path[-2]
                        # Prevent immediate back-and-forth: A->B->A
                        if (nx, ny) == (prev_prev_x, prev_prev_y):
                            continue
                
                # Check for longer cycles (visiting same position too often)
                if recent_positions.count((nx, ny)) >= 2:
                    continue
                
                # Check if already visited this state
                next_state = (nx, ny, node.ring_on, node.mithril_on)
                if next_state in visited:
                    continue
                
                # Safe to move there?
                if self._is_cell_safe(nx, ny, node.ring_on, node.mithril_on):
                    # Check if we pick up mithril
                    new_mithril = node.mithril_on
                    cell_info = self.map_no_ring.get((nx, ny)) or self.map_with_ring.get((nx, ny))
                    if cell_info and cell_info.cell_type == ItemType.C:
                        new_mithril = True
                    
                    new_path = node.path + [(nx, ny, node.ring_on)]
                    new_node = BacktrackNode(nx, ny, node.ring_on, new_mithril, new_path, node.depth + 1)
                    successors.append(new_node)
            
            # Try toggling ring - with VERY strict anti-cycle checks
            can_toggle = self._can_toggle_ring(node.x, node.y, node.ring_on, node.mithril_on)
            
            if can_toggle:
                # Don't toggle if we JUST toggled (last action was a toggle at same position)
                allow_toggle = True
                if len(node.path) >= 2:
                    last_x, last_y, last_ring = node.path[-1]
                    prev_x, prev_y, prev_ring = node.path[-2]
                    # If both last positions are the same and ring changed, we just toggled
                    if (last_x, last_y) == (node.x, node.y) == (prev_x, prev_y) and last_ring != prev_ring:
                        allow_toggle = False
                
                # Also count total toggles at this position
                if allow_toggle:
                    toggles_at_position = 0
                    for i in range(len(node.path) - 1, max(-1, len(node.path) - 8), -1):
                        if i > 0:
                            x, y, ring_state = node.path[i]
                            prev_x, prev_y, prev_ring = node.path[i-1]
                            if (x, y) == (node.x, node.y) == (prev_x, prev_y) and ring_state != prev_ring:
                                toggles_at_position += 1
                    
                    # Max 1 toggle at any position
                    if toggles_at_position >= 1:
                        allow_toggle = False
                
                if allow_toggle:
                    new_ring_state = not node.ring_on
                    toggle_state = (node.x, node.y, new_ring_state, node.mithril_on)
                    
                    if toggle_state not in visited:
                        new_path = node.path + [(node.x, node.y, new_ring_state)]
                        new_node = BacktrackNode(node.x, node.y, new_ring_state, node.mithril_on, 
                                                new_path, node.depth + 1)
                        successors.append(new_node)
            
            # Add successors to stack, prioritizing those closer to goal
            if successors:
                successors.sort(key=lambda n: manhattan_distance((n.x, n.y), goal), reverse=True)
                stack.extend(successors)
        
        return None

    def _is_cell_safe(self, x: int, y: int, ring_on: bool, mithril_on: bool) -> bool:
        """Проверяем безопасность клетки с учетом известных врагов и состояния"""
        # Сначала проверяем явные флаги безопасности из карты
        current_map = self.map_with_ring if ring_on else self.map_no_ring
        cell = current_map.get((x, y))
        
        if cell:
            if ring_on and cell.is_safe_with_ring is not None:
                return cell.is_safe_with_ring
            elif not ring_on and cell.is_safe_no_ring is not None:
                return cell.is_safe_no_ring
        
        # Если нет явной информации, проверяем против известных врагов
        for enemy in self.known_enemies:
            lethal_zone = enemy.calculate_lethal_zone(ring_on, mithril_on)
            if (x, y) in lethal_zone:
                return False
        
        # Если нет информации о врагах и нет явных флагов, считаем безопасной
        # Это позволяет исследовать неизвестные территории
        return True
        
    def _can_toggle_ring(self, x: int, y: int, current_ring: bool, mithril: bool) -> bool:
        # check if we can safely toggle ring at this position
        new_ring_state = not current_ring
        return self._is_cell_safe(x, y, new_ring_state, mithril)

    def _get_current_goal(self) -> Tuple[int, int]:
        if self.found_gollum and self.doom_pos:
            return self.doom_pos
        else:
            return self.gollum_pos
            
    def update_after_move(self, new_x: int, new_y: int):
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
        self.ring_on = new_ring_state
        
        # Mark current cell as visited with new ring state
        current_map = self.get_current_map()
        cell = current_map.get(self.current_pos)
        if cell:
            if self.ring_on:
                cell.visited_with_ring = True
            else:
                cell.visited_no_ring = True

    def _recently_toggled_here(self) -> bool:
        """Check if we recently toggled ring at current position"""
        if len(self.action_history) < 2:
            return False
        
        count = 0
        for i in range(len(self.action_history) - 1, max(-1, len(self.action_history) - 5), -1):
            x, y, ring = self.action_history[i]
            if (x, y) == self.current_pos:
                if i > 0:
                    prev_x, prev_y, prev_ring = self.action_history[i-1]
                    if (prev_x, prev_y) == self.current_pos and ring != prev_ring:
                        count += 1
        
        return count >= 1
    

    def _explore_new_direction(self) -> str:
        """Try to move to a safe, preferably unvisited cell"""
        goal = self._get_current_goal()
        best_move = None
        best_score = float('inf')
        
        # Get recently visited positions to avoid them
        recent_positions = set()
        if len(self.action_history) >= 5:
            recent_positions = {(x, y) for x, y, _ in self.action_history[-5:]}
        
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = self.current_pos[0] + dx, self.current_pos[1] + dy
            
            if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                continue
            
            if not self._is_cell_safe(nx, ny, self.ring_on, self.has_mithril):
                continue
            
            # Calculate score (lower is better)
            current_map = self.get_current_map()
            cell = current_map.get((nx, ny))
            
            is_visited = False
            if cell:
                is_visited = cell.visited_with_ring if self.ring_on else cell.visited_no_ring
            
            dist_to_goal = manhattan_distance((nx, ny), goal)
            
            # Score: prefer unvisited, not recently visited, closer to goal
            score = dist_to_goal
            if is_visited:
                score += 50
            if (nx, ny) in recent_positions:
                score += 100
            
            if score < best_score:
                best_score = score
                best_move = f"m {nx} {ny}"
        
        if best_move:
            return best_move
        
        # If all else fails, try toggling ring if safe
        if self._can_toggle_ring(self.current_pos[0], self.current_pos[1], 
                                  self.ring_on, self.has_mithril):
            if not self._recently_toggled_here():
                return "r" if not self.ring_on else "rr"
        
        return "e -1"

    def _is_in_cycle(self) -> bool:
        """Check if we're repeating the same states"""
        if len(self.action_history) < 4:
            return False
        
        # Check for A->B->A->B pattern (oscillation)
        recent = self.action_history[-4:]
        if (recent[0] == recent[2] and recent[1] == recent[3] and 
            recent[0] != recent[1]):
            return True
        
        # Check if we're stuck at same position with ring toggles
        if len(self.action_history) >= 3:
            last_3 = self.action_history[-3:]
            positions = [(x, y) for x, y, _ in last_3]
            if len(set(positions)) == 1:  # Same position
                ring_states = [r for _, _, r in last_3]
                if len(set(ring_states)) > 1:  # Ring toggling
                    return True
        
        return False
                
        
    def execute_move(self) -> Optional[str]:
        """Figure out what to do next and return the command"""
        try:
            goal = self._get_current_goal()
            if goal is None:
                return "e -1"
            
            # Already at goal
            if self.current_pos == goal:
                if not self.found_gollum:
                    return None
                else:
                    return None
            
            # Check if we're stuck in a cycle
            if self._is_in_cycle():
                # Try to break out of cycle by exploring
                return self._explore_new_direction()
            
            # Find path to goal with current ring state
            path = self.backtracking_search(
                self.current_pos, goal, 
                self.ring_on, self.has_mithril,
                max_depth=50
            )
            
            # If no path found, try with ring toggled
            if not path or len(path) < 2:
                # But don't toggle if we just toggled recently at this position
                if not self._recently_toggled_here():
                    alt_path = self.backtracking_search(
                        self.current_pos, goal, 
                        not self.ring_on, self.has_mithril,
                        max_depth=50
                    )
                    if alt_path and len(alt_path) > 2:
                        return "r" if not self.ring_on else "rr"
                
                # Try to explore
                return self._explore_new_direction()
            
            # Take next step from path
            next_x, next_y, next_ring_state = path[1]
            
            if next_ring_state != self.ring_on:
                return "r" if next_ring_state else "rr"
            else:
                return f"m {next_x} {next_y}"
                
        except Exception as e:
            return "e -1"

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