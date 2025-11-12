# The Lord of the Rings: Interactive Pathfinding Simulator

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Pygame](https://img.shields.io/badge/Pygame-2.0+-green.svg)
![AI](https://img.shields.io/badge/AI-Pathfinding-orange.svg)

An interactive simulation and visualization of pathfinding algorithms in a Middle-earth inspired environment. Watch as Frodo navigates through dangerous territories while avoiding Nazgûl and other obstacles using different AI algorithms.

<img src="https://github.com/Eeeee13/TheLordOfTheRingsPathFinder/blob/main/demoVideo.gif" width="500">

## 🧭 Overview

This project simulates the journey of Frodo Baggins from The Shire to Mount Doom, implementing various pathfinding algorithms to navigate through a dynamic environment filled with enemies, obstacles, and hidden dangers. The simulation provides real-time visualization of algorithm decision-making processes.

## ✨ Features

### 🎮 Interactive Simulation
- **Real-time Visualization**: Watch algorithms in action with Pygame-based graphics
- **Dynamic Environment**: Randomly generated maps with enemies, obstacles, and the One Ring
- **Multiple Algorithms**: Compare A* and Backtracking algorithms
- **Adaptive Difficulty**: Enemies with perception systems that can detect the player

### 🎯 Algorithm Implementation
- **A* Algorithm**: Optimal pathfinding with heuristic optimization
- **Backtracking**: Exhaustive search algorithm for complete path exploration
- **Perception Systems**: Enemy AI that can detect and track the player
- **Fog of War**: Limited visibility options for increased challenge

### 🎨 User Interface
- **Adaptive Window**: Resizable interface that scales to any screen size
- **Control Panel**: Interactive speed controls and algorithm selection
- **Real-time Stats**: Live tracking of moves, enemies detected, and game state
- **Visual Feedback**: Color-coded perception zones and danger areas

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Pygame 2.0+

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/lotr-pathfinding-simulator.git
cd lotr-pathfinding-simulator
```

2. **Install dependencies**
```bash
pip install pygame
```

3. **Run the simulation**
```bash
python interactor3000.py
```

## 🎮 How to Use

### Basic Controls
- **Enter/Auto**: Run automatic pathfinding with current algorithm
- **Ctrl+R**: Generate new random map
- **A**: Switch to A* algorithm
- **B**: Switch to Backtracking algorithm  
- **F**: Toggle fog of war
- **ESC**: Exit the simulation

### UI Controls
- **Speed Slider**: Adjust simulation speed (1-10)
- **Algorithm Buttons**: Switch between A* and Backtracking
- **Fog of War Checkbox**: Enable/disable limited visibility

### Command Line Interface
```bash
# Auto-solve with current algorithm
auto

# Move to specific coordinates
m x y

# Put on or take off the Ring
ring
unring

# Calculate path to current goal
path

# Run automated tests
test 10

# Run comparative analysis
compare 5

# Exit the program
exit
```

## 🔧 Technical Details

### Algorithms

#### A* Algorithm
- **Heuristic**: Manhattan distance with enemy avoidance
- **Optimization**: Priority queue for efficient node selection
- **Features**: Dynamic weight adjustment based on danger levels

#### Backtracking Algorithm
- **Approach**: Depth-first search with pruning
- **Optimization**: Memoization and early termination
- **Use Case**: Guaranteed path finding in complex environments

### Entity System

- **Agent**: Player-controlled character with inventory system
- **Enemies**: NPCs with perception zones and pursuit behavior  
- **Objects**: Interactive items with special properties
- **Environment**: Grid-based world with dynamic obstacles

### Visualization Engine

- **Real-time Rendering**: Pygame-based graphics with smooth animations
- **Adaptive Scaling**: UI elements that adjust to window size
- **Color Coding**: Visual representation of algorithm states and danger zones
- **Information Display**: Live statistics and algorithm metrics

## 📊 Performance Analysis

The simulator includes built-in performance tracking for:
- Pathfinding efficiency comparison
- Algorithm success rates
- Average completion times
- Memory usage statistics
- Enemy detection and avoidance rates

## 🎯 Use Cases

### 🎓 Educational
- **Computer Science**: Algorithm visualization and analysis
- **AI Courses**: Pathfinding algorithm implementation examples
- **Game Development**: AI behavior and perception systems

### 🔬 Research
- **Algorithm Comparison**: A* vs Backtracking performance analysis
- **AI Behavior**: Enemy perception and pursuit patterns
- **Optimization**: Heuristic function effectiveness

### 🎮 Entertainment
- **Interactive Demo**: Engaging visualization of classic algorithms
- **Game Prototyping**: Foundation for turn-based strategy games
- **AI Demonstration**: Showcase of intelligent agent behavior

## 🛠️ Customization

### Adding New Algorithms
1. Implement the algorithm in a new file
2. Add algorithm selection in the control panel
3. Update the command processor for new commands

### Modifying Game Rules
- Adjust enemy perception ranges in `enemy.py`
- Modify movement costs in the pathfinding algorithms
- Change victory conditions in the game state manager

### Visual Customization
- Modify color schemes in the drawing functions
- Add new entity types with custom sprites
- Adjust UI layout and controls

## 🤝 Contributing

We welcome contributions! Please feel free to submit pull requests for:
- New pathfinding algorithms
- Improved visualization features
- Additional game entities and behaviors
- Performance optimizations
- Bug fixes and documentation

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by J.R.R. Tolkien's "The Lord of the Rings"
- Built with Pygame community resources
- Algorithm implementations based on classic computer science principles

---

**Ready to begin your journey to Mount Doom? Run the simulator and may the algorithms be with you!** 🧙‍♂️💍
