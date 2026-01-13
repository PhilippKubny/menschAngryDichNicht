# Mensch ärgere dich nicht (Ludo) MCTS Agent

A lightweight Python implementation of a two-player *Mensch ärgere dich nicht* (Ludo) game state plus a Monte Carlo Tree Search (MCTS) agent that selects the best move for the Blue player. The project focuses on core game logic (token positions, captures, finish lanes) and a self-contained MCTS rollout policy so you can simulate decision making without any external dependencies.

## :books: Table of Contents
:bulb: [About the project](#bulb-about-the-project)  
:rocket: [Quickstart](#rocket-quickstart)  
:wrench: [Usage](#wrench-usage)  
:link: [Links](#link-links)  

## :bulb: About the project
The repository provides a small, readable implementation of a two-player Ludo variant with a built-in Monte Carlo Tree Search planner. It models the board as a 40-position track plus a 4-position finish lane, enforces entry on a six, supports captures, and keeps track of whose turn it is.

The main entry point is `choose_blue_move`, which runs a configurable number of simulations and returns the recommended action for the Blue player. This makes the repo a compact reference for experimenting with search-based game AI without bringing in a full game engine or UI.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## :rocket: Quickstart
```bash
# (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Run a quick interactive snippet
python - <<'PY'
from ludo_mcts import LudoState, choose_blue_move

# Four tokens per player, all in home, Blue starts
state = LudoState(blue_tokens=[-1, -1, -1, -1], red_tokens=[-1, -1, -1, -1], current_player="Blue")
move = choose_blue_move(state, n_simulations=200)
print("Suggested move:", move)
PY
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## :wrench: Usage
### Key concepts
- **Token positions**: `-1` means a token is at home, `0-39` are track positions, and `40-43` are finish positions.
- **Actions**: tuples of `(token_index, target_position)`.
- **Turns**: `LudoState.current_player` toggles between `"Blue"` and `"Red"` as moves are applied.

### Example: applying a move
```python
from ludo_mcts import LudoState

state = LudoState(blue_tokens=[-1, -1, -1, -1], red_tokens=[-1, -1, -1, -1], current_player="Blue")
legal = state.get_legal_actions(dice_roll=6)
print("Legal actions:", legal)

if legal:
    state.apply_action(legal[0])
    print("Next player:", state.current_player)
```

### Customizing simulations
```python
from ludo_mcts import LudoState, choose_blue_move

state = LudoState(blue_tokens=[0, -1, -1, -1], red_tokens=[-1, -1, -1, -1], current_player="Blue")
move = choose_blue_move(state, n_simulations=1500)
print("Deeper search move:", move)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## :link: Links
- [Rules overview for Mensch ärgere dich nicht (Wikipedia)](https://en.wikipedia.org/wiki/Mensch_%C3%A4rgere_Dich_nicht)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
