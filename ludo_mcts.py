"""Monte Carlo Tree Search agent for a 2-player Ludo / Mensch ärgere dich nicht.

Blue is always the planning player. Red is simulated by the rollout policy.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, List, Optional, Tuple

Player = str  # "Blue" or "Red"
Action = Tuple[int, int]  # (token_index, target_position)


TRACK_LEN = 40
FINISH_LEN = 4
BLUE_START = 0
RED_START = TRACK_LEN // 2
HOME = -1


def other_player(player: Player) -> Player:
    return "Red" if player == "Blue" else "Blue"


@dataclass
class LudoState:
    """State for a 2-player Ludo game (Blue vs Red)."""

    blue_tokens: List[int]
    red_tokens: List[int]
    current_player: Player

    def clone(self) -> "LudoState":
        return LudoState(
            blue_tokens=self.blue_tokens.copy(),
            red_tokens=self.red_tokens.copy(),
            current_player=self.current_player,
        )

    def is_terminal(self) -> bool:
        return self._all_finished(self.blue_tokens) or self._all_finished(self.red_tokens)

    def winner(self) -> Optional[Player]:
        if self._all_finished(self.blue_tokens):
            return "Blue"
        if self._all_finished(self.red_tokens):
            return "Red"
        return None

    def get_legal_actions(self, dice_roll: int) -> List[Action]:
        tokens = self._current_tokens()
        own_track_positions = self._track_positions(tokens, self.current_player)
        actions: List[Action] = []
        for idx, pos in enumerate(tokens):
            if pos == HOME:
                if dice_roll != 6:
                    continue
                target = 0
                if target in own_track_positions:
                    continue
                actions.append((idx, target))
                continue
            if pos >= TRACK_LEN + FINISH_LEN:
                continue
            target = pos + dice_roll
            if target >= TRACK_LEN + FINISH_LEN:
                continue
            if target >= TRACK_LEN:
                if target in tokens:
                    continue
                actions.append((idx, target))
                continue
            if target in own_track_positions:
                continue
            actions.append((idx, target))
        return actions

    def apply_action(self, action: Action) -> None:
        token_index, target = action
        tokens = self._current_tokens()
        opponent_tokens = self._opponent_tokens()
        if tokens[token_index] == HOME and target != 0:
            raise ValueError("Illegal action: token in home must enter at position 0.")
        if target < TRACK_LEN:
            target_global = self._track_to_global(target, self.current_player)
            opponent_track_positions = self._track_positions(
                opponent_tokens, other_player(self.current_player)
            )
            if target_global in opponent_track_positions:
                self._capture_opponent(target_global)
        tokens[token_index] = target
        self.current_player = other_player(self.current_player)

    def key(self) -> Tuple[Player, Tuple[int, ...], Tuple[int, ...]]:
        return (
            self.current_player,
            tuple(self.blue_tokens),
            tuple(self.red_tokens),
        )

    def _current_tokens(self) -> List[int]:
        return self.blue_tokens if self.current_player == "Blue" else self.red_tokens

    def _opponent_tokens(self) -> List[int]:
        return self.red_tokens if self.current_player == "Blue" else self.blue_tokens

    def _all_finished(self, tokens: List[int]) -> bool:
        return all(pos >= TRACK_LEN + FINISH_LEN - 1 for pos in tokens)

    def _track_to_global(self, pos: int, player: Player) -> int:
        start = BLUE_START if player == "Blue" else RED_START
        return (start + pos) % TRACK_LEN

    def _track_positions(self, tokens: List[int], player: Player) -> List[int]:
        positions = []
        for pos in tokens:
            if 0 <= pos < TRACK_LEN:
                positions.append(self._track_to_global(pos, player))
        return positions

    def _capture_opponent(self, target_global: int) -> None:
        opponent = self._opponent_tokens()
        opponent_player = other_player(self.current_player)
        for idx, pos in enumerate(opponent):
            if 0 <= pos < TRACK_LEN:
                if self._track_to_global(pos, opponent_player) == target_global:
                    opponent[idx] = HOME


class MCTSNode:
    def __init__(self, state_key: Tuple[Player, Tuple[int, ...], Tuple[int, ...]]):
        self.state_key = state_key
        self.children: Dict[Action, Tuple[Player, Tuple[int, ...], Tuple[int, ...]]] = {}
        self.N: Dict[Action, int] = {}
        self.W: Dict[Action, float] = {}
        self.total_visits = 0

    def q_value(self, action: Action) -> float:
        if self.N.get(action, 0) == 0:
            return 0.0
        return self.W.get(action, 0.0) / self.N[action]

    def uct_score(self, action: Action, cpuct: float, player: Player) -> float:
        n_sa = self.N.get(action, 0)
        if n_sa == 0:
            return float("inf")
        sign = 1.0 if player == "Blue" else -1.0
        exploitation = sign * self.q_value(action)
        exploration = cpuct * math.sqrt(math.log(self.total_visits + 1) / n_sa)
        return exploitation + exploration


class MCTS:
    def __init__(self, cpuct: float = 1.4, n_simulations: int = 1000) -> None:
        self.cpuct = cpuct
        self.n_simulations = n_simulations
        self.nodes: Dict[Tuple[Player, Tuple[int, ...], Tuple[int, ...]], MCTSNode] = {}

    def search(self, root_state: LudoState) -> Optional[Action]:
        root_key = root_state.key()
        if root_key not in self.nodes:
            self.nodes[root_key] = MCTSNode(root_key)
        for _ in range(self.n_simulations):
            self._simulate(root_state.clone())
        root_node = self.nodes[root_key]
        if not root_node.N:
            return None
        return max(root_node.N.items(), key=lambda item: item[1])[0]

    def _simulate(self, state: LudoState) -> float:
        path: List[Tuple[MCTSNode, Action, Player]] = []
        while True:
            if state.is_terminal():
                return self._backpropagate(path, state.winner())
            node = self.nodes.get(state.key())
            if node is None:
                node = MCTSNode(state.key())
                self.nodes[state.key()] = node
                value = self._rollout(state)
                return self._backpropagate(path, self._value_to_winner(value))
            dice_roll = random.randint(1, 6)
            legal_actions = state.get_legal_actions(dice_roll)
            if not legal_actions:
                state.current_player = other_player(state.current_player)
                continue
            untried = [action for action in legal_actions if action not in node.children]
            if untried:
                action = random.choice(untried)
                next_state = state.clone()
                next_state.apply_action(action)
                node.children[action] = next_state.key()
                path.append((node, action, state.current_player))
                value = self._rollout(next_state)
                return self._backpropagate(path, self._value_to_winner(value))
            action = max(
                legal_actions,
                key=lambda act: node.uct_score(act, self.cpuct, state.current_player),
            )
            path.append((node, action, state.current_player))
            state.apply_action(action)

    def _rollout(self, state: LudoState, max_depth: int = 200) -> float:
        for _ in range(max_depth):
            if state.is_terminal():
                winner = state.winner()
                if winner == "Blue":
                    return 1.0
                if winner == "Red":
                    return -1.0
                return 0.0
            dice_roll = random.randint(1, 6)
            legal_actions = state.get_legal_actions(dice_roll)
            if not legal_actions:
                state.current_player = other_player(state.current_player)
                continue
            action = random.choice(legal_actions)
            state.apply_action(action)
        return 0.0

    def _value_to_winner(self, value: float) -> Optional[Player]:
        if value > 0:
            return "Blue"
        if value < 0:
            return "Red"
        return None

    def _backpropagate(
        self, path: List[Tuple[MCTSNode, Action, Player]], winner: Optional[Player]
    ) -> float:
        if winner == "Blue":
            outcome = 1.0
        elif winner == "Red":
            outcome = -1.0
        else:
            outcome = 0.0
        for node, action, _ in path:
            node.total_visits += 1
            node.N[action] = node.N.get(action, 0) + 1
            node.W[action] = node.W.get(action, 0.0) + outcome
        return outcome


def choose_blue_move(state: LudoState, n_simulations: int = 1000) -> Optional[Action]:
    """Run MCTS from the given state and return the best Blue action."""
    if state.current_player != "Blue":
        raise ValueError("choose_blue_move expects Blue to be the current player.")
    mcts = MCTS(n_simulations=n_simulations)
    return mcts.search(state)
