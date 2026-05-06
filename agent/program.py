# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Coord, Direction, CARDINAL_DIRECTIONS, \
    Action, PlaceAction, MoveAction, EatAction, CascadeAction
from referee.game.board import Board, CellState, GamePhase
from referee.game.constants import BOARD_N

import math
import random


# Search depth configuration
PLACEMENT_DEPTH = 2   # Shallower: branching is very high during placement
PLAY_DEPTH      = 3   # Increase if timing allows; tune against the 180s limit

def get_legal_actions(board: Board, color: PlayerColor) -> list[Action]:
    """
    Generate all legal actions for `color` in the current board state.
 
    Placement phase: every empty cell (adjacency restriction is enforced
    by the board's apply_action, so illegal ones get filtered at apply time).
    Play phase: MOVE, EAT, and CASCADE for every owned stack.
    """
    actions: list[Action] = []
 
    if board.phase == GamePhase.PLACEMENT:
        for r in range(BOARD_N):
            for c in range(BOARD_N):
                coord = Coord(r, c)
                if board[coord].is_empty:
                    actions.append(PlaceAction(coord))
        return actions
 
    # ── Play phase ────────────────────────────────────────────────────────────
    opponent = color.opponent
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            coord = Coord(r, c)
            cell  = board[coord]
            if cell.color != color:
                continue
 
            height = cell.height
 
            for direction in CARDINAL_DIRECTIONS:
                nr = r + direction.r
                nc = c + direction.c
 
                if not (0 <= nr < BOARD_N and 0 <= nc < BOARD_N):
                    continue
 
                dest = board[Coord(nr, nc)]
 
                if dest.is_empty or dest.color == color:
                    # MOVE: relocate (empty) or merge (friendly)
                    actions.append(MoveAction(coord, direction))
 
                elif dest.color == opponent and height >= dest.height:
                    # EAT: capture enemy stack (height constraint checked)
                    actions.append(EatAction(coord, direction))
 
            # CASCADE: valid for any stack with height >= 2, all four directions
            if height >= 2:
                for direction in CARDINAL_DIRECTIONS:
                    actions.append(CascadeAction(coord, direction))
 
    return actions
def _placement_candidates(board: Board, color: PlayerColor) -> list[Coord]:
    """
    Return a small set of sensible placement coordinates to keep the
    placement-phase branching factor manageable.
 
    Strategy:
      - Always include the 4x4 central region (rows/cols 2-5)
      - Also include cells within 2 steps of any existing friendly stack
      - Exclude occupied cells (the board's apply_action will reject
        adjacency violations, so we let it filter those)
    """
    candidates = set()
 
    # Central 4×4 region
    for r in range(2, 6):
        for c in range(2, 6):
            coord = Coord(r, c)
            if board[coord].is_empty:
                candidates.add(coord)
 
    # Cells within 2 steps of existing friendly stacks
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            cell = board[Coord(r, c)]
            if cell.color != color:
                continue
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < BOARD_N and 0 <= nc < BOARD_N:
                        coord = Coord(nr, nc)
                        if board[coord].is_empty:
                            candidates.add(coord)
 
    # Fallback: if somehow empty, allow any empty cell
    if not candidates:
        for r in range(BOARD_N):
            for c in range(BOARD_N):
                coord = Coord(r, c)
                if board[coord].is_empty:
                    candidates.add(coord)
 
    return list(candidates)

def evaluate(board: Board, color: PlayerColor) -> float:
    """
    Static evaluation from `color`'s perspective (higher = better for `color`).
 
    Components:
      - Token advantage   : net token count difference
      - Eat threats       : bonus for each enemy token we can immediately eat
      - Edge penalty      : penalty for own tokens sitting on board edges
                            (vulnerable to being cascaded off)
    """
    opponent = color.opponent
 
    my_tokens  = 0
    opp_tokens = 0
    eat_bonus  = 0.0
    edge_pen   = 0.0
 
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            coord = Coord(r, c)
            cell  = board[coord]
            if cell.is_empty:
                continue
 
            h = cell.height
 
            if cell.color == color:
                my_tokens += h
 
                # Penalise stacks on the perimeter (cascade risk)
                if r == 0 or r == BOARD_N - 1 or c == 0 or c == BOARD_N - 1:
                    edge_pen += h * 0.3
 
                # Reward adjacency to attackable enemies
                for direction in CARDINAL_DIRECTIONS:
                    nr = r + direction.r
                    nc = c + direction.c
                    if not (0 <= nr < BOARD_N and 0 <= nc < BOARD_N):
                        continue
                    adj = board[Coord(nr, nc)]
                    if adj.color == opponent and h >= adj.height:
                        eat_bonus += adj.height * 0.5
 
            else:
                opp_tokens += h
 
    if opp_tokens == 0:
        return math.inf
    if my_tokens == 0:
        return -math.inf
 
    return (my_tokens - opp_tokens) * 1.0 + eat_bonus - edge_pen
 

def order_actions(actions: list[Action]) -> list[Action]:
    def priority(action):
        if isinstance(action, EatAction):
            return 0   # try first
        if isinstance(action, CascadeAction):
            return 1
        if isinstance(action, MoveAction):
            return 2
        return 3       # PlaceAction

    return sorted(actions, key=priority)


def minimax(
    board:      Board,
    depth:      int,
    alpha:      float,
    beta:       float,
    maximising: bool,
    my_color:   PlayerColor,
) -> tuple[float, Action | None]:
    """
    Minimax with alpha-beta pruning.
 
    The board is mutated in-place (apply_action / undo_action) to avoid
    expensive deep copies at every node.
 
    Returns (score, best_action) where score is from `my_color`'s perspective.
    """
    if board.game_over or depth == 0:
        return evaluate(board, my_color), None
 
    current_color = board.turn_color
    actions = get_legal_actions(board, current_color)
 
    if not actions:
        return 0.0, None  # Stalemate — draw
 
    actions = order_actions(actions)
    best_action: Action | None = None
 
    if maximising:
        best_score = -math.inf
        for action in actions:
            try:
                board.apply_action(action)
            except Exception:
                continue  # Illegal action (e.g. placement adjacency violation)
 
            score, _ = minimax(board, depth - 1, alpha, beta, False, my_color)
            board.undo_action()
 
            if score > best_score:
                best_score = score
                best_action = action
 
            alpha = max(alpha, best_score)
            if alpha >= beta:
                break  # β-cutoff
 
        return best_score, best_action
 
    else:
        best_score = math.inf
        for action in actions:
            try:
                board.apply_action(action)
            except Exception:
                continue
 
            score, _ = minimax(board, depth - 1, alpha, beta, True, my_color)
            board.undo_action()
 
            if score < best_score:
                best_score = score
                best_action = action
 
            beta = min(beta, best_score)
            if beta <= alpha:
                break  # α-cutoff
 
        return best_score, best_action
 

class Agent:
    """
    This class is the "entry point" for your agent, providing an interface to
    respond to various Cascade game events.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        """
        This constructor method runs when the referee instantiates the agent.
        Any setup and/or precomputation should be done here.
        """
        self._color = color
        self._board = Board()
        self._turn_count = 0
        match color:
            case PlayerColor.RED:
                print("Testing: I am playing as RED (first player)")
            case PlayerColor.BLUE:
                print("Testing: I am playing as BLUE")

    
    def action(self, **referee: dict) -> Action:
        """Called by the referee when it's our turn."""
        is_placement = self._board.phase == GamePhase.PLACEMENT
        depth = PLACEMENT_DEPTH if is_placement else PLAY_DEPTH
 
        _, best = minimax(
            self._board,
            depth,
            alpha=-math.inf,
            beta=math.inf,
            maximising=True,
            my_color=self._color,
        )
 
        if best is None:
            # Fallback safety net — should rarely trigger in practice
            actions = get_legal_actions(self._board, self._color)
            best = random.choice(actions) if actions else PlaceAction(Coord(0, 0))
 
        return best
    
        # """
        # This method is called by the referee each time it is the agent's turn
        # to take an action. It must always return an action object.
        # """

        # # Below we have hardcoded actions to be played depending on whether
        # # the agent is playing as BLUE or RED. Obviously this won't work beyond
        # # the initial moves of the game, so you should use some game playing
        # # technique(s) to determine the best action to take.

        # # During placement phase (first 8 turns total, 4 per player)
        # if self._turn_count < 4:
        #     self.minimax_placement()
        # else:
        #     self.minimax_play()
        
        # if self._turn_count < 4:
        #     match self._color:
        #         case PlayerColor.RED:
        #             print("Testing: RED is playing a PLACE action")
        #             return PlaceAction(Coord(0, self._turn_count))
        #         case PlayerColor.BLUE:
        #             print("Testing: BLUE is playing a PLACE action")
        #             return PlaceAction(Coord(7, self._turn_count))

        # # During play phase
        # match self._color:
        #     case PlayerColor.RED:
        #         print("Testing: RED is playing a MOVE action")
        #         return MoveAction(Coord(0, 0), Direction.Down)
        #     case PlayerColor.BLUE:
        #         print("Testing: BLUE is playing a MOVE action")
        #         return MoveAction(Coord(7, 0), Direction.Up)

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after a player has taken their
        turn. You should use it to update the agent's internal game state.
        """
        # if color == self._color:
        #     self._turn_count += 1

        # There are four possible action types: PLACE, MOVE, EAT, and CASCADE.
        # Below we check which type of action was played and print out the
        # details of the action for demonstration purposes. You should replace
        # this with your own logic to update your agent's internal game state.
        # match action:
        #     case PlaceAction(coord):
        #         print(f"Testing: {color} played PLACE action at {coord}")
        #     case MoveAction(coord, direction):
        #         print(f"Testing: {color} played MOVE action:")
        #         print(f"  Coord: {coord}")
        #         print(f"  Direction: {direction}")
        #     case EatAction(coord, direction):
        #         print(f"Testing: {color} played EAT action:")
        #         print(f"  Coord: {coord}")
        #         print(f"  Direction: {direction}")
        #     case CascadeAction(coord, direction):
        #         print(f"Testing: {color} played CASCADE action:")
        #         print(f"  Coord: {coord}")
        #         print(f"  Direction: {direction}")
        #     case _:
        #         raise ValueError(f"Unknown action type: {action}")
        """Called by the referee after every turn to keep our board in sync."""
        try:
            self._board.apply_action(action)
        except Exception as e:
            print(f"Warning: could not apply {action}: {e}")