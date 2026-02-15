import streamlit as st
import time
import math
from typing import List, Tuple, Optional

# ── Rust Engine (optional, near-instant AI) ──
try:
    import connect4_engine as _rust
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

# ----------------------------
#       Global Constants
# ----------------------------
ROW_COUNT = 6
COLUMN_COUNT = 7
COL_HEIGHT = ROW_COUNT + 1  # Using an extra bit per column as a sentinel
WINDOW_LENGTH = 4
EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2
MAX_DEPTH = 14  # AI search depth (iterative deepening will use 1..MAX_DEPTH)
WIN_SCORE = 1_000_000  # Base score for winning positions

# Transposition table entry types
EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2

# Static move ordering: center columns first (optimal heuristic for Connect 4)
COLUMN_ORDER = [3, 2, 4, 1, 5, 0, 6]

# Streamlit Page Configuration
st.set_page_config(page_title='Connect 4 AI', page_icon='🔴')

# Precompute all window masks
def generate_window_masks() -> List[int]:
    masks = []
    # Horizontal windows
    for row in range(ROW_COUNT):
        for col in range(COLUMN_COUNT - 3):
            mask = 0
            for i in range(WINDOW_LENGTH):
                pos = (col + i) * COL_HEIGHT + row
                mask |= (1 << pos)
            masks.append(mask)
    # Vertical windows
    for col in range(COLUMN_COUNT):
        for row in range(ROW_COUNT - 3):
            mask = 0
            for i in range(WINDOW_LENGTH):
                pos = col * COL_HEIGHT + (row + i)
                mask |= (1 << pos)
            masks.append(mask)
    # Positive diagonal (bottom-left to top-right)
    for row in range(ROW_COUNT - 3):
        for col in range(COLUMN_COUNT - 3):
            mask = 0
            for i in range(WINDOW_LENGTH):
                pos = (col + i) * COL_HEIGHT + (row + i)
                mask |= (1 << pos)
            masks.append(mask)
    # Negative diagonal (top-left to bottom-right)
    for row in range(3, ROW_COUNT):
        for col in range(COLUMN_COUNT - 3):
            mask = 0
            for i in range(WINDOW_LENGTH):
                pos = (col + i) * COL_HEIGHT + (row - i)
                mask |= (1 << pos)
            masks.append(mask)
    return masks

WINDOW_MASKS = generate_window_masks()

# Precompute a mask for the center column (for bonus scoring)
def generate_center_mask() -> int:
    center_col = COLUMN_COUNT // 2
    mask = 0
    for row in range(ROW_COUNT):
        pos = center_col * COL_HEIGHT + row
        mask |= (1 << pos)
    return mask

CENTER_MASK = generate_center_mask()

class Connect4Game:
    """
    A Connect 4 game that uses a bitboard representation for the board state
    and a hash table for caching minimax evaluations.
    """
    def __init__(self):
        # Bitboards for the two players (all bits initially 0)
        self.player_board = 0
        self.ai_board = 0
        # For each column, the next available row index (0-indexed)
        self.heights = [0] * COLUMN_COUNT
        self.game_over = False
        self.winner: Optional[int] = None
        # turn: 0 means player's turn; 1 means AI's turn.
        self.turn = 0
        # Cache for minimax: key -> (depth, best_col, score)
        self.cache = {}

    # ---------------
    #  Board Helpers
    # ---------------
    def is_valid_location(self, col: int) -> bool:
        return 0 <= col < COLUMN_COUNT and self.heights[col] < ROW_COUNT

    def get_valid_locations(self) -> List[int]:
        return [c for c in range(COLUMN_COUNT) if self.is_valid_location(c)]

    def is_board_full(self) -> bool:
        return all(h == ROW_COUNT for h in self.heights)

    def drop_piece(self, col: int, piece: int) -> None:
        if not self.is_valid_location(col):
            return
        row = self.heights[col]
        pos = col * COL_HEIGHT + row
        if piece == PLAYER_PIECE:
            self.player_board |= (1 << pos)
        elif piece == AI_PIECE:
            self.ai_board |= (1 << pos)
        self.heights[col] += 1

    def undo_move(self, col: int, piece: int) -> None:
        if not (0 <= col < COLUMN_COUNT) or self.heights[col] <= 0:
            return
        self.heights[col] -= 1
        row = self.heights[col]
        pos = col * COL_HEIGHT + row
        if piece == PLAYER_PIECE:
            self.player_board &= ~(1 << pos)
        elif piece == AI_PIECE:
            self.ai_board &= ~(1 << pos)

    def get_piece_at(self, row: int, col: int) -> int:
        pos = col * COL_HEIGHT + row
        if self.player_board & (1 << pos):
            return PLAYER_PIECE
        elif self.ai_board & (1 << pos):
            return AI_PIECE
        else:
            return EMPTY

    # ---------------
    #   Win Checking (via Bitboards)
    # ---------------
    def winning_move_bitboard(self, board: int) -> bool:
        # Vertical
        m = board & (board >> 1)
        if m & (m >> 2):
            return True
        # Horizontal
        m = board & (board >> COL_HEIGHT)
        if m & (m >> (2 * COL_HEIGHT)):
            return True
        # Diagonal (bottom-left to top-right)
        m = board & (board >> (COL_HEIGHT - 1))
        if m & (m >> (2 * (COL_HEIGHT - 1))):
            return True
        # Diagonal (top-left to bottom-right)
        m = board & (board >> (COL_HEIGHT + 1))
        if m & (m >> (2 * (COL_HEIGHT + 1))):
            return True
        return False

    def winning_move(self, piece: int) -> bool:
        if piece == PLAYER_PIECE:
            return self.winning_move_bitboard(self.player_board)
        elif piece == AI_PIECE:
            return self.winning_move_bitboard(self.ai_board)
        return False

    # ---------------
    #   Scoring (Bitboard-based Heuristic)
    # ---------------
    def score_window(self, window_mask: int, piece: int) -> int:
        board = self.ai_board if piece == AI_PIECE else self.player_board
        opp_board = self.player_board if piece == AI_PIECE else self.ai_board
        count_piece = (board & window_mask).bit_count()
        count_opp = (opp_board & window_mask).bit_count()
        # Windows with both players' pieces are blocked — no strategic value
        if count_piece > 0 and count_opp > 0:
            return 0
        empty_count = WINDOW_LENGTH - count_piece - count_opp
        if count_piece == 4:
            return 100
        if count_piece == 3 and empty_count == 1:
            return 50
        if count_piece == 2 and empty_count == 2:
            return 10
        if count_opp == 3 and empty_count == 1:
            return -45
        if count_opp == 2 and empty_count == 2:
            return -8
        return 0

    def score_position(self, piece: int) -> int:
        score = 0
        board = self.ai_board if piece == AI_PIECE else self.player_board
        center_count = (board & CENTER_MASK).bit_count()
        score += center_count * 12

        for window_mask in WINDOW_MASKS:
            score += self.score_window(window_mask, piece)
        return score

    # ---------------
    #   AI Mechanics
    # ---------------
    def order_moves(self, valid_locations: List[int], tt_move: Optional[int] = None) -> List[int]:
        """Order moves: TT best move first, then center-biased static order."""
        ordered = []
        if tt_move is not None and tt_move in valid_locations:
            ordered.append(tt_move)
        for col in COLUMN_ORDER:
            if col in valid_locations and col != tt_move:
                ordered.append(col)
        return ordered

    def minimax(self, depth: int, alpha: float, beta: float, maximizing_player: bool) -> Tuple[Optional[int], float]:
        orig_alpha = alpha
        orig_beta = beta

        # Transposition table lookup
        board_key = (self.player_board, self.ai_board)
        tt_move = None
        if board_key in self.cache:
            cached_depth, cached_col, cached_score, cached_flag = self.cache[board_key]
            tt_move = cached_col
            if cached_depth >= depth:
                if cached_flag == EXACT:
                    return cached_col, cached_score
                elif cached_flag == LOWERBOUND:
                    alpha = max(alpha, cached_score)
                elif cached_flag == UPPERBOUND:
                    beta = min(beta, cached_score)
                if alpha >= beta:
                    return cached_col, cached_score

        valid_locations = self.get_valid_locations()

        # Terminal node checks
        if self.winning_move(AI_PIECE):
            return None, WIN_SCORE + depth
        if self.winning_move(PLAYER_PIECE):
            return None, -(WIN_SCORE + depth)
        if not valid_locations:
            return None, 0
        if depth == 0:
            return None, self.score_position(AI_PIECE)

        # Check for immediate winning move (avoids unnecessary recursion)
        piece = AI_PIECE if maximizing_player else PLAYER_PIECE
        for col in valid_locations:
            self.drop_piece(col, piece)
            if self.winning_move(piece):
                self.undo_move(col, piece)
                score = (WIN_SCORE + depth - 1) if maximizing_player else -(WIN_SCORE + depth - 1)
                self.cache[board_key] = (depth, col, score, EXACT)
                return col, score
            self.undo_move(col, piece)

        ordered = self.order_moves(valid_locations, tt_move)

        if maximizing_player:
            value = -math.inf
            best_col = ordered[0]
            for col in ordered:
                self.drop_piece(col, AI_PIECE)
                _, new_score = self.minimax(depth - 1, alpha, beta, False)
                self.undo_move(col, AI_PIECE)
                if new_score > value:
                    value = new_score
                    best_col = col
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:
            value = math.inf
            best_col = ordered[0]
            for col in ordered:
                self.drop_piece(col, PLAYER_PIECE)
                _, new_score = self.minimax(depth - 1, alpha, beta, True)
                self.undo_move(col, PLAYER_PIECE)
                if new_score < value:
                    value = new_score
                    best_col = col
                beta = min(beta, value)
                if alpha >= beta:
                    break

        # Store in transposition table with bound type
        if value <= orig_alpha:
            flag = UPPERBOUND
        elif value >= orig_beta:
            flag = LOWERBOUND
        else:
            flag = EXACT
        if len(self.cache) > 1_000_000:
            self.cache.clear()
        self.cache[board_key] = (depth, best_col, value, flag)
        return best_col, value

    # ---------------
    #   Turn Management
    # ---------------
    def handle_player_move(self, col: int) -> None:
        if not self.game_over and self.turn == 0 and self.is_valid_location(col):
            self.drop_piece(col, PLAYER_PIECE)
            if self.winning_move(PLAYER_PIECE):
                self.game_over = True
                self.winner = PLAYER_PIECE
            elif self.is_board_full():
                self.game_over = True
            else:
                self.turn = 1  # Switch to AI
            st.session_state["game"] = self
            st.rerun()

    def ai_move(self) -> None:
        if not self.game_over and self.turn == 1:
            valid = self.get_valid_locations()
            if not valid:
                self.game_over = True
                st.session_state["game"] = self
                st.rerun()
                return

            # Check for immediate winning move (instant response)
            for col in valid:
                self.drop_piece(col, AI_PIECE)
                if self.winning_move(AI_PIECE):
                    self.game_over = True
                    self.winner = AI_PIECE
                    st.session_state["game"] = self
                    st.rerun()
                    return
                self.undo_move(col, AI_PIECE)

            if _HAS_RUST:
                # ── Rust engine path (near-instant) ──
                if "rust_engine" not in st.session_state:
                    st.session_state["rust_engine"] = _rust.Engine(ROW_COUNT, COLUMN_COUNT, WINDOW_LENGTH)
                    print("[connect4] ENGINE=RUST", flush=True)
                engine = st.session_state["rust_engine"]
                best_col = int(engine.find_best_move_bb(
                    self.ai_board, self.player_board, self.heights, 500
                ))
            else:
                # ── Python fallback ──
                if not hasattr(self, '_logged_py'):
                    print("[connect4] ENGINE=PYTHON", flush=True)
                    self._logged_py = True
                best_col = valid[0]
                start_time = time.time()
                for d in range(1, MAX_DEPTH + 1):
                    col, score = self.minimax(d, -math.inf, math.inf, True)
                    if col is not None:
                        best_col = col
                    if score >= WIN_SCORE:
                        break
                    if time.time() - start_time > 2.0:
                        break

            if self.is_valid_location(best_col):
                self.drop_piece(best_col, AI_PIECE)
                if self.winning_move(AI_PIECE):
                    self.game_over = True
                    self.winner = AI_PIECE
                elif self.is_board_full():
                    self.game_over = True
                else:
                    self.turn = 0
            st.session_state["game"] = self
            st.rerun()

    # ---------------
    #   Board Display
    # ---------------
    def draw_board(self) -> None:
        st.markdown("<h1 style='text-align: center;'>Connect 4 AI</h1>", unsafe_allow_html=True)
        with st.expander("CONNECT 4"):
            st.write("""
            Hello and welcome to my Connect 4 AI demonstrator,

            This simple Python/Streamlit app allows you to play against an agent that I created for the [ConnectX](https://www.kaggle.com/competitions/connectx/leaderboard) Kaggle competition. My best submission consistently ranked in the Top 20 on the leaderboard (Top 10%).

            **Rules:**
            You play by dropping the red pieces, try to connect 4 of them in a row (either vertically, horizontally, or diagonally) before the AI does it with the yellow pieces.

            *Note:
            To keep the game enjoyable, you are playing against an early, tamed-down version of the AI with limited Minimax depth and a less advanced heuristic. Although not as "smart" as the final submission, you will see that it is already quite challenging to beat.*

            Have fun!

            Corentin de Tilly - [LinkedIn](https://www.linkedin.com/in/corentin-de-tilly/?locale=en_US) - [GitHub](https://github.com/Tcha182) - [CV](https://resume-corentindetilly.streamlit.app/)

            """)

        col1, col2 = st.columns([1, 1])
        if col1.button('New Game (You Start)', use_container_width=True):
            st.session_state["game"] = Connect4Game()
            st.rerun()
        if col2.button('New Game (AI Starts)', use_container_width=True):
            new_game = Connect4Game()
            new_game.turn = 1
            st.session_state["game"] = new_game
            st.rerun()

        st.write("")
        drop_columns = st.columns(COLUMN_COUNT)
        for c in range(COLUMN_COUNT):
            disabled = (not self.is_valid_location(c) or self.game_over or (self.turn == 1))
            if drop_columns[c].button('🔻', key=f"drop_col_{c}", disabled=disabled, use_container_width=True):
                self.handle_player_move(c)

        # Draw the board (top row to bottom)
        for row in range(ROW_COUNT - 1, -1, -1):
            row_display = st.columns(COLUMN_COUNT)
            for col in range(COLUMN_COUNT):
                piece = self.get_piece_at(row, col)
                row_display[col].markdown(self.get_piece_html(piece), unsafe_allow_html=True)

        if self.game_over:
            if self.winner == PLAYER_PIECE:
                st.success("Congratulations! You won the game! 🎉")
            elif self.winner == AI_PIECE:
                st.error("Game Over. The AI won this time! 🤖")
            else:
                st.info("It's a tie! 🤝")

    @staticmethod
    def get_piece_html(piece: int) -> str:
        if piece == PLAYER_PIECE:
            emoji = '🔴'
        elif piece == AI_PIECE:
            emoji = '🟡'
        else:
            emoji = '⚪️'
        return f"<div style='text-align: center; font-size: 50px;'>{emoji}</div>"

def main():
    st.markdown(
        """
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .css-1v0mbdj, .css-18e3th9 {
                flex: 1 !important;
                max-width: 100% !important;
                min-width: 100% !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    if "game" not in st.session_state:
        st.session_state["game"] = Connect4Game()

    game: Connect4Game = st.session_state["game"]

    if game.turn == 1 and not game.game_over:
        game.ai_move()

    game.draw_board()

if __name__ == "__main__":
    main()
