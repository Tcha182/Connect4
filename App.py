import streamlit as st
import numpy as np
import random
import math
from typing import List, Tuple, Optional

# ----------------------------
#       Global Constants
# ----------------------------
ROW_COUNT = 6
COLUMN_COUNT = 7
WINDOW_LENGTH = 4
EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2
MAX_DEPTH = 5  # AI search depth

# Streamlit Page Configuration
st.set_page_config(page_title='Connect 4 AI', page_icon='🔴')


class Connect4Game:
    """
    A Connect 4 game class with a minimax-based AI opponent.
    Tracks and updates board state, detects wins, and handles user and AI moves.
    """

    def __init__(self):
        # The board is a ROW_COUNT x COLUMN_COUNT numpy array of zeros.
        self.board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=int)
        self.game_over = False
        self.winner: Optional[int] = None
        # turn = 0 means the player goes first; turn = 1 means the AI goes first.
        self.turn = 0

    # ---------------
    #  Board Helpers
    # ---------------
    def drop_piece(self, row: int, col: int, piece: int) -> None:
        """Place the given piece at the specified (row, col)."""
        self.board[row][col] = piece

    def is_valid_location(self, col: int) -> bool:
        """
        A column is valid if its topmost cell is still empty
        (i.e., the last row in that column is 0).
        """
        return self.board[ROW_COUNT - 1][col] == EMPTY

    def get_next_open_row(self, col: int) -> int:
        """
        Return the first open row in a given column (from bottom to top).
        Raises ValueError if the column is full.
        """
        for r in range(ROW_COUNT):
            if self.board[r][col] == EMPTY:
                return r
        raise ValueError(f"Column {col} is full")

    def get_valid_locations(self) -> List[int]:
        """Return a list of columns that are valid for the next move."""
        return [c for c in range(COLUMN_COUNT) if self.is_valid_location(c)]

    def is_board_full(self) -> bool:
        """Check if the board is completely full (no valid moves left)."""
        return all(self.board[ROW_COUNT - 1][c] != EMPTY for c in range(COLUMN_COUNT))

    # ---------------
    #   Win Checking
    # ---------------
    def winning_move(self, piece: int) -> bool:
        """
        Checks if there is a winning (4 in a row) configuration on the board for 'piece'.
        """
        # Horizontal
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT):
                if all(self.board[r][c + i] == piece for i in range(WINDOW_LENGTH)):
                    return True

        # Vertical
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT - 3):
                if all(self.board[r + i][c] == piece for i in range(WINDOW_LENGTH)):
                    return True

        # Positive Diagonal
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT - 3):
                if all(self.board[r + i][c + i] == piece for i in range(WINDOW_LENGTH)):
                    return True

        # Negative Diagonal
        for c in range(COLUMN_COUNT - 3):
            for r in range(3, ROW_COUNT):
                if all(self.board[r - i][c + i] == piece for i in range(WINDOW_LENGTH)):
                    return True

        return False

    def is_terminal_node(self) -> bool:
        """
        Determines if the current board state is a terminal node:
        either player or AI has won, or no valid locations remain.
        """
        return (
            self.winning_move(PLAYER_PIECE)
            or self.winning_move(AI_PIECE)
            or len(self.get_valid_locations()) == 0
        )

    # ---------------
    #   Scoring
    # ---------------
    def evaluate_window(self, window: List[int], piece: int) -> int:
        """
        Evaluate the score of a 4-element window for the given piece.
        Higher scores mean more advantageous positions.
        """
        score = 0
        opp_piece = PLAYER_PIECE if piece == AI_PIECE else AI_PIECE

        if window.count(piece) == 4:
            score += 100
        elif window.count(piece) == 3 and window.count(EMPTY) == 1:
            score += 10
        elif window.count(piece) == 2 and window.count(EMPTY) == 2:
            score += 5

        # Discourage letting the opponent get 3 in a row
        if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
            score -= 80

        return score

    def score_position(self, piece: int) -> int:
        """
        Score the entire board for the specified piece.
        """
        score = 0
        # Center column preference
        center_col = COLUMN_COUNT // 2
        center_array = list(self.board[:, center_col])
        center_count = center_array.count(piece)
        score += center_count * 6

        # Horizontal
        for r in range(ROW_COUNT):
            row_array = list(self.board[r, :])
            for c in range(COLUMN_COUNT - 3):
                window = row_array[c : c + WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)

        # Vertical
        for c in range(COLUMN_COUNT):
            col_array = list(self.board[:, c])
            for r in range(ROW_COUNT - 3):
                window = col_array[r : r + WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)

        # Positive Diagonal
        for r in range(ROW_COUNT - 3):
            for c in range(COLUMN_COUNT - 3):
                window = [self.board[r + i][c + i] for i in range(WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece)

        # Negative Diagonal
        for r in range(3, ROW_COUNT):
            for c in range(COLUMN_COUNT - 3):
                window = [self.board[r - i][c + i] for i in range(WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece)

        return score

    # -----------------
    #   AI Mechanics
    # -----------------
    def order_moves(self, valid_locations: List[int], piece: int) -> List[int]:
        """
        Pre-sorts columns by their potential 'score' for the current piece,
        so the minimax search explores most promising moves first.
        """
        scores = []
        for col in valid_locations:
            row = self.get_next_open_row(col)
            self.drop_piece(row, col, piece)
            score = self.score_position(piece)
            self.board[row][col] = EMPTY  # Undo move
            scores.append((score, col))

        scores.sort(reverse=True, key=lambda x: x[0])
        # Extract columns in descending order of score
        return [move[1] for move in scores]

    def minimax(self, depth: int, alpha: float, beta: float, maximizing_player: bool) -> Tuple[Optional[int], int]:
        """
        Minimax with alpha-beta pruning. Returns a tuple (best_column, score).
        If best_column is None, it means the position is terminal or no moves exist.
        """
        valid_locations = self.get_valid_locations()
        is_terminal = self.is_terminal_node()

        # Base conditions
        if depth == 0 or is_terminal:
            if is_terminal:
                if self.winning_move(AI_PIECE):
                    return None, math.inf
                elif self.winning_move(PLAYER_PIECE):
                    return None, -math.inf
                else:
                    return None, 0  # Board is full or tie
            else:
                return None, self.score_position(AI_PIECE)

        if maximizing_player:
            value = -math.inf
            best_col = None
            # Order moves to explore best first
            ordered_locations = self.order_moves(valid_locations, AI_PIECE)
            for col in ordered_locations:
                row = self.get_next_open_row(col)
                self.drop_piece(row, col, AI_PIECE)
                _, new_score = self.minimax(depth - 1, alpha, beta, False)
                self.board[row][col] = EMPTY  # Undo move

                if new_score > value:
                    value = new_score
                    best_col = col
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return best_col, value
        else:
            value = math.inf
            best_col = None
            ordered_locations = self.order_moves(valid_locations, PLAYER_PIECE)
            for col in ordered_locations:
                row = self.get_next_open_row(col)
                self.drop_piece(row, col, PLAYER_PIECE)
                _, new_score = self.minimax(depth - 1, alpha, beta, True)
                self.board[row][col] = EMPTY

                if new_score < value:
                    value = new_score
                    best_col = col
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return best_col, value

    # -----------------
    #  Turn Management
    # -----------------
    def handle_player_move(self, col: int) -> None:
        """
        Handle the event of a player dropping a piece into a chosen column.
        """
        if not self.game_over and self.turn == 0 and self.is_valid_location(col):
            row = self.get_next_open_row(col)
            self.drop_piece(row, col, PLAYER_PIECE)

            if self.winning_move(PLAYER_PIECE):
                self.game_over = True
                self.winner = PLAYER_PIECE
            elif self.is_board_full():
                self.game_over = True

            else:
                # Switch to AI turn
                self.turn = 1

            # Update session state and rerun
            st.session_state["game"] = self
            st.rerun()

    def ai_move(self) -> None:
        """
        Execute the AI move using the minimax algorithm.
        """
        if not self.game_over and self.turn == 1:
            col, _ = self.minimax(MAX_DEPTH, -math.inf, math.inf, True)
            if col is not None and self.is_valid_location(col):
                row = self.get_next_open_row(col)
                self.drop_piece(row, col, AI_PIECE)

                if self.winning_move(AI_PIECE):
                    self.game_over = True
                    self.winner = AI_PIECE
                elif self.is_board_full():
                    self.game_over = True
                else:
                    self.turn = 0  # Switch back to player's turn

            st.session_state["game"] = self
            st.rerun()

    # -----------------
    #   Board Display
    # -----------------
    def draw_board(self) -> None:
        """
        Renders the top-level UI elements (new game buttons, drop buttons) and displays the board.
        """
    def draw_board(self):
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

        # game controls
        col1, col2 = st.columns([1, 1])
        if col1.button('New Game (You Start)', use_container_width=True):
            st.session_state["game"] = Connect4Game()
            st.rerun()

        if col2.button('New Game (AI Starts)', use_container_width=True):
            new_game = Connect4Game()
            new_game.turn = 1  # AI moves first
            st.session_state["game"] = new_game
            st.rerun()

        st.write("")  # Spacer

        # Draw the top row of drop-piece buttons
        drop_columns = st.columns(COLUMN_COUNT)
        for c in range(COLUMN_COUNT):
            disabled = (
                not self.is_valid_location(c)
                or self.game_over
                or (self.turn == 1)  # AI's turn
            )
            if drop_columns[c].button('🔻', key=f"drop_col_{c}", disabled=disabled, use_container_width=True):
                self.handle_player_move(c)

        # Display the board in row-major form (top row last)
        for row in range(ROW_COUNT - 1, -1, -1):
            row_display = st.columns(COLUMN_COUNT)
            for col in range(COLUMN_COUNT):
                piece = self.board[row][col]
                row_display[col].markdown(self.get_piece_html(piece), unsafe_allow_html=True)

        # Post-game feedback
        if self.game_over:
            if self.winner == PLAYER_PIECE:
                st.success("Congratulations! You won the game! 🎉")
            elif self.winner == AI_PIECE:
                st.error("Game Over. The AI won this time! 🤖")
            else:
                st.info("It's a tie! 🤝")

    @staticmethod
    def get_piece_html(piece: int) -> str:
        """Return the HTML string for rendering a piece (red, yellow, or empty)."""
        if piece == PLAYER_PIECE:
            emoji = '🔴'
        elif piece == AI_PIECE:
            emoji = '🟡'
        else:
            emoji = '⚪️'
        return f"<div style='text-align: center; font-size: 50px;'>{emoji}</div>"


def main():
    """
    Main Streamlit app entry point.
    Creates/loads the Connect4Game object, forces an AI move if it's the AI's turn,
    and then draws the board.
    """
    # Hide Streamlit's default UI elements
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

    # Create or retrieve an existing game instance from session state
    if "game" not in st.session_state:
        st.session_state["game"] = Connect4Game()

    game: Connect4Game = st.session_state["game"]

    # If it's the AI's turn and game isn't over, let the AI make a move
    if game.turn == 1 and not game.game_over:
        game.ai_move()

    # Draw the game board and user interface
    game.draw_board()


if __name__ == "__main__":
    main()
