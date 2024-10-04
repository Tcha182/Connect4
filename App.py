import streamlit as st
import numpy as np
import random
import math
from typing import List, Tuple, Optional

# Game Constants
ROW_COUNT = 6
COLUMN_COUNT = 7
WINDOW_LENGTH = 4
EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2
DEPTH = 5  # Hardcoded AI search depth

# Streamlit Page Configuration
st.set_page_config(page_title='Connect 4 AI', page_icon='🔴')

class Connect4Game:
	def __init__(self):
		self.board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=int)
		self.game_over = False
		self.winner: Optional[int] = None
		self.turn = 0  # 0 = Player's turn, 1 = AI's turn
		self.transposition_table = {}

	def create_board(self):
		self.board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=int)

	def drop_piece(self, row: int, col: int, piece: int):
		self.board[row][col] = piece

	def is_valid_location(self, col: int) -> bool:
		return self.board[ROW_COUNT - 1][col] == EMPTY

	def get_next_open_row(self, col: int) -> int:
		for r in range(ROW_COUNT):
			if self.board[r][col] == EMPTY:
				return r
		raise ValueError("Column is full")

	def winning_move(self, piece: int) -> bool:
		# Horizontal check
		for c in range(COLUMN_COUNT - 3):
			for r in range(ROW_COUNT):
				if all(self.board[r][c + i] == piece for i in range(WINDOW_LENGTH)):
					return True

		# Vertical check
		for c in range(COLUMN_COUNT):
			for r in range(ROW_COUNT - 3):
				if all(self.board[r + i][c] == piece for i in range(WINDOW_LENGTH)):
					return True

		# Positive diagonal check
		for c in range(COLUMN_COUNT - 3):
			for r in range(ROW_COUNT - 3):
				if all(self.board[r + i][c + i] == piece for i in range(WINDOW_LENGTH)):
					return True

		# Negative diagonal check
		for c in range(COLUMN_COUNT - 3):
			for r in range(3, ROW_COUNT):
				if all(self.board[r - i][c + i] == piece for i in range(WINDOW_LENGTH)):
					return True

		return False

	def evaluate_window(self, window: List[int], piece: int) -> int:
		score = 0
		opp_piece = PLAYER_PIECE if piece == AI_PIECE else AI_PIECE

		if window.count(piece) == 4:
			score += 100
		elif window.count(piece) == 3 and window.count(EMPTY) == 1:
			score += 10
		elif window.count(piece) == 2 and window.count(EMPTY) == 2:
			score += 5

		if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
			score -= 80

		return score

	def score_position(self, piece: int) -> int:
		score = 0

		# Center column preference
		center_array = list(self.board[:, COLUMN_COUNT // 2])
		center_count = center_array.count(piece)
		score += center_count * 6

		# Horizontal scoring
		for r in range(ROW_COUNT):
			row_array = list(self.board[r, :])
			for c in range(COLUMN_COUNT - 3):
				window = row_array[c:c + WINDOW_LENGTH]
				score += self.evaluate_window(window, piece)

		# Vertical scoring
		for c in range(COLUMN_COUNT):
			col_array = list(self.board[:, c])
			for r in range(ROW_COUNT - 3):
				window = col_array[r:r + WINDOW_LENGTH]
				score += self.evaluate_window(window, piece)

		# Positive diagonal scoring
		for r in range(ROW_COUNT - 3):
			for c in range(COLUMN_COUNT - 3):
				window = [self.board[r + i][c + i] for i in range(WINDOW_LENGTH)]
				score += self.evaluate_window(window, piece)

		# Negative diagonal scoring
		for r in range(ROW_COUNT - 3):
			for c in range(COLUMN_COUNT - 3):
				window = [self.board[r + 3 - i][c + i] for i in range(WINDOW_LENGTH)]
				score += self.evaluate_window(window, piece)

		return score

	def is_terminal_node(self) -> bool:
		return (self.winning_move(PLAYER_PIECE) or
				self.winning_move(AI_PIECE) or
				len(self.get_valid_locations()) == 0)

	def get_valid_locations(self) -> List[int]:
		return [col for col in range(COLUMN_COUNT) if self.is_valid_location(col)]

	def order_moves(self, valid_locations: List[int], piece: int) -> List[int]:
		scores = []
		for col in valid_locations:
			row = self.get_next_open_row(col)
			temp_board = self.board.copy()
			self.drop_piece(row, col, piece)
			score = self.score_position(piece)
			scores.append((score, col))
			self.board[row][col] = EMPTY  # Undo move
		scores.sort(reverse=True)
		ordered_moves = [col for score, col in scores]
		return ordered_moves

	def minimax(self, depth: int, alpha: float, beta: float, maximizingPlayer: bool) -> Tuple[Optional[int], int]:
		valid_locations = self.get_valid_locations()
		is_terminal = self.is_terminal_node()
		if depth == 0 or is_terminal:
			if is_terminal:
				if self.winning_move(AI_PIECE):
					return (None, math.inf)
				elif self.winning_move(PLAYER_PIECE):
					return (None, -math.inf)
				else:
					return (None, 0)
			else:
				return (None, self.score_position(AI_PIECE))

		if maximizingPlayer:
			value = -math.inf
			best_col = random.choice(valid_locations)
			ordered_locations = self.order_moves(valid_locations, AI_PIECE)
			for col in ordered_locations:
				row = self.get_next_open_row(col)
				self.drop_piece(row, col, AI_PIECE)
				new_score = self.minimax(depth - 1, alpha, beta, False)[1]
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
			best_col = random.choice(valid_locations)
			ordered_locations = self.order_moves(valid_locations, PLAYER_PIECE)
			for col in ordered_locations:
				row = self.get_next_open_row(col)
				self.drop_piece(row, col, PLAYER_PIECE)
				new_score = self.minimax(depth - 1, alpha, beta, True)[1]
				self.board[row][col] = EMPTY  # Undo move
				if new_score < value:
					value = new_score
					best_col = col
				beta = min(beta, value)
				if alpha >= beta:
					break
			return best_col, value

	def handle_click(self, col: int):
		if self.is_valid_location(col) and not self.game_over:
			row = self.get_next_open_row(col)
			self.drop_piece(row, col, PLAYER_PIECE)
			if self.winning_move(PLAYER_PIECE):
				self.game_over = True
				self.winner = PLAYER_PIECE
			self.turn = (self.turn + 1) % 2
			if not self.game_over and self.turn == AI_PIECE - 1:
				self.ai_move()

	def ai_move(self):
		col, _ = self.minimax(DEPTH, -math.inf, math.inf, True)
		if col is not None and self.is_valid_location(col):
			row = self.get_next_open_row(col)
			self.drop_piece(row, col, AI_PIECE)
			if self.winning_move(AI_PIECE):
				self.game_over = True
				self.winner = AI_PIECE
			self.turn = (self.turn + 1) % 2

	def draw_board(self):
		st.markdown("<h1 style='text-align: center;'>Connect 4 AI</h1>", unsafe_allow_html=True)
		with st.expander("CONNECT 4"):
			st.write("""
			Hello and welcome to my Connect 4 AI demonstrator,

			This simple Python/Streamlit app allows you to play against an agent that I created for the [ConnnectX](https://www.kaggle.com/competitions/connectx/leaderboard) Kaggle competition. My best submission consistently ranked in the Top 20 on the leaderboard (Top 10%). \n

			**Rules:**
			You play by dropping the red pieces, try to connect 4 of them in a row (either vertically, horizontally, or diagonally) before the AI does it with the yellow pieces.

			*Note:
			To keep the game enjoyable, you are playing against an early, tamed down version of the AI with limited Minimax depth of 2 moves per player and a less advanced heuristic. Although not as "smart" as the final submission, you will see that it is already quite challenging to beat.*


			have fun!

			Corentin de Tilly - [LinkedIn](https://www.linkedin.com/in/corentin-de-tilly/?locale=en_US) - [GitHub](https://github.com/Tcha182) - [CV](https://resume-corentindetilly.streamlit.app/)

			""")
			
		col1, col2 = st.columns([1,1])
		if col1.button('New Game (You Start)', use_container_width=True):
			self.__init__()
		if col2.button('New Game (AI Starts)', use_container_width=True):
			self.__init__()
			self.turn = AI_PIECE - 1
			self.ai_move()

		st.write('')  # Spacer

		valid_locations = self.get_valid_locations() if not self.game_over else []
		cols = st.columns(COLUMN_COUNT)

		# Draw the top buttons for dropping pieces
		for col in range(COLUMN_COUNT):
			if cols[col].button('🔻', key=f'drop_{col}',use_container_width=True ,disabled=col not in valid_locations):
				self.handle_click(col)

		# Draw the board
		for row in range(ROW_COUNT - 1, -1, -1):
			cols = st.columns(COLUMN_COUNT)
			for col in range(COLUMN_COUNT):
				piece = self.board[row][col]
				cols[col].markdown(self.get_piece_html(piece), unsafe_allow_html=True)

		if self.game_over:
			if self.winner == PLAYER_PIECE:
				st.success("Congratulations! You won the game! 🎉")
			elif self.winner == AI_PIECE:
				st.error("Game Over. The AI won the game. 🤖")
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


	if 'game' not in st.session_state:
		st.session_state.game = Connect4Game()

	game: Connect4Game = st.session_state.game
	game.draw_board()

if __name__ == '__main__':
    main()
