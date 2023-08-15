import streamlit as st
import numpy as np
import random
import math

st.set_page_config(page_title='Connect 4 AI', page_icon='🔴')

ROW_COUNT = 6
COLUMN_COUNT = 7

PLAYER = 0
AI = 1

EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2

WINDOW_LENGTH = 4

DEPTH = 5

def init(start=False):
	st.session_state.game_over = False
	create_board()
	st.session_state.winner = None
	if start:
		st.session_state.turn = 1

def create_board():
	st.session_state.turn = 0
	st.session_state.board = np.zeros((ROW_COUNT,COLUMN_COUNT))
	return st.session_state.board

def drop_piece(board, row, col, piece):
	board[row][col] = piece

def is_valid_location(board, col):
	return board[ROW_COUNT-1][col] == 0

def get_next_open_row(board, col):
	for r in range(ROW_COUNT):
		if board[r][col] == 0:
			return r

# def print_board(board):
# 	st.write(np.flip(board, 0))

def winning_move(board, piece):
	# Check horizontal locations for win
	for c in range(COLUMN_COUNT-3):
		for r in range(ROW_COUNT):
			if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
				return True

	# Check vertical locations for win
	for c in range(COLUMN_COUNT):
		for r in range(ROW_COUNT-3):
			if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
				return True

	# Check positively sloped diaganols
	for c in range(COLUMN_COUNT-3):
		for r in range(ROW_COUNT-3):
			if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
				return True

	# Check negatively sloped diaganols
	for c in range(COLUMN_COUNT-3):
		for r in range(3, ROW_COUNT):
			if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
				return True

def evaluate_window(window, piece):
	score = 0
	opp_piece = PLAYER_PIECE
	if piece == PLAYER_PIECE:
		opp_piece = AI_PIECE

	if window.count(piece) == 4:
		score += 100
	elif window.count(piece) == 3 and window.count(EMPTY) == 1:
		score += 5
	elif window.count(piece) == 2 and window.count(EMPTY) == 2:
		score += 2

	if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
		score -= 4

	return score

def score_position(board, piece):
	score = 0

	## Score center column
	center_array = [int(i) for i in list(board[:, COLUMN_COUNT//2])]
	center_count = center_array.count(piece)
	score += center_count * 3

	## Score Horizontal
	for r in range(ROW_COUNT):
		row_array = [int(i) for i in list(board[r,:])]
		for c in range(COLUMN_COUNT-3):
			window = row_array[c:c+WINDOW_LENGTH]
			score += evaluate_window(window, piece)

	## Score Vertical
	for c in range(COLUMN_COUNT):
		col_array = [int(i) for i in list(board[:,c])]
		for r in range(ROW_COUNT-3):
			window = col_array[r:r+WINDOW_LENGTH]
			score += evaluate_window(window, piece)

	## Score posiive sloped diagonal
	for r in range(ROW_COUNT-3):
		for c in range(COLUMN_COUNT-3):
			window = [board[r+i][c+i] for i in range(WINDOW_LENGTH)]
			score += evaluate_window(window, piece)

	## Score negative sloped diagonal
	for r in range(ROW_COUNT-3):
		for c in range(COLUMN_COUNT-3):
			window = [board[r+3-i][c+i] for i in range(WINDOW_LENGTH)]
			score += evaluate_window(window, piece)

	return score

def is_terminal_node(board):
	return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(get_valid_locations(board)) == 0

def minimax(board, depth, alpha, beta, maximizingPlayer):
	valid_locations = get_valid_locations(board)
	is_terminal = is_terminal_node(board)
	if depth == 0 or is_terminal:
		if is_terminal:
			if winning_move(board, AI_PIECE):
				return (None, 100000000000000)
			elif winning_move(board, PLAYER_PIECE):
				return (None, -10000000000000)
			else: # Game is over, no more valid moves
				return (None, 0)
		else: # Depth is zero
			return (None, score_position(board, AI_PIECE))
	if maximizingPlayer:
		value = -math.inf
		column = random.choice(valid_locations)
		for col in valid_locations:
			row = get_next_open_row(board, col)
			b_copy = board.copy()
			drop_piece(b_copy, row, col, AI_PIECE)
			new_score = minimax(b_copy, depth-1, alpha, beta, False)[1]
			if new_score > value:
				value = new_score
				column = col
			alpha = max(alpha, value)
			if alpha >= beta:
				break
		return column, value

	else: # Minimizing player
		value = math.inf
		column = random.choice(valid_locations)
		for col in valid_locations:
			row = get_next_open_row(board, col)
			b_copy = board.copy()
			drop_piece(b_copy, row, col, PLAYER_PIECE)
			new_score = minimax(b_copy, depth-1, alpha, beta, True)[1]
			if new_score < value:
				value = new_score
				column = col
			beta = min(beta, value)
			if alpha >= beta:
				break
		return column, value

def get_valid_locations(board):
	valid_locations = []
	for col in range(COLUMN_COUNT):
		if is_valid_location(board, col):
			valid_locations.append(col)
	return valid_locations


def handle_click(col):

		if is_valid_location(st.session_state.board, col):
			row = get_next_open_row(st.session_state.board, col)
			drop_piece(st.session_state.board, row, col, PLAYER_PIECE)
			if winning_move(st.session_state.board, PLAYER_PIECE):
				st.session_state.game_over = True
				st.session_state.winner = PLAYER

			st.session_state.turn += 1
			st.session_state.turn = st.session_state.turn % 2


def ai_turn():
	col, st.session_state.minimax_score = minimax(st.session_state.board, DEPTH, -math.inf, math.inf, True)
	if is_valid_location(st.session_state.board, col):	
		row = get_next_open_row(st.session_state.board, col)
		drop_piece(st.session_state.board, row, col, AI_PIECE)		

		if winning_move(st.session_state.board, AI_PIECE):
			st.session_state.game_over = True
			st.session_state.winner = AI

		st.session_state.turn += 1
		st.session_state.turn = st.session_state.turn % 2


def draw_board(board):

	col1, col2, _ = st.columns([1.4, 2.5, 5])
	col1.button('New game', on_click=init)
	col2.button('New game (AI starts)', on_click=init, args=(True,))

	if st.session_state.game_over:
		valid = []
	else:
		valid = get_valid_locations(board)

	st.write(' ')

	cols = st.columns([30,10,10,10,10,10,10,10,31])
	for column in range(board.shape[1]):
		disabled = column not in valid
		cols[column + 1].button('🔻',
		key=f'B{column}',
		on_click=handle_click,
		args=(column,),
		disabled = disabled)

	for i, row in reversed(list(enumerate(board))):
		cols = st.columns([33,10,10,10,10,10,10,10,33])
		for j, field in enumerate(row):
			cols[j + 1].markdown(f'{draw_piece(board[i,j])}', unsafe_allow_html=True)


def draw_piece(player):
	if player == 1:
		piece = '🔴'
	elif player == 2:
		piece = '🟡'
	else:
		piece = '⚪'
	
	return f'<p style="text-align:center; font-size: 25px">{piece}</p>'


def main():

	st.markdown("""<style> 
	#MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
	header {visibility: hidden;} </style>
	""", unsafe_allow_html=True)

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

	if "board" not in st.session_state:
		init()

	draw_board(st.session_state.board)

	if st.session_state.turn == 1 and not st.session_state.game_over:
		ai_turn()
		st.experimental_rerun()

#	st.write(st.session_state)

	if get_valid_locations(st.session_state.board) == []:
		st.session_state.game_over = True

	if st.session_state.game_over:
		if st.session_state.winner == PLAYER:
			st.success("Congrats! you won the game! 🎈")
		elif st.session_state.winner == AI:
			st.error("AI won the game.")
		else:
			st.info('It\'s a tie 📍')


if __name__ == '__main__':
	main()
