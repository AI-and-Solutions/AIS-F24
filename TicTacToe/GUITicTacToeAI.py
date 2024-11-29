import pygame
import torch
from NNTicTacToe import TicTacToeNN  # Assuming your model is saved in a file named 'model.py'

# Initialize Pygame
pygame.init()

# Constants
SCREEN_SIZE = 600
GRID_SIZE = 3
CELL_SIZE = SCREEN_SIZE // GRID_SIZE
LINE_WIDTH = 5
CIRCLE_RADIUS = CELL_SIZE // 3
CIRCLE_WIDTH = 10
CROSS_WIDTH = 10
FONT_SIZE = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Initialize the screen
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("Tic Tac Toe: AI vs Human")

# Font
font = pygame.font.Font(None, FONT_SIZE)

# Board
EMPTY = 0
HUMAN = 1
AI = -1

def draw_board(board):
    screen.fill(WHITE)

    # Draw grid lines
    for i in range(1, GRID_SIZE):
        pygame.draw.line(screen, BLACK, (0, i * CELL_SIZE), (SCREEN_SIZE, i * CELL_SIZE), LINE_WIDTH)
        pygame.draw.line(screen, BLACK, (i * CELL_SIZE, 0), (i * CELL_SIZE, SCREEN_SIZE), LINE_WIDTH)

    # Draw Xs and Os
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            center = (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2)
            if board[row][col] == HUMAN:
                pygame.draw.line(screen, BLUE, (center[0] - CIRCLE_RADIUS, center[1] - CIRCLE_RADIUS),
                                 (center[0] + CIRCLE_RADIUS, center[1] + CIRCLE_RADIUS), CROSS_WIDTH)
                pygame.draw.line(screen, BLUE, (center[0] - CIRCLE_RADIUS, center[1] + CIRCLE_RADIUS),
                                 (center[0] + CIRCLE_RADIUS, center[1] - CIRCLE_RADIUS), CROSS_WIDTH)
            elif board[row][col] == AI:
                pygame.draw.circle(screen, RED, center, CIRCLE_RADIUS, CIRCLE_WIDTH)

def check_winner(board):
    # Check rows, columns, and diagonals for a winner
    for row in range(GRID_SIZE):
        if abs(sum(board[row])) == 3:
            return board[row][0]

    for col in range(GRID_SIZE):
        if abs(sum(board[row][col] for row in range(GRID_SIZE))) == 3:
            return board[0][col]

    if abs(sum(board[i][i] for i in range(GRID_SIZE))) == 3:
        return board[0][0]

    if abs(sum(board[i][GRID_SIZE - i - 1] for i in range(GRID_SIZE))) == 3:
        return board[0][GRID_SIZE - 1]

    if all(board[row][col] != EMPTY for row in range(GRID_SIZE) for col in range(GRID_SIZE)):
        return 0  # Draw

    return None  # No winner yet

def get_ai_move(board, model):
    flat_board = [cell for row in board for cell in row]
    board_tensor = torch.tensor(flat_board, dtype=torch.float32).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(board_tensor)
    move = torch.argmax(output).item()
    return divmod(move, GRID_SIZE)  # Convert index to (row, col)

def main():
    # Initialize board
    board = [[EMPTY] * GRID_SIZE for _ in range(GRID_SIZE)]

    # Load the trained model
    
    model = TicTacToeNN()
    state_dict = torch.load("C:/Users/vadiwa/Documents/GitHub/AIS-F24/NNTicTacToe.py")

# Load the state dictionary into the model
    model.load_state_dict(state_dict)

    # Main game loop
    running = True
    current_player = HUMAN
    winner = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN and current_player == HUMAN and winner is None:
                x, y = event.pos
                row, col = y // CELL_SIZE, x // CELL_SIZE
                if board[row][col] == EMPTY:
                    board[row][col] = HUMAN
                    winner = check_winner(board)
                    current_player = AI

        if current_player == AI and winner is None:
            row, col = get_ai_move(board, model)
            if board[row][col] == EMPTY:
                board[row][col] = AI
                winner = check_winner(board)
                current_player = HUMAN

        draw_board(board)

        if winner is not None:
            text = "Draw!" if winner == 0 else ("You Win!" if winner == HUMAN else "AI Wins!")
            text_surface = font.render(text, True, BLACK)
            screen.blit(text_surface, (SCREEN_SIZE // 2 - text_surface.get_width() // 2, SCREEN_SIZE // 2 - FONT_SIZE // 2))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
