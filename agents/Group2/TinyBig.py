import socket
from random import choice
from time import sleep
import heapq


class TinyBigAgent():
    """Uses minimax search with alpha-beta pruning, a low depth factor and a heuristic based on number of unopposed moves to win
    """

    HOST = "127.0.0.1"
    PORT = 1234

    def __init__(self, board_size=11):
        self.s = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        )

        self.s.connect((self.HOST, self.PORT))
        
        self.last_message = None  # Initialize last_message
        self.board_size = board_size
        self.board = []
        self.colour = ""
        self.turn_count = 0

    def run(self):
        """Reads data until it receives an END message or the socket closes."""

        while True:
            data = self.s.recv(1024)
            if not data:
                break
            # print(f"{self.colour} {data.decode('utf-8')}", end="")
            if (self.interpret_data(data)):
                break

        # print(f"Naive agent {self.colour} terminated")

    def interpret_data(self, data):
        """Checks the type of message and responds accordingly. Returns True
        if the game ended, False otherwise.
        """
        
        messages = data.decode("utf-8").strip().split("\n")
        messages = [x.split(";") for x in messages]
        # print(messages)
        for s in messages:
            self.last_message = s[0]  # Update last_message with the current message type
            if s[0] == "START":
                self.board_size = int(s[1])
                self.colour = s[2]
                self.board = [
                    [0]*self.board_size for i in range(self.board_size)]

                if self.colour == "R":
                    self.make_move()

            elif s[0] == "END":
                return True

            elif s[0] == "CHANGE":
                if s[3] == "END":
                    return True

                elif s[1] == "SWAP":
                    self.colour = self.opp_colour()
                    if s[3] == self.colour:
                        self.make_move()

                elif s[3] == self.colour:
                    action = [int(x) for x in s[1].split(",")]
                    self.board[action[0]][action[1]] = self.opp_colour()

                    self.make_move()

        return False

    def apply_temp_move(self, move, colour, board):
        x, y = move
        temp_board = [row[:] for row in board]  # Create a deep copy of the given board
        temp_board[x][y] = colour  # Apply the move
        return temp_board


    def minimax(self, board, depth, alpha, beta, maximizing_player):
        if depth == 0 or self.is_game_over(self.last_message):
            return self.heuristic(board) 

        if maximizing_player:
            max_eval = float('-inf')
            for move in self.get_moves(board):  
                temp_board = self.apply_temp_move(move, self.colour, board)  # Simulate the move on the given board
                eval = self.minimax(temp_board, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in self.get_moves(board):
                temp_board = self.apply_temp_move(move, self.opp_colour(), board) # Use the opponent's colour
                eval = self.minimax(temp_board, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

        
    def make_move(self):
        best_score = float('-inf')
        best_move = None

        for move in self.get_moves(self.board):  # Pass the current board as an argument
            temp_board = self.apply_temp_move(move, self.colour, self.board)  # Pass the board to apply_temp_move
            score = self.minimax(temp_board, 3, float('-inf'), float('inf'), True)
            if score > best_score:
                best_score = score
                best_move = move

        if best_move:
            self.s.sendall(bytes(f"{best_move[0]},{best_move[1]}\n", "utf-8"))
            self.board[best_move[0]][best_move[1]] = self.colour



    def is_game_over(self, last_message):
        """Check if the game is over based on the last received message.""" #idk if this needed
        parts = last_message.split(';')
        if parts[0] == "END":
            return True
        else:
            return False



    def get_moves(self, board):
        moves = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                if board[x][y] == 0:
                    moves.append((x, y))
        return moves

    def opp_colour(self):
        """Returns the char representation of the colour opposite to the
        current one.
        """
        if self.colour == "R":
            return "B"
        elif self.colour == "B":
            return "R"
        else:
            return "None"

    def djikstra(self, start_pos, board, colour):
        opp_colour = "R" if colour == "B" else "B"

        #Return infinity if impossible path
        if (board[start_pos[0]][start_pos[1]] == opp_colour):
            return float("inf")

        #Holds the distance of the hex (r, c) from the source, where the distance to the opposite end of the board represents
        #the number of moves needed to win
        distance_start = dict()
        #Holds the predecessor of hex (r, c)
        prev_nodes = dict()
        #Holds the hexes sorted by distance
        queue = []

        #Add the starting hex to the queue
        heapq.heappush(queue, (0, start_pos))

        #Set the distance of each hex from the start pos
        for r in range(self.board_size):
            for c in range(self.board_size):
                hex = (r, c)
                distance_start[hex] = float("inf")
                prev_nodes[hex] = None
        
        if board[start_pos[0]][start_pos[1]] == colour:
            distance_start[start_pos] = 0
        else:
            distance_start[start_pos] = 1

        while len(queue) != 0:
            current_hex = heapq.heappop(queue)[1]

            #I think this should work now? Cause it's always gonna find traverse the shortest path first with the priority queue
            if (colour == "R" and current_hex[0] == self.board_size-1) or (colour == "B" and current_hex[1] == self.board_size-1):
                return distance_start[current_hex]

            neighbours = []
            
            #A hex is a neighbour only if it is adjacent and not the opponent's colour
            if current_hex[1] > 1 or (colour == "R" and current_hex[1] > 0):
                #Get the neighbour on the left
                if board[current_hex[0]][current_hex[1]-1] != opp_colour:
                    neighbours.append((current_hex[0], current_hex[1]-1))
            if current_hex[1] < self.board_size-1:
                #Get the neighbour on the right
                if board[current_hex[0]][current_hex[1]+1] != opp_colour:
                    neighbours.append((current_hex[0], current_hex[1]+1))
            
            if current_hex[0] > 1 or (colour == "B" and current_hex[0] > 0):
                #Get the neighbour directly above
                if board[current_hex[0]-1][current_hex[1]] != opp_colour:
                    neighbours.append((current_hex[0]-1, current_hex[1]))
                #Get the neighbour to the top right
                if current_hex[1] < self.board_size-1 and board[current_hex[0]-1][current_hex[1]+1] != opp_colour:
                    neighbours.append((current_hex[0]-1, current_hex[1]+1))
            if current_hex[0] < self.board_size-1:
                #Get the neighbour directly below
                if board[current_hex[0]+1][current_hex[1]] != opp_colour:
                    neighbours.append((current_hex[0]+1, current_hex[1]))
                #Get the neighbour to the bottom left
                if current_hex[1] > 1 and board[current_hex[0]+1][current_hex[1]-1] != opp_colour:
                    neighbours.append((current_hex[0]+1, current_hex[1]-1))

            for neighbour in neighbours:
                #If the neighbour is our colour, don't increase the distance, otherwise increase it by 1
                alt_distance = distance_start[current_hex]
                if board[neighbour[0]][neighbour[1]] != colour:
                    alt_distance = alt_distance + 1
                
                #If the new distance to this hex is less that current known distance, update it and add it to the queue
                if alt_distance < distance_start[neighbour]:
                    distance_start[neighbour] = alt_distance
                    prev_nodes[neighbour] = current_hex
                    #Remove node if in queue
                    queue = [q for q in queue if q[1] != neighbour]
                    heapq.heapify(queue)
                    heapq.heappush(queue, (alt_distance, neighbour))
        
    #Takes a board and returns the number of plays the opponent needs to win - the number of plays the opponent needs
    def djikstra_heuristic(self, board):
        least_moves_win = float("inf")
        least_moves_lose = float("inf")
        for i in range(0, self.board_size):
            if self.colour == "B":
                start_pos = (0, i)
                opp_start_pos = (i, 0)
            else:
                start_pos = (i, 0)
                opp_start_pos = (0, i)
            
            self_moves = self.djikstra(start_pos, board, self.colour)
            if self_moves < least_moves_win:
                least_moves_win = self_moves

            opp_moves = self.djikstra(opp_start_pos, board, self.opp_colour)
            if opp_moves < least_moves_lose:
                least_moves_lose = opp_moves
        
        return least_moves_win - least_moves_lose

if (__name__ == "__main__"):
    agent = TinyBigAgent()
    agent.run()
