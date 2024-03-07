from TinyBig import TinyBigAgent  
import heapq


class DjikstraTest():

    def __init__(self, board_size=11):
        self.colour = "B"
        self.opp_colour = "R"
        self.board_size = board_size
        self.board = [[0]*self.board_size for i in range(self.board_size)]

        #RIGHT NOW THIS IS ONLY FOR THE AGENT SEEING HOW MANY MOVES IT NEEDS, IT WONT WORK FOR THE OPPONENT
        # pass colour to make independent of player
        # use board not self.board
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
                
                #Check for bridges
                predecessor = prev_nodes[current_hex]
                #If our current hex has no predecessor, its an edge piece, so we can assume our predecessor is our colour
                if predecessor == None:
                    predecessor_colour = colour
                else:
                    predecessor_colour = board[predecessor[0]][predecessor[1]]
                #If the current neighbour and previous node are our colour, and the current hex is empty
                if board[current_hex[0]][current_hex[1]] != colour and board[neighbour[0]][neighbour[1]] == colour and predecessor_colour == colour and alt_distance == distance_start[neighbour]:
                    #And if the grandparent (2nd predecessor) of the current neighbour is the same as the current hexes predecessor
                    if prev_nodes[neighbour] != None and prev_nodes[prev_nodes[neighbour]] == predecessor:
                        alt_distance = alt_distance - 0.5

                #If the new distance to this hex is less that current known distance, update it and add it to the queue
                if alt_distance < distance_start[neighbour]:
                    distance_start[neighbour] = alt_distance
                    prev_nodes[neighbour] = current_hex
                    #Remove node if in queue
                    queue = [q for q in queue if q[1] != neighbour]
                    heapq.heapify(queue)
                    heapq.heappush(queue, (alt_distance, neighbour))

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

agent = DjikstraTest()
agent.board[1][1] = "B"
print(agent.djikstra((0, 0), agent.board, "B"))
