class TicTacToe:
    def __init__(self, board=None, is_x_turn=None):
        if board is None:
            self.board = [' '] * 9
        else:
            self.board = board.copy()  # board can only contain ' ', 'X', 'O'
        if is_x_turn is None:
            self.is_x_turn = True
        else:
            self.is_x_turn = is_x_turn

    def display_board(self):
        print('+-----------------+')
        print('|  %s  |  %s  |  %s  |' % (self.board[0], self.board[1], self.board[2]))
        print('+-----------------+')
        print('|  %s  |  %s  |  %s  |' % (self.board[3], self.board[4], self.board[5]))
        print('+-----------------+')
        print('|  %s  |  %s  |  %s  |' % (self.board[6], self.board[7], self.board[8]))
        print('+-----------------+')

    def is_board_full(self):
        return not any([space == ' ' for space in self.board])

    def is_player_win(self, player):
        check_pos_list = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                          (0, 3, 6), (1, 4, 7), (2, 5, 8),
                          (0, 4, 8), (2, 4, 6)]
        for a, b, c in check_pos_list:
            if player == self.board[a] == self.board[b] == self.board[c]:
                return True
        else:
            return False

    def is_game_end(self):
        return self.is_board_full() or self.is_player_win('X') or self.is_player_win('O')

    def get_legal_actions(self):
        return [i for i in range(0, 9) if self.board[i] == ' ']

    def is_valid_move(self, action):
        return self.board[action] == ' '

    def make_move(self, move):
        if self.is_x_turn:
            chess = 'X'
        else:
            chess = 'O'

        if self.is_valid_move(move):
            self.board[move] = chess
            self.is_x_turn = not self.is_x_turn
        else:
            raise ValueError("Move is not valid. ")

    def generate_successors(self):
        res = []
        moves = self.get_legal_actions()
        for move in moves:
            game_state_copy = TicTacToe(self.board, self.is_x_turn)
            game_state_copy.make_move(move)
            res.append(game_state_copy)
        return res

    def play_game(self):
        while not self.is_game_end():
            self.display_board()
            move = int(input('position: '))
            self.make_move(move)
            # move = MinimaxAgent().get_optimal_move(self)
        else:
            self.display_board()
            if self.is_player_win('X'):
                print('X wins. ')
            elif self.is_player_win('O'):
                print('O wins. ')


def utility(state):
    if state.is_player_win('X'):
        return 1
    elif state.is_player_win('O'):
        return -1
    elif not state.is_board_full():
        return None
    else:
        return 0


class MinimaxAgent:
    # def get_optimal_move(self, state):
    #     board = state.board
    #     best_value = 0
    #     possible_moves = state.get_legal_actions()
    #     for move in possible_moves:
    #         board[move] = 'X'
    #         value = self.min_value(state)
    #         board[move] = ' '
    #         if value < best_value:
    #             best_value = value
    #             optimal_action = move
    #     return optimal_action
    def minimax(self, state):
        if state.is_x_turn:
            return self.max_value(state)
        else:
            return self.min_value(state)

    def max_value(self, state):
        if utility(state) is not None:
            return utility(state)

        successors = state.generate_successors()
        max_value = max([self.min_value(s) for s in successors] + [float('-inf')])

        return max_value

    def min_value(self, state):
        if utility(state) is not None:
            return utility(state)

        successors = state.generate_successors()

        min_value = min([self.max_value(s) for s in successors] + [float('inf')])

        return min_value


if __name__ == '__main__':
    s1 = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X']
    s2 = ['O', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X']
    s3 = ['O', ' ', ' ', 'X', ' ', ' ', ' ', ' ', 'X']
    s4 = ['O', 'O', ' ', 'X', ' ', ' ', ' ', ' ', 'X']
    s5 = ['O', 'O', 'X', 'X', ' ', ' ', ' ', ' ', 'X']
    s6 = ['O', 'O', 'X', 'X', ' ', 'O', ' ', ' ', 'X']

    st0 = TicTacToe()
    st1 = TicTacToe(s1, False)
    st2 = TicTacToe(s2, True)
    st3 = TicTacToe(s3, False)
    st4 = TicTacToe(s4, True)
    st5 = TicTacToe(s5, False)
    st6 = TicTacToe(s6, True)

    minimax_agent = MinimaxAgent()

    # to play the game, run the following command:
    # st0.play_game()

    # 3.
    print('S6: ', minimax_agent.minimax(st6))
    print('S0: ', minimax_agent.minimax(st0))

    # 4.
    print('S0: ', minimax_agent.minimax(st0))
    print('S1: ', minimax_agent.minimax(st1))
    print('S2: ', minimax_agent.minimax(st2))
    print('S3: ', minimax_agent.minimax(st3))
    print('S4: ', minimax_agent.minimax(st4))
    print('S5: ', minimax_agent.minimax(st5))
    print('S6: ', minimax_agent.minimax(st6))

    ''' 
    s1 -> s2 is a suboptimal move for O, the optimal move would be:
    +-----------------+
    |     |     |     |
    +-----------------+
    |     |  O  |     |
    +-----------------+
    |     |     |  X  |
    +-----------------+
    s2 -> s3 is a suboptimal move for X, the optimal move would be:
    +-----------------+          +-----------------+
    |  O  |     |     |          |  O  |     |  X  |
    +-----------------+          +-----------------+
    |     |     |     |    or    |     |     |     |
    +-----------------+          +-----------------+
    |  X  |     |  X  |          |     |     |  X  |
    +-----------------+          +-----------------+
    s3 -> s4 is a suboptimal move for X, the optimal move would be:
    +-----------------+          +-----------------+          +-----------------+
    |  O  |     |  O  |          |  O  |     |     |          |  O  |     |     |
    +-----------------+          +-----------------+          +-----------------+
    |  X  |     |     |    or    |  X  |     |  O  |    or    |  X  |  O  |     |
    +-----------------+          +-----------------+          +-----------------+
    |     |     |  X  |          |     |     |  X  |          |     |     |  X  |
    +-----------------+          +-----------------+          +-----------------+
    
    '''
