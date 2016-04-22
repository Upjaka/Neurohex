import argparse
from program import Program
import threading
import time
from gamestate import gamestate
import sys

class agent:
	def __init__(self, exe):
		self.exe = exe 
		self.program = Program(self.exe, True)
		self.name = self.program.sendCommand("name").strip()
		self.lock  = threading.Lock()

	def sendCommand(self, command):
		self.lock.acquire()
		answer = self.program.sendCommand(command)
		self.lock.release()
		return answer

	def reconnect(self):
		self.program.terminate()
		self.program = Program(self.exe,True)
		self.lock = threading.Lock()

def move_to_cell(move):
	x =	ord(move[0].lower())-ord('a')
	y = int(move[1:])-1
	return (x,y)

def run_game(blackAgent1, blackAgent2, whiteAgent1, whiteAgent2, num_moves, boardsize, verbose = False):
	game = gamestate(boardsize)
	winner = None
	moves = []
	blackAgent1.sendCommand("clear_board")
	if(blackAgent2): blackAgent2.sendCommand("clear_board")
	whiteAgent1.sendCommand("clear_board")
	if(whiteAgent2): whiteAgent2.sendCommand("clear_board")
	move_count = 0
	while(True):
		move_count +=1
		if(move_count <= num_moves):
			move = blackAgent1.sendCommand("genmove black").strip()
			if(blackAgent2): blackAgent2.sendCommand("play black "+move)
		else:
			if(blackAgent2): 
				move = blackAgent2.sendCommand("genmove black").strip()
				blackAgent1.sendCommand("play black "+move)
			else:
				move = blackAgent1.sendCommand("genmove black").strip()
		if( move == "resign"):
			winner = game.PLAYERS["white"]
			return winner
		moves.append(move)
		game.place_black(move_to_cell(move))
		whiteAgent1.sendCommand("play black "+move)
		if(whiteAgent2): whiteAgent2.sendCommand("play black "+move)
		if verbose:
			print(blackAgent1.name+(" and "+blackAgent2.name if blackAgent2 else " " )+" v.s. "+whiteAgent1.name+(" and " + whiteAgent2.name if whiteAgent2 else " "))
			print(game)
		if(game.winner() != game.PLAYERS["none"]):
			winner = game.winner()
			break
		sys.stdout.flush()
		if(move_count <= num_moves):
			move = whiteAgent1.sendCommand("genmove white").strip()
			if(whiteAgent2): whiteAgent2.sendCommand("play white "+move)
		else:
			if(whiteAgent2):
				move = whiteAgent2.sendCommand("genmove white").strip()
				whiteAgent1.sendCommand("play white "+move)
			else:
				move = whiteAgent1.sendCommand("genmove white").strip()
	        if( move == "resign"):
                	winner = game.PLAYERS["black"] 
        		return winner
		moves.append(move)
		game.place_white(move_to_cell(move))
		blackAgent1.sendCommand("play white "+move)
		if(blackAgent2): blackAgent2.sendCommand("play white "+move)
		if verbose: 
			print(blackAgent1.name+(" and "+blackAgent2.name if blackAgent2 else " " )+" v.s. "+whiteAgent1.name+(" and " + whiteAgent2.name if whiteAgent2 else " "))
			print(game)
		if(game.winner() != game.PLAYERS["none"]):
			winner = game.winner()
			break
		sys.stdout.flush()
	winner_name = blackAgent1.name+(" and "+blackAgent2.name if blackAgent2 else " ") if winner == game.PLAYERS["black"] else whiteAgent1.name+(" and "+whiteAgent2.name if whiteAgent2 else " ")
	loser_name =  whiteAgent1.name+(" and "+whiteAgent2.name if whiteAgent2 else " ") if winner == game.PLAYERS["black"] else blackAgent1.name+(" and "+blackAgent2.name if blackAgent2 else " ")
	print("Game over, " + winner_name+ " ("+game.PLAYER_STR[winner]+") " + "wins against "+loser_name)
	print(game)
	print(" ".join(moves))
	return winner

mohex_exe = "/cshome/kjyoung/Summer_2015/benzene-vanilla/src/mohex/mohex 2>/dev/null"
neurohex_exe = "/cshome/kjyoung/Summer_2015/Neurohex/playerAgents/program.py 2>/dev/null"

parser = argparse.ArgumentParser(description="Run tournament against mohex and output results.")
parser.add_argument("num_games", type=int, help="number of *pairs* of games (one as black, one as white) to play between each pair of agents.")
parser.add_argument("--time", "-t", type=int, help="total time allowed for gitkeach move in seconds.")
parser.add_argument("--num_moves", "-n", type=int, help="number of opening moves generated by neuohex")
args = parser.parse_args()

print("Starting tournament...")
mohex1 = agent(mohex_exe)
mohex2 = agent(mohex_exe)
num_games = args.num_games
if(args.time):
	time = args.time
else:
	time = 5
if(args.num_moves):
	num_moves = args.num_moves
else:
	num_moves = 10
mohex1.sendCommand("param_mohex max_time "+str(time))
mohex2.sendCommand("param_mohex max_time "+str(time))

neurohex = agent(neurohex_exe)
white_wins = 0
black_wins = 0
for game in range(num_games):
	winner = run_game(mohex1, neurohex, mohex2, None, num_moves, 13, True)
	if(winner == gamestate.PLAYERS["black"]):
		white_wins += 1
	winner = run_game(mohex2, None	, mohex1, neurohex, num_moves, 13, True)
	if(winner == gamestate.PLAYERS["white"]):
		black_wins += 1

print "win_rate as white: "+str(white_wins/float(num_games)*100)[0:5]+"%"
print "win_rate as black: "+str(black_wins/float(num_games)*100)[0:5]+"%"



