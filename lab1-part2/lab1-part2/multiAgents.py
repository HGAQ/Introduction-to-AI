# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
from math import sqrt, log

from game import Agent



class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        # NOTE: this is an incomplete function, just showing how to get current state of the Env and Agent.

        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 1)
    """

    def getAction(self, gameState):
      currentdepth = self.depth
      legalMoves = gameState.getLegalActions(0)
      max_val = -1145141919810
      chosen_action = None
      if currentdepth != 0:
        for action in legalMoves:
          value = self.min_value(gameState.generateSuccessor(0,action),currentdepth,1)
          if max_val < value:
            max_val = value
            chosen_action = action
      return chosen_action
    def max_value(self, gameState, currentdepth=0,agentindex=0):
      max_val = -1145141919810
      legalMoves = gameState.getLegalActions(0)
      if len(gameState.getLegalActions(0)) == 0:
        return self.evaluationFunction(gameState)
      if currentdepth != 0:
        for action in legalMoves:
          value = self.min_value(gameState.generateSuccessor(0,action),currentdepth,1)
          if max_val < value:
            max_val = value
        return max_val
      else:
        return self.evaluationFunction(gameState)
      
    def min_value(self, gameState, currentdepth=0,agentindex=1):
      min_val = 1145141919810
      legalMoves = gameState.getLegalActions(agentindex)
      if len(gameState.getLegalActions(agentindex)) == 0:
            return self.evaluationFunction(gameState)
      if currentdepth != 0:
        for action in legalMoves:
          if(agentindex==gameState.getNumAgents()-1):
            value = self.max_value(gameState.generateSuccessor(agentindex,action),currentdepth-1,0)
          else:
            value = self.min_value(gameState.generateSuccessor(agentindex,action),currentdepth,agentindex+1)
          if min_val > value:
            min_val = value
        return min_val
      else:
        return self.evaluationFunction(gameState)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 2)
    """

    #alpha is the max of the evaluation
    #beta is the max of the component evaluation
    def getAction(self, gameState):
      currentdepth = self.depth
      legalMoves = gameState.getLegalActions(0)
      max_val = -1145141919810
      alpha=-1145141919810
      beta=1145141919810
      chosen_action = None
      if currentdepth != 0:
        for action in legalMoves:
          value = self.min_value(gameState.generateSuccessor(0,action),currentdepth,1,alpha, beta)
          if max_val < value:
            max_val = value
            chosen_action = action
          alpha = max(alpha,max_val)
          if beta <= alpha:
            break
      return chosen_action
    
    
    def max_value(self, gameState, currentdepth=0, agentindex=0, alpha=-1145141919810, beta=11451419191810):
      max_val = -1145141919810
      legalMoves = gameState.getLegalActions(0)
      if len(gameState.getLegalActions(0)) == 0:
        return self.evaluationFunction(gameState)
      if currentdepth != 0:
        for action in legalMoves:
          value = self.min_value(gameState.generateSuccessor(0,action),currentdepth,1,alpha,beta)
          max_val = max(max_val,value)
          alpha = max(alpha,max_val)
          if beta < alpha:
            break
        return max_val
      else:
        return self.evaluationFunction(gameState)
      
      
      
    def min_value(self, gameState, currentdepth=0, agentindex=1, alpha=-1145141919810, beta=11451419191810):
      min_val = 1145141919810
      legalMoves = gameState.getLegalActions(agentindex)
      if len(gameState.getLegalActions(agentindex)) == 0:
            return self.evaluationFunction(gameState)
      if currentdepth != 0:
        for action in legalMoves:
          if(agentindex==gameState.getNumAgents()-1):
            value = self.max_value(gameState.generateSuccessor(agentindex,action),currentdepth-1,0,alpha,beta)
          else:
            value = self.min_value(gameState.generateSuccessor(agentindex,action),currentdepth,agentindex+1,alpha,beta)
          min_val = min(min_val,value)
          beta = min(beta,min_val)
          if alpha > beta:
            break
        return min_val
      else:
        return self.evaluationFunction(gameState)

class MCTSAgent(MultiAgentSearchAgent):
    """
      Your MCTS agent with Monte Carlo Tree Search (question 3)
    """

    def getAction(self, gameState):

        class Node:
            def __init__(self, data):
                self.child = [None, None, None, None, None]
                self.parent = None
                self.gamestate = data[0]
                self.agentid = data[1]
                self.numerator = data[2]
                self.denominator = data[3]
        
        
        agent_num = gameState.getNumAgents()
        
        
        def HeuristicFunction(node):
          heuristic=0
          #pos_food=node.gamestate.getFood()
          pos_pacman=node.gamestate.getPacmanPosition()
          pos_ghost_not_scared = []
          for index in range(agent_num):
            if index != 0:
              if gameState.getGhostState(index).scaredTimer == 0:
                pos_ghost_not_scared.append(gameState.getGhostPosition(index))
          '''min_food_dis=100000000000
          for food in pos_food:
            food_dis=manhattanDistance(pos_pacman, food)
            if (min_food_dis>food_dis):
              min_food_dis=food_dis
          heuristic-=min_food_dis'''
          min_dis = 1145141919810
          for ghost in pos_ghost_not_scared:
            dis = manhattanDistance(pos_pacman, ghost)
            if min_dis > dis:
              min_dis = dis
              heuristic = dis
          return heuristic
        
        
        
#################################  INITIALIZE  ###################################################

        index_position={0:'North',1:'East',2:'West',3:'South',4:'Stop'}
        position_index={'North':0,'East':1,'West':2,'South':3,'Stop':4}
        
        
        search_times=200 #搜索最多次数
        dangerous_length=3 #危险预警
        Depth=16 #模拟深度
        
        #设置根节点
        data = [gameState, 0, 0, 0]
        root = Node(data) 
        root_heur=HeuristicFunction(root)
        
        #一些全局变量
        agent_num = gameState.getNumAgents()
        pos_ghost_not_scared0 = []
        for index in range(agent_num):
            if index != 0:
              if gameState.getGhostState(index).scaredTimer == 0:
                pos_ghost_not_scared0.append(gameState.getGhostPosition(index))
        pos_pacman0 = gameState.getPacmanPosition()
        
        pos_capsule = gameState.getCapsules()  
        pos_food = gameState.getFood().asList()
        can_eaten = pos_capsule + pos_food
                
        
############################  FUNCTION OF CLASSES  #############################################################
        
        def is_dangerous(dangerous_length):
          is_danger = 0
          for ghost_pos in pos_ghost_not_scared0:
            if manhattanDistance(pos_pacman0, ghost_pos) <= dangerous_length:
              is_danger = 1
          return is_danger


        #定义选择函数（Selection）：
        #从节点的子节点中选择最优的节点，并计算其UCB值，
        #如果最优节点存在，则继续递归调用Selection函数，否则返回该节点。
        def Selection(node):
          best_childnode = None
          max_UCB = -1145141919810
          if node.gamestate.isWin():
            return 1, node
          elif node.gamestate.isLose():
            return 0, node
          else:
            for childnode in node.child:
              if childnode!=None:
                if childnode.denominator == 0:
                  return -1, childnode
                UCB=Evaluate(childnode)
                if max_UCB < UCB:
                  best_childnode = childnode
                  max_UCB = UCB  
            if best_childnode != None:
              return Selection(best_childnode)
            else:
              return -1, node
          
        #定义计算UCB值函数（Evaluate）
        def Evaluate(node):
          if node.denominator != 0 and node.parent != None:
            return sqrt(2*log(node.parent.denominator)/node.denominator)+node.numerator/node.denominator
          elif node.denominator == 0:
            return 1145141919810
          elif node.parent == None:
            print("wtf")


        #定义扩展函数（Expansion）：
        #获取当前节点的合法动作，如果没有合法动作则返回False
        #否则根据合法动作生成子节点，并且绑定亲子关系，最后从子节点中随机选择一个返回。
        def Expansion(node):
          legalMoves = node.gamestate.getLegalActions(node.agentid)
          if len(legalMoves) == 0:
            return False,node
          else:
            children = []
            next_agentid = (node.agentid + 1) % agent_num
            for move in legalMoves:
              if( move == 'Stop'):
                continue
              child_data = [node.gamestate.generateSuccessor(node.agentid, move), next_agentid, 0, 0]
              child_node = Node(child_data)
              #绑定亲子关系
              node.child[position_index[move]] = child_node
              children.append(child_node)
              child_node.parent=node
            if len(children)==0:
              return False, node
            else:
              return True, children[random.randint(0,len(children)-1)]
        
        
        #定义模拟函数（Simulation）：
        # 给定一个节点，模拟游戏的进行，模拟的深度为Depth
        # 模拟的过程中，每一步都从当前状态的合法行动中随机选择一个行动，如果游戏结束，则返回1或0
        # 否则，返回当前节点的启发函数值是否大于根节点的启发函数值减去模拟深度。
        def Simulation(node):
          stimulate_state=node.gamestate
          stimulate_agent=node.agentid
          for step in range(Depth):
            if stimulate_state.isWin():
              return 1
            elif stimulate_state.isLose():
              return 0
            else:
              legalMoves = stimulate_state.getLegalActions(stimulate_agent)
              if len(legalMoves) == 0:
                return 0
              else:
                stimulate_state = stimulate_state.generateSuccessor(stimulate_agent,random.choice(legalMoves))
                stimulate_agent = (stimulate_agent+1) % agent_num
          #return 1
          return HeuristicFunction(node) >  root_heur
            

        def Backpropagation(node, WinorLose):
          while node is not None:
            if node.agentid == 0:
              node.numerator += WinorLose
            else:
              if WinorLose == 0:
                node.numerator += 1
            node.denominator += 1
            node = node.parent

            
        
        #定义搜索函数（search）：
        #  循环times次：
        #    选择没有子节点的节点；
        #    扩展该节点；
        #    如果可以扩展，则进行模拟；
        #    否则，设置胜负结果为0；
        #    回溯。
        def search(times, root):
          for i in range(times):
            win_or_lose, node_with_no_child = Selection(root)
            if win_or_lose == 1:
              Backpropagation(node_with_no_child, win_or_lose)
              continue
            elif win_or_lose == 0:
              Backpropagation(node_with_no_child, win_or_lose)
              continue
            else:
              can_expand,random_child_node=Expansion(node_with_no_child)
              if can_expand:
                win_or_lose = Simulation(random_child_node)
              else:
                win_or_lose = 0
              Backpropagation(random_child_node, win_or_lose)
        
        
        #定义final_Selection（root）函数：
        # 遍历root的子节点，计算每个子节点的比率，找出比率最大的子节点，并返回其索引位置。
        def final_Selection(root):
          best = None
          i = 0
          max_rate = -1145141919810
          for i in range(5):
            if root.child[i] != None:
              root.child[i].numerator=root.child[i].denominator-root.child[i].numerator
              rate = Evaluate(root.child[i])
              if max_rate < rate:
                best = i
                max_rate = rate
          return index_position[best]
          
        def find_food(root):
          Frontier = util.Queue()
          Visited = []
          Frontier.push( (root.gamestate, []) )
          Visited.append( root.gamestate )
          while Frontier.isEmpty() == 0:
              state, actions = Frontier.pop()
              legalMoves = state.getLegalActions(0)
              if state.getPacmanPosition() in can_eaten:
                  #print 'Find Goal'
                  return actions[0]
              for next in legalMoves:
                  n_state = state.generateSuccessor(0,next)
                  n_direction = next
                  if n_state not in Visited:
                      Frontier.push( (n_state, actions + [n_direction]) )
                      Visited.append( n_state )
        
############################  MAIN FUNCTION  #######################################################
        if is_dangerous(dangerous_length) == 1:
          search(search_times,root)
          return final_Selection(root)
        else:
          return find_food(root)
##################################################################################################