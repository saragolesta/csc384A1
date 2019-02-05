#   Look for #IMPLEMENT tags in this file. These tags indicate what has
#   to be implemented to complete the warehouse domain.

#   You may add only standard python imports---i.e., ones that are automatically
#   available on TEACH.CS
#   You may not remove any imports.
#   You may not import or otherwise source any of your own files

import os #for time functions
from search import * #for search engines
from sokoban import SokobanState, Direction, PROBLEMS #for Sokoban specific classes and problems
import numpy as np
from scipy.optimize import linear_sum_assignment

def sokoban_goal_state(state):
  '''
  @return: Whether all boxes are stored.
  '''
  for box in state.boxes:
    if box not in state.storage:
      return False
  return True

def heur_manhattan_distance(state):
#IMPLEMENT
    '''admissible sokoban puzzle heuristic: manhattan distance'''
    '''INPUT: a sokoban state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state to the goal.'''
    #We want an admissible heuristic, which is an optimistic heuristic.
    #It must never overestimate the cost to get from the current state to the goal.
    #The sum of the Manhattan distances between each box that has yet to be stored and the storage point nearest to it is such a heuristic.
    #When calculating distances, assume there are no obstacles on the grid.
    #You should implement this heuristic function exactly, even if it is tempting to improve it.
    #Your function should return a numeric value; this is the estimate of the distance to the goal.
    distances = 0
    for box in state.boxes:
        min_distance = float("inf")
        for goal in state.storage:
            temp = abs(box[0] - goal[0]) + abs(box[1] - goal[1])
            if(temp < min_distance):
                min_distance = temp
        distances += min_distance

    return distances


#SOKOBAN HEURISTICS
def trivial_heuristic(state):
  '''trivial admissible sokoban heuristic'''
  '''INPUT: a sokoban state'''
  '''OUTPUT: a numeric value that serves as an estimate of the distance of the state (# of moves required to get) to the goal.'''
  count = 0
  for box in state.boxes:
    if box not in state.storage:
        count += 1
  return count

def box_on_edge(state, box, storage):
  # Check if the box is next to the border of the grid and there is no storage along the wall
  if ((((box[0] == 0) or (box[0] == state.width - 1)) and (box[0] - storage[0] != 0)) or
    (((box[1] == state.height - 1) or (box[1] == 0)) and (box[1] - storage[1] != 0))) :
      return True  
  return False

def box_in_corner(state, box):
  obstacle_left =  (((box[0] - 1, box[1]) in state.obstacles))
  obstacle_right = (((box[0] + 1, box[1]) in state.obstacles))
  obstacle_above = (((box[0], box[1] + 1) in state.obstacles))
  obstacle_below = (((box[0], box[1] - 1) in state.obstacles))
  #check if the box is in corner 
  return (obstacle_left or obstacle_right) and (obstacle_above or obstacle_below)

def heur_alternate(state):
#IMPLEMENT
    '''a better heuristic'''
    '''INPUT: a sokoban state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state to the goal.'''
    #heur_manhattan_distance has flaws.
    #Write a heuristic function that improves upon heur_manhattan_distance to estimate distance between the current state and the goal.
    #Your function should return a numeric value for the estimate of the distance to the goal.

    # If all boxes were in a storage spot return 0 immediately to save time
    if(state.boxes.issubset(state.storage)):
      return 0
    remaining_boxes = state.boxes - (state.boxes & state.storage)
    remaining_storages = state.storage - (state.boxes & state.storage)

    costs = np.zeros((len(remaining_boxes), len(remaining_storages)))
    for box_idx, box in enumerate(remaining_boxes):
      for s_idx, s in enumerate(remaining_storages):
        manhattan_dist = abs(box[0] - s[0]) + abs(box[1] - s[1])
        costs[box_idx][s_idx] = manhattan_dist
        if(box_on_edge(state, box, s) or box_in_corner(state, box)):
          #Assign a high cost for these deadlock cases
          costs[box_idx][s_idx] = 2**31
    row_ind, col_ind = linear_sum_assignment(costs)

    # We have to account for robot moves when they are on a stoarge spot
    robot_moves = len(frozenset(state.robots) & state.storage)
    return costs[row_ind, col_ind].sum() + robot_moves

def heur_zero(state):
    '''Zero Heuristic can be used to make A* search perform uniform cost search'''
    return 0

def fval_function(sN, weight):
#IMPLEMENT
    """
    Provide a custom formula for f-value computation for Anytime Weighted A star.
    Returns the fval of the state contained in the sNode.

    @param sNode sN: A search node (containing a SokobanState)
    @param float weight: Weight given by Anytime Weighted A star
    @rtype: float
    """
  
    #Many searches will explore nodes (or states) that are ordered by their f-value.
    #For UCS, the fvalue is the same as the gval of the state. For best-first search, the fvalue is the hval of the state.
    #You can use this function to create an alternate f-value for states; this must be a function of the state and the weight.
    #The function must return a numeric f-value.
    #The value will determine your state's position on the Frontier list during a 'custom' search.
    #You must initialize your search engine object as a 'custom' search engine if you supply a custom fval function.
    return sN.gval + (weight * sN.hval) 

def anytime_weighted_astar(initial_state, heur_fn, weight=1., timebound = 10):
#IMPLEMENT
  '''Provides an implementation of anytime weighted a-star, as described in the HW1 handout'''
  '''INPUT: a sokoban state that represents the start state and a timebound (number of seconds)'''
  '''OUTPUT: A goal state (if a goal is found), else False'''
  '''implementation of weighted astar algorithm'''
  curr_time = os.times()[0]
  wrapped_fval_function = (lambda sN : fval_function(sN,weight))
  searcher = SearchEngine(strategy='custom', cc_level='default')
  #searcher.trace_on()
  searcher.init_search(initial_state, sokoban_goal_state, heur_fn, wrapped_fval_function)
  # Initial costbound all set to infinity
  costbound = (float("inf"), float("inf"), float("inf"))
  path = searcher.search(timebound)
  end_time = curr_time + timebound
  remaining_timebound = timebound
  best_path = False
  if (weight > 0):
    weight -= 1
  #Making sure the function returns when the timebound is reached
  while (curr_time < end_time):
    # If there was no solution (best_path was false)
    if (not path):
      print(False)
      return best_path
    if (path.gval < costbound[0]):
        costbound = (path.gval, path.gval, path.gval)
        best_path = path
    prev_time = curr_time
    curr_time = os.times()[0]
    delta_t = curr_time - prev_time
    remaining_timebound = remaining_timebound - delta_t
    path = searcher.search(remaining_timebound, costbound)
    if (weight > 0):
        weight -= 1
  print(best_path.gval)
  return best_path

def anytime_gbfs(initial_state, heur_fn, timebound = 10):
#IMPLEMENT
  '''Provides an implementation of anytime greedy best-first search, as described in the HW1 handout'''
  '''INPUT: a sokoban state that represents the start state and a timebound (number of seconds)'''
  '''OUTPUT: A goal state (if a goal is found), else False'''
  '''implementation of weighted astar algorithm'''
  curr_time = os.times()[0]
  searcher = SearchEngine(strategy='best_first', cc_level='default')
  searcher.init_search(initial_state, sokoban_goal_state, heur_fn)

  # Initial costbound all set to infinity
  costbound = (float("inf"), float("inf"), float("inf"))
  path = searcher.search(timebound)
  end_time = curr_time + timebound
  remaining_timebound = timebound
  best_path = False

    # Perform the search while the timebound has not been reached
  while (curr_time < end_time):
    # If there was no solution (best_path was false)
    if (not path):
      return best_path
    if (path.gval < costbound[0]):
      costbound = (path.gval, path.gval, path.gval)          
      best_path = path
    prev_time = curr_time
    curr_time = os.times()[0]
    delta_t = curr_time - prev_time
    remaining_timebound = remaining_timebound - delta_t
    path = searcher.search(remaining_timebound, costbound)
  return best_path

