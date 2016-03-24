import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

from collections import OrderedDict
import numpy as np


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # Initialize any additional variables here

        # create dictionary with indicies to point to matrix of (state, action) pairs
        self.state_table = OrderedDict()
        self.state_table['light'] = {'green':0, 'red':1}
        self.state_table['oncoming'] = {None:0, 'forward':1, 'left':1, 'right':1}
        self.state_table['waypoint'] = {'forward':0, 'left':1, 'right':2}
        self.state_table['action'] = {None: 0, 'forward':1, 'left':2, 'right':3}
        dims = [2,2,3,4]
        # initialize the policy table with small random numbers
        self.q_table = np.random.uniform(0.0001, 0.0002, dims)
        self.q_visits = np.zeros(dims)
        
        # Q Learning parameters
        self.gamma = 0.3
        self.epsilon = 1.
        self.alpha = .3
        
        # tracking variables
        self.n_moves = 0
        self.moves = []
        self.success = []
        self.sum_penalties = 0
        self.penalties = []
        self.deadlines = []

        # initial parameter setting
        #self.reset()

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.destination = destination
        self.n_moves = 0
        self.sum_penalties = 0

        self.new = 1
        self.alpha = .3
        self.epsilon *= .9

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Update state
        if self.new == 1: #new trip!
            self.deadlines.append(self.env.get_deadline(self))
            self.new = 0
        self.alpha *= .95 
        
        def state_sense():
            i = self.env.sense(self)
            # use 'oncoming' state to represent the presence of ANY traffic
            if i['left'] is not None:
                i['oncoming'] = 'forward'
            if i['right'] is not None:
                i['oncoming'] = 'forward'
            i['waypoint'] = self.next_waypoint
            return i

        def make_state_vector(x):
            # only use keys in state_table for indicies in matrix
            # for determining indicies for q_table list
            v = []
            for key in self.state_table.keys()[:-1]:
                v.append(self.state_table[key][x[key]])
            return tuple(v)

        # Update state
        state0 = state_sense()
        s0 = make_state_vector(state0)
        self.state = state0 

        
        # Select action according to your policy
        visits = [self.q_visits.item(s0+tuple([i])) for i in range(4)]
        pos_q = [self.q_table.item(s0+tuple([i])) for i in range(4)]
        if sum(visits) == 4:
            #print "visited ALL"
            # perform action with epsilon pct
            if random.random() < self.epsilon:
                action_ind = int(random.random() * 3)+1
                #print "* RANDOM *", self.env.valid_actions[action_ind]
            else:
                action_ind = pos_q.index(max(pos_q))
                #print "best", self.env.valid_actions[action_ind]
        elif sum(visits) == 0:
            #print "visited NONE"
            action_ind = np.random.choice(range(4))
        else:
            #print "visited SOME"
            pos_q_ma = np.ma.array(pos_q, mask=visits)
            action_ind = pos_q_ma.argmax()
            
        action = self.env.valid_actions[action_ind]
        # the (state, action) pair
        sa0 = s0+tuple([action_ind])
        self.q_visits.itemset(sa0, 1)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # keep track of variables
        if action is not None:
            self.n_moves +=1
        if reward < 0:
            self.sum_penalties += reward
        if reward >= 10 and deadline > 0:
            print "*** SUCCESS ***"
            self.success.append(1)
            self.moves.append(self.n_moves)
            self.penalties.append(self.sum_penalties)
        elif deadline <= 0:
            print "*** FAIL ***"
            self.success.append(0)
            self.moves.append(self.n_moves)
            self.penalties.append(self.sum_penalties)

        # Learn policy based on state, action, reward
        state1 = state_sense()
        self.state = state1
        s1 = make_state_vector(state1)

        # * Update self.q_table by:
        # * Qhat(s,a) <-alpha-  r + gamma * max(a')Qhat(s',a')
        # rewards for possible actions of new state
        pos_q = [self.q_table.item(s1+tuple([i])) for i in range(4)]
        # right hand side of equation
        x = reward + (self.gamma * max(pos_q))
        cur_q = self.q_table.item(sa0)
        # update based on alpha
        new_q = (1.-self.alpha)*cur_q + (self.alpha)*x
        # store updated q value
        self.q_table.itemset(sa0, new_q)

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        print "LearningAgent.update(): state ={}".format(self.state)
        print "LearningAgent.update(): action ={}".format(action)

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=10)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
