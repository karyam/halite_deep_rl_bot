from multiprocessing import Process
import random
from .halite_env import HaliteEnv
from .league import *
from .agent import Agent

class Actor(Process):
  """
  Actor worker which simulates matches for corresponding player against each player in the league.
  It sends the collected trajectories to the replay buffer associated with each player.
  """
  def __init__(self, num_players):
    self.num_players = num_players
    self.train_agent = train_agent
    self.env = HaliteEnv()

  def run(self):
    for i in range(num_games):
      # 1. Get initial game state
      self.train_agent.pos = random.randint(0,3)
      opponents = self.player.get_match() 
      self.env.reset(num_players)
      my_obs = self.env.get_player_obs(self.player)
      
      for i in range(self.num_opps):
        opp_obs[i] = self.env.get_player_obs(opps[i])

      transitions = []
      
      # 2. Loop through an episode (400 game steps) to collect the trajectory
      while not self.env.is_done(): 
        actions, logits, lstm_states = [], [], []
        for player_idx in range(self.num_players):
          player_obs = self.env.get_player_obs(player_idx)
          if player_idx == self.train_agent.pos:
            action, logit, lstm_state = self.train_agent.step(player_obs)
          
        my_action, my_logits, lstm_state = self.player.train_agent.step(my_obs, global_obs, opp_obs, initial_lstm_state)
        for i in range(self.num_players): 
          opp_action[i], opp_logits[i], opp_lstm_state[i] = opps[i].train_agent.step(opp_obs[i], global_obs, ?opp_obs)

        # step in the environment
        done, rewards = self.env.step(my_action, opp_action)
        #TODO: What information I need for one data point (one transition)
        
        transitions.append(Trajectory(
          global_obs=
        ))

        # update the buffer
        if len(transitions) > TRANSITION_LENGTH:
          buffer.send_data(transitions)
          transitions = []

      self.coordinator.send_outcome()

class Learner():
  """
  Learner worker that updates train_agent parameters based on trajectories.
  """
  def __init__(self, player, buffer):
    self.player = player
    self.buffer = buffer
    pass

  def update_train_agent(self):
    pass

  def run(self):
    pass




def run(args):
    # initialize the league
    league = League(
      num_players=args.num_players,
      initial_agents={
        race: Agent(args, race)
        for race in ("convexOptimization", "")
    })
    # initialize learner and actors for each player
    for i in range(args.num_players):
      player = league.get_player(idx)
      learner = Learner(player)
      actors.extend([ActorLoop(player, coordinator) for _ in range(16000)])


    # start the actor and learner jobs


if __name__ == '__main__':
  run(args)