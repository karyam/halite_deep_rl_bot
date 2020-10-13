import numpy as np
import tensorflow as tf
from agent import *



class Player(object):

  def get_match(self):
    pass

  def ready_to_checkpoint(self):
    return False

  def _create_checkpoint(self):
    return Historical(self, self.payoff)

  @property
  def payoff(self):
    return self._payoff

  @property
  def race(self):
    return self._race

  def checkpoint(self):
    raise NotImplementedError


class MainPlayer(Player):

  def __init__(self, race, weights, payoff):
    self.agent = Agent(race, weights())
    self._payoff = payoff
    self._race = race
    self._checkpoint_step = 0

  def _pfsp_branch(self):
    historical = [
        player for player in self._payoff.players
        if isinstance(player, Historical)
    ]
    win_rates = self._payoff[self, historical]
    return np.random.choice(
        historical, p=pfsp(win_rates, weighting="squared")), True

  def _selfplay_branch(self, opponent):
    # Play self-play match
    if self._payoff[self, opponent] > 0.3:
      return opponent, False

    # If opponent is too strong, look for a checkpoint
    # as curriculum
    historical = [
        player for player in self._payoff.players
        if isinstance(player, Historical) and player.parent == opponent
    ]
    win_rates = self._payoff[self, historical]
    return np.random.choice(
        historical, p=pfsp(win_rates, weighting="variance")), True

  def _verification_branch(self, opponent):
    # Check exploitation
    exploiters = set([
        player for player in self._payoff.players
        if isinstance(player, MainExploiter)
    ])
    exp_historical = [
        player for player in self._payoff.players
        if isinstance(player, Historical) and player.parent in exploiters
    ]
    win_rates = self._payoff[self, exp_historical]
    if len(win_rates) and win_rates.min() < 0.3:
      return np.random.choice(
          exp_historical, p=pfsp(win_rates, weighting="squared")), True

    # Check forgetting
    historical = [
        player for player in self._payoff.players
        if isinstance(player, Historical) and player.parent == opponent
    ]
    win_rates = self._payoff[self, historical]
    win_rates, historical = remove_monotonic_suffix(win_rates, historical)
    if len(win_rates) and win_rates.min() < 0.7:
      return np.random.choice(
          historical, p=pfsp(win_rates, weighting="squared")), True

    return None

  def get_match(self):
    coin_toss = np.random.random()

    # Make sure you can beat the League
    if coin_toss < 0.5:
      return self._pfsp_branch()

    main_agents = [
        player for player in self._payoff.players
        if isinstance(player, MainPlayer)
    ]
    opponent = np.random.choice(main_agents)

    # Verify if there are some rare players we omitted
    if coin_toss < 0.5 + 0.15:
      request = self._verification_branch(opponent)
      if request is not None:
        return request

    return self._selfplay_branch(opponent)

  def ready_to_checkpoint(self):
    steps_passed = self.agent.get_steps() - self._checkpoint_step
    if steps_passed < 2e9:
      return False

    historical = [
        player for player in self._payoff.players
        if isinstance(player, Historical)
    ]
    win_rates = self._payoff[self, historical]
    return win_rates.min() > 0.7 or steps_passed > 4e9

  def checkpoint(self):
    self._checkpoint_step = self.agent.get_steps()
    return self._create_checkpoint()


class MainExploiter(Player):

  def __init__(self, race, agent, payoff):
    self.agent = Agent(race, agent.get_weights())
    self._initial_weights = agent.get_weights()
    self._payoff = payoff
    self._race = agent.race
    self._checkpoint_step = 0

  def get_match(self):
    main_agents = [
        player for player in self._payoff.players
        if isinstance(player, MainPlayer)
    ]
    opponent = np.random.choice(main_agents)

    if self._payoff[self, opponent] > 0.1:
      return opponent, True

    historical = [
        player for player in self._payoff.players
        if isinstance(player, Historical) and player.parent == opponent
    ]
    win_rates = self._payoff[self, historical]

    return np.random.choice(
        historical, p=pfsp(win_rates, weighting="variance")), True

  def checkpoint(self):
    self.agent.set_weights(self._initial_weights)
    self._checkpoint_step = self.agent.get_steps()
    return self._create_checkpoint()

  def ready_to_checkpoint(self):
    steps_passed = self.agent.get_steps() - self._checkpoint_step
    if steps_passed < 2e9:
      return False

    main_agents = [
        player for player in self._payoff.players
        if isinstance(player, MainPlayer)
    ]
    win_rates = self._payoff[self, main_agents]
    return win_rates.min() > 0.7 or steps_passed > 4e9



class Payoff():
    def __init__(self):
        pass
    



class League():
    def __init__(self, num_players:int, initial_agents:dict):
        self.num_players = num_players
        self.initial_agents = initial_agents
        self.payoff = Payoff()
    