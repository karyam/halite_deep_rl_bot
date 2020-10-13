
from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
from utils import *

class HaliteEnv(obs):
    """ 
    Wrapper for the Halite environment.

    Attributes:
      - map:tf.Tensor(num_players, map_x, map_y, ch)
        - halite cell: varying shades of blue depending on halite amount
        - my units: red cell
        - opponent units: green

      - ship_obs:tf.Tensor(num_players, map_x, map_y, num_ships)
        The frame is the same as map with the exception that 
        each frame contains one ships colored in varying shades of pink
        depending on halite amount. Each frame is seen from the corresponding player p.o.v.

     - shipyards_obs:tf.Tensor(num_players, map_x, map_y, num_shipyards)
        The frame is the same as map with the exception that 
        each frame contains one shipyards colored in varying shades of purple
        depending on halite amount. Each frame is seen from the corresponding player p.o.v.

    
    
    
    
    """

    def __init__():
        self.env = make("halite")
        self.parser = DataParser()
        self.map = None
        self.ship_obs = None
        self.shipyard_obs = None
        self.meta
        self.done = false

 
    
    def step(my_agent_out, opp_agents_out):
        # propagate actions
        my_agent_out.ships_action
        my_agent_out.shipyards_action
        self.board = self.board.next()
        self.map, self.ship_obs, self.shipyard_obs, self.meta, self.done = parser.get_step_frames(self.board)
    
    
    def reset(num_players:int):
        self.env.reset(num_players) 
        
    
    def get_player_obs(player_idx:int):
        """ Return the input structure for the """
        slice_begin = 
        slice_end = 
        obs = AgentInput(
            map=tf.slice(self.map, [player_idx,0,0,0], [player_idx+1,-1,-1]),
            my_ships=(self.ship_obs, [player_idx,0,0,0], [player_idx+1,-1,-1])
            my_shipyards=self.shipyard_obs[player],
            close_opps_obs,
            far_opp_obs,
            meta,

        )
        return obs