# from utils import *
# import matplotlib.pyplot as plt
# import math
# import numpy as np
# from kaggle_environments import make
# from kaggle_environments.envs.halite.helpers import *
# import tensorflow as tf
# import json
# from kaggle_environments import get_episode_replay, list_episodes, list_episodes_for_team, list_episodes_for_submission
# import time
from data_parser import *


class DataParserTest(tf.test.TestCase):
    
    def setUp(self):
        super(DataParserTest, self).setUp()
        self.data_parser = DataParser()
        env = make("halite", debug=True)
        self.env_config = env.configuration
        self.training_env = env.train([None, "random", "random", "random"])
        self.obs = self.training_env.reset()
        self.board = Board(self.obs, self.env_config)
    
    def tearDown(self):
        del self.data_parser

    def test_player_map_out_init_state(self):
        ''' Test whether the curren player's units are red and the opponents' are green. '''
        for player_id in range(4):
            player = self.board.players[player_id]
            player_map, _ = self.data_parser.get_player_map(self.board, player_id, pad=False)
            my_units = player.ships + player.shipyards

            for unit in my_units:
                coords = unit.position
                self.assertAllEqual(player_map[coords], self.data_parser.cmap['player_ship'])
            opp_units = []
            
            for opp_id in range(4):
                if opp_id == player_id: continue
                opp_units.extend(self.board.players[opp_id].ships)
                opp_units.extend(self.board.players[opp_id].shipyards)

            for unit in opp_units:
                coords = unit.position
                self.assertAllEqual(player_map[coords], self.data_parser.cmap['enemy_ship'])

    def test_player_map_out_state(self):
        ''' Test whether the curren player's units are red and the opponents' are green. '''
        for i in range(100):
            actions = test_agent(self.obs, self.env_config)  
            self.obs, reward, done, info = self.training_env.step(actions)
        self.board = Board(self.obs, self.env_config)
        
        for player_id in range(4):
            player = self.board.players[player_id]
            player_map, _ = self.data_parser.get_player_map(self.board, player_id, pad=False)
            my_units = player.ships + player.shipyards

            for unit in my_units:
                coords = unit.position
                self.assertAllEqual(player_map[coords], self.data_parser.cmap['player_ship'])
            opp_units = []
            
            for opp_id in range(4):
                if opp_id == player_id: continue
                opp_units.extend(self.board.players[opp_id].ships)
                opp_units.extend(self.board.players[opp_id].shipyards)

            for unit in opp_units:
                coords = unit.position
                self.assertAllEqual(player_map[coords], self.data_parser.cmap['enemy_ship'])

    def test_player_map_normalised(self):
        _, player_map = self.data_parser.get_player_map(self.board, player_id=0, pad=False)
        self.assertAllInRange(player_map, 0, 1)

    def test_player_map_shape_no_pad(self):
        _, player_map = self.data_parser.get_player_map(self.board, player_id=0, pad=False)
        self.assertSequenceEqual(player_map.shape, (21, 21, 3))

    def test_player_map_shape_pad(self):
        _, player_map = self.data_parser.get_player_map(self.board, player_id=0, pad=True)
        self.assertSequenceEqual(player_map.shape, (32, 32, 3))  

    #######################################

    def test_player_entity_obs_normalised(self):
        player_map = self.data_parser.get_player_entity(self.board, map=player_map, player_id=0, entity="ship", pad=False)
        self.assertAllInRange(player_map, 0, 1)



if __name__ == '__main__':
    tf.test.main()

    

