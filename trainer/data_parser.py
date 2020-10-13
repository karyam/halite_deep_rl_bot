from .utils import *
import matplotlib.pyplot as plt
import math
import numpy as np
from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
import tensorflow as tf
import json
from kaggle_environments import get_episode_replay, list_episodes, list_episodes_for_team, list_episodes_for_submission
import time

PATH = "./data/"


class DataParser(object):
    """ Util class to parse replay episodes and obtain training frames """
    def __init__(self):
        self.cmap = {'empty': np.array([0,0,0]), # empty cells
                       'hlt_75_100': np.array([0,0,255]), # halite
                       'hlt_50_75': np.array([0,0,212]),
                       'hlt_25_50': np.array([0,0,170]),
                       'hlt_0_25': np.array([0,0,128]),

                       'player_ship': np.array([255,0,0]), # player units
                       'player_crt_ship_cargo_0_25': np.array([255,0,128]),
                       'player_crt_ship_cargo_25_50': np.array([255,0,170]),
                       'player_crt_ship_cargo_50_75': np.array([255,0,212]),
                       'player_crt_ship_cargo_75_100': np.array([255,0,255]),
                       'player_yard': np.array([128,0,0]),
                       'player_crt_yard': np.array([128,0,128]),

                       'enemy_ship': np.array([0,255,0]), # enemy units
                       'enemy_yard': np.array([0,128,0])}
        self.amap = {
            "NORTH": 1,
            "EAST": 2,
            "SOUTH": 3, 
            "WEST": 4,
            "CONVERT": 0 # should use 5 instead and cross entropy?
        }

    def get_trajectory_frames(self, replay_path:str, my_player_id:int=0, one_hot:bool=False, pad:bool=True):
        """
        Function to generate one supervised training data sample.
        Return:
          - trajectory:tf.Tensor(AgentInput) 
        """
        start = time.time()

        # replay = tf.io.gfile.GFile.read(replay_path)
        # replay = json.loads(replay)
        # replay = json.loads(replay['result']['replay'])
        replay_path = "/." + replay_path 
        with open(replay_path) as replay_file:
            replay = json.load(replay_file)
        replay = json.loads(replay['result']['replay'])

        init_board = Board(replay["steps"][0][0]["observation"], replay["configuration"])
        close_opps_id, far_opp_id = self.get_opp_ids(init_board, my_player_id)

        if len(replay["steps"]) != 400: raise NameError("Not enough steps")

        map, my_ships, my_shipyards, close_opps_obs, far_opp_obs, z, meta, dones, ship_moves, shipyard_moves = [], [], [], [], [], [], [], [], [], []
        
        for step in range(len(replay["steps"])-1):
            obs = replay["steps"][step][0]["observation"]
            board = Board(obs, replay["configuration"])
            close_opp_obs = []

            # my_player
            player_map, player_ship_obs, player_shipyard_obs = self.get_step_frames(board, my_player_id, tensor=True)
            map.append(player_map)
            my_ships.append(player_ship_obs)
            my_shipyards.append(player_shipyard_obs)
            my_z, my_ship_moves, my_shipyard_moves = self.get_player_step_moves(board, 
                                                     replay["steps"][step+1][my_player_id]["action"], my_player_id, one_hot=one_hot, tensor=True)
            z.append(my_z)
            ship_moves.append(my_ship_moves)
            shipyard_moves.append(my_shipyard_moves)
            
            # close opponents
            player_map, player_ship_obs, player_shipyard_obs = self.get_step_frames(board, close_opps_id[0], tensor=True)
            close_opps_obs_tmp = tf.expand_dims(player_ship_obs, 0)
            close_opps_obs_tmp = tf.concat([close_opps_obs_tmp, tf.expand_dims(player_shipyard_obs,0)], axis=0)

            player_map, player_ship_obs, player_shipyard_obs = self.get_step_frames(board, close_opps_id[1], tensor=True)
            close_opps_obs_tmp = tf.concat([close_opps_obs_tmp, tf.expand_dims(player_ship_obs,0)], axis=0)
            close_opps_obs_tmp = tf.concat([close_opps_obs_tmp, tf.expand_dims(player_shipyard_obs,0)], axis=0)

            # far opponent
            player_map, player_ship_obs, player_shipyard_obs = self.get_step_frames(board, far_opp_id, tensor=True)
            far_opp_obs_tmp = tf.expand_dims(player_ship_obs, 0)
            far_opp_obs_tmp = tf.concat([far_opp_obs_tmp,  tf.expand_dims(player_shipyard_obs,0)], axis=0)

            close_opps_obs.append(close_opps_obs_tmp)
            far_opp_obs.append(far_opp_obs_tmp)
            dones.append(False)

        dones.append(True)
        z = [z] * 399
        z = np.array(z)
        # print(z.shape)
        z = z[:, 0:100, :]
        # print(z.shape)
        # TODO: save winning strategies

        trajectory = AgentInput(
            map=tf.stack(map),
            my_ships=tf.stack(my_ships),
            my_shipyards=tf.stack(my_shipyards),
            close_opps_obs=tf.stack(close_opps_obs),
            far_opp_obs=tf.stack(far_opp_obs),
            dones=tf.stack(dones),
            z = tf.cast(tf.constant(z), dtype=tf.float32),
            ship_moves=tf.stack(ship_moves),
            shipyard_moves=tf.stack(shipyard_moves)
        )

        end = time.time()
        # print(end - start)
        return trajectory

    def get_opp_ids(self, init_board, player_id):
        close_opps_id = []
        far_opp_id = []
        player_ship = init_board.players[player_id].ships[0]
        opps = []

        def take_first(x:tuple):
            return x[0]
        
        for opp_id, opp in init_board.players.items():
            if (opp_id == player_id): continue
            ship = opp.ships[0]
            dx = ship.position.x - player_ship.position.x
            dy = ship.position.y - player_ship.position.y
            dist = dx * dx + dy * dy
            opps.append((dist, opp_id))

        opps.sort(key=take_first)
        return [opps[0][1], opps[1][1]], opps[2][1]


    def get_step_frames(self, board, player_id:int = None, pad:bool=True, tensor:bool=False):
        """
        Function to obtain agent_input for one single step.
        """
        map, ship_obs, shipyard_obs = [], [], []
        if pad == True: map_size = 32
        else: map_size = 21

        if player_id is not None:
            raw_map, map = self.get_player_map(board, player_id, "ship", pad)
            ship_obs = self.get_player_entity_obs(board, raw_map, player_id, "ship", pad)
            shipyard_obs = self.get_player_entity_obs(board, raw_map, player_id, "shipyard", pad)
        else:
            for i in range(4):
                raw_player_map, player_map = self.get_player_map(board, i, "ship", pad)
                player_ship_obs = self.get_player_entity_obs(board, raw_player_map, i, "ship", pad)
                player_shipyard_obs = self.get_player_entity_obs(board, raw_player_map, i, "shipyard", pad)
                
                map.append(player_map)
                ship_obs.append(player_ship_obs)
                shipyard_obs.append(player_shipyard_obs)

        if tensor == True:
            map = tf.constant(map)
            ship_obs = tf.constant(ship_obs)
            shipyard_obs = tf.constant(shipyard_obs)
        else:
            map = np.array(map)
            ship_obs = np.array(ship_obs); #ship_obs = np.moveaxis(ship_obs, 0, -1)
            shipyard_obs = np.array(shipyard_obs); #shipyard_obs = np.moveaxis(shipyard_obs, 0, -1)
        
        if player_id is None:
            assert(map.shape == (4, map_size, map_size, 3))
            assert(ship_obs.shape[0] == 4)
            assert(shipyard_obs.shape[0] == 4)

        return map, ship_obs, shipyard_obs


    def get_player_map(self, board, player_id:int, vers:str="ship", pad:bool=True):
        if pad: map_size = 32
        else: map_size = 21
        
        board_size = board.configuration.size
        max_cell_halite = board.configuration.max_cell_halite
        map = np.zeros((board_size,board_size,3))

        for coords, cell in board.cells.items():    
            if cell.ship is not None:
                role = 'player' if cell.ship.player_id == player_id else 'enemy'
                map[coords] = self.cmap[f'{role}_ship']

            elif cell.shipyard is not None:
                role = 'player' if cell.shipyard.player_id == player_id else 'enemy'
                map[coords] = self.cmap[f'{role}_{vers}']

            elif cell.halite > 0:
                hlt_percent = cell.halite / max_cell_halite * 100
                hlt_interval = self.get_hlt_percent_interval(hlt_percent)
                map[coords] = self.cmap[f'hlt_{hlt_interval}']
        
        raw_map = map
        map = self.rotate_board_img(map)
        if pad == True: map = self.get_pad_frame(map)
        map = self.normalize_board_img(map)   
        return raw_map, map

    def get_player_entity_obs(self, board, map, player_id, entity:str, pad:bool):
        if pad: map_size = 32
        else: map_size = 21
        
        player = board.players[player_id]
        if entity == "ship": entities = player.ships
        else: entities = player.shipyards

        if len(entities) == 0:
            entity_obs = self.rotate_board_img(map)
            if pad == True: entity_obs = self.get_pad_frame(entity_obs)
        else:
            map_cp = map.copy()
            for e in entities:
                if entity == "ship": 
                    cargo_interval = self.get_cargo_percent_interval(e.halite)
                    map_cp[e.position] = self.cmap[f'player_crt_ship_cargo_{cargo_interval}']
                else:
                    map_cp[e.position] = self.cmap[f'player_crt_yard']
            map_cp = self.rotate_board_img(map_cp)
            if pad == True: map_cp = self.get_pad_frame(map_cp)
            entity_obs = map_cp

        entity_obs = np.array(entity_obs) 
        assert(entity_obs.shape == (map_size, map_size, 3))
        entity_obs = self.normalize_board_img(entity_obs)
        return entity_obs


    def get_player_ships_obs(self, board, map, player_id, pad:bool):
        if pad: map_size = 32
        else: map_size = 21
        ship_obs = []
        player = board.players[player_id]

        if len(player.ships) == 0:
            ship_obs = self.rotate_board_img(map)
            if pad == True: ship_obs = self.get_pad_frame(ship_obs)
            ship_obs = np.expand_dims(ship_obs, 0)
        else:
            for ship in player.ships:
                cargo_interval = self.get_cargo_percent_interval(ship.halite)
                map_cp = map.copy()
                map_cp[ship.position] = self.cmap[f'player_crt_ship_cargo_{cargo_interval}']
                map_cp = self.rotate_board_img(map_cp)
                if pad == True: map_cp = self.get_pad_frame(map_cp)
                ship_obs.append(map_cp)

        ship_obs = np.array(ship_obs) 
        
        if len(player.ships): assert(ship_obs.shape == (len(player.ships), map_size, map_size, 3))
        else: assert(ship_obs.shape == (1, map_size, map_size, 3))
        ship_obs = self.normalize_board_img(ship_obs)
        
        return ship_obs

    def get_player_shipyards_obs(self, board, map, player_id, pad:bool):
        if pad: map_size = 32
        else: map_size = 21
        
        shipyard_obs = []
        player = board.players[player_id]

        if len(player.shipyards) == 0:
            shipyard_obs = self.rotate_board_img(map)
            if pad == True: shipyard_obs = self.get_pad_frame(shipyard_obs)
            shipyard_obs = np.expand_dims(shipyard_obs, 0)
        else:
            for shipyard in player.shipyards: 
                map_cp = map.copy()
                map_cp[shipyard.position] = self.cmap['player_crt_yard']
                map_cp = self.rotate_board_img(map_cp)
                if pad == True: map_cp = self.get_pad_frame(map_cp)
                shipyard_obs.append(map_cp)

        shipyard_obs = np.array(shipyard_obs)

        if len(player.shipyards): assert(shipyard_obs.shape == (len(player.shipyards), map_size, map_size, 3))
        else: assert(shipyard_obs.shape == (1, map_size, map_size, 3))
        
        shipyard_obs = self.normalize_board_img(shipyard_obs) 
        return shipyard_obs

    def get_player_step_moves(self, board, actions, player_id, one_hot:bool=False, pad:bool=False, tensor:bool=False):
        """ Function to get one-hot encoded moves """
        if pad: map_size = 32
        else: map_size = 21

        ship_moves = np.zeros((21, 21, 6))
        shipyard_moves = np.zeros((21, 21, 2))
        
        player = board.players[player_id]
        player_ships = player.ships
        player_shipyards = player.shipyards
        new_ships, new_shipyards = 0, 0
        
        for ship in player_ships:
            # print(ship.id)
            # print(ship.position)
            if (actions is not None) and ship.id in actions.keys():
                if actions[ship.id] == "CONVERT":
                    new_shipyards += 1
                ship_moves[ship.position, self.amap[actions[ship.id]]] = 1
            else: # 'collect' action
                ship_moves[ship.position.x, ship.position.y, 5] = 1


        for shipyard in player_shipyards:
            if (actions is not None) and shipyard.id in actions.keys():
                new_ships += 1
                shipyard_moves[shipyard.position, 1] = 1
            else:
                shipyard_moves[shipyard.position, 0] = 1

        z = np.array([len(player_ships), new_ships, len(player_shipyards), new_shipyards])
        ship_moves = self.rotate_board_img(ship_moves)
        shipyard_moves = self.rotate_board_img(shipyard_moves)

        # if pad == True:
        #     ship_moves = self.get_pad_frame(ship_moves)
        #     shipyard_moves = self.get_pad_frame(shipyard_moves)

        if one_hot == False:
            ship_moves = np.argmax(ship_moves, axis=-1)
            shipyard_moves = np.argmax(shipyard_moves, axis=-1)

            assert (ship_moves.shape == (map_size, map_size))
            assert (shipyard_moves.shape == (map_size, map_size))
        else:
            assert(ship_moves.shape == (map_size, map_size, 6))
            assert(shipyard_moves.shape == (map_size, map_size, 2))

        if tensor:
            ship_moves = tf.constant(ship_moves)
            shipyard_moves = tf.constant(shipyard_moves)

        return z, ship_moves, shipyard_moves
    

    def get_hlt_percent_interval(self, hlt_percent):
        interval_dict = {(0,25):'0_25', (25,50):'25_50', (50,75):'50_75', (75,np.inf):'75_100'}
        for interval in interval_dict.keys():
            if interval[0] < hlt_percent <= interval[1]:
                return interval_dict[interval]

    def get_cargo_percent_interval(self, cargo_amount):
        interval_dict = {(0,250):'0_25', (250,500):'25_50', (500,1000):'50_75', (1000,np.inf):'75_100'}
        for interval in interval_dict.keys():
            if interval[0] <= cargo_amount < interval[1]:
                return interval_dict[interval]

    def get_pad_frame(self, frame):
        max_dim = 32
        pad_y1 = (min(max_dim, frame.shape[0]*3) - frame.shape[0])//2
        pad_y2 = (min(max_dim, frame.shape[0]*3) - frame.shape[0]) - pad_y1
        frame = np.concatenate([frame[-pad_y1:], frame, frame[:pad_y2]], axis=0)

        pad_x1 = (min(max_dim, frame.shape[1]*3) - frame.shape[1])//2
        pad_x2 = (min(max_dim, frame.shape[1]*3) - frame.shape[1]) - pad_x1
        frame = np.concatenate([frame[:, -pad_x1:], frame, frame[:, :pad_x2]], axis=1)
        # print(frame.shape)
        assert(frame.shape == (32, 32, 3))
        return frame

    def render_map(self, board_img, player_id):
        plt.figure(figsize=(5,5))
        plt.subplot(1,1,1)
        plt.imshow(board_img)
        plt.axis('off')
        #plt.title(f"Player: {player_id}, General view", fontsize=20)
        plt.show()

    def render_ship_obs(self, ship_names, board_img, player_id):
        ships_count = board_img.shape[0]
        if ships_count > 0:
            row_count = math.ceil(ships_count / 3)
            plt.figure(figsize=(6*3,5*row_count))
            for i in range(ships_count):
                ax = plt.subplot(row_count,3,i+1)
                ax.imshow(board_img[i])
                plt.axis('off')
                ship_id = ship_names[i]
                plt.title(f"Player: {player_id}, Ship ID: {ship_id}", fontsize=20)        
            plt.show()
        else:
            print("There are no ships to render")
    
    def render_shipyard_obs(self, shipyard_names, board_img, player_id):
        shipyards_count = board_img.shape[0]
        if shipyards_count > 0:
            row_count = math.ceil(shipyards_count / 3)
            plt.figure(figsize=(6*3,5*row_count))
            for i in range(shipyards_count):
                ax = plt.subplot(row_count,3,i+1)
                ax.imshow(board_img[i])
                plt.axis('off')
                shipyard_id = shipyard_names[i]
                plt.title(f"Player: {player_id}, Shipyard ID: {shipyard_id}", fontsize=20)        
            plt.show()
        else:
            print("There are no shipyards to render")
    

    def apply_func_to_board_img(self, board_img, func):
        if len(board_img.shape) == 0:
            return 
        if len(board_img.shape) > 3:
            for i in range(board_img.shape[0]):
                board_img[i] = func(board_img[i,:,:])
        else:
            board_img = func(board_img)
        return board_img
         
        
    def normalize_board_img(self, board_img):
        func = lambda x: np.round(x / 255.0, 3)
        return self.apply_func_to_board_img(board_img, func)
    
    
    def rotate_board_img(self, board_img):
        func = lambda x: np.rot90(x)
        return self.apply_func_to_board_img(board_img, func)

if __name__ == "__main__":
    parser = DataParser()
    

    

  

      
  
