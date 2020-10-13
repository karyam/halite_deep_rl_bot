import tensorflow as tf
import collections
import numpy as np
from numpy.random import choice

replay_root = ""
step_size = 399

InputFields = [
    "map",
    "my_ships",
    "my_shipyards",
    "close_opps_obs",
    "far_opp_obs",
    "z",
    "dones",
    "ship_moves",
    "shipyard_moves",

]
AgentInput = collections.namedtuple("AgentInput", InputFields)


OutputFields = [
    "ship_logits", # tf.Tensor(map_x, map_y, 6)
    "shipyard_logits", # tf.Tensor(map_x, map_y, 2)
    # "ships_action", # tf.Tensor(map_x, map_y, 6)
    # "shipyards_action", # tf.Tensor(map_x, map_y, 2)
    # "baselines", # tf.Tensor()
]

AgentOutput = collections.namedtuple("AgentOutput", OutputFields)

def make_time_major(x:tf.Tensor):
    return tf.nest.map_structure(lambda t: tf.transpose(t, [1, 0] + list(range(t.shape.ndims))[2:]), x)

def merge_leading_dim(x:tf.Tensor, n_dims:int) -> tf.Tensor:
    return tf.nest.map_structure(lambda t: tf.reshape(t, shape=[np.prod(t.shape[:n_dims])] + t.shape[n_dims:]), x)

def split_leading_dim(x:tf.Tensor, first_dim, second_dim) -> tf.Tensor:
    return tf.nest.map_structure(lambda t: tf.reshape(t, shape=[first_dim, second_dim] + t.shape[1:]), x)

def make_batch_major(x:tf.Tensor):
    return tf.nest.map_structure(lambda t: tf.transpose(t, [1, 0] + list(range(t.shape.ndims))[2:]), x)

def stack_key(x:list, key:int, batch_size):
    key_shape = x[0][key].shape
    key_list = [sample[key] for sample in x]
    result = tf.stack(key_list)
    assert(result.shape == (batch_size, *key_shape))
    return result

def stack_namedtuple(x:list, batch_size) -> AgentInput:
    result = AgentInput(
        map=stack_key(x, 0, batch_size),
        my_ships=stack_key(x, 1, batch_size),
        my_shipyards=stack_key(x, 2, batch_size),
        close_opps_obs=stack_key(x, 3, batch_size),
        far_opp_obs=stack_key(x, 4, batch_size),
        z=stack_key(x, 5, batch_size),
        dones=stack_key(x, 6, batch_size),
        ship_moves=stack_key(x, 7, batch_size),
        shipyard_moves=stack_key(x, 8, batch_size),   
    )
    return result


def test_agent(obs, config):    
    halite, shipyards, ships = obs['players'][obs['player']]
    
    actions = {}

    if len(ships) == 0 and len(shipyards) > 0:
        shipyard_id = list(shipyards.keys())[0]
        actions[shipyard_id] = 'SPAWN'   
    elif len(ships) > 0 and len(shipyards) == 0:
        ship_id = list(ships.keys())[0]
        actions[ship_id] = 'CONVERT'
    else:    
        for ship_id in ships.keys():
            ship_action = choice(["NORTH", "SOUTH", "EAST", "WEST", "CONVERT", None], 1, 
                                 p=[0.2, 0.2, 0.2, 0.2, 0.05, 0.15])[0]
            if ship_action is not None:
                actions[ship_id] = ship_action
                
        for shipyard_id in shipyards.keys():
            shipyard_action = choice(["SPAWN", None], 1, 
                                     p=[0.1, 0.9])[0]
            if shipyard_action is not None:
                actions[shipyard_id] = shipyard_action

    return actions




