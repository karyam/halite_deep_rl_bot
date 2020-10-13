import collections
import numpy as np
from threading import Thread, Lock, currentThread
from multiprocessing import Process, Queue, current_process
from data_parser import DataParser
from agent import *
import os
import copy
import time
import tensorflow as tf
from utils import *
import json
from policy import *

###############
### GLOBALS ###
###############

start = time.perf_counter()

replay_root = "drive/My Drive/Halite/code/Halite/data/"
meta_path = "drive/My Drive/Halite/code/Halite/data/meta/"
checkpoint_path = "drive/My Drive/Halite/code/Halite/data/meta/checkpoints"

num_saved_games = 4
batch_size = 2
num_batches = 2
step_size = 399
num_epochs = 1
num_data_loaders = 3

files = os.listdir(replay_root)
agent_ids = {}
with open(meta_path+'agent_ids_5234835.txt') as readfile:
    agent_ids = json.load(readfile)


train_loss_results = []
train_ship_accuracy_results = []
train_shipyard_accuracy_results = []

parser = DataParser()

loader_lock = Lock()
sampler_lock = Lock()

sample_counter = [0]
batch_counter = [0]

def build_data_loader(queue):
  """
    Function to serve as a worker for loading each saved replay sampled 
    from RAM, parsing it and sending it to the queue for later batching.
  """
  process_name = current_process().name
  while True:
    game_no = np.random.randint(0, 50)
    game_path = os.path.join(replay_root, files[game_no])
    
    try:
      agent_input = copy.deepcopy(parser.get_trajectory_frames(game_path, my_player_id=agent_ids[game_path.split('_')[1][:-4]], one_hot=True))
    except:
      continue 
  
    queue.put(copy.deepcopy(agent_input))
    print(f"Process {process_name} processed sample no: {game_no}")
    
  print(f"Return from build_data_loader: {process_name}")


def build_batch_sampler(queue, buffer):
  batch = []
  
  while True:
    while queue.empty(): time.sleep(1)
    trajectory = queue.get()
    batch.append(trajectory)
    
    if len(batch) == batch_size:
      batch = stack_namedtuple(batch)
      #[B, T, D] => [T, B, D]
      batch = batch._replace(
        map=make_time_major(batch[0]),
        my_ships=make_time_major(batch[1]),
        my_shipyards=make_time_major(batch[2]),
        close_opps_obs=make_time_major(batch[3]),
        far_opp_obs=make_time_major(batch[4]),
        z=make_time_major(batch[5]),
        dones=make_time_major(batch[6]))

      assert(batch[0].shape == (step_size, batch_size, 32, 32, 3))
      assert(batch[3].shape == (step_size, batch_size, 4, 32, 32, 3))
      
      buffer.put(copy.deepcopy(batch))
      batch_counter[0] += 1
    
      print(f"Queued batch no: {batch_counter[0]}")

      del batch
      batch = []

  print("Return from build_batch_sampler")

  
class SLLearner():
  """
    Learner class for initialising each agent type with a learned supervised policy.
  """
  def __init__(self, buffer):
    # Process.__init__(self)
    self.epoch = 0
    self.agent = Agent()
    self.lr = 1e-3
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr,
                                            beta_1=0.9,
                                            beta_2=0.999,
                                            epsilon=1e-8)
    self.loss_object = tf.keras.losses.KLDivergence()
    self.buffer = buffer
  
  def loss(self, preds, targets):
    ships_loss = self.loss_object(targets[0], preds[0])
    shipyards_loss = self.loss_object(targets[1], preds[1])
    return ships_loss * 0.5 + shipyards_loss * 0.5

  def run(self):
    epoch_loss_avg = tf.keras.metrics.Mean()
    ship_epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
    shipyard_epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
    self.epoch+=1

    for step in range(2):
      while self.buffer.empty(): time.sleep(1)
      # sample a batch of replays
      trajectories = self.buffer.get()
      
      with tf.GradientTape() as tape:
        # unroll the agent's network over each trajectory to obtain action logits
        agent_output = self.agent.rollout(trajectories)
      
        # get loss and gradients 
        loss_val = self.loss(preds=(agent_output.ship_logits, agent_output.shipyard_logits),  
                        targets=(trajectories.ship_moves, trajectories.shipyard_moves))
      
      # get the current parameters of the agent's model
      agent_params = self.agent.get_weights()
      grads = tape.gradient(loss_val, agent_params)
      # optimise the model's parameters
      self.optimizer.apply_gradients(zip(grads, agent_params))
      
      epoch_loss_avg.update_state(loss_val)
      ship_epoch_accuracy.update_state(trajectories.ship_moves, agent_output.ship_logits)
      shipyard_epoch_accuracy.update_state(trajectories.shipyard_moves, agent_output.shipyard_logits)
      
      print(f'Step {step+1} finished!')

    print(f"Epoch: {self.epoch}, " +
           f"Loss: {epoch_loss_avg.result()}, " + 
           f"Ship accuracy: {ship_epoch_accuracy.result()}, " +
           f"Shipyard accuracy: {shipyard_epoch_accuracy.result()}")

    # if len(train_loss_results) == 0 or \
    #   epoch_loss_avg.result() < train_loss_results[-1]: # when to save weights?
    #   self.agent.save_weights(checkpoint_path)
    
    train_loss_results.append(epoch_loss_avg.result())
    train_ship_accuracy_results.append(ship_epoch_accuracy.result())
    train_shipyard_accuracy_results.append(shipyard_epoch_accuracy.result())


def main():
  
  for epoch in range(num_epochs):
    print(f'Start epoch: {epoch}')
    
    queue = Queue(100) # queue of transitions
    buffer = Queue(100) # buffer of bached trajectories
    learner = SLLearner(buffer)
    data_loaders = [Process(name=f'data_loader{i}', target=build_data_loader, args=(queue,)) for i in range(num_data_loaders)]
    batch_sampler = Process(name='batch_sampler', target=build_batch_sampler, args=(queue, buffer))
        
    # start processes
    for data_loader in data_loaders: data_loader.start()
    batch_sampler.start()
    learner.run()
    
    # terminate the threads/processes once the learning is done
    for data_loader in data_loaders: data_loader.terminate()
    batch_sampler.terminate()
    print('Terminated')
    # bacause join blocks untill the porcess
    # terminates it makes no sense to call join
    # for data_loader in data_loaders: data_loader.join()
    # batch_sampler.join()

    del queue
    del buffer
    del learner
    del data_loaders
    del batch_sampler

    print(f'Finish epoch: {epoch}')
    
  finish = time.perf_counter()
  print(f'Finished in {round(finish-start,2)} seconds')

if __name__ == "__main__":
  main()
