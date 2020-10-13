import collections
import numpy as np
from threading import Thread, Lock, currentThread
from queue import Queue
# from multiprocessing import Process, Queue, current_process
from .data_parser import *
from .agent import *
import os
import copy
import time
import tensorflow as tf
from .utils import *
import json
from .policy import *
from argparse import ArgumentParser
import pickle

###############
### GLOBALS ###
###############

start = time.perf_counter()

meta_path = "gs://halite-storage/meta/"

# agent_ids = tf.io.gfile.GFile.read(str(meta_path+'agent_ids_5234835.txt'))
with tf.io.gfile.GFile(meta_path+'agent_ids_5234835.txt', "r") as f:
    agent_ids = f.read() 
agent_ids = json.loads(agent_ids)
# agent_ids = {}
# with open(meta_path+'agent_ids_5234835.txt') as readfile:
#     agent_ids = json.load(readfile)


train_loss_results = []
train_ship_accuracy_results = []
train_shipyard_accuracy_results = []

eval_loss_results = []
eval_ship_accuracy_results = []
eval_shipyard_accuracy_results = []

parser = DataParser()

# def save_metrics(path):
#   with tf.gfile.GFile(path+"/metrics/train_loss.", "w") as f:
#     f.write()   
#   with open(os.path.join(path, "eval_loss"), "wb") as fp:   
#      pickle.dump(eval_loss_results, fp)
  
#   with open(os.path.join(path, "train_ship_accuracy"), "wb") as fp:   
#      pickle.dump(train_ship_accuracy_results, fp)
#   with open(os.path.join(path, "eval_ship_accuracy"), "wb") as fp:   
#      pickle.dump(eval_ship_accuracy_results, fp)

#   with open(os.path.join(path, "train_shipyard_accuracy"), "wb") as fp:   
#      pickle.dump(train_shipyard_accuracy_results, fp)
#   with open(os.path.join(path, "eval_shipyard_accuracy"), "wb") as fp:   
#      pickle.dump(eval_shipyard_accuracy_results, fp)



def build_data_loader(queue, path, files):
  """
    Function to serve as a worker for loading each saved replay sampled 
    from RAM, parsing it and sending it to the queue for later batching.
  """
  thread_name = currentThread()
  while True:
    game_no = np.random.randint(0, len(files))
    game_path = os.path.join(path, files[game_no])
    
    try:
      agent_input = copy.deepcopy(parser.get_trajectory_frames(game_path, my_player_id=agent_ids[game_path.split('_')[1][:-4]], one_hot=True))
    except:
      continue 
  
    queue.put(copy.deepcopy(agent_input))
    # print(f"Thread {thread_name} processed sample no: {game_no}")
    del agent_input
    del game_no

    # print(f"Return from build_data_loader: {thread_name}")


def build_batch_sampler(queue, buffer, batch_size):
  batch = []
  
  while True:

    while queue.empty(): time.sleep(1)
    trajectory = queue.get()
    batch.append(trajectory)
    
    if len(batch) == batch_size:
      batch = stack_namedtuple(batch, batch_size)
      #[B, T, D] => [T, B, D]
      batch = batch._replace(
          map = make_time_major(batch.map),
          my_ships = make_time_major(batch.my_ships),
          my_shipyards = make_time_major(batch.my_shipyards),
          close_opps_obs = make_time_major(batch.close_opps_obs),
          far_opp_obs = make_time_major(batch.far_opp_obs),
          z = make_time_major(batch.z),
          dones = make_time_major(batch.dones),
          ship_moves = batch.ship_moves,
          shipyard_moves = batch.shipyard_moves,
      )
      assert(batch[0].shape == (step_size, batch_size, 32, 32, 3))
      assert(batch[3].shape == (step_size, batch_size, 4, 32, 32, 3))
      assert(batch.ship_moves.shape == (batch_size, step_size, 21, 21, 6))
      assert(batch.shipyard_moves.shape == (batch_size, step_size, 21, 21, 2))
      
      buffer.put(copy.deepcopy(batch))

      del batch
      batch = []
  
  # print("Return from build_batch_sampler")

  
class SLLearner():
  """
    Learner class for initialising each agent type with a learned supervised policy.
  """
  def __init__(self, buffer, v_buffer, args):
    self.agent = Agent(args)
    self.lr = 1e-3
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr,
                                            beta_1=0.9,
                                            beta_2=0.999,
                                            epsilon=1e-8)
    self.loss_object = tf.keras.losses.KLDivergence()
    self.buffer = buffer
    self.v_buffer = v_buffer
    self.best_loss = None
  
  def loss(self, preds, targets):
    ships_loss = self.loss_object(targets[0], preds[0])
    shipyards_loss = self.loss_object(targets[1], preds[1])
    return ships_loss * 0.5 + shipyards_loss * 0.5

  def train(self, epoch, args):
    train_loss_avg = tf.keras.metrics.Mean()
    ship_train_accuracy = tf.keras.metrics.CategoricalAccuracy()
    shipyard_train_accuracy = tf.keras.metrics.CategoricalAccuracy()

    for step in range(args.train_steps):
      while self.buffer.empty(): time.sleep(1)
      trajectories = self.buffer.get()
      
      with tf.GradientTape() as tape:
        # unroll the agent's network over each trajectory to obtain action logits
        agent_output = self.agent.rollout(trajectories) 
        loss_val = self.loss(preds=(agent_output.ship_logits, agent_output.shipyard_logits),  
                        targets=(trajectories.ship_moves, trajectories.shipyard_moves))
      
      agent_params = self.agent.get_weights()
      grads = tape.gradient(loss_val, agent_params)
      self.optimizer.apply_gradients(zip(grads, agent_params))
      
      train_loss_avg.update_state(loss_val)
      ship_train_accuracy.update_state(trajectories.ship_moves, agent_output.ship_logits)
      shipyard_train_accuracy.update_state(trajectories.shipyard_moves, agent_output.shipyard_logits)
      
      print(f'Step {step+1} finished!')
    
    print("Epoch: {:03d}, Train Loss: {:.3f}, Train Ship accuracy: {:.3%}, Train Shipyard accuracy: {:.3%}".format(epoch, train_loss_avg.result(), ship_train_accuracy.result(), shipyard_train_accuracy.result()))
    
    train_loss_results.append(train_loss_avg.result())
    train_ship_accuracy_results.append(ship_train_accuracy.result())
    train_shipyard_accuracy_results.append(shipyard_train_accuracy.result())

  def eval(self, epoch, args):
    eval_loss_avg = tf.keras.metrics.Mean()
    ship_eval_accuracy = tf.keras.metrics.CategoricalAccuracy()
    shipyard_eval_accuracy = tf.keras.metrics.CategoricalAccuracy()
    
    for step in range(args.eval_steps):
      trajectories = self.v_buffer.get()
      agent_output = self.agent.rollout(trajectories)
      loss_val = self.loss(preds=(agent_output.ship_logits, agent_output.shipyard_logits),  
                           targets=(trajectories.ship_moves, trajectories.shipyard_moves))
      # update metrics
      eval_loss_avg.update_state(loss_val)
      ship_eval_accuracy.update_state(trajectories.ship_moves, agent_output.ship_logits)
      shipyard_eval_accuracy.update_state(trajectories.shipyard_moves, agent_output.shipyard_logits)
    
    if self.best_loss is None or (eval_loss_avg.result() < self.best_loss):
      self.best_loss = eval_loss_avg.result()
      checkpoint_path = args.job_dir + f"checkpoints/cp{epoch}.ckpt"
      self.agent.save_weights(checkpoint_path)
    
    eval_loss_results.append(eval_loss_avg.result())
    eval_ship_accuracy_results.append(ship_eval_accuracy.result())
    eval_shipyard_accuracy_results.append(shipyard_eval_accuracy.result())


def run(args):
  # train_files = os.listdir(args.train_files)
  # eval_files = os.listdir(args.eval_files)
  train_files = tf.io.gfile.listdir(args.train_files)
  eval_files = tf.io.gfile.listdir(args.eval_files)

  for epoch in range(args.num_epochs):
    print(f'Start epoch: {epoch}')
    
    queue = Queue(2) # queue of transitions
    buffer = Queue(2) # buffer of bached trajectories
    v_queue = Queue(2)
    v_buffer = Queue(2)

    learner = SLLearner(buffer, v_buffer, args)

    data_loaders = [Thread(target=build_data_loader, args=(queue, args.train_files, train_files,)) for i in range(args.num_data_loaders)]
    batch_sampler = Thread(target=build_batch_sampler, args=(queue, buffer, args.train_batch_size,))
    # v_data_loaders = [Thread(target=build_data_loader, args=(v_queue, args.eval_files, eval_files,)) for i in range(args.num_data_loaders)]
    # v_batch_sampler = Thread(target=build_batch_sampler, args=(v_queue, v_buffer, args.eval_batch_size,))

    # start training threads
    for data_loader in data_loaders: data_loader.start()
    batch_sampler.start()
    learner.train(epoch, args)
    # clean up after training is done
    del queue
    del buffer
    del data_loaders
    del batch_sampler
    
    # if epoch == 0 or (epoch % args.eval_freq) == 0:
    #   # start eval threads
    #   for v_data_loader in v_data_loaders: v_data_loader.start()
    #   v_batch_sampler.start()
    #   learner.eval(epoch, args)
    #   # clean up after eval is done
    #   del v_queue
    #   del v_buffer
    #   del v_data_loaders
    #   del v_batch_sampler

    del learner
    print(f'Finish epoch: {epoch}')
  
  # save_metrics(args.job_dir)
  finish = time.perf_counter()
  print(f'Finished in {round(finish-start,2)} seconds')

if __name__ == "__main__":
  PARSER = ArgumentParser()
  
  # Input Arguments
  PARSER.add_argument(
        '--train-files',
        help='GCS file or local paths to training data',
        nargs='+',
        default='gs://halite-storage/train/')
        #default='data/train/'
  PARSER.add_argument(
        '--eval-files',
        help='GCS file or local paths to evaluation data',
        nargs='+',
        default='gs://halite-storage/eval/')
        # default='data/eval/')
  PARSER.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        # default='/tmp/census-estimator'
        default='gs://halite-storage/first-job-dir/')
  PARSER.add_argument(
        '--num-epochs',
        help="""\
      Maximum number of training data epochs on which to train.
      If both --max-steps and --num-epochs are specified,
      the training job will run for --max-steps or --num-epochs,
      whichever occurs first. If unspecified will run for --max-steps.\
      """,
        type=int,
        default=2)
  PARSER.add_argument(
        '--train-batch-size',
        help='Batch size for training steps',
        type=int,
        default=2)
  PARSER.add_argument(
        '--eval-batch-size',
        help='Batch size for evaluation steps',
        type=int,
        default=2)
  PARSER.add_argument(
        '--train-steps',
        help="""\
      Steps to run the training job for. If --num-epochs is not specified,
      this must be. Otherwise the training job will run indefinitely.""",
        default=5,
        type=int)
  PARSER.add_argument(
        '--eval-steps',
        help='Number of steps to run evalution for at each checkpoint',
        default=3,
        type=int)
  PARSER.add_argument(
        '--eval-freq',
        help='Number of epoch after which to perform validation',
        default=1,
        type=int)
  PARSER.add_argument(
        '--num-data-loaders',
        help='Number of workers to sample data',
        default=2,
        type=int)
  PARSER.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')
  
  args, _ = PARSER.parse_known_args()
  tf.io.gfile.mkdir(args.job_dir+"checkpoints/")
  # tf.logging.set_verbosity(args.verbosity)
  run(args)
