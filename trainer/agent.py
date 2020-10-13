from .utils import *
import functools
from .policy import *
import os


class Agent(object):
    def __init__(self, args=None, race=None, checkpoint=None):
        self.model = Model(args.train_batch_size)
        if checkpoint is not None:
            self.set_weights(checkpoint)
        if args: # if there are previous checkpoints get the most recent one
            # files = tf.io.gfile.listdir(args.job_dir+"checkpoints/")
            latest_checkpoint = tf.train.latest_checkpoint(args.job_dir + "checkpoints/")
            if latest_checkpoint is not None:
                self.set_weights(latest_checkpoint)
        
    def set_weights(self, checkpoint:str):
        self.model.load_weights(checkpoint)
    
    def save_weights(self, checkpoint):
        self.model.save_weights(checkpoint)
    
    def get_weights(self):
        return self.model.trainable_variables
    
    def step(self, global_obs, my_obs, opponents_obs, lstm_state):
        pass

    def rollout(self, trajectories:tf.Tensor, supervised:bool=False):
        return self.model(trajectories)

    