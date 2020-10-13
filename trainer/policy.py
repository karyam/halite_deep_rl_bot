import numpy as np
import tensorflow as tf 
from .utils import *


########################
##### MODEL BLOCKS #####
########################


class ConvBlock(tf.keras.Model):
  def __init__(self, filters, kernel_size, activation, shape, ragged:bool=False):
    super(ConvBlock, self).__init__()
    self.conv1 = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=shape, dtype="float64", ragged=ragged),
                                      tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation)])
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation)
    self.bn2 = tf.keras.layers.BatchNormalization()

  def call(self, X):
    X_conv = self.conv1(X)
    X_conv = self.bn1(X_conv)
    X_conv = self.conv2(X_conv)
    out = self.bn2(X_conv)
    return X_conv, out


class SpatialEncoder(tf.keras.Model):
  def __init__(self, shape, ragged:bool=False):
    super(SpatialEncoder, self).__init__()
    self.block1 = ConvBlock(filters=4, kernel_size=3, activation='relu', shape=shape, ragged=ragged)
    self.block2 = ConvBlock(8, kernel_size=3, activation='relu', shape=(28,28,4), ragged=ragged)
    self.block3 = ConvBlock(16, kernel_size=3, activation='relu', shape=(24,24,8), ragged=ragged)

  def call(self, X):
    X_conv1, out1 = self.block1(X)
    X_conv2, out2 = self.block2(out1)
    X_conv3, out3 = self.block3(out2)
    return out3, [X_conv1, X_conv2, X_conv3]

class AttBlock(tf.keras.Model):
  def __init__(self, filters):
    super(AttBlock, self).__init__()
    self.Wskip = tf.keras.layers.Conv2D(filters=filters, kernel_size=1)
    self.Wx = tf.keras.layers.Conv2D(filters=filters, kernel_size=1)

    self.conv = tf.keras.Sequential([
      tf.keras.layers.Conv2D(filters=1, kernel_size=1),
      tf.keras.layers.BatchNormalization(),
      ])

  def call(self, X, X_skip):
    # change the number of channels
    X_skip1 = self.Wskip(X_skip)
    X = self.Wx(X)
    assert(X.shape == X_skip.shape)
    # compute att scores
    att_scores = tf.keras.activations.sigmoid(self.conv(tf.keras.activations.relu(X_skip1 + X)))
    return tf.math.multiply(X, att_scores)


class AttUpConvBlock(tf.keras.Model):
  def __init__(self, filters, up_filters, kernel_size, up_kernel_size, activation="relu"):
    super(AttUpConvBlock, self).__init__()
    self.upsample = tf.keras.layers.Conv2DTranspose(filters=up_filters, kernel_size=up_kernel_size, strides=(1,1))
    self.att = AttBlock(filters)
    self.conv1 = tf.keras.Sequential([tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation),
                                     tf.keras.layers.BatchNormalization()])
    self.conv2 = tf.keras.Sequential([tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation),
                                     tf.keras.layers.BatchNormalization()])                                 

  def call(self, X, X_skip):
    X = self.upsample(X)
    X_att = self.att(X, X_skip)
    assert(X.shape[1:] == X_skip.shape[1:])
    X = tf.concat([X_att, X], axis=-1)
    X = self.conv1(X)
    X = self.conv2(X)
    return X


class Upsample(tf.keras.Model):
    def __init__(self, filters=[64,32], kernel_size=[4,4], strides=[4,4]):
      super(Upsample, self).__init__()
      self.up_conv1 = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,1,128), dtype="float32"),
        tf.keras.layers.Conv2DTranspose(filters=filters[0], kernel_size=kernel_size[0], strides=strides[0], activation="relu"),
        tf.keras.layers.BatchNormalization()
      ])
      
      self.up_conv2 = tf.keras.Sequential([
        tf.keras.layers.Conv2DTranspose(filters=filters[1], kernel_size=kernel_size[1], strides=strides[1], activation="relu"),
        tf.keras.layers.BatchNormalization()
      ])

    def call(self, X):
      X = self.up_conv1(X)
      X = self.up_conv2(X)
      assert(X.shape[1:] == (16, 16, 32))
      return X

class ScalarEncoder(tf.keras.Model):
  def __init__(self, filters=[8,8,8], kernel_size=[16,8,4]):
    super(ScalarEncoder, self).__init__()
    self.conv1 = tf.keras.Sequential([
      tf.keras.layers.Input(shape=[100, 4], dtype=tf.float32),
      tf.keras.layers.Conv1D(filters=filters[0], kernel_size=kernel_size[0], activation="relu")
    ])
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.conv2 = tf.keras.layers.Conv1D(filters=filters[1], kernel_size=kernel_size[1], activation="relu")
    self.bn2 = tf.keras.layers.BatchNormalization()
    self.conv3 = tf.keras.layers.Conv1D(filters=filters[2], kernel_size=kernel_size[2], activation="relu")
    self.bn3 = tf.keras.layers.BatchNormalization()

  def call(self, X):
    X = self.bn1(self.conv1(X))
    X = self.bn2(self.conv2(X))
    return self.bn3(self.conv3(X))
    

############################
##### MODEL COMPONENTS #####
############################


class Encoder(tf.keras.Model):
  def __init__(self):
    super(Encoder, self).__init__()
    self.map_encoder = SpatialEncoder(shape=(32,32,3))
    self.ship_encoder = SpatialEncoder(shape=(32, 32, 3))
    self.shipyard_encoder = SpatialEncoder(shape=(32, 32, 3))
    self.far_opp_encoder = SpatialEncoder(shape=(32, 32, 3))
    self.close_opps_encoder = SpatialEncoder(shape=(32, 32, 3))
    self.scalar_encoder = ScalarEncoder()
    self.lin1 = tf.keras.layers.Dense(128)
    self.flatten = tf.keras.layers.Flatten()   

  def call(self, input):
    #input: [T*B, max_x, map_y, num_ch]
    map_out, _ = self.map_encoder(input.map)
    ships_out, ship_embed = self.ship_encoder(input.my_ships)
    shipyards_out, shipyard_embed = self.shipyard_encoder(input.my_shipyards)
    close_opps_out, _ = self.close_opps_encoder(input.close_opps_obs)
    far_opp_out, _ = self.far_opp_encoder(input.far_opp_obs)
    scalar_out = self.scalar_encoder(input.z)

    # output: [T*B, map_x-12, map_y-12, 16]
    assert(map_out.shape == (input.map.shape[0],20,20,16))
    assert(close_opps_out.shape == (*input.close_opps_obs.shape[0:2],20,20,16))
    assert(far_opp_out.shape == (*input.far_opp_obs.shape[0:2],20,20,16))
    assert(scalar_out.shape == (input.z.shape[0],75,8))

    # latent game state: flatten each embedding and concatenate the results across last dimension => [B, D']
    map_out = self.flatten(map_out)
    ships_out = self.flatten(ships_out)
    shipyards_out = self.flatten(shipyards_out)
    close_opps_out = self.flatten(close_opps_out)
    far_opp_out = self.flatten(far_opp_out)
    scalar_out = self.flatten(scalar_out)
    
    latent_game_state = tf.concat([map_out, ships_out, shipyards_out, close_opps_out, far_opp_out, scalar_out], axis=-1)
    latent_game_state = self.lin1(latent_game_state)
    return latent_game_state, ship_embed, shipyard_embed

    
class Decoder(tf.keras.Model):
  def __init__(self):
    super(Decoder, self).__init__()
    # self.upsample_core = Upsample()
    self.block1 = AttUpConvBlock(filters=16, up_filters=16, kernel_size=3, up_kernel_size=5)
    self.block2 = AttUpConvBlock(filters=8, up_filters=8, kernel_size=3, up_kernel_size=9)
    self.block3 = AttUpConvBlock(filters=4, up_filters=4, kernel_size=3, up_kernel_size=9)

  def call(self, core_output, X_skip):
    # core_output = self.upsample_core(core_output) #=> 16,16,32
    X = self.block1(core_output, X_skip[-1]) #=> 16,16,16
    X = self.block2(X, X_skip[-2]) #=> 20,20,8
    X = self.block3(X, X_skip[-3]) #=> 24,24,4
    return X


class Core(tf.keras.Model):
  def __init__(self, units, return_state=True):
    super(Core, self).__init__()
    self.memory = tf.keras.layers.LSTM(units, return_sequences=True, return_state=return_state, time_major=True)
    
  def call(self, X):
    return self.memory(X)


class Head(tf.keras.Model):
  def __init__(self):
    super(Head, self).__init__()
    self.ship_output = tf.keras.layers.Conv2D(filters=6, kernel_size=4, activation=None) #6 action types for each ship
    self.shipyard_output = tf.keras.layers.Conv2D(filters=2, kernel_size=4, activation=None) #2 action types for each shipyard
    # self.baseline = 

  def call(self, ship_input, shipyard_input):
    ship_logits = self.ship_output(ship_input)
    shipyard_logits = self.shipyard_output(shipyard_input)
    
    assert(ship_logits.shape[1:] == (21,21,6))
    assert(shipyard_logits.shape[1:] == (21,21,2))
    
    output = AgentOutput(ship_logits=ship_logits, shipyard_logits=shipyard_logits)
    return output


class Model(tf.keras.Model):
  
  def __init__(self, batch_size):
    super(Model, self).__init__()
    self.encoder = Encoder()
    self.core = Core(128)
    self.upsample_core = Upsample()
    self.ship_decoder = Decoder()
    self.shipyard_decoder = Decoder()
    self.head = Head()
    self.batch_size = batch_size

  def call(self, trajectories:tf.Tensor, supervised:bool=False):
    step_size = trajectories[0].shape[0]

    # [T, B, D] -> [T*B, D] compress because the model does only accept 2D tensors
    trajectories = merge_leading_dim(trajectories, 2)
    assert(trajectories[0].shape.ndims == 4)
  
    latent_game, ship_embed, shipyard_embed = self.encoder(trajectories)
    # [T*B, D] -> [T, B, D]
    print(latent_game.shape)
    latent_game = split_leading_dim(latent_game, step_size, self.batch_size)
    assert(latent_game.shape == (step_size, self.batch_size, 128))

    # get outputs from the memory block
    core_output, core_memory, core_carry = self.core(latent_game)
    assert(core_output.shape == (step_size, self.batch_size, 128))
        
    # core_output [T, B, lstm_units] -> [T*B, 1, 1, lstm_units]
    core_output = merge_leading_dim(core_output, 2)
    core_output = tf.reshape(core_output, shape=core_output.shape[0:1] + [1,1] + core_output.shape[-1])
    assert(core_output.shape == (self.batch_size*step_size, 1, 1, 128))

    core_output = self.upsample_core(core_output)
    assert(core_output.shape == (self.batch_size*step_size, 16, 16, 32))

    ship_decode = self.ship_decoder(core_output, ship_embed)
    shipyard_decode = self.shipyard_decoder(core_output, shipyard_embed)
        
    agent_output = self.head(ship_decode, shipyard_decode)
        
    # [T*B, D] -> [B, T, D]
    agent_output = split_leading_dim(agent_output, step_size, self.batch_size)
    agent_output = make_batch_major(agent_output)
    return agent_output
    



    
