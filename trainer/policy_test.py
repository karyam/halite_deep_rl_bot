
from policy import *
from data_parser import *
import os
from argparse import ArgumentParser

data_parser = DataParser()
path = "./data/train/"
files = os.listdir(path)
test_path = os.path.join(path, files[0])
step_size = 399
trajectories = []
bt = 0

def setUpModule():
    global trajectories, bt
    for i in range(args.batch_size):
        trajectories.append(data_parser.get_trajectory_frames(os.path.join(path, files[i]), one_hot=True))
    trajectories = stack_namedtuple(trajectories, args.batch_size)
    trajectories = make_time_major(trajectories)
    bt = trajectories[0].shape[0]*trajectories[0].shape[1]



class PolicyTest(tf.test.TestCase):
    
    def setUp(self):
        super(PolicyTest, self).setUp()
        # self.trajectories = []
        # for i in range(args.batch_size):
        #     self.trajectories.append(data_parser.get_trajectory_frames(os.path.join(path, files[0]), one_hot=True))
        # self.trajectories = stack_namedtuple(self.trajectories)
        # self.trajectories = make_time_major(self.trajectories)
        # self.bt = self.trajectories[0].shape[0]*self.trajectories[0].shape[1]

    def test_spatial_encoder_shape(self):
        pass

    def test_scalar_encoder_shape(self):
        pass

    def test_encoder_shape(self):
        encoder = Encoder()
        input = merge_leading_dim(trajectories, 2)
        latent, ship, shipyard = encoder(input)
        self.assertSequenceEqual(latent.shape, (bt, 128))
        self.assertEqual(len(ship), 3)
        self.assertEqual(len(shipyard), 3)

    def test_core_shape(self):
        '''Since I test the shape in the model class this method is redundant'''
        pass

    def test_upsample_shape(self):
        '''Since I test the shape in the model class this method is redundant'''
        pass

    def test_decoder_batch_shape(self):
        encoder = Encoder()
        input = merge_leading_dim(trajectories, 2)
        decoder = Decoder()
        pass
    
    def test_model_shape(self):
        model = Model(batch_size=args.batch_size)
        global trajectories
        out = model(trajectories)
        self.assertSequenceEqual(out[0].shape, (args.batch_size, step_size, 21, 21, 6))
        self.assertSequenceEqual(out[1].shape, (args.batch_size, step_size, 21, 21, 2))

if __name__ == '__main__':
    PARSER = ArgumentParser()
    PARSER.add_argument(
        '--batch-size',
        help='Batch size for training steps',
        type=int,
        default=3)
    global args
    args, _ = PARSER.parse_known_args()
    tf.test.main()

    