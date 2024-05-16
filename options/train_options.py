import argparse
from .train_options_rs import ISPRS_Potsdam_RGB, ISPRS_Vaihingen_IRRG, LoveDA, DeepGlobe_Landcover


class TrainOptions():
    def initialize(self):
        parser = argparse.ArgumentParser( description="training script for FDA" )        

        # dataset
        # parser = Cityscapes(parser)
        # parser = ISPRS_Potsdam_RGB(parser)
        # parser = LoveDA(parser)
        parser = DeepGlobe_Landcover(parser)

        # model
        parser.add_argument("--model", type=str, default='EfficientUNet', 
                                choices=['DeepLab_V3plus', 'HRNet', 'FCDenseNet67', 'EfficientUNet', \
                                         'Swin_Transformer', 'SegFormer', 'BuildFormer'])
        parser.add_argument("--method", type=str, default='DistMatch', \
                                choices=['Sup', 'ST', 'ST+DL', 'ST+WL', 'FixMatch', 'SepFixMatch', 'DistMatch', 'DebiasPL'])
        parser.add_argument("--alpha", type=float, default=1, help="set trade-off between label and unlabeled set.")
        parser.add_argument("--gpu", type=int, default=0, help="set GPU.")

        # optimization
        parser.add_argument("--seed", type=int, default=1234)
        parser.add_argument("--momentum", type=float, default=0.9, help="Momentum component of the optimiser.")
        parser.add_argument("--weight-decay", type=float, default=0.0005, help="Regularisation parameter for L2-loss.")
        parser.add_argument("--power", type=float, default=0.9, help="Decay parameter to compute the learning rate (only for deeplab).")

        parser.add_argument("--restore", type=bool, default=False, help="restore checkpoint or not.")
        parser.add_argument("--save-dir", type=str, default='./checkpoints', help="Where to save snapshots of the model.")
        parser.add_argument("--num-workers", type=int, default=4)
        parser.add_argument("--print-freq", type=int, default=50)

        return parser.parse_args()
    
    def print_options(self, args):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)






