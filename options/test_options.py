import argparse

class TestOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description="test segmentation network")
        parser.add_argument("--gpu", type=int, default=0, help="selected GPU device")
        parser.add_argument("--data-dir", type=str, default='/home/Datasets/RSSeg')
        parser.add_argument("--dataset", type=str, default='ISPRS_Potsdam_RGB',        
                                choices=['ISPRS_Potsdam_RGB', 'DeepGlobe_Landcover', 'LoveDA'] )     
        parser.add_argument("--batch-size", type=int, default=12, help="test batch size.")
        parser.add_argument("--num-classes", type=int, default=6, help="Number of classes for ISPRS.")
                                # ISPRS:6, DeepGlobe_Landuse:7, LoveDA: 8
        parser.add_argument("--ignore-index", type=int, default=5)
                                # ISPRS:5, DeepGlobe_Landuse:6, LoveDA: 0
        parser.add_argument("--percent", type=float, default=5.0, help="percent of training set.")
        parser.add_argument("--model", type=str, default='SegFormer', 
                                choices=['DeepLab_V3plus', 'HRNet', 'FCDenseNet67', 'EfficientUNet', \
                                         'Swin_Transformer', 'SegFormer', 'BuildFormer'])
        parser.add_argument("--method", type=str, default='AEL', \
                                choices=['Sup', 'ST', 'ST+DL', 'ST+WL', 'FixMatch', 'SepFixMatch', 'DistMatch', 
                                         'CPS', 'CCT', 'CutMix', 'UniMatch', 'CCVC', 'U2PL', 'AEL', 'LSST'])
        parser.add_argument("--alpha", type=float, default=0.5, help="set trade-off between label and unlabeled set.")
        parser.add_argument("--save-dir", type=str, default='./checkpoints', help="Where to save tSNEs.")
        parser.add_argument("--num-workers", type=int, default=4, help="number of threads.")

        return parser.parse_args()
    
    def print_options(self, args):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

