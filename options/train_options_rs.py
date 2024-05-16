import os


def ISPRS_Potsdam_RGB(parser):
    parser.add_argument("--dataset", type=str, default='ISPRS_Potsdam_RGB')       
    parser.add_argument("--batch-size", type=int, default=4, help="input batch size.")
    parser.add_argument("--num-classes", type=int, default=6, help="Number of classes for ISPRS.")
    parser.add_argument("--percent", type=float, default=1, help="percent of training set.")
    parser.add_argument("--data-dir", type=str, default='/home/Datasets/RSSeg')
    parser.add_argument("--train-lbl-list", type=str, default='lists/train_percent%_labeled.txt')
    parser.add_argument("--train-unl-list", type=str, default='lists/train_percent%_unlabeled.txt')
    parser.add_argument("--val-list", type=str, default='lists/val.txt')
    parser.add_argument("--ignore-index", type=int, default=5)
    parser.add_argument("--confidence-thr", type=float, default=0.95, help="confidence threshold of pseudo-labels")
    parser.add_argument("--learning_rate", type=float, default=2.5e-4, help="initial learning rate for the segmentation network.")
    parser.add_argument("--lr-multi", type=float, default=10, help="initial learning rate for the segmentation network.")
    parser.add_argument("--num-epochs", type=int, default=50)
    return parser

def ISPRS_Vaihingen_IRRG(parser):
    parser.add_argument("--dataset", type=str, default='ISPRS_Vaihingen_IRRG')       
    parser.add_argument("--batch-size", type=int, default=4, help="input batch size.")
    parser.add_argument("--num-classes", type=int, default=6, help="Number of classes for ISPRS.")
    parser.add_argument("--percent", type=float, default=5, help="percent of training set.")
    parser.add_argument("--data-dir", type=str, default='/home/Datasets/RSSeg')
    parser.add_argument("--train-lbl-list", type=str, default='lists/train_percent%_labeled.txt')
    parser.add_argument("--train-unl-list", type=str, default='lists/train_percent%_unlabeled.txt')
    parser.add_argument("--val-list", type=str, default='lists/val.txt')
    parser.add_argument("--ignore-index", type=int, default=5)
    parser.add_argument("--confidence-thr", type=float, default=0.95, help="confidence threshold of pseudo-labels")
    parser.add_argument("--learning_rate", type=float, default=2.5e-4, help="initial learning rate for the segmentation network.")
    parser.add_argument("--lr-multi", type=float, default=10, help="initial learning rate for the segmentation network.")
    parser.add_argument("--num-epochs", type=int, default=100)
    return parser


def LoveDA(parser):
    parser.add_argument("--dataset", type=str, default='LoveDA')       
    parser.add_argument("--batch-size", type=int, default=4, help="input batch size.")
    parser.add_argument("--num-classes", type=int, default=8, help="Number of classes for ISPRS.")
    parser.add_argument("--percent", type=float, default=5, help="percent of training set.")
    parser.add_argument("--data-dir", type=str, default='/home/Datasets/RSSeg')
    parser.add_argument("--train-lbl-list", type=str, default='lists/train_percent%_labeled.txt')
    parser.add_argument("--train-unl-list", type=str, default='lists/train_percent%_unlabeled.txt')
    parser.add_argument("--val-list", type=str, default='lists/val_short.txt')
    parser.add_argument("--ignore-index", type=int, default=0)
    parser.add_argument("--confidence-thr", type=float, default=0.95, help="confidence threshold of pseudo-labels")
    parser.add_argument("--learning_rate", type=float, default=2.5e-4, help="initial learning rate for the segmentation network.")
    parser.add_argument("--lr-multi", type=float, default=10, help="initial learning rate for the segmentation network.")
    parser.add_argument("--num-epochs", type=int, default=20)
    return parser


def DeepGlobe_Landcover(parser):
    parser.add_argument("--dataset", type=str, default='DeepGlobe_Landcover')       
    parser.add_argument("--batch-size", type=int, default=4, help="input batch size.")
    parser.add_argument("--num-classes", type=int, default=7, help="Number of classes for ISPRS.")
    parser.add_argument("--percent", type=float, default=5, help="percent of training set.")
    parser.add_argument("--data-dir", type=str, default='/home/Datasets/RSSeg')
    parser.add_argument("--train-lbl-list", type=str, default='lists/train_percent%_labeled.txt')
    parser.add_argument("--train-unl-list", type=str, default='lists/train_percent%_unlabeled.txt')
    parser.add_argument("--val-list", type=str, default='lists/val.txt')
    parser.add_argument("--ignore-index", type=int, default=6)
    parser.add_argument("--confidence-thr", type=float, default=0.95, help="confidence threshold of pseudo-labels")
    parser.add_argument("--learning_rate", type=float, default=2.5e-4, help="initial learning rate for the segmentation network.")
    parser.add_argument("--lr-multi", type=float, default=10, help="initial learning rate for the segmentation network.")
    parser.add_argument("--num-epochs", type=int, default=20)
    return parser



