from data.base_dataset import BaseDataset

class CIFAR10Dataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, mode):
        # =============== Data =================
        parser.set_defaults(image_size=32)
        parser.set_defaults(input_nc=3)
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)