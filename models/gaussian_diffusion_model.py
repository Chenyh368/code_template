from .base_model import BaseModel

class GaussianDiffusionModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, mode):
        return parser

    def __init__(self, opt, manager):
        BaseModel.__init__(self, opt, manager)