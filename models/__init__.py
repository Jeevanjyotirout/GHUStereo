from .GHUStereo import GHUStereo
from .GHUStereo_trt import GHUStereo_trt
from .loss import model_loss_train, model_loss_test
__models__ = {
    "GHUStereo": GHUStereo,
    "GHUStereo_trt": GHUStereo_trt
}
