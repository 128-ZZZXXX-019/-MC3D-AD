import importlib.util

# MinkowskiEngine 只有 uca3dal_mink backbone 需要，缺失时不阻塞其他 backbone 的导入
if importlib.util.find_spec("MinkowskiEngine") is not None:
    from .uca3dal_mink import *
from .pointmae import *

backbone_info = {
    "uca3dal_mink": {},
    "pointmae": {},
}
