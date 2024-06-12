from openood.evaluators.mos_evaluator import MOSEvaluator
from openood.utils import Config

from .ad_evaluator import ADEvaluator
from .arpl_evaluator import ARPLEvaluator
from .base_evaluator import BaseEvaluator
from .dump_evaluator import DumpEvaluator
from .ece_evaluator import ECEEvaluator
from .fsood_evaluator import FSOODEvaluator
from .ood_evaluator import OODEvaluator
from .osr_evaluator import OSREvaluator
from .patchcore_evaluator import PatchCoreEvaluator


def get_evaluator(config: Config):
    evaluators = {
        'base': BaseEvaluator,
        'dump': DumpEvaluator,
        'ood': OODEvaluator,
        'fsood': FSOODEvaluator,
        'patch': PatchCoreEvaluator,
        'arpl': ARPLEvaluator,
        'ad': ADEvaluator,
        'mos': MOSEvaluator,
        'ece': ECEEvaluator,
        'osr': OSREvaluator
    }
    return evaluators[config.evaluator.name](config)
