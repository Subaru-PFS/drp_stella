"""Claude: the old name of `pfs.drp.stella.armLsf`, kept so that pickled models can be read

The ``lsfModel`` datasets written before the rename are pickles of
``pfs.drp.stella.lsfModel.ArmLsfModel``.
"""
from .armLsf import *  # noqa: F401,F403
from .armLsf import ArmLsf

ArmLsfModel = ArmLsf
