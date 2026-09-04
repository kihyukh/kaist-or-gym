"""KAIST RL Lab Gymnasium environments."""

from .version import __version__
from .envs import *
from .registration import _register

_register()
