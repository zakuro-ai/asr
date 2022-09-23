__version__ = "0.3.2"
import os

from gnutools import fs

from .functional import *
from .vars import *

try:
    cfg = fs.load_config(os.environ["ZAK_ASR_CONFIG"])
except:
    cfg = fs.load_config(f"{fs.parent(__file__)}/config.yml")
