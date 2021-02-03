from gnutools.test import test_imports
from gnutools.fs import parent
import os
test_imports(parent(os.path.realpath(__file__), level=2))