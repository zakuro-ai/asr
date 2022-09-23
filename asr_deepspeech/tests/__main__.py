from gnutools.tests import test_imports
from gnutools.fs import parent
import os
import torch
# import  warpctc_pytorch as warp_ctc
test_imports(parent(os.path.realpath(__file__), level=2))
# try:
#     warp_ctc.gpu_ctc if torch.cuda.is_available() else warp_ctc.cpu_ctc
#     print("=1= TEST PASSED : warp_ctc.gpu_ctc")
# except Exception as e:
#     print("=0= TEST FAILED : warp_ctc.gpu_ctc")
#     raise e