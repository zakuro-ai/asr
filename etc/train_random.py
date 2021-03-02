import sys
import subprocess
from gnutools.utils import listfiles, name
import random
argslist = list(sys.argv)[1:]
root_manifest = argslist[argslist.index('--root-manifest') + 1]
manifests = list(set([name(manifest).replace('train_', '').replace('val_', '')
                      for manifest in listfiles(root_manifest, ['.json'])]))
random.shuffle(manifests)
for manifest in manifests:
    argslist_k = argslist + ['--manifest', name(manifest), '--epochs', '1']
    print(argslist_k)
    p = subprocess.Popen([str(sys.executable)] + ['train.py'] + argslist_k)
    p.wait()
    if p.returncode != 0:
        raise subprocess.CalledProcessError(returncode=p.returncode,
                                            cmd=p.args)
