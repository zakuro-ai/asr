rm -r .deps; mkdir .deps; cd .deps
#Install this fork for Warp-CTC bindings:
git clone https://github.com/SeanNaren/warp-ctc.git
cd warp-ctc; mkdir build; cd build; cmake ..; make -j32
export CUDA_HOME="/usr/local/cuda"
cd ../pytorch_binding && python setup.py install
cd ../../

#Install NVIDIA apex:
git clone --recursive https://github.com/NVIDIA/apex.git
cd apex && pip install .
cd ../

#Beam search
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install .
cd ../../
