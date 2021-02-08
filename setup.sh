##Install NVIDIA apex:
git clone --recursive https://github.com/NVIDIA/apex.git
cd apex && pip install .
cd ../

#Beam search
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install .
cd ../../

# Install asr_deepspeech
python setup.py install
