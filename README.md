# LDSAGE
LDSAGE implement

## environment
```shell
conda create -n bole python=3.7.9

conda activate bole

conda install -c aibox recbole=1.1.1

pip uninstall torch

pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install dgl-cu113 dglgo -f https://data.dgl.ai/wheels/repo.html

possible problem:
ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found 
(required by /home/dw/miniconda3/envs/bole/lib/python3.7/site-packages/scipy/spatial/ckdtree.cpython-37m-x86_64-linux-gnu.so)

solution:
find your up to date libstdc++.so

cp /home/dw/miniconda3/envs/bole/lib/libstdc++.so.6.0.30 /usr/lib/x86_64-linux-gnu/

cd /usr/lib/x86_64-linux-gnu

rm -rf libstdc++.so.6

ln -s libstdc++.so.6.0.30 libstdc++.so.6
```

## Dataset
Create a new folder named 'dataset' at the same level as 'args'. Then Download dataset follow https://recbole.io/docs/user_guide/data/dataset_download.html

## Quick Start
```shell
python main.py
```
