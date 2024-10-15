# export PYTHONPATH=
# conda deactivate
# conda activate vida
DATA_DIR="/root/autodl-tmp/data"
data_dir=$DATA_DIR

python imagenetc.py --cfg ./cfgs/vit/cotta.yaml --data_dir $data_dir