# export PYTHONPATH=
# conda deactivate
# conda activate vida
DATA_DIR="/mnt/data/xugao"
data_dir=$DATA_DIR

python imagenetc.py --cfg ./cfgs/vit/dpcore.yaml  --data_dir $data_dir


