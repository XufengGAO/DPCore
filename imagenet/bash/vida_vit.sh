# export PYTHONPATH=
# conda deactivate
# conda activate vida 
# --checkpoint [your-ckpt-path]
DATA_DIR="/root/autodl-tmp/data"
data_dir=$DATA_DIR

python imagenetc.py --cfg ./cfgs/vit/vida.yaml  --data_dir $data_dir --checkpoint /root/autodl-tmp/imagent_vit_vida.pt
