# DPCore
[Dynamic Domains, Dynamic Solutions: DPCore for Continual Test-Time Adaptation](https://arxiv.org/pdf/2406.10737)

Yunbei Zhang, Akshay Mehra, Jihun Hamm

## Requirements
```bash
conda update conda
conda env create -f environment.yml
conda activate dpcore 
```

## ImageNet-C Experiments

```bash
cd imagenet
sh ./bash/dpcore_vit.sh
```


## Citation
Please cite our work if you find it useful.
```bibtex
@article{zhang2024dynamic,
  title={Dynamic Domains, Dynamic Solutions: DPCore for Continual Test-Time Adaptation},
  author={Zhang, Yunbei and Mehra, Akshay and Hamm, Jihun},
  journal={arXiv preprint arXiv:2406.10737},
  year={2024}
}
```

## Acknowlegdement
[CoTTA](https://github.com/qinenergy/cotta) code is heavily used. \
[Robustbench](https://github.com/RobustBench/robustbench) official.