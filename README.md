# Stochastic Marginal Actor-Critic

This is the pytorch implementation for our ICLR 2023 paper: 

[Latent State Marginalization as a Low-cost Approach for Improving Exploration](https://arxiv.org/abs/2210.00999).  
[Dinghuai Zhang](https://zdhnarsil.github.io/), Aaron Courville, [Yoshua Bengio](https://yoshuabengio.org/), [Qinqing Zheng](https://enosair.github.io/), [Amy Zhang](https://amyzhang.github.io/), [Ricky T. Q. Chen](https://rtqichen.github.io//).


We also provide an efficient pytorch [Dreamer](https://arxiv.org/abs/1912.01603)  implementation at `visual_control/dreamer.py`, which takes about 7 hours for 1 million frames (action_repeat=2) on a single NVIDIA V100. 

---

<!-- <p align="center"> -->
<img src="https://i.328888.xyz/2023/02/10/RVkHc.png" alt="RVkHc.png" border="0" width=70% height=70% class="center" />
<!-- </p> -->

We show how to efficiently marginalize latent variable policies for MaxEnt RL to enable better exploration and more robust training at very minimal additional cost. The proposed algorithm is coined as Stochastic Marginal Actor-Critic (SMAC).



## Dependencies
```
pip install -r requirements.txt
```

## Run
Codes for running visual control experiments with Dreamer (model-based RL), latent SAC (model-free SAC in the latent space of RSSM), and SMAC (the proposed method).
```
python -m visual_control.main env=cheetah_run dreamer=0  # Dreamer
python -m visual_control.main env=cheetah_run dreamer=1  # Latent SAC
python -m visual_control.main env=cheetah_run dreamer=2  # SMAC
```

## Citation
If you find our paper / repository helpful, please consider citing:
```
@inproceedings{zhang2023latent,
    title={Latent State Marginalization as a Low-cost Approach to Improving Exploration},
    author={Dinghuai Zhang and Aaron Courville and Yoshua Bengio and Qinqing Zheng and Amy Zhang and Ricky T. Q. Chen},
    booktitle={International Conference on Learning Representations},
    year={2023},
    url={https://openreview.net/forum?id=b0UksKFcTOL}
}
```

## Acknowledgement
Part of this repo builds upon Zhixuan Lin's [code](https://github.com/zhixuan-lin/dreamer-pytorch).
