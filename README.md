# PyTorch fpsample

PyTorch efficient farthest point sampling (FPS) implementation, adopted from [fpsample](https://github.com/leonardodalinky/fpsample).

**This project currently has a GPU implementation, but its performance is lower than the CPU version in most tested [cases](https://github.com/leonardodalinky/pytorch_fpsample/issues/1#issuecomment-4327426027). Therefore, I recommend using the original [pytorch_fpsample](https://github.com/leonardodalinky/pytorch_fpsample)'s CPU implementation by default. This CUDA implementation should only be considered when CPU performance is limited or when GPU-side execution is specifically required.**


> [!NOTE]
> Since the PyTorch capsules the native multithread implementation, this project is expected to have a much better performance than the *fpsample* implementation.

## Installation

```bash
# Install from github
pip install git+https://github.com/hanm2019/pytorch_fpsample

# Build locally
pip install .
```

## Usage

```python
import torch_fpsample

x = torch.rand(64, 2048, 3)
# random sample
sampled_points, indices = torch_fpsample.sample(x, 1024)
# random sample with specific tree height
sampled_points, indices = torch_fpsample.sample(x, 1024, h=5)
# random sample with start point index (int)
sampled_points, indices = torch_fpsample.sample(x, 1024, start_idx=0)

> sampled_points.size(), indices.size()
Size([64, 1024, 3]), Size([64, 1024])
```

> [!WARNING]
> By default, if the input `x` is on the GPU, the CUDA implementation will be used; otherwise, the CPU implementation will be used.
> However, the CUDA implementation may be slower than the CPU implementation in most cases, so please benchmark both backends before using it in performance-critical workloads.

## Reference
Bucket-based farthest point sampling (QuickFPS) is proposed in the following paper. The implementation is based on the author's Repo ([CPU](https://github.com/hanm2019/bucket-based_farthest-point-sampling_CPU) & [GPU](https://github.com/hanm2019/bucket-based_farthest-point-sampling_GPU)).
```bibtex
@article{han2023quickfps,
  title={QuickFPS: Architecture and Algorithm Co-Design for Farthest Point Sampling in Large-Scale Point Clouds},
  author={Han, Meng and Wang, Liang and Xiao, Limin and Zhang, Hao and Zhang, Chenhao and Xu, Xiangrong and Zhu, Jianfeng},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
  year={2023},
  publisher={IEEE}
}
```

Thanks to the authors for their great works.
