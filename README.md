# benchmarks
Computational/deep learning benchmarks.

Currently, there is the only script called `super_burner.py`. It supports `matmul`, `conv2d` and `rnn` tasks in both `PyTorch` and `TensorFlow`.

Example usage:

```
CUDA_VISIBLE_DEVICES=0 python super_burner.py \
    --framework=tf --task=conv2d \
    --batch-size=32 --iterations=100000 \
    --kernel-size=3 --input-size-2d=224 --depth=64
```

To get information about other supported options, type `python super_burner.py --help`.
