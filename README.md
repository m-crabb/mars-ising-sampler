# mars-ising-sampler
Repository for MARS V application - Models of Sparsity and Mean Field Theory, Dmitry Vaintrob

## Task Brief
Write a model (transformer or your favorite architecture) that learns to approximately sample an N x N Ising
lattice at critical temperature (feel free to choose N here). Explain how you can check whether its outputs
are a reasonable sampler (assign some numerical score).

## Acknowledgements
The work here builds directly on that presented in https://github.com/J-zin/DNFS, https://github.com/yuchen-zhu-zyc/MDNS, and https://github.com/malbergo/leaps. Thanks to the authors for providing their papers and code - my implementation is mostly influenced by Zijing's work, with some inspiration from the other two. Most of my code was written from paper reference only, consulting the repositories when I became blocked by poor training performance.
