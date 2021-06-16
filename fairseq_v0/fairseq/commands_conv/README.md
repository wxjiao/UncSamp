# Configure envir for LightConv and DynamicConv

## On jizhi
- Update gcc and g++:

> conda install gcc_linux-64

> conda install gxx_linux-64

> conda install cython

- Create soft symbol link:

> ln -s x86_64-conda_cos6-linux-gnu-gcc gcc

> ln -s x86_64-conda_cos6-linux-gnu-g++ g++

## In fairseq
- Add an argument '--float-valid' in `option.py`;

- Alternate between `fp16` and `float` in `valid_step` of `trainer.py`.
