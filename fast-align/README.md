# Basic Operations of *fast-align*

### Alignment Model
- Install [fast_align](https://github.com/wxjiao/fast_align) and compile, just follow the official repo:
```
git clone https://github.com/wxjiao/fast_align.git

cd fast_align
mkdir build
cd build
cmake ..
make
```

- Prepare input data in the format ``src ||| tgt":
```
paste wmt14_en_de_bitext/train.en wmt14_en_de_bitext/train.de  | sed 's/ *\t */ ||| /g' > wmt14_en_de_bitext/train.en-de
```

- Train a forward alignment model:
```
./fast_align -i wmt14_en_de_bitext/train.en-de -d -v -o -p wmt14_en_de_bitext/train.fwd_params > wmt14_en_de_bitext/train.fwd_align 2> wmt14_en_de_bitext/train.fwd_err
```

- Train a reversed alignment model:
```
./fast_align -i wmt14_en_de_bitext/train.en-de -r -d -v -o -p wmt14_en_de_bitext/train.rev_params > wmt14_en_de_bitext/train.rev_align 2> wmt14_en_de_bitext/train.rev_err
```

- Symmetrize the alignment:
```
./atools -i wmt14_en_de_bitext/train.fwd_align -j wmt14_en_de_bitext/train.rev_align -c grow-diag-final-and > wmt14_en_de_bitext/train.sym_align
```

- Apply the trained alignment to infer:
```
./force_align.py wmt14_en_de_bitext/train.fwd_params wmt14_en_de_bitext/train.fwd_err wmt14_en_de_bitext/train.rev_params wmt14_en_de_bitext/train.rev_err < wmt14_en_de_bitext/test.en-de > wmt14_en_de_bitext/test.en-de.sym_align
```

- Tips: run commands in background:
> nohup [commands] &


