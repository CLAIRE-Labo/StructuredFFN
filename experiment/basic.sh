#!/bin/bash
s_token=2200000000
m_token=6700000000
l_token=14580000000
xl_token=25500000000
s_lr=0.0006
m_lr=0.0003
l_lr=0.00025
xl_lr=0.0002
s_train_batch=64
s_test_batch=64
m_train_batch=32
m_test_batch=32
l_train_batch=16
l_test_batch=32
xl_train_batch=16
xl_test_batch=16


# dense
./run_gpt.sh "gpt2s" 1 "torchrun --nnodes=1 --nproc_per_node=1 refinedweb_experiment.py model=gpt2 method=linear optimization.optimizer.kwargs.lr=${s_lr} optimization.max_tokens=${s_token} data.train.train_batch=${s_train_batch} data.test.test_batch=${s_test_batch} optimization.log_interval=100"
./run_gpt.sh "gpt2m" 1 "torchrun --nnodes=1 --nproc_per_node=1 refinedweb_experiment.py model=gpt2m method=linear optimization.optimizer.kwargs.lr=${s_lr} optimization.max_tokens=${s_token} data.train.train_batch=${m_train_batch} data.test.test_batch=${m_test_batch} optimization.log_interval=100"
./run_gpt.sh "gpt2l" 1 "torchrun --nnodes=1 --nproc_per_node=1 refinedweb_experiment.py model=gpt2l method=linear optimization.optimizer.kwargs.lr=${s_lr} optimization.max_tokens=${s_token} data.train.train_batch=${l_train_batch} data.test.test_batch=${l_test_batch} optimization.log_interval=100"
./run_gpt.sh "gpt2xl" 1 "torchrun --nnodes=1 --nproc_per_node=1 refinedweb_experiment.py model=gpt2xl method=linear optimization.optimizer.kwargs.lr=${s_lr} optimization.max_tokens=${s_token} data.train.train_batch=${xl_train_batch} data.test.test_batch=${xl_test_batch} optimization.log_interval=100"

# LowRank
./run_gpt.sh "gpt2s-lr-0x63" 1 "torchrun --nnodes=1 --nproc_per_node=1 refinedweb_experiment.py model=gpt2 method=lowrank method.kwargs.rank=384 optimization.optimizer.kwargs.lr=${s_lr} optimization.max_tokens=${s_token} data.train.train_batch=${s_train_batch} data.test.test_batch=${s_test_batch} optimization.log_interval=100"
./run_gpt.sh "gpt2s-lr-0x33" 1 "torchrun --nnodes=1 --nproc_per_node=1 refinedweb_experiment.py model=gpt2 method=lowrank method.kwargs.rank=192 optimization.optimizer.kwargs.lr=${s_lr} optimization.max_tokens=${s_token} data.train.train_batch=${s_train_batch} data.test.test_batch=${s_test_batch} optimization.log_interval=100"
./run_gpt.sh "gpt2m-lr-0x63" 1 "torchrun --nnodes=1 --nproc_per_node=1 refinedweb_experiment.py model=gpt2m method=lowrank method.kwargs.rank=512 optimization.optimizer.kwargs.lr=${m_lr} optimization.max_tokens=${m_token} data.train.train_batch=${m_train_batch} data.test.test_batch=${m_test_batch} optimization.log_interval=100"
./run_gpt.sh "gpt2m-lr-0x33" 1 "torchrun --nnodes=1 --nproc_per_node=1 refinedweb_experiment.py model=gpt2m method=lowrank method.kwargs.rank=256 optimization.optimizer.kwargs.lr=${m_lr} optimization.max_tokens=${m_token} data.train.train_batch=${m_train_batch} data.test.test_batch=${m_test_batch} optimization.log_interval=100"
./run_gpt.sh "gpt2l-lr-0x63" 1 "torchrun --nnodes=1 --nproc_per_node=1 refinedweb_experiment.py model=gpt2l method=lowrank method.kwargs.rank=768 optimization.optimizer.kwargs.lr=${l_lr} optimization.max_tokens=${l_token} data.train.train_batch=${l_train_batch} data.test.test_batch=${l_test_batch} optimization.log_interval=100"
./run_gpt.sh "gpt2l-lr-0x33" 1 "torchrun --nnodes=1 --nproc_per_node=1 refinedweb_experiment.py model=gpt2l method=lowrank method.kwargs.rank=384  optimization.optimizer.kwargs.lr=${l_lr} optimization.max_tokens=${l_token} data.train.train_batch=${l_train_batch} data.test.test_batch=${l_test_batch} optimization.log_interval=100"
./run_gpt.sh "gpt2xl-lr-0x63" 1 "torchrun --nnodes=1 --nproc_per_node=1 refinedweb_experiment.py model=gpt2xl method=lowrank method.kwargs.rank=1024 optimization.optimizer.kwargs.lr=${xl_lr} optimization.max_tokens=${xl_token} data.train.train_batch=${xl_train_batch} data.test.test_batch=${xl_test_batch} optimization.log_interval=100"
./run_gpt.sh "gpt2xl-lr-0x33" 1 "torchrun --nnodes=1 --nproc_per_node=1 refinedweb_experiment.py model=gpt2xl method=lowrank method.kwargs.rank=512 optimization.optimizer.kwargs.lr=${xl_lr} optimization.max_tokens=${xl_token} data.train.train_batch=${xl_train_batch} data.test.test_batch=${xl_test_batch} optimization.log_interval=100"

# BlockDense
./run_gpt.sh "gpt2s-bld-0x63" 1 "torchrun --nnodes=1 --nproc_per_node=1 refinedweb_experiment.py model=gpt2 method=blockdense method.kwargs.nblocks=2 method.kwargs.rank=512 optimization.optimizer.kwargs.lr=${s_lr} optimization.max_tokens=${s_token} data.train.train_batch=${s_train_batch} data.test.test_batch=${s_test_batch} optimization.log_interval=100"
./run_gpt.sh "gpt2s-bld-0x33" 1 "torchrun --nnodes=1 --nproc_per_node=1 refinedweb_experiment.py model=gpt2 method=blockdense method.kwargs.nblocks=2 method.kwargs.rank=256 optimization.optimizer.kwargs.lr=${s_lr} optimization.max_tokens=${s_token} data.train.train_batch=${s_train_batch} data.test.test_batch=${s_test_batch} optimization.log_interval=100"
./run_gpt.sh "gpt2m-bld-0x63" 1 "torchrun --nnodes=1 --nproc_per_node=1 refinedweb_experiment.py model=gpt2m method=blockdense method.kwargs.nblocks=4 method.kwargs.rank=768 optimization.optimizer.kwargs.lr=${m_lr} optimization.max_tokens=${m_token} data.train.train_batch=${m_train_batch} data.test.test_batch=${m_test_batch} optimization.log_interval=100"
./run_gpt.sh "gpt2m-bld-0x33" 1 "torchrun --nnodes=1 --nproc_per_node=1 refinedweb_experiment.py model=gpt2m method=blockdense method.kwargs.nblocks=4 method.kwargs.rank=384 optimization.optimizer.kwargs.lr=${m_lr} optimization.max_tokens=${m_token} data.train.train_batch=${m_train_batch} data.test.test_batch=${m_test_batch} optimization.log_interval=100"
./run_gpt.sh "gpt2l-bld-0x63" 1 "torchrun --nnodes=1 --nproc_per_node=1 refinedweb_experiment.py model=gpt2l method=blockdense method.kwargs.nblocks=2 method.kwargs.rank=1024 optimization.optimizer.kwargs.lr=${l_lr} optimization.max_tokens=${l_token} data.train.train_batch=${l_train_batch} data.test.test_batch=${l_test_batch} optimization.log_interval=100"
./run_gpt.sh "gpt2l-bld-0x33" 1 "torchrun --nnodes=1 --nproc_per_node=1 refinedweb_experiment.py model=gpt2l method=blockdense method.kwargs.nblocks=2 method.kwargs.rank=512  optimization.optimizer.kwargs.lr=${l_lr} optimization.max_tokens=${l_token} data.train.train_batch=${l_train_batch} data.test.test_batch=${l_test_batch} optimization.log_interval=100"
./run_gpt.sh "gpt2xl-bld-0x63" 1 "torchrun --nnodes=1 --nproc_per_node=1 refinedweb_experiment.py model=gpt2xl method=blockdense method.kwargs.nblocks=4 method.kwargs.rank=1536 optimization.optimizer.kwargs.lr=${xl_lr} optimization.max_tokens=${xl_token} data.train.train_batch=${xl_train_batch} data.test.test_batch=${xl_test_batch} optimization.log_interval=100"
./run_gpt.sh "gpt2xl-bld-0x33" 1 "torchrun --nnodes=1 --nproc_per_node=1 refinedweb_experiment.py model=gpt2xl method=blockdense method.kwargs.nblocks=4 method.kwargs.rank=768 optimization.optimizer.kwargs.lr=${xl_lr} optimization.max_tokens=${xl_token} data.train.train_batch=${xl_train_batch} data.test.test_batch=${xl_test_batch} optimization.log_interval=100"

# BlockShuffle
./run_gpt.sh "gpt2s-bls-0x63" 1 "torchrun --nnodes=1 --nproc_per_node=1 refinedweb_experiment.py model=gpt2 method=blockshuffle method.kwargs.nblocks=2 optimization.optimizer.kwargs.lr=${s_lr} optimization.max_tokens=${s_token} data.train.train_batch=${s_train_batch} data.test.test_batch=${s_test_batch} optimization.log_interval=100"
./run_gpt.sh "gpt2s-bls-0x33" 1 "torchrun --nnodes=1 --nproc_per_node=1 refinedweb_experiment.py model=gpt2 method=blockshuffle method.kwargs.nblocks=4 optimization.optimizer.kwargs.lr=${s_lr} optimization.max_tokens=${s_token} data.train.train_batch=${s_train_batch} data.test.test_batch=${s_test_batch} optimization.log_interval=100"
./run_gpt.sh "gpt2m-bls-0x63" 1 "torchrun --nnodes=1 --nproc_per_node=1 refinedweb_experiment.py model=gpt2m method=blockshuffle method.kwargs.nblocks=2 optimization.optimizer.kwargs.lr=${m_lr} optimization.max_tokens=${m_token} data.train.train_batch=${m_train_batch} data.test.test_batch=${m_test_batch} optimization.log_interval=100"
./run_gpt.sh "gpt2m-bls-0x33" 1 "torchrun --nnodes=1 --nproc_per_node=1 refinedweb_experiment.py model=gpt2m method=blockshuffle method.kwargs.nblocks=4 optimization.optimizer.kwargs.lr=${m_lr} optimization.max_tokens=${m_token} data.train.train_batch=${m_train_batch} data.test.test_batch=${m_test_batch} optimization.log_interval=100"
./run_gpt.sh "gpt2l-bls-0x63" 1 "torchrun --nnodes=1 --nproc_per_node=1 refinedweb_experiment.py model=gpt2l method=blockshuffle method.kwargs.nblocks=2 optimization.optimizer.kwargs.lr=${l_lr} optimization.max_tokens=${l_token} data.train.train_batch=${l_train_batch} data.test.test_batch=${l_test_batch} optimization.log_interval=100"
./run_gpt.sh "gpt2l-bls-0x33" 1 "torchrun --nnodes=1 --nproc_per_node=1 refinedweb_experiment.py model=gpt2l method=blockshuffle method.kwargs.nblocks=4 optimization.optimizer.kwargs.lr=${l_lr} optimization.max_tokens=${l_token} data.train.train_batch=${l_train_batch} data.test.test_batch=${l_test_batch} optimization.log_interval=100"
./run_gpt.sh "gpt2xl-bls-0x63" 1 "torchrun --nnodes=1 --nproc_per_node=1 refinedweb_experiment.py model=gpt2xl method=blockshuffle method.kwargs.nblocks=2 optimization.optimizer.kwargs.lr=${xl_lr} optimization.max_tokens=${xl_token} data.train.train_batch=${xl_train_batch} data.test.test_batch=${xl_test_batch} optimization.log_interval=100"
./run_gpt.sh "gpt2xl-bls-0x33" 1 "torchrun --nnodes=1 --nproc_per_node=1 refinedweb_experiment.py model=gpt2xl method=blockshuffle method.kwargs.nblocks=4 optimization.optimizer.kwargs.lr=${xl_lr} optimization.max_tokens=${xl_token} data.train.train_batch=${xl_train_batch} data.test.test_batch=${xl_test_batch} optimization.log_interval=100"
