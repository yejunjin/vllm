#!/bin/bash

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export VLLM_USE_MODELSCOPE=True

# control num_gpu_blocks_override to avoid miscomputing avaiable gpu blocks
# since we have two engines now!
vllm serve Qwen/Qwen2.5-0.5B-Instruct -tp 2 \
    --num_gpu_blocks_override 4096