#!/bin/bash

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export VLLM_USE_MODELSCOPE=True

# control num_gpu_blocks_override to avoid miscomputing avaiable gpu blocks
# since we have two engines now!
vllm serve /home/admin/resource/model/16394f66.qwen2.5-270ba37b-instruct/128k-20250112/ -tp 8 \
    --gpu_memory_utilization 0.9 \
    --num_gpu_blocks_override 4096

# Qwen/Qwen2.5-0.5B-Instruct
#/home/admin/resource/model/16394f66.qwen2.5-270ba37b-instruct/128k-20250112/