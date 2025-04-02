# SPDX-License-Identifier: Apache-2.0
import time
from collections import deque
from collections.abc import Iterable
from typing import Optional, Union

from vllm.config import CacheConfig, SchedulerConfig
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import (CachedRequestData, NewRequestData,
                                       SchedulerOutput)
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.engine import (EngineCoreEventType, EngineCoreOutput,
                            EngineCoreOutputs)
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus


class SlidingWindowScheduler(SchedulerInterface):

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        log_stats: bool = False,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.log_stats = log_stats

        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = (
            self.scheduler_config.max_num_batched_tokens)
        self.max_model_len = self.scheduler_config.max_model_len
        self.sliding_window_size = self.scheduler_config.sliding_window_size

        num_gpu_blocks = cache_config.num_gpu_blocks
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        # Create the KV cache manager.
        self.kv_cache_manager = KVCacheManager(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=num_gpu_blocks,
            max_model_len=self.max_model_len,
            sliding_window=self.cache_config.sliding_window,
            enable_caching=self.cache_config.enable_prefix_caching,
            caching_hash_algo=self.cache_config.prefix_caching_hash_algo,
            log_stats=self.log_stats,
        )
        self.block_size = self.cache_config.block_size

        self.requests: dict[str, Request] = {}
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []

        self.scheduled_req_ids: deque[set[str]] = deque(
            maxlen=self.sliding_window_size)
        self.finished_req_ids: set[str] = set()

        self._cached_reqs_data: dict[str, CachedRequestData] = {}

    def schedule(self) -> SchedulerOutput:
        scheduled_new_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []

        req_to_new_block_ids: dict[str, list[int]] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        current_scheduled_req_ids: set[str] = set()

        # For logging.
        scheduled_timestamp = time.monotonic()

        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]
            if request.request_id in current_scheduled_req_ids:
                req_index += 1
                continue

            num_new_tokens = (request.num_tokens_with_spec -
                              request.num_computed_tokens)
            assert num_new_tokens >= 0

            if num_new_tokens == 0:
                assert len(self.scheduled_req_ids) < self.sliding_window_size
                num_new_tokens = 1

            if (0 < self.scheduler_config.long_prefill_token_threshold <
                    num_new_tokens):
                num_new_tokens = (
                    self.scheduler_config.long_prefill_token_threshold)
            num_new_tokens = min(num_new_tokens, token_budget)

            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request, num_new_tokens)
                # TODO(yejunjin): implement preempted
                assert new_blocks is not None
                break

            scheduled_running_reqs.append(request)
            current_scheduled_req_ids.add(request.request_id)
            req_to_new_block_ids[request.request_id] = [
                b.block_id for b in new_blocks
            ]

            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

        # TODO(yejunjin): implement preempted_reqs
        while self.waiting and token_budget > 0:
            if len(self.running) == self.max_num_running_reqs:
                break

            request = self.waiting[0]
            # TODO(yejunjin): implement structured output request

            # Get already-cached tokens.
            computed_blocks, num_computed_tokens = (
                self.kv_cache_manager.get_computed_blocks(request))
            # Number of tokens to be scheduled.
            # We use `request.num_tokens` instead of
            # `request.num_prompt_tokens` to consider the resumed requests,
            # which have output tokens.
            num_new_tokens = request.num_tokens - num_computed_tokens
            if num_new_tokens == 0:
                # This happens when prompt length is divisible by the block
                # size and all blocks are cached. Now we force to recompute
                # the last block. Note that we have to re-compute an entire
                # block because allocate_slots() assumes num_computed_tokens
                # is always a multiple of the block size. This limitation
                # can potentially be removed in the future to slightly
                # improve the performance.
                num_computed_tokens -= self.block_size
                num_new_tokens = self.block_size
                computed_blocks.pop()
            if (0 < self.scheduler_config.long_prefill_token_threshold <
                    num_new_tokens):
                num_new_tokens = (
                    self.scheduler_config.long_prefill_token_threshold)
            num_new_tokens = min(num_new_tokens, token_budget)
            assert num_new_tokens > 0
            if num_new_tokens == 0:
                break

            new_blocks = self.kv_cache_manager.allocate_slots(
                request, num_new_tokens, computed_blocks)
            if new_blocks is None:
                # The request cannot be scheduled.
                break

            self.waiting.popleft()
            req_index += 1
            self.running.append(request)
            current_scheduled_req_ids.add(request.request_id)
            if self.log_stats:
                request.record_event(EngineCoreEventType.SCHEDULED,
                                     scheduled_timestamp)
            if request.status == RequestStatus.WAITING:
                scheduled_new_reqs.append(request)
            # TODO(yejunjin): implement preempted
            else:
                raise RuntimeError(f"Invalid request status: {request.status}")

            req_to_new_block_ids[request.request_id] = [
                b.block_id for b in computed_blocks + new_blocks
            ]
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            request.status = RequestStatus.RUNNING
            request.num_computed_tokens = num_computed_tokens

        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # TODO(yejunjin): implement preempted
        assert len(scheduled_new_reqs) + len(scheduled_running_reqs) <= len(
            self.running)

        # TODO(yejunjin): implement num_common_prefix_blocks
        new_reqs_data = [
            NewRequestData.from_request(req,
                                        req_to_new_block_ids[req.request_id])
            for req in scheduled_new_reqs
        ]
        running_reqs_data = [
            self._make_cached_request_data(
                req,
                num_scheduled_tokens[req.request_id],
                0,
                req_to_new_block_ids[req.request_id],
                resumed_from_preemption=False,
            ) for req in scheduled_running_reqs
        ]
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=running_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=0,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_input_ids=[],
            structured_output_request_ids={},
            grammar_bitmask=None,
        )

        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            self.requests[req_id].num_computed_tokens += num_scheduled_token

        self.scheduled_req_ids.append(current_scheduled_req_ids)
        self.finished_req_ids = set()
        return scheduler_output

    def _make_cached_request_data(
        self,
        request: Request,
        num_scheduled_tokens: int,
        num_scheduled_spec_tokens: int,
        new_block_ids: list[int],
        resumed_from_preemption: bool,
    ) -> CachedRequestData:
        # OPTIMIZATION: Cache the CachedRequestData objects to avoid creating
        # them at each scheduling step.
        num_computed_tokens = request.num_computed_tokens
        num_regular_tokens = num_scheduled_tokens - num_scheduled_spec_tokens
        new_token_idices = list(
            range(num_computed_tokens,
                  num_computed_tokens + num_regular_tokens))
        req_data = self._cached_reqs_data.get(request.request_id)
        if req_data is not None:
            req_data.resumed_from_preemption = resumed_from_preemption
            req_data.new_token_ids = []
            req_data.new_block_ids = new_block_ids
            req_data.num_computed_tokens = num_computed_tokens
            req_data.new_token_idices = new_token_idices
        else:
            req_data = CachedRequestData.from_request(
                request,
                resumed_from_preemption,
                [],
                new_block_ids,
                new_token_idices,
            )
            self._cached_reqs_data[request.request_id] = req_data
        return req_data

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> EngineCoreOutputs:
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens

        new_running: list[Request] = []
        outputs: list[EngineCoreOutput] = []

        for request in self.running:
            req_id = request.request_id
            num_tokens_scheduled = num_scheduled_tokens.get(req_id, 0)
            if num_tokens_scheduled == 0:
                # The request was not scheduled in this step.
                new_running.append(request)
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[req_index]

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids

            # Append generated tokens and check for stop. Note that if
            # a request is still being prefilled, we expect the model runner
            # to return empty token ids for the request.
            for num_new, output_token_id in enumerate(new_token_ids, 1):
                request.append_output_token_ids(output_token_id)

                # Check for stop and update request state.
                # This must be called before we make the EngineCoreOutput.
                stopped = check_stop(request, self.max_model_len)
                if stopped:
                    self._free_request(request)
                    del new_token_ids[num_new:]  # Trim new tokens if needed.
                    break

            # Extract sample logprobs if needed.
            if request.sampling_params.logprobs is not None and logprobs:
                # NOTE: once we support N tokens per step (spec decode),
                # the outer lists can be of length > 1.
                new_logprobs = logprobs.slice(req_index, req_index + 1)

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids:
                # Add EngineCoreOutput for this Request.
                outputs.append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                    ))
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

            if not stopped:
                new_running.append(request)

        self.running = new_running
        self.scheduled_req_ids.pop()
        engine_core_outputs = EngineCoreOutputs(
            outputs=outputs,
            scheduler_stats=self.make_stats(),
        )

        return engine_core_outputs

    def add_request(self, request: Request) -> None:
        self.waiting.append(request)
        self.requests[request.request_id] = request
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)

    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str]],
        finished_status: RequestStatus,
    ) -> None:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids, )
        else:
            request_ids = set(request_ids)

        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None:
                # Invalid request ID.
                continue

            if request.status == RequestStatus.RUNNING:
                self.running.remove(request)
                for s_req_ids in self.scheduled_req_ids:
                    if req_id in s_req_ids:
                        s_req_ids.remove(req_id)
            else:
                self.waiting.remove(request)
            request.status = finished_status
            self._free_request(request)

    def _free_request(self, request: Request) -> None:
        assert request.is_finished()
        self.kv_cache_manager.free(request)
        self.kv_cache_manager.free_block_hashes(request)
        self._cached_reqs_data.pop(request.request_id, None)
        del self.requests[request.request_id]
        self.finished_req_ids.add(request.request_id)

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def has_finished_requests(self) -> bool:
        return len(self.finished_req_ids) > 0

    def get_num_unscheduled_requests(self) -> int:
        """Number of requests that are not being processed by the executor."""
        return self.get_num_unfinished_requests() - len(self.scheduled_req_ids)

    def reset_prefix_cache(self) -> bool:
        return self.kv_cache_manager.reset_prefix_cache()

    def make_stats(self) -> Optional[SchedulerStats]:
        if not self.log_stats:
            return None
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            gpu_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=self.kv_cache_manager.make_prefix_cache_stats(),
        )
