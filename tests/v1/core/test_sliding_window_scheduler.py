# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import pytest

from vllm.config import CacheConfig, SchedulerConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.sliding_window_scheduler import SlidingWindowScheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

EOS_TOKEN_ID = 50256


def create_sliding_window_scheduler(
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    enable_prefix_caching: Optional[bool] = None,
    long_prefill_token_threshold: int = 0,
    sliding_window_size: int = 1,
) -> SlidingWindowScheduler:
    '''Create sliding windows scheduler under test.

    Args:
      model: model under test
      max_num_seqs: max sequences to schedule
      max_num_batch_tokens: max num tokens to batch
      enable_prefix_caching: optionally force APC config
                             (True/False) or use default
                             (None)

    Returns:
      :class:`SlidingWindowScheduler` instance
    '''
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_num_batched_tokens,
        long_prefill_token_threshold=long_prefill_token_threshold,
        sliding_window_size=sliding_window_size,
    )
    # Cache config, optionally force APC
    kwargs_cache = ({} if enable_prefix_caching is None else {
        'enable_prefix_caching': enable_prefix_caching
    })
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
        **kwargs_cache,
    )
    cache_config.num_gpu_blocks = 10000
    return SlidingWindowScheduler(scheduler_config,
                                  cache_config,
                                  log_stats=True)


def create_requests(num_requests: int,
                    num_tokens: int = 10,
                    max_tokens: int = 16,
                    stop_token_ids: Optional[list[int]] = None,
                    prompt_logprobs: Optional[int] = None) -> list[Request]:
    sampling_params = SamplingParams(ignore_eos=False,
                                     max_tokens=max_tokens,
                                     stop_token_ids=stop_token_ids,
                                     prompt_logprobs=prompt_logprobs)
    requests = []
    for i in range(num_requests):
        request = Request(
            request_id=f"{i}",
            prompt=None,
            prompt_token_ids=[i] * num_tokens,
            sampling_params=sampling_params,
            multi_modal_inputs=None,
            multi_modal_placeholders=None,
            multi_modal_hashes=None,
            eos_token_id=EOS_TOKEN_ID,
            arrival_time=0,
        )
        requests.append(request)
    return requests


def test_add_requests():
    scheduler = create_sliding_window_scheduler()
    requests = create_requests(num_requests=10)

    for i, request in enumerate(requests):
        scheduler.add_request(request)
        assert request.request_id in scheduler.requests
        assert len(scheduler.waiting) == i + 1


def test_finish_request():
    scheduler = create_sliding_window_scheduler()
    requests = create_requests(num_requests=10)
    for request in requests:
        scheduler.add_request(request)

    for i, request in enumerate(requests):
        scheduler.finish_requests(request.request_id,
                                  RequestStatus.FINISHED_ABORTED)
        assert request.request_id not in scheduler.requests
        assert len(scheduler.waiting) == 9 - i


def test_get_num_unfinished_requests():
    scheduler = create_sliding_window_scheduler()
    requests = create_requests(num_requests=10)
    for request in requests:
        scheduler.add_request(request)

    for i, request in enumerate(requests):
        scheduler.finish_requests(request.request_id,
                                  RequestStatus.FINISHED_STOPPED)
        assert scheduler.get_num_unfinished_requests() == len(requests) - i - 1


@pytest.mark.parametrize("enable_prefix_caching, prompt_logprobs", [
    (None, None),
    (True, 5),
])
def test_schedule(enable_prefix_caching: Optional[bool],
                  prompt_logprobs: Optional[int]):
    '''Test scheduling. 
    Two cases: default APC/no prompt logprobs; APC=True + prompt logprobs
    '''
    scheduler = create_sliding_window_scheduler(
        enable_prefix_caching=enable_prefix_caching)
    requests = create_requests(num_requests=10,
                               prompt_logprobs=prompt_logprobs)
    for request in requests:
        scheduler.add_request(request)

    # Test initial scheduling
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == len(requests)
    assert len(output.scheduled_cached_reqs) == 0
    assert len(output.finished_req_ids) == 0
    # Verify all requests are scheduled.
    for req_id, num_tokens in output.num_scheduled_tokens.items():
        assert num_tokens == len(requests[int(req_id)].prompt_token_ids)

    # Verify requests moved from waiting to running
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == len(requests)
    for i, request in enumerate(requests):
        assert scheduler.running[i] == request


def test_schedule_partial_requests():
    """Test scheduling behavior with partial requests.

    This test verifies that:
    1. The scheduler can handle multiple partial requests in a single step when
       constrained by budget.
    2. A request in RUNNING state may be unscheduled in subsequent steps if
       there is insufficient budget.
    """
    scheduler = create_sliding_window_scheduler(max_num_batched_tokens=1024, )
    requests = create_requests(
        num_requests=3,
        num_tokens=800,
    )
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 2
    assert len(output.scheduled_cached_reqs) == 0
    assert len(output.finished_req_ids) == 0

    # The first request is scheduled fully.
    assert output.num_scheduled_tokens[requests[0].request_id] == 800
    # The second request is scheduled partially.
    assert output.num_scheduled_tokens[requests[1].request_id] == 224

    scheduled_requests = requests[:2]
    req_to_index = {
        request.request_id: i
        for i, request in enumerate(scheduled_requests)
    }
    model_runner_output = ModelRunnerOutput(
        req_ids=[request.request_id for request in scheduled_requests],
        req_id_to_index=req_to_index,
        # Only the first request has a sampled token id because
        # the rest requests are still being prefilled.
        sampled_token_ids=[[0], []],
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
    )
    scheduler.update_from_output(output, model_runner_output)

    # Schedule the next step.
    output = scheduler.schedule()
    assert len(scheduler.running) == 3
    assert len(output.scheduled_new_reqs) == 1
    assert len(output.scheduled_cached_reqs) == 2
    assert len(output.finished_req_ids) == 0
    assert output.num_scheduled_tokens[requests[0].request_id] == 1
    assert output.num_scheduled_tokens[requests[1].request_id] == 576
    assert output.num_scheduled_tokens[requests[2].request_id] == 447

    scheduled_requests = requests
    req_to_index = {
        request.request_id: i
        for i, request in enumerate(scheduled_requests)
    }
    model_runner_output = ModelRunnerOutput(
        req_ids=[request.request_id for request in scheduled_requests],
        req_id_to_index=req_to_index,
        # Only the first request has a sampled token id because
        # the rest requests are still being prefilled.
        sampled_token_ids=[[0], [0], []],
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
    )
    scheduler.update_from_output(output, model_runner_output)

    # Schedule the next step.
    output = scheduler.schedule()
    assert len(scheduler.running) == 3
    assert len(output.scheduled_new_reqs) == 0
    assert len(output.scheduled_cached_reqs) == 3
    assert len(output.finished_req_ids) == 0
    assert output.num_scheduled_tokens[requests[0].request_id] == 1
    assert output.num_scheduled_tokens[requests[1].request_id] == 1
    assert output.num_scheduled_tokens[requests[2].request_id] == 353

    scheduled_requests = requests
    req_to_index = {
        request.request_id: i
        for i, request in enumerate(scheduled_requests)
    }
    model_runner_output = ModelRunnerOutput(
        req_ids=[request.request_id for request in scheduled_requests],
        req_id_to_index=req_to_index,
        # Only the first request has a sampled token id because
        # the rest requests are still being prefilled.
        sampled_token_ids=[[0], [0], [0]],
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
    )
    scheduler.update_from_output(output, model_runner_output)

    # Schedule the next step.
    output = scheduler.schedule()
    assert len(scheduler.running) == 3
    assert len(output.scheduled_new_reqs) == 0
    assert len(output.scheduled_cached_reqs) == 3
    assert len(output.finished_req_ids) == 0
    assert output.num_scheduled_tokens[requests[0].request_id] == 1
    assert output.num_scheduled_tokens[requests[1].request_id] == 1
    assert output.num_scheduled_tokens[requests[2].request_id] == 1


@pytest.mark.parametrize("enable_prefix_caching", [True, False])
def test_schedule_concurrent_partial_requests(enable_prefix_caching: bool):
    """Test scheduling behavior with concurrent partial requests.

    This test verifies that: there are multiple long prefill requests in the
    RUNNING state, and we can schedule them together.

    """
    scheduler = create_sliding_window_scheduler(
        max_num_batched_tokens=1024,
        long_prefill_token_threshold=400,
        enable_prefix_caching=enable_prefix_caching,
    )
    requests = create_requests(
        num_requests=3,
        num_tokens=800,
    )
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 3
    assert len(output.scheduled_cached_reqs) == 0
    assert len(output.finished_req_ids) == 0

    # The first request is scheduled partially - 400.
    assert output.num_scheduled_tokens[requests[0].request_id] == 400
    # The second request is scheduled partially - 400.
    assert output.num_scheduled_tokens[requests[1].request_id] == 400
    # The third request is also scheduled partially - 1024 - 400 - 400 = 224.
    assert output.num_scheduled_tokens[requests[2].request_id] == 224
    req_to_index = {
        request.request_id: i
        for i, request in enumerate(requests)
    }
    model_runner_output = ModelRunnerOutput(
        req_ids=[request.request_id for request in requests],
        req_id_to_index=req_to_index,
        sampled_token_ids=[[] for _ in range(len(requests))],
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
    )
    scheduler.update_from_output(output, model_runner_output)

    # Schedule the next step. All three requests are running.
    # Processed the remaining prefills of the first and second requests.
    output1 = scheduler.schedule()
    assert len(scheduler.running) == 3
    assert len(output1.scheduled_new_reqs) == 0
    assert len(output1.scheduled_cached_reqs) == 3
    assert len(output1.finished_req_ids) == 0
    assert output1.num_scheduled_tokens[requests[0].request_id] == 400
    assert output1.num_scheduled_tokens[requests[1].request_id] == 400
    assert output1.num_scheduled_tokens[requests[2].request_id] == 224

    # Schedule the third step. All three requests are running.
    # First and second requests are in the decode stage.
    # All the remaining tokens in the third request are processed.
    model_runner_output = ModelRunnerOutput(
        req_ids=[request.request_id for request in requests],
        req_id_to_index=req_to_index,
        sampled_token_ids=[[0], [0]] + [[] for _ in range(len(requests) - 2)],
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
    )
    scheduler.update_from_output(output1, model_runner_output)
    output2 = scheduler.schedule()
    assert len(scheduler.running) == 3
    assert len(output2.scheduled_new_reqs) == 0
    assert len(output2.scheduled_cached_reqs) == 3
    assert len(output2.finished_req_ids) == 0
    assert output2.num_scheduled_tokens[requests[0].request_id] == 1
    assert output2.num_scheduled_tokens[requests[1].request_id] == 1
    assert output2.num_scheduled_tokens[
        requests[2].request_id] == 800 - 224 - 224


def test_stop_via_update_from_output():
    """Test stopping behavior through update_from_output"""
    scheduler = create_sliding_window_scheduler()

    # Test case 1: Stop on EOS token
    requests = create_requests(num_requests=2, max_tokens=10)
    current_scheduled_req_ids: set[str] = set()
    for req in requests:
        req.num_computed_tokens = req.num_tokens
        scheduler.requests[req.request_id] = req
        scheduler.running.append(req)
        current_scheduled_req_ids.add(req.request_id)
    scheduler.scheduled_req_ids.append(current_scheduled_req_ids)

    scheduler_output = SchedulerOutput(scheduled_new_reqs=[],
                                       scheduled_cached_reqs=[],
                                       num_scheduled_tokens={
                                           requests[0].request_id: 1,
                                           requests[1].request_id: 1
                                       },
                                       total_num_scheduled_tokens=2,
                                       scheduled_encoder_inputs={},
                                       scheduled_spec_decode_tokens={},
                                       num_common_prefix_blocks=0,
                                       finished_req_ids=set(),
                                       free_encoder_input_ids=[],
                                       structured_output_request_ids={},
                                       grammar_bitmask=None)

    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={
            req.request_id: i
            for i, req in enumerate(requests)
        },
        sampled_token_ids=[[EOS_TOKEN_ID],
                           [10]],  # First request hits EOS, second continues
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={})

    scheduler.update_from_output(scheduler_output, model_output)

    # Verify first request stopped, second continues
    assert len(scheduler.running) == 1
    assert scheduler.running[0].request_id == requests[1].request_id
    assert requests[0].status == RequestStatus.FINISHED_STOPPED
    assert requests[0].request_id in scheduler.finished_req_ids
    assert list(requests[0].output_token_ids) == [EOS_TOKEN_ID]
    assert list(requests[1].output_token_ids) == [10]

    # Test case 2: Stop on custom stop token
    scheduler = create_sliding_window_scheduler()
    requests = create_requests(num_requests=2,
                               max_tokens=10,
                               stop_token_ids=[42, 43])
    current_scheduled_req_ids: set[str] = set()
    for req in requests:
        req.num_computed_tokens = req.num_tokens
        scheduler.requests[req.request_id] = req
        scheduler.running.append(req)
        current_scheduled_req_ids.add(req.request_id)
    scheduler.scheduled_req_ids.append(current_scheduled_req_ids)

    scheduler_output = SchedulerOutput(scheduled_new_reqs=[],
                                       scheduled_cached_reqs=[],
                                       num_scheduled_tokens={
                                           requests[0].request_id: 1,
                                           requests[1].request_id: 1
                                       },
                                       total_num_scheduled_tokens=2,
                                       scheduled_encoder_inputs={},
                                       scheduled_spec_decode_tokens={},
                                       num_common_prefix_blocks=0,
                                       finished_req_ids=set(),
                                       free_encoder_input_ids=[],
                                       structured_output_request_ids={},
                                       grammar_bitmask=None)

    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={
            req.request_id: i
            for i, req in enumerate(requests)
        },
        sampled_token_ids=[[42], [13]],  # First request hits stop token
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={})

    scheduler.update_from_output(scheduler_output, model_output)

    # Verify first request stopped on custom token
    assert len(scheduler.running) == 1
    assert scheduler.running[0].request_id == requests[1].request_id
    assert requests[0].status == RequestStatus.FINISHED_STOPPED
    assert requests[0].stop_reason == 42
    assert requests[0].request_id in scheduler.finished_req_ids
    assert list(requests[0].output_token_ids) == [42]
    assert list(requests[1].output_token_ids) == [13]

    # Test case 3: Stop on max tokens
    scheduler = create_sliding_window_scheduler()
    requests = create_requests(num_requests=2, max_tokens=2)
    current_scheduled_req_ids: set[str] = set()
    for req in requests:
        req.num_computed_tokens = req.num_tokens
        scheduler.requests[req.request_id] = req
        scheduler.running.append(req)
        current_scheduled_req_ids.add(req.request_id)
    scheduler.scheduled_req_ids.append(current_scheduled_req_ids)

    scheduler_output = SchedulerOutput(scheduled_new_reqs=[],
                                       scheduled_cached_reqs=[],
                                       num_scheduled_tokens={
                                           requests[0].request_id: 2,
                                           requests[1].request_id: 1
                                       },
                                       total_num_scheduled_tokens=3,
                                       scheduled_encoder_inputs={},
                                       scheduled_spec_decode_tokens={},
                                       num_common_prefix_blocks=0,
                                       finished_req_ids=set(),
                                       free_encoder_input_ids=[],
                                       structured_output_request_ids={},
                                       grammar_bitmask=None)

    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={
            req.request_id: i
            for i, req in enumerate(requests)
        },
        sampled_token_ids=[[10, 11], [13]],  # First request exceeds max_tokens
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={})

    scheduler.update_from_output(scheduler_output, model_output)

    # Verify first request stopped due to length
    assert len(scheduler.running) == 1
    assert scheduler.running[0].request_id == requests[1].request_id
    assert requests[0].status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert requests[0].request_id in scheduler.finished_req_ids
    assert list(requests[0].output_token_ids) == [10, 11
                                                  ]  # Truncated to max_tokens
    assert list(requests[1].output_token_ids) == [13]

    # Test case 4: Ignore EOS flag
    scheduler = create_sliding_window_scheduler()
    requests = create_requests(num_requests=1, max_tokens=10)
    requests[0].sampling_params.ignore_eos = True
    requests[0].num_computed_tokens = requests[0].num_tokens
    scheduler.requests[requests[0].request_id] = requests[0]
    scheduler.running.append(requests[0])
    current_scheduled_req_ids: set[str] = set()
    current_scheduled_req_ids.add(requests[0].request_id)
    scheduler.scheduled_req_ids.append(current_scheduled_req_ids)

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=[],
        num_scheduled_tokens={requests[0].request_id: 3},
        total_num_scheduled_tokens=3,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={},
        num_common_prefix_blocks=0,
        finished_req_ids=set(),
        free_encoder_input_ids=[],
        structured_output_request_ids={},
        grammar_bitmask=None)

    model_output = ModelRunnerOutput(
        req_ids=[requests[0].request_id],
        req_id_to_index={requests[0].request_id: 0},
        sampled_token_ids=[[EOS_TOKEN_ID, 10, 11]],
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={})

    scheduler.update_from_output(scheduler_output, model_output)

    # Verify request continues past EOS
    assert len(scheduler.running) == 1
    assert not requests[0].is_finished()
    assert list(requests[0].output_token_ids) == [EOS_TOKEN_ID, 10, 11]


@pytest.mark.parametrize("enable_prefix_caching, prompt_logprobs", [
    (None, None),
    (True, 5),
])
def test_schedule_concurrent_batches(enable_prefix_caching: Optional[bool],
                                     prompt_logprobs: Optional[int]):
    scheduler = create_sliding_window_scheduler(
        max_num_batched_tokens=1024,
        max_num_seqs=2,
        enable_prefix_caching=enable_prefix_caching,
        sliding_window_size=2,
    )
    requests = create_requests(
        num_requests=2,
        num_tokens=512,
        prompt_logprobs=prompt_logprobs,
    )

    # Schedule the step 0.
    scheduler.add_request(requests[0])
    scheduler_output0 = scheduler.schedule()
    assert len(scheduler_output0.scheduled_new_reqs) == 1
    assert scheduler_output0.num_scheduled_tokens[
        requests[0].request_id] == 512

    # Schedule the step 1.
    scheduler.add_request(requests[1])
    scheduler_output1 = scheduler.schedule()
    assert len(scheduler_output1.scheduled_new_reqs) == 1
    assert len(scheduler_output1.scheduled_cached_reqs) == 1
    assert scheduler_output1.num_scheduled_tokens[requests[0].request_id] == 1
    assert scheduler_output1.num_scheduled_tokens[
        requests[1].request_id] == 512

    # Model output of the step 0.
    model_runner_output = ModelRunnerOutput(
        req_ids=[requests[0].request_id],
        req_id_to_index={requests[0].request_id: 0},
        sampled_token_ids=[[0]],
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
    )
    scheduler.update_from_output(scheduler_output0, model_runner_output)

    # Schedule the step 2.
    scheduler_output2 = scheduler.schedule()
    assert len(scheduler_output2.scheduled_new_reqs) == 0
    assert len(scheduler_output2.scheduled_cached_reqs) == 2
    assert scheduler_output2.num_scheduled_tokens[requests[0].request_id] == 1
    assert scheduler_output2.num_scheduled_tokens[requests[1].request_id] == 1

    # Model output of the step 1.
    model_runner_output = ModelRunnerOutput(
        req_ids=[requests[0].request_id, requests[1].request_id],
        req_id_to_index={
            requests[0].request_id: 0,
            requests[1].request_id: 1
        },
        sampled_token_ids=[[0], [0]],
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
    )
    scheduler.update_from_output(scheduler_output1, model_runner_output)

    # Schedule the step 3.
    scheduler_output3 = scheduler.schedule()
    assert len(scheduler_output3.scheduled_cached_reqs) == 2
    assert scheduler_output3.num_scheduled_tokens[requests[0].request_id] == 1
    assert scheduler_output3.num_scheduled_tokens[requests[1].request_id] == 1

    # Model output of the step 2.
    model_runner_output = ModelRunnerOutput(
        req_ids=[requests[0].request_id, requests[1].request_id],
        req_id_to_index={
            requests[0].request_id: 0,
            requests[1].request_id: 1
        },
        sampled_token_ids=[[0], [0]],
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
    )
    scheduler.update_from_output(scheduler_output2, model_runner_output)
