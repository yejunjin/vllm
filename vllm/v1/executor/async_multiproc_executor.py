# SPDX-License-Identifier: Apache-2.0
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional, Union

from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc
from vllm.v1.outputs import ModelRunnerOutput


class AsyncMultiprocExecutor(MultiprocExecutor):

    def _init_executor(self) -> None:
        super()._init_executor()
        self.executor = ThreadPoolExecutor(max_workers=1)

    def execute_model(
        self,
        scheduler_output,
    ) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:
        start_time = time.monotonic()
        future: Future[ModelRunnerOutput] = Future()
        try:
            self.rpc_broadcast_mq.enqueue(
                ("execute_model", (scheduler_output, ), {}))
            self.executor.submit(
                self._collect_response,
                start_time,
                future,
            )
        except TimeoutError as e:
            raise TimeoutError("RPC call to execute_model timed out.") from e
        except Exception as e:
            # Re-raise any other exceptions
            raise e

        return future

    def _collect_response(self,
                          start_time: float,
                          future: Future[ModelRunnerOutput],
                          timeout: Optional[float] = None) -> None:
        responses = [None] * self.world_size
        for w in self.workers:
            dequeue_timeout = timeout - (
                time.monotonic() - start_time) if timeout is not None else None
            status, result = w.worker_response_mq.dequeue(
                timeout=dequeue_timeout)

            if status != WorkerProc.ResponseStatus.SUCCESS:
                if isinstance(result, Exception):
                    future.set_exception(result)
                else:
                    future.set_exception(RuntimeError("Worker failed"))
                return

            responses[w.rank] = result

        future.set_result(responses[0])  # type: ignore

    def shutdown(self):
        super().shutdown()
        self.executor.shutdown()
