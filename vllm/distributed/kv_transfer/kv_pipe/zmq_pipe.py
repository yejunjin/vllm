from typing import Optional

import torch
import zmq

from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_pipe.base import KVPipeBase
from vllm.logger import init_logger

logger = init_logger(__name__)

class ZMQPipe(KVPipeBase):
        def __init__(self,
                 local_rank: int,
                 config: KVTransferConfig,
                 device: Optional[str] = None,
                 port_offset: int = 0):
            self.local_rank = local_rank
            self.config = config
            self.kv_rank = self.config.kv_rank
            self.kv_parallel_size = self.config.kv_parallel_size
            self.port_offset = port_offset
            assert self.kv_parallel_size == 2
            self.context = zmq.Context()
            self.p2c_ipc_endpoint = f"ipc:///tmp/zmq_pipe_l{self.local_rank}p{self.port_offset}_p2c"
            self.c2p_ipc_endpoint = f"ipc:///tmp/zmq_pipe_l{self.local_rank}p{self.port_offset}_c2p"
            if self.config.is_kv_producer:
                self.send_socket = self.context.socket(zmq.constants.PUSH)
                self.send_socket.connect(self.p2c_ipc_endpoint)
                self.recv_socket = self.context.socket(zmq.constants.PULL)
                self.recv_socket.bind(self.c2p_ipc_endpoint)
                logger.info(f"KV Producer ZMQPipe connect to {self.p2c_ipc_endpoint}, bind to {self.c2p_ipc_endpoint}")
            else:
                self.send_socket = self.context.socket(zmq.constants.PUSH)
                self.send_socket.connect(self.c2p_ipc_endpoint)
                self.recv_socket = self.context.socket(zmq.constants.PULL)
                self.recv_socket.bind(self.p2c_ipc_endpoint)
                logger.info(f"kV Consumer ZMQPipe connect to {self.c2p_ipc_endpoint}, bind to {self.p2c_ipc_endpoint}")

            


        def send_tensor(self, tensor: Optional[torch.Tensor]) -> None:
            self.send_socket.send_pyobj(tensor, zmq.NOBLOCK)


        def recv_tensor(self) -> Optional[torch.Tensor]:
            tensor = self.recv_socket.recv_pyobj()
            return tensor
            

        def close(self) -> None:
            self.send_socket.close()
            self.recv_socket.close()
