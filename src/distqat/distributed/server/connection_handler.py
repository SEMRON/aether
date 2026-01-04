import multiprocessing as mp
import os
import pickle
from typing import Dict

import grpc
import torch

from hivemind.compression import deserialize_torch_tensor, serialize_torch_tensor
from hivemind.proto import runtime_pb2
from hivemind.utils import get_logger, nested_flatten
from hivemind.utils.asyncio import switch_to_uvloop

from distqat.distributed.proto import swarm_runtime_pb2_grpc as swarm_runtime_grpc
from distqat.distributed.server.expert_backend import ExpertBackend
from distqat.distributed.utils.networking import Endpoint
from distqat.distributed.utils.grpc import GRPC_KEEPALIVE_OPTIONS

logger = get_logger(__name__)


class ConnectionHandler(mp.context.ForkProcess):
    """
    A process that accepts incoming requests to experts and submits them into the corresponding TaskPool.

    :note: ConnectionHandler is designed so as to allow using multiple handler processes for the same port.
    :param listen_on: network interface, e.g. "0.0.0.0:1337" or "localhost:*" (* means pick any port) or "[::]:7654"
    :param experts: a dict [UID -> ExpertBackend] with all active experts
    """

    def __init__(self, listen_on: Endpoint, experts: Dict[str, ExpertBackend]):
        super().__init__()
        self.listen_on, self.experts = listen_on, experts
        self.ready = mp.Event()

    def run(self):
        """
        Start the gRPC server to handle incoming expert requests.
        
        Sets up an asyncio event loop with gRPC server that can handle multiple
        concurrent requests for expert forward/backward passes and info queries.
        """
        torch.set_num_threads(1)
        loop = switch_to_uvloop()

        async def _run():
            grpc.aio.init_grpc_aio()
            logger.debug(f"Starting, pid {os.getpid()}")
            server = grpc.aio.server(
                options=GRPC_KEEPALIVE_OPTIONS
                + (
                    ("grpc.so_reuseport", 1),
                    ("grpc.max_send_message_length", -1),
                    ("grpc.max_receive_message_length", -1),
                )
            )
            swarm_runtime_grpc.add_SwarmConnectionHandlerServicer_to_server(self, server)

            found_port = server.add_insecure_port(self.listen_on)
            assert found_port != 0, f"Failed to listen to {self.listen_on}"

            await server.start()
            self.ready.set()
            await server.wait_for_termination()
            logger.debug(f"ConnectionHandler terminated: (pid={os.getpid()})")

        try:
            loop.run_until_complete(_run())
        except KeyboardInterrupt:
            logger.debug("Caught KeyboardInterrupt, shutting down")

    async def info(self, request: runtime_pb2.ExpertUID, context: grpc.ServicerContext):
        """
        Handle gRPC requests for expert information.
        
        :param request: ExpertUID message containing the expert identifier
        :param context: gRPC service context
        :returns: ExpertInfo containing serialized expert metadata
        """
        return runtime_pb2.ExpertInfo(serialized_info=pickle.dumps(self.experts[request.uid].get_info()))

    async def forward(self, request: runtime_pb2.ExpertRequest, context: grpc.ServicerContext):
        """
        Handle gRPC requests for expert forward passes.
        
        Deserializes input tensors, submits them to the expert's forward pool,
        and returns serialized outputs.
        
        :param request: ExpertRequest containing expert UID and input tensors
        :param context: gRPC service context
        :returns: ExpertResponse containing serialized output tensors
        """
        inputs = [deserialize_torch_tensor(tensor) for tensor in request.tensors]
        future = self.experts[request.uid].forward_pool.submit_task(*inputs)
        # NOTE: hivemind serialization sends bf16 as fp32-sized payloads under CompressionType.NONE,
        # and Float16Compression does not support bf16 tensors. To avoid 2x network traffic under bf16-mixed,
        # we downcast bf16 -> fp16 for transport (compute still happens under autocast on the expert).
        serialized_response = [
            serialize_torch_tensor(
                (tensor.to(dtype=torch.float16) if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.bfloat16 else tensor),
                proto.compression,
                allow_inplace=True,
            )
            for tensor, proto in zip(await future, nested_flatten(self.experts[request.uid].outputs_schema))
        ]

        return runtime_pb2.ExpertResponse(tensors=serialized_response)

    async def backward(self, request: runtime_pb2.ExpertRequest, context: grpc.ServicerContext):
        """
        Handle gRPC requests for expert backward passes.
        
        Deserializes input tensors and gradient outputs, submits them to the expert's
        backward pool for gradient computation and parameter updates, and returns
        serialized input gradients.
        
        :param request: ExpertRequest containing expert UID, inputs and gradient outputs
        :param context: gRPC service context
        :returns: ExpertResponse containing serialized input gradients
        """
        inputs_and_grad_outputs = [deserialize_torch_tensor(tensor) for tensor in request.tensors]
        future = self.experts[request.uid].backward_pool.submit_task(*inputs_and_grad_outputs)
        serialized_response = [
            serialize_torch_tensor(
                (tensor.to(dtype=torch.float16) if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.bfloat16 else tensor),
                proto.compression,
                allow_inplace=True,
            )
            for tensor, proto in zip(await future, nested_flatten(self.experts[request.uid].grad_inputs_schema))
        ]
        return runtime_pb2.ExpertResponse(tensors=serialized_response)
