# SPDX-License-Identifier: Apache-2.0
"""Unified tests for relay implementations (NIXLRelay and SHMRelay)."""

import pytest
import torch

from sglang_omni.relay.descriptor import Descriptor


@pytest.fixture(params=["nixl", "shm"])
def relay_class(request):
    if request.param == "nixl":
        from sglang_omni.relay.relays.nixl import NIXLRelay

        return NIXLRelay
    else:
        from sglang_omni.relay.relays.shm import SHMRelay

        return SHMRelay


@pytest.fixture
def relay_configs(relay_class):
    if relay_class.__name__ == "NIXLRelay":
        if torch.cuda.is_available() and torch.cuda.device_count() < 2:
            pytest.skip("NIXLRelay requires at least 2 GPUs")
        return [
            {
                "host": "127.0.0.1",
                "metadata_server": "http://127.0.0.1:8080/metadata",
                "device_name": "",
                "gpu_id": 0,
                "worker_id": "worker0",
            },
            {
                "host": "127.0.0.1",
                "metadata_server": "http://127.0.0.1:8080/metadata",
                "device_name": "",
                "gpu_id": 1 if torch.cuda.is_available() else 0,
                "worker_id": "worker1",
            },
        ]
    return [{}, {}]


def _get_device(relay_class, config):
    if relay_class.__name__ == "NIXLRelay":
        return f'cuda:{config["gpu_id"]}' if torch.cuda.is_available() else "cpu"
    return "cpu"


def _get_received_data(relay_class, read_op, buffer=None):
    if relay_class.__name__ == "SHMRelay":
        data = read_op.data
        return (
            data.cpu() if isinstance(data, torch.Tensor) else torch.tensor(data).cpu()
        )
    return buffer.cpu() if buffer is not None else None


def _create_connectors(relay_class, configs):
    try:
        return relay_class(configs[0]), relay_class(configs[1])
    except (ImportError, RuntimeError) as e:
        pytest.skip(f"Failed to initialize {relay_class.__name__}: {e}")


class TestRelayUnified:
    def test_transfer(self, relay_class, relay_configs):
        connector0, connector1 = _create_connectors(relay_class, relay_configs)
        try:
            device0, device1 = _get_device(relay_class, relay_configs[0]), _get_device(
                relay_class, relay_configs[1]
            )
            test_tensor = torch.randn(100000, dtype=torch.float32, device=device0)
            original = test_tensor.cpu().clone()

            readable_op = connector0.put([Descriptor(test_tensor)])
            metadata = readable_op.metadata()

            buffer = None
            if relay_class.__name__ == "SHMRelay":
                read_op = connector1.get(metadata, [])
            else:
                desc_meta = (
                    metadata.descriptors[0]
                    if hasattr(metadata, "descriptors")
                    else metadata
                )
                buffer = torch.empty(
                    desc_meta.size // test_tensor.element_size(),
                    dtype=test_tensor.dtype,
                    device=device1,
                )
                read_op = connector1.get(metadata, [Descriptor(buffer)])

            if hasattr(read_op, "wait_for_completion"):
                coro = read_op.wait_for_completion()
                if coro:
                    if hasattr(connector1, "_run_maybe_async"):
                        connector1._run_maybe_async(coro)
                    else:
                        import asyncio

                        asyncio.run(coro)

            received = _get_received_data(relay_class, read_op, buffer)
            assert torch.allclose(original, received, rtol=1e-5, atol=1e-5)
            assert connector0._metrics["puts"] >= 1
            assert connector1._metrics["gets"] >= 1
        finally:
            connector0.close()
            connector1.close()

    @pytest.mark.asyncio
    async def test_transfer_async(self, relay_class, relay_configs):
        connector0, connector1 = _create_connectors(relay_class, relay_configs)
        try:
            device0, device1 = _get_device(relay_class, relay_configs[0]), _get_device(
                relay_class, relay_configs[1]
            )
            test_tensor = torch.randn(100000, dtype=torch.float32, device=device0)
            original = test_tensor.cpu().clone()

            readable_op = await connector0.put_async([Descriptor(test_tensor)])
            metadata = readable_op.metadata()

            if relay_class.__name__ == "SHMRelay":
                read_op = await connector1.get_async(metadata, [])
            else:
                desc_meta = (
                    metadata.descriptors[0]
                    if hasattr(metadata, "descriptors")
                    else metadata
                )
                buffer = torch.empty(
                    desc_meta.size // test_tensor.element_size(),
                    dtype=test_tensor.dtype,
                    device=device1,
                )
                read_op = await connector1.get_async(metadata, [Descriptor(buffer)])

            if hasattr(read_op, "wait_for_completion"):
                await read_op.wait_for_completion()

            buffer = buffer if relay_class.__name__ != "SHMRelay" else None
            received = _get_received_data(relay_class, read_op, buffer)
            assert torch.allclose(original, received, rtol=1e-5, atol=1e-5)
        finally:
            connector0.close()
            connector1.close()

    def test_health(self, relay_class, relay_configs):
        connector = _create_connectors(relay_class, relay_configs)[0]
        try:
            health = connector.health()
            assert health["status"] == "healthy"
        finally:
            connector.close()

    def test_cleanup(self, relay_class, relay_configs):
        connector = _create_connectors(relay_class, relay_configs)[0]
        try:
            connector.cleanup("test_request_id")
        finally:
            connector.close()
