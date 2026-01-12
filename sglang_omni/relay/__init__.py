# SPDX-License-Identifier: Apache-2.0
"""Relay module for inter-stage data transfer.

This module provides various relay implementations for transferring data
between pipeline stages:
- SHMRelay: Shared memory relay for local (same-machine) transfers
- NIXLRelay: NIXL-based RDMA relay for distributed transfers
"""

from sglang_omni.relay.operations.base import BaseReadableOperation, BaseReadOperation
from sglang_omni.relay.operations.shm import SHMReadableOperation, SHMReadOperation
from sglang_omni.relay.relays.base import Relay
from sglang_omni.relay.relays.shm import SHMRelay

__all__ = [
    "BaseReadOperation",
    "BaseReadableOperation",
    "Relay",
    "SHMRelay",
    "SHMReadableOperation",
    "SHMReadOperation",
]
