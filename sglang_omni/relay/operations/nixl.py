# SPDX-License-Identifier: Apache-2.0
"""NIXL operation classes - re-exported from nixl.connector module.

NIXL operations are tightly coupled with the Connection and Connector classes,
so they remain in the nixl.connector module and are re-exported here for convenience.
"""

# Re-export NIXL operation classes
# These classes are defined in sglang_omni.relay.nixl.connector
# and are re-exported here for API consistency

from sglang_omni.relay.nixl import (
    RdmaMetadata,
    ReadableOperation,
    ReadOperation,
    WritableOperation,
    WriteOperation,
)

__all__ = [
    "ReadOperation",
    "ReadableOperation",
    "WriteOperation",
    "WritableOperation",
    "RdmaMetadata",
]
