"""Federated learning strategies and simulation utilities."""

from fedderm.federated.client import DermClient, get_parameters, set_parameters
from fedderm.federated.simulation import run_federated
from fedderm.federated.scaffold import (
    PersistentControlVariates,
    ScaffoldClient,
    ScaffoldStrategy,
    run_scaffold,
)
from fedderm.federated.dp_client import DPDermClient
from fedderm.federated.dp_simulation import run_federated_dp
from fedderm.federated.dp_lora_simulation import run_federated_dp_lora

__all__ = [
    "DermClient",
    "get_parameters",
    "set_parameters",
    "run_federated",
    "PersistentControlVariates",
    "ScaffoldClient",
    "ScaffoldStrategy",
    "run_scaffold",
    "DPDermClient",
    "run_federated_dp",
    "run_federated_dp_lora",
]
