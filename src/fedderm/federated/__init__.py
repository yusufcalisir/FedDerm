"""Federated learning strategies and simulation utilities."""

from fedderm.federated.client import DermClient, get_parameters, set_parameters
from fedderm.federated.simulation import run_federated

__all__ = ["DermClient", "get_parameters", "set_parameters", "run_federated"]

