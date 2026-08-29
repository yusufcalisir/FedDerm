"""Differential privacy wrappers and accountants using Opacus."""

from fedderm.privacy.accountant import FederatedPrivacyAccountant
from fedderm.privacy.engine import (
    check_opacus_compatibility,
    make_private_client,
    train_one_epoch_dp,
)

__all__ = [
    "FederatedPrivacyAccountant",
    "check_opacus_compatibility",
    "make_private_client",
    "train_one_epoch_dp",
]
