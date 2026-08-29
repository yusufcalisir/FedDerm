"""
Phase 1 test: verify package can be imported and environment is intact.
Run with: pytest tests/test_environment.py -v
"""

import importlib
import sys


REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("flwr", "flwr"),
    ("opacus", "opacus"),
    ("medmnist", "medmnist"),
    ("sklearn", "sklearn"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("tqdm", "tqdm"),
    ("PIL", "PIL"),
    ("pandas", "pandas"),
    ("hydra", "hydra"),
    ("omegaconf", "omegaconf"),
]


class TestEnvironment:
    def test_fedderm_package_importable(self) -> None:
        """The fedderm package itself must be importable."""
        import fedderm  # noqa: PLC0415

        assert hasattr(fedderm, "__version__")

    def test_fedderm_subpackages_importable(self) -> None:
        """All declared subpackages must be importable."""
        subpkgs = ["fedderm.data", "fedderm.models", "fedderm.federated",
                   "fedderm.privacy", "fedderm.utils"]
        for pkg in subpkgs:
            mod = importlib.import_module(pkg)
            assert mod is not None, f"{pkg} failed to import"


class TestDependencies:
    def test_required_packages_importable(self) -> None:
        """Each dependency in REQUIRED_PACKAGES must import without error."""
        failures = []
        for friendly_name, import_name in REQUIRED_PACKAGES:
            try:
                importlib.import_module(import_name)
            except ImportError as exc:
                failures.append(f"{friendly_name}: {exc}")
        assert not failures, "Missing packages:\n" + "\n".join(failures)

    def test_torch_version_meets_minimum(self) -> None:
        """Torch must be at least 2.x."""
        import torch  # noqa: PLC0415

        major = int(torch.__version__.split(".")[0])
        assert major >= 2, f"torch >= 2.0 required, got {torch.__version__}"

    def test_python_version_meets_minimum(self) -> None:
        """Python must be 3.10+."""
        assert sys.version_info >= (3, 10), (
            f"Python 3.10+ required, got {sys.version}"
        )

    def test_torch_device_available(self) -> None:
        """At minimum, CPU device must be available (sanity check)."""
        import torch  # noqa: PLC0415

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Create a tiny tensor on the device to confirm it works
        t = torch.zeros(2, 2, device=device)
        assert t.shape == (2, 2)
