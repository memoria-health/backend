"""GPU-host identity service package.

Currently holds the /state CONTRACT only; the service implementation
(src/services/identity + tools/run_identity.py) relocates here in one
move together with its reid-side dependencies during the legacy
deletion phase — moving it earlier would leave a gpu/ package that
still imports src/* and pays the import churn twice.
"""
