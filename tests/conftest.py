"""Shared pytest configuration.

* Exposes the startup-harness fixtures (e.g. runtime_container).
* api.main refuses to import unless the process runs as UID 10001 with a valid
  auth configuration (F-UID-1). In-process tests import it once here with the
  runtime identity simulated for that single import, and AUTH_ENABLED cleared
  so a developer's shell cannot change the result. Nothing in production code
  is relaxed; real-identity behaviour is covered by subprocess tests in
  tests/test_serving_identity.py, run as real UIDs 0, 1000, 65534 and 10001.
"""

import os
from unittest import mock

from api.runtime_identity import RUNTIME_UID

pytest_plugins = ["tests.startup_harness"]

with mock.patch.object(os, "geteuid", return_value=RUNTIME_UID), \
     mock.patch.dict(os.environ, {"AUTH_ENABLED": ""}):
    import api.main  # noqa: F401 -- deliberate import under the simulated runtime identity
