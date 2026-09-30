"""Shared test settings.

On a continuous-integration runner (the CI environment variable, which GitHub Actions sets), Hypothesis
runs under a profile without its input-speed health check and without per-example deadlines. Shared
runners are slow and uneven, and a fresh checkout has no example database, so the first inputs can take
over a second to draw. The properties tested are the same; only the timing guards are relaxed.
"""

import os

from hypothesis import HealthCheck, settings

settings.register_profile("ci", suppress_health_check=[HealthCheck.too_slow], deadline=None)
if os.environ.get("CI"):
    settings.load_profile("ci")
