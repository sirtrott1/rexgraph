"""
agent.hfguard: whether a model repository's own code may execute in this process.

``trust_remote_code=True`` makes transformers download and RUN Python that ships with
the model repository. Where the repository is named in operator configuration that is a
choice someone made deliberately. Where it is named in a request body it is remote code
execution as the server user, and /api/v1/huggingface/analyze took the name from a
request body and passed it straight to ``from_pretrained``.

The flag appeared at six call sites across three modules with no policy behind it, which
is how one of them stayed reachable from a request. The policy lives here instead.

Off for anything a caller named. An operator who needs an architecture that ships its
own modeling code turns it on for the deployment:

    REXGRAPH_TRUST_REMOTE_CODE=1
"""

from __future__ import annotations

import os

TRUST_ENV = "REXGRAPH_TRUST_REMOTE_CODE"


def remote_code_allowed(*, caller_named: bool = False) -> bool:
    """Whether a model repository's own code may run.

    `caller_named` is the whole question. Operator configuration keeps the behavior it
    has always had; a repository named in a request does not run its own code unless the
    deployment has said so.
    """
    if not caller_named:
        return True
    return os.environ.get(TRUST_ENV, "").strip().lower() in {"1", "true", "yes", "on"}
