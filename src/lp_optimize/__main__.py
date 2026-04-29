"""Allow ``python -m lp_optimize`` to invoke the CLI."""

from .cli import main

raise SystemExit(main())
