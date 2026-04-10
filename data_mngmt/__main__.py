"""Entry point for ``python -m data_mngmt`` (unified build CLI in ``pipeline.build_data``)."""
from data_mngmt.pipeline.build_data import main

if __name__ == "__main__":
    raise SystemExit(main())
