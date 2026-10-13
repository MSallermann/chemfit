#!/usr/bin/env bash
set -euo pipefail

sphinx-apidoc -o ./docs/src/api ./src/chemfit -f --remove-old --separate
sphinx-build -W -b doctest ./docs ./docs/build/doctest
sphinx-autobuild -M html ./docs ./docs/build
