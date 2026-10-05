sphinx-apidoc -o ./docs/src/api ./src/chemfit -f --remove-old --separate
sphinx-build -b doctest ./docs ./docs/build
sphinx-autobuild -M html ./docs ./docs/build
