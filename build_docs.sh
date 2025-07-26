sphinx-apidoc -o ./docs/src/api ./src/chemfit -f --remove-old --separate
sphinx-autobuild -M html ./docs ./docs/build