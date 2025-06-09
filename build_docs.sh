sphinx-apidoc -o ./docs/src/api ./src/scme_fitting -f --remove-old --separate
sphinx-autobuild -M html ./docs ./docs/build