#!/bin/bash -eu

root_dir=$(readlink -f $(dirname $0)/..)
poetry run python $root_dir/src/main.py start_server
