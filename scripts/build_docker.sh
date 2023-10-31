#!/bin/bash -eu

docker_image_name=${1:-""}

root_dir=$(readlink -f $(dirname $0)/..)
source $root_dir/config/set_constants.sh

cd $root_dir
docker build -t $docker_image_name .
