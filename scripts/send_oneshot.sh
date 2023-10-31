#!/bin/bash -eu

root_dir=$(readlink -f $(dirname $0)/..)
source $root_dir/config/set_constants.sh

image_path=${1:-${root_dir}/scripts/sample_data/posi_center.jpg}
instruction_path=${4:-${root_dir}/scripts/sample_data/instruction.json}

cmd="curl -X POST \
    -F image=@$image_path \
    -F instruction=@$instruction_path \
    http://localhost:$server_listen_port/oneshot/all"

exec time -p $cmd
