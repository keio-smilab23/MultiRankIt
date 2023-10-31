#!/bin/bash -eu

docker_container_name=${1:-""}
docker_network_name=${2:-""}

root_dir=$(readlink -f $(dirname $0)/..)
source $root_dir/config/set_constants.sh


# dockerネットワークを設定
if ! docker network inspect $docker_network_name > /dev/null 2>&1; then
    docker network create $docker_network_name > /dev/null
fi

# dockerコンテナを起動
if ! docker container inspect $docker_container_name > /dev/null 2>&1; then
    docker run -it -d \
        --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        --mount src=${root_dir},dst=/src,type=bind \
        --name $docker_container_name \
        --net $docker_network_name \
        -p $server_listen_port:8080 \
        --ipc=host --shm-size=64g \
        $docker_image_name
else
    docker start $docker_container_name > /dev/null
fi
