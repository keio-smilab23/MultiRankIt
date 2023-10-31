#!/bin/bash -eu

if [ -z ${CUDA_VISIBLE_DEVICES:-""} ]; then
    CUDA_VISIBLE_DEVICES='all'
fi

if [ -z ${docker_image_name:-""} ]; then
    docker_image_name=tir2:focal-11.1
fi

if [ -z ${docker_container_name:-""} ]; then
    docker_container_name=tir2-server
fi

if [ -z ${docker_network_name:-""} ]; then
    docker_network_name=ml-network
fi

if [ -z ${server_listen_port:-""} ]; then
    server_listen_port=58001
fi
