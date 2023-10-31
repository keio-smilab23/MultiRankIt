#!/bin/bash -eu

docker_container_name=${1:-""}
docker_network_name=${2:-""}

root_dir=$(readlink -f $(dirname $0)/..)
source $root_dir/config/set_constants.sh

log_dir=${root_dir}/log
datetime=`date '+%Y%m%d_%H%M%S'`
log_prefix=${log_dir}/${datetime}_server
echo "logging to [${log_prefix}.log]"

# launch container
$root_dir/scripts/launch_container.sh $docker_container_name $docker_network_name

# docker exec
trap "pkill -f 'docker exec $docker_container_name'" SIGINT SIGTERM
set +e  # 以下のdocker execを強制終了させた場合にもdocker killが実行されてほしいため必要
docker exec $docker_container_name /bin/bash -c "export TZ=Asia/Tokyo && ./scripts/start_server.sh" |& tee ${log_prefix}.log

# kill docker container
docker kill $docker_container_name > /dev/null
