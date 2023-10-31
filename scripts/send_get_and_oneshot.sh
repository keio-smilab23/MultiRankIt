#!/bin/bash -eu

root_dir=$(readlink -f $(dirname $0)/..)
source $root_dir/config/set_constants.sh

#### 引数の説明 ####
# 第1引数: 環境情報共有サーバのIPアドレス
# 第2引数: 環境情報共有サーバから得られた画像リストのうち、何番目のデータを利用するか
# 第3引数: 送信する指示分を含むJSONファイルのパス
env_info_server=$1
number_of_env_info=${2:-0}
instruction_path=${3:-${root_dir}/scripts/sample_data/186_instruction.json}

# パスの生成
env_info_server_url=http://$env_info_server:58100
tmp_root=$(mktemp -d /tmp/tir-oneshot-XXXXXX)
env_list_path=$tmp_root/env_info_list.json
send_json_path=$tmp_root/env_info_list.json

# 環境情報リストを取得
curl -qsSL $env_info_server_url/get_env_info_list > $env_list_path

# 画像のキーを取得
key1=$(cat $env_list_path | jq --sort-keys -r -c "keys[$(($number_of_env_info*3+0))]")
key2=$(cat $env_list_path | jq --sort-keys -r -c "keys[$(($number_of_env_info*3+1))]")
key3=$(cat $env_list_path | jq --sort-keys -r -c "keys[$(($number_of_env_info*3+2))]")

# 画像のURLを取得
url1=$(cat $env_list_path | jq -r -c ".\"$key1\".files[] | select(.name | test(\".(jpg|png)$\", \"sx\")) | .url_path")
url2=$(cat $env_list_path | jq -r -c ".\"$key2\".files[] | select(.name | test(\".(jpg|png)$\", \"sx\")) | .url_path")
url3=$(cat $env_list_path | jq -r -c ".\"$key3\".files[] | select(.name | test(\".(jpg|png)$\", \"sx\")) | .url_path")

# 画像のURLを送信するjsonに追加する
cat $instruction_path \
    | jq '. |= .+ {"img_url": {}}' \
    | jq ".img_url |= .+ {\"left_img\": \"${env_info_server_url}$url1\"}" \
    | jq ".img_url |= .+ {\"center_img\": \"${env_info_server_url}$url2\"}" \
    | jq ".img_url |= .+ {\"right_img\": \"${env_info_server_url}$url3\"}" \
    | jq --compact-output > $send_json_path

# 一発打ちを実行するコマンドの生成
cmd="curl -X POST \
    -F input_json=@$send_json_path \
    http://localhost:$server_listen_port/get_and_oneshot"


echo "======== sent file(below) ==========="
cat $send_json_path
echo "======== curl command(below) ========"
echo $cmd
echo "======== oneshot result(below) ======"
exec time -p $cmd

rm -rf $tmp_root
