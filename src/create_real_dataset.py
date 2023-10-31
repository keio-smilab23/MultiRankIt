"""評価用の実機データセットを作成"""

import argparse
import glob
import json
import os
import sys
import pathlib

import clip
import numpy as np
from PIL import Image
from tqdm import tqdm

from stanford_parser import get_all_np


ANNOTATION_DIR = "annotations"
IMAGE_DIR = "images"
IMAGE_SUFFIX = "_headrgbdcamera_color.jpg"
ANNOTATION_FILE_SUFFIX = "*.json"
CLIP_BASE_MODEL = "ViT-L/14"
DEVICE = "cuda:0"
ENV_NAME_PATTERN = "real_{0}"
REAL_IMAGE_PER_WAYPOINT = 3
IMAGE_IDX_MAP = {
    "center": "id01",  # 正面方向の画像ID
    "left": "id00",  # 左方向の画像ID
    "right": "id02",  # 右方向の画像ID
}
SPLIT_NAME = "real_test"
GT_IMG_ID_PATTERN = "image_{0}_{1:02d}"
DATABASE_NAME = f"reverie_{SPLIT_NAME}_by_image_" + "{0}.json"
ALL_FEATURE_FILE_NAME = f"eval_features_{SPLIT_NAME}_" + "{0}.npy"
IMAGE_LIST_FILE_NAME = f"eval_features_full_image_list_{SPLIT_NAME}_" + "{0}.json"


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate_json_files", "-v", action="store_true")
    parser.add_argument("--create_numpy_files", "-n", action="store_true")
    parser.add_argument("--create_dataset_files", "-d", action="store_true")

    parser.add_argument("--real_raw_data", type=str, default="./data/tir_wrs_test_A_set",
                        help="the directory path to the real TIR dataset")
    parser.add_argument("--np_save_dir", type=str, default="./data/full_image_features",
                        help="the directory path to the numpy image data")
    parser.add_argument("--tir_dataset_dir", type=str, default="./data/reverie_retrieval_dataset",
                        help="the directory path to the reverie retrieval dataset")
    args = parser.parse_args()

    return args


def validate_json(args):
    """入力データセットに不備がないかを確認する"""
    annotation_dir_path = os.path.join(args.real_raw_data, ANNOTATION_DIR)
    if not os.path.isdir(annotation_dir_path):
        print(f"directory [{annotation_dir_path}] must exist.", file=sys.stderr)
        return False

    anno_files = glob.glob(os.path.join(annotation_dir_path, ANNOTATION_FILE_SUFFIX))
    for anno_file in tqdm(anno_files, desc="Validate input files"):
        with open(anno_file, "r", encoding="utf-8") as file:
            anno_js = json.load(file)
        for idx, anno in enumerate(anno_js["annotations"]):
            for gt_img in anno["gt_center_image_id"]:
                if int(gt_img) % 3 != 0:
                    err_str = "gt_center_image_id must be a multiple of 3.: "
                    err_str += f"'{gt_img}' in {idx}th annotation (in {anno_file})"
                    print(err_str, file=sys.stderr)
                    return False

                img_path = os.path.join(
                    args.real_raw_data,
                    IMAGE_DIR,
                    anno_js["environment"],
                    f"{gt_img}{IMAGE_SUFFIX}"
                )
                if not os.path.exists(img_path):
                    print(
                        "specified image by gt_center_image_id is not exist.: {img_path} (in {anno_file})",
                        file=sys.stderr
                    )
                    return False
    return True


def create_numpy(args):
    """画像を前処理して、numpy化したものを出力する"""
    clip_model, preprocess_clip = clip.load(CLIP_BASE_MODEL, device=DEVICE)

    def save_image_as_numpy(img_path, np_path):
        """CLIPによる前処理を行う"""
        clip_feature = clip_model.encode_image(
            preprocess_clip(Image.open(img_path)).unsqueeze(0).to(DEVICE)
        )
        output_data = clip_feature.to("cpu").detach().numpy().copy()
        np.save(np_path, output_data)
        return output_data

    def real_img_path_to_np_path(img_path, args):
        """実画像のファイル名からTIR向けのnumpyデータのファイルパスを求める"""
        raw_path = pathlib.Path(img_path)
        env = ENV_NAME_PATTERN.format(raw_path.parts[-2])
        img_full_idx = int(raw_path.parts[-1].split("_")[0])
        waypoint_idx = img_full_idx // REAL_IMAGE_PER_WAYPOINT
        img_id = f"id{img_full_idx % REAL_IMAGE_PER_WAYPOINT:02d}"
        gt_img_idx = GT_IMG_ID_PATTERN.format(env, int(img_full_idx) // REAL_IMAGE_PER_WAYPOINT)
        return os.path.join(args.np_save_dir, env, f"wp{waypoint_idx:02d}", f"{img_id}.npy"), env, gt_img_idx

    # タイムスタンプディレクトリ(TIRではenv相当)へのパスのみを取り出す
    img_dir_path = os.path.join(args.real_raw_data, IMAGE_DIR)
    env_path_list = glob.glob(os.path.join(img_dir_path, "*"))
    env_path_list = [env for env in env_path_list if os.path.isdir(env)]

    for env_path in tqdm(env_path_list, desc="Create numpy files  "):
        center_images = []
        left_images = []
        right_images = []
        image_list = []

        img_path_list = glob.glob(os.path.join(env_path, f"*{IMAGE_SUFFIX}"))
        for img_path in sorted(img_path_list):
            np_path, env_name, gt_img_idx = real_img_path_to_np_path(img_path, args)
            os.makedirs(os.path.dirname(np_path), exist_ok=True)
            np_data = save_image_as_numpy(img_path, np_path)

            if IMAGE_IDX_MAP["center"] in np_path:
                center_images.append(np_data)
                image_list.append(gt_img_idx)
            elif IMAGE_IDX_MAP["left"] in np_path:
                left_images.append(np_data)
            elif IMAGE_IDX_MAP["right"] in np_path:
                right_images.append(np_data)

        # 画像特徴量をevnごとにまとめて保存
        image_features_path = os.path.join(args.tir_dataset_dir, ALL_FEATURE_FILE_NAME.format(env_name))
        np.save(image_features_path, np.array(center_images))
        np.save(image_features_path.replace(".npy", "_left.npy"), np.array(left_images))
        np.save(image_features_path.replace(".npy", "_right.npy"), np.array(right_images))

        # image listファイルを保存
        image_list_path = os.path.join(args.tir_dataset_dir, IMAGE_LIST_FILE_NAME.format(env_name))
        with open(image_list_path, "w", encoding="utf-8") as file:
            image_list_data = {
                "data_path": image_features_path,
                "imgId_list": image_list
            }
            json.dump(image_list_data, file, indent=4)


def create_dataset(args):
    """データセットの定義ファイルを出力"""
    def get_tir_annotation(env, anno, inst):
        """TIR形式のアノテーション情報を返す"""
        env_name = ENV_NAME_PATTERN.format(env)
        anno = {
            "gt_img_id": [
                GT_IMG_ID_PATTERN.format(env_name, int(full_img_idx) // REAL_IMAGE_PER_WAYPOINT)
                for full_img_idx in anno["gt_center_image_id"]
            ],
            "instruction_id": 0,  # 利用しない値
            "instruction": inst,
            "noun_phrases": list(get_all_np(inst)),
            "environment": env_name,
            "gt_waypoint": "unused",  # 利用しない値,
            "gt_view": "id01",  # 固定値
            "target_objId": "unused",  # 利用しない値
            "target_object": "unused",  # 利用しない値
            "gt_bbox": [[0,0,0,0]],  # 利用しない値
        }
        return anno

    # JSON形式のアノテーション情報を作成
    annotation_dir_path = os.path.join(args.real_raw_data, ANNOTATION_DIR)
    anno_files = glob.glob(os.path.join(annotation_dir_path, ANNOTATION_FILE_SUFFIX))

    for anno_file in tqdm(sorted(anno_files), desc="Create real dataset "):
        with open(anno_file, "r", encoding="utf-8") as file:
            anno_js = json.load(file)

        tir_annos = []
        for anno in anno_js["annotations"]:
            for inst in anno["instructions"]:
                tir_annos.append(get_tir_annotation(anno_js["environment"], anno, inst))

        database_name = ENV_NAME_PATTERN.format(anno_js["environment"])
        database_path = os.path.join(args.tir_dataset_dir, DATABASE_NAME.format(database_name))
        with open(database_path, "w", encoding="utf-8") as file:
            json.dump(tir_annos, file, indent=4)


def create_real_dataset(args):
    """評価用の実機データセットを作成"""
    # データの検証を実施
    if args.validate_json_files:
        if not validate_json(args):
            return False

    # numpy形式のデータを出力
    if args.create_numpy_files:
        create_numpy(args)

    # dataset定義ファイルを出力
    if args.create_dataset_files:
        create_dataset(args)


if __name__ == "__main__":
    args = parse_args()
    create_real_dataset(args)
