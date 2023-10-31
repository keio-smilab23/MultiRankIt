"""REVERIEデータセットのデータローダモジュール"""
import json

import clip
import numpy as np
import torch
from torch.utils.data import Dataset


class reverie_dataset(Dataset):
    """REVERIEデータセットを管理するクラス"""

    def __init__(self, model, args, split="train", env="train", num_bbox=16, eval=False, N=30, baseline_dataset=""):
        self.model = model
        self.args = args
        self.bbox = True
        self.num_bbox = num_bbox
        self.eval = eval
        self.N = N

        if baseline_dataset != "":
            self.baseline = f"_baseline_{baseline_dataset}"
        else:
            self.baseline = ""

        dataset_path = f"data/reverie_retrieval_dataset/reverie_{split}_by_bbox_{env}.json"
        print(dataset_path)

        self.clip_model, self.preprocess_clip = clip.load("ViT-L/14", device="cuda:0")

        if self.eval:
            self.get_dataset(dataset_path)
            eval_features_path_base = f"data/reverie_retrieval_dataset/eval_features_{split}_{env}"

            eval_id_path = f"data/reverie_retrieval_dataset/eval_features_bbox_list_{split}_{env}.json"

            self.all_image_features = self.open_npy(f"{eval_features_path_base}.npy")
            self.all_left_image_features = self.open_npy(f"{eval_features_path_base}_left.npy")
            self.all_right_image_features = self.open_npy(f"{eval_features_path_base}_right.npy")

            self.all_bbox_image_features = self.open_npy(f"{eval_features_path_base}_bbox.npy")
            self.imageId_list = json.load(open(eval_id_path, "r"))["bboxId_list"]

        else:
            self.get_dataset(dataset_path)

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        raw_instruction = self.db[idx]["instruction"]

        tokenized_instruction_clip = clip.tokenize(raw_instruction).squeeze(0)
        tokenized_np_clip = self.get_noun_phrases_clip(self.db[idx]["noun_phrases"], n=self.N)

        if not self.eval:
            bbox_image_feature = self.open_npy(self.db[idx]["bbox_image_feature_path"][0])
            entire_image_feature = self.open_npy(self.db[idx]["full_image_feature_path"][0])
            left_image_feature = self.open_npy(self.db[idx]["left_image_feature_path"][0])
            right_image_feature = self.open_npy(self.db[idx]["right_image_feature_path"][0])


            ret = (
                bbox_image_feature,
                entire_image_feature,
                tokenized_instruction_clip,
                tokenized_np_clip,
                left_image_feature,
                right_image_feature
            )

        else:
            gt_img_id = self.db[idx]["gt_bbox_id"]
            instId = self.db[idx]["instruction_id"]
            gt_bbox = self.db[idx]["gt_bbox"]
            ret = (
                self.all_bbox_image_features,
                self.all_image_features,
                tokenized_instruction_clip,
                tokenized_np_clip,
                self.all_left_image_features,
                self.all_right_image_features,
                raw_instruction,
                gt_img_id,
                self.imageId_list,
                instId,
                gt_bbox
            )

        return ret

    @staticmethod
    def get_unique_np(noun_phrases):
        """名詞句リストから重複を削除する"""
        unique_phrases = []
        for phrase in sorted(noun_phrases, key=len, reverse=True):
            if phrase not in " ".join(unique_phrases):
                unique_phrases.append("A photo of " + phrase)
        return unique_phrases

    @classmethod
    def get_noun_phrases_clip(cls, noun_phrases, n=1):
        """名詞句をtensorとして取得"""
        noun = cls.get_unique_np(noun_phrases)
        tokenized_noun_phrases_clip = clip.tokenize(noun)

        # id 4019 : np を n つに制限
        if tokenized_noun_phrases_clip.shape[0] > n:
            return tokenized_noun_phrases_clip[:n, :].to(torch.int32)

        tokenized_noun_phrases_clip = torch.cat(
            [
                tokenized_noun_phrases_clip,
                torch.zeros(n - tokenized_noun_phrases_clip.shape[0], tokenized_noun_phrases_clip.shape[1]),
            ],
            dim=0,
        )
        return tokenized_noun_phrases_clip.to(torch.int32)

    def get_dataset(self, dataset_path):
        """JSON形式のデータセット情報を読み出す"""
        self.db = json.load(open(dataset_path, "r"))

    def open_npy(self, path):
        """numpy形式データを読み出す"""
        return np.load(path)
