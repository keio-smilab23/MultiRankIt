"""
Text-Image Retrieval

TO ADD :
- Gradient Checkpointing
- Filter out bias from weight decay
- Decaying learning rate with cosine schedule
- Half-precision Adam statistics
- Half-precision stochastically rounded text encoder weights were used

Note:
1. BATCH_SIZE must larger than 1
"""

import argparse
import itertools
import json
import os
import warnings

import numpy as np
import torch
import wandb
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import callback_server
import clip
import dataloader
import performance_timer
from model import ClipReverie
from stanford_parser import get_all_np

warnings.simplefilter("ignore")


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str)

    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--wandb_name", "-w", default="")

    parser.add_argument("--clip_base_model", default="ViT-L/14")

    parser.add_argument("--frcnn", action="store_true")
    parser.add_argument("--bbox", action="store_true")
    parser.add_argument("--num_bbox", default=16)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", default="5e-4")
    parser.add_argument("--bs", default=128)
    parser.add_argument("--epochs", default=20)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval_metric", default="recall10+mrr")
    parser.add_argument("--eval_test", action="store_true")
    parser.add_argument("--eval_seen", action="store_true")
    parser.add_argument("--eval_unseen", action="store_true")

    parser.add_argument("--model_output_prefix", default="model/model_tir")
    parser.add_argument("--infer_model_path", default=None)
    parser.add_argument("--output_file", default=None)  # results/id4024 (no .json)
    parser.add_argument("--server_config", default="config/server_config.json")

    parser.add_argument("--N", default="30")
    parser.add_argument("--baseline_dataset", default="")  # train or destination
    parser.add_argument("--cut_environment", action="store_true")
    parser.add_argument("--no_2d", action="store_true")

    parser.add_argument("--profiling", action="store_true")

    args = parser.parse_args()

    if args.wandb_name != "":
        args.log_wandb = True

    if not args.infer_model_path:
        args.infer_model_path = args.model_output_prefix + "_best.pth"

    return args


class TextImageRetrievalMain:
    """Text-Image Retrievalを実行するクラス"""

    VAL_SEEN_ENVIRONMENT = ["8WUmhLawc2A", "1pXnuDYAj8r", "VzqfbhrpDEA", "ac26ZMwG7aT", "rPc6DW4iMge"]
    VAL_UNSEEN_ENVIRONMENT = ["EU6Fwq7SyZv", "x8F5xyUWy9e", "zsNo4HB9uLZ", "oLBMNvg9in8"]
    TEST_ENVIRONMENT = ["2azQ1b91cZZ", "QUCTc6BB5sX", "X7HyMhZNoso", "TbHJrupSAjP"]
    REAL_TEST_ENVIRONMENT = [
        "real_20230118T172959",
        "real_20230118T173924",
        "real_20230119T153429",
        "real_20230119T154113",
        "real_20230119T155732",
        "real_20230119T160333",
        "real_20230119T161046",
        "real_20230119T161732",
        "real_20230119T162429",
        "real_20230119T163059",
    ]

    def __init__(self, args_):
        self.device = "cuda:0"
        self.args = args_

        self.model = ClipReverie(args_.clip_base_model, self.device).cuda(self.device)

    def load_model(self, path):
        """モデルをロードする"""
        self.model.load_state_dict(torch.load(path))
        print(f"model file was loaded from {path}.")

    def save_model(self, path):
        """モデルを保存する"""
        torch.save(self.model.state_dict(), path)

    def train_model(self):
        """学習を実行する"""
        print("Currently loading train dataset ... ")
        train_dataset = dataloader.reverie_dataset(
            self.model,
            self.args,
            split="train",
            env="train",
            eval=False,
            N=int(self.args.N),
            baseline_dataset=self.args.baseline_dataset,
        )
        train_dataloader = DataLoader(train_dataset, batch_size=int(self.args.bs))

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(self.args.lr), betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2
        )  # Params from paper

        best_score = 0
        best_epoch = 0
        for epoch in range(int(self.args.epochs)):
            print(f"\n==== Epoch {epoch} ================")
            loss = self.train_epoch(train_dataloader, optimizer, frcnn_flag=self.args.frcnn)
            print(f"Epoch: {epoch},  Loss: {loss:.4f}")

            if self.args.eval_seen:
                val_seen = self.evaluate("val_seen", self.VAL_SEEN_ENVIRONMENT)
                print(", ".join([f"{value:.2f}" for value in val_seen]))

            if self.args.eval_unseen:
                val_unseen = self.evaluate("val_unseen", self.VAL_UNSEEN_ENVIRONMENT)
                val_unseen_mrr = val_unseen[0]
                val_unseen_recall1 = val_unseen[1]
                val_unseen_recall5 = val_unseen[2]
                val_unseen_recall10 = val_unseen[3]
                val_unseen_recall20 = val_unseen[4]  # noqa
                print(", ".join([f"{value:.2f}" for value in val_unseen]))

                if self.args.eval_metric == "recall10+mrr":
                    eval_score = val_unseen_recall10 + val_unseen_mrr
                elif self.args.eval_metric == "recall10":
                    eval_score = val_unseen_recall10
                elif self.args.eval_metric == "recall5":
                    eval_score = val_unseen_recall5
                elif self.args.eval_metric == "recall1":
                    eval_score = val_unseen_recall1
                elif self.args.eval_metric == "mrr":
                    eval_score = val_unseen_mrr

                if eval_score > best_score:
                    best_epoch = epoch
                    best_score = eval_score
                    model_path = f"{self.args.model_output_prefix}_best.pth"
                    print(f"save model file as best.: {model_path}")
                    self.save_model(model_path)

            self.save_model(f"{self.args.model_output_prefix}_{epoch:03d}.pth")

            if self.args.eval_test:
                test_result = self.evaluate("test", self.TEST_ENVIRONMENT)
                print(", ".join([f"{value:.2f}" for value in test_result]))

        print(f"\n==== RESULTS for best epoch {best_epoch} =====")
        self.load_model(self.args.infer_model_path)
        self.test_model()

    def test_model(self):
        """評価を実行する"""
        header = "mrr, recall1, recall5, recall10, recall20 = "
        test_result = self.evaluate("test", self.TEST_ENVIRONMENT)
        print(header + ", ".join([f"{val:.2f}" for val in test_result]))

        if self.args.log_wandb:
            wandb.log(
                {
                    "mrr": test_result[0],
                    "R@1": test_result[1],
                    "R@5": test_result[2],
                    "R@10": test_result[3],
                    "R@20": test_result[4],
                }
            )

    def test_model_real(self):
        """評価を実行する"""
        header = "mrr, recall1, recall5, recall10, recall20 = "
        test_result = self.evaluate("real_test", self.REAL_TEST_ENVIRONMENT)
        print(header + ", ".join([f"{val:.2f}" for val in test_result]))

    def predict_oneshot(self, image_embeddings, instruction, mode):
        """一発打ちを実行する"""
        self.model.eval()

        if hasattr(instruction, "read"):
            raw_st_instruction = json.load(instruction)["instruction"]
        elif isinstance(instruction, str):
            raw_st_instruction = instruction
        else:
            raise RuntimeError("unsupported type for argument 'instruction' of function 'predict_oneshot'")

        with performance_timer.get_timer("get information from ChatGPT", self.args.profiling):
            chat_gpt_data = self.interpret_with_chat_gpt()

        with performance_timer.get_timer("compose data about instruction", self.args.profiling):
            raw_instruction = mode + " " + raw_st_instruction
            if mode == "<target>":
                raw_instruction_chatgpt = chat_gpt_data["instruction_chatgpt"]
            elif mode == "<destination>":
                raw_instruction_chatgpt = chat_gpt_data["llm_data"]["destination"]
            else:
                raise RuntimeError(f"unknown mode for {mode}")

            if raw_instruction_chatgpt == "NA":
                raw_instruction_chatgpt = ""
                modified_instruction = raw_instruction
            else:
                modified_instruction = mode + " " + chat_gpt_data["llm_data"]["modified_instruction"]

            with performance_timer.get_timer("tokenize with clip tokenizer", self.args.profiling):
                tokenized_instruction_clip = clip.tokenize(raw_instruction).squeeze(0)
                tokenized_instruction_modified_clip = clip.tokenize(modified_instruction).squeeze(0)
                tokenized_instruction_chatgpt_clip = clip.tokenize("A photo of " + raw_instruction_chatgpt).squeeze(0)
            with performance_timer.get_timer("get noun phrase with stanford parser", self.args.profiling):
                tokenized_np_clip = dataloader.reverie_dataset.get_noun_phrases_clip(
                    list(get_all_np(raw_st_instruction)), n=self.model.N
                )
            gpt3_embeddings = torch.tensor(chat_gpt_data["gpt3_embeddings"])

        with performance_timer.get_timer("forwad with model", self.args.profiling):
            logits, _ = self.model(
                None,
                image_embeddings.unsqueeze(0).to(self.device),
                tokenized_instruction_clip.unsqueeze(0).to(self.device),
                tokenized_instruction_modified_clip.unsqueeze(0).to(self.device),
                tokenized_instruction_chatgpt_clip.unsqueeze(0).to(self.device),
                tokenized_np_clip.unsqueeze(0).to(self.device),
                None,
                None,
                None,
                None,
                None,
                None,
                gpt3_embeddings.unsqueeze(0).to(self.device),
            )
            score = logits.detach().cpu().numpy().tolist()[0]
        return score[0]

    def embed_image(self, image, np_return=False):
        """画像をTIR向けにembeddingする"""
        clip_feature = self.model.clip_model.encode_image(
            self.model.preprocess_clip(Image.open(image)).unsqueeze(0).to(self.device)
        )

        if np_return:
            return clip_feature.to("cpu").detach().numpy().copy()
        return clip_feature

    def interpret_with_chat_gpt(self):
        """ChatGPTに解釈させたデータを生成する"""
        ret = {
            "instruction_chatgpt": "lamp",
            "llm_data": {"destination": "white table", "modified_instruction": "Carry lamp to white table"},
            "gpt3_embeddings": [0.0] * 1536,
        }
        return ret

    def cross_entropy(self, preds, targets, reduction="none"):
        """Cross entropyを定義"""
        log_softmax = torch.nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()

    def train_epoch(self, dataloader, optimizer, frcnn_flag=False):
        """1epoch分のtrainを実行する"""
        self.model.train()
        t_loss = 0
        n_ex = 0
        for (
            bbox_image_feature,
            entire_image_feature,
            tokenized_instruction_clip,
            # tokenized_instruction_modified_clip,
            # tokenized_instruction_chatgpt_clip,
            tokenized_np_clip,
            left_image_feature,
            right_image_feature
        ) in tqdm(dataloader):
            optimizer.zero_grad()
            logits, targets = self.model(
                bbox_image_feature.to(self.device),
                entire_image_feature.to(self.device),
                tokenized_instruction_clip.to(self.device),
                # tokenized_instruction_modified_clip.to(self.device),
                # tokenized_instruction_chatgpt_clip.to(self.device),
                tokenized_np_clip.to(self.device),
                left_image_feature.to(self.device),
                right_image_feature.to(self.device)
            )
            texts_loss = self.cross_entropy(logits, targets)
            images_loss = self.cross_entropy(logits.T, targets.T)
            loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)
            loss = loss.mean()

            t_loss += loss
            loss.backward()
            optimizer.step()
            n_ex += 1

        return loss / n_ex

    def calc_score(self, probs, gt_id, imgId_list_full, eval_baseline=False):
        """Scoreを計算"""
        mrr, recall1, recall5, recall10, recall20 = 0, 0, 0, 0, 0

        ranks = []
        if not eval_baseline:
            imgId_list_full = [i[0] for i in imgId_list_full]

        imgId_list = [i.split("_")[-1] for i in imgId_list_full]

        for gt in gt_id:
            idx = imgId_list.index(gt[0].split("_")[-1])
            rank = sorted(probs, reverse=True).index(probs[idx])
            while rank in ranks:  # 同じ値の場合に
                rank += 1
            ranks.append(rank)

        # find top 20
        top20 = []
        top20_rank = np.argsort(probs)[-20:][::-1]
        for i in top20_rank:
            top20.append(imgId_list_full[i])

        for i, rank in enumerate(sorted(ranks)):
            if rank < 20:
                recall20 += 1
            if rank < 10:
                recall10 += 1
            if rank < 5:
                recall5 += 1
            if rank < 1:
                recall1 += 1

            if i == 0:  # first time
                mrr = 100 / (rank + 1)

        recall20 = 100 * recall20 / len(ranks)
        recall10 = 100 * recall10 / len(ranks)
        recall5 = 100 * recall5 / len(ranks)
        recall1 = 100 * recall1 / len(ranks)

        return mrr, recall1, recall5, recall10, recall20, ranks, top20

    @torch.no_grad()
    def evaluate(self, split, environments, file_output=None):
        """
        評価処理を実行

        返り値は、mrr, recall1, recall5, recall10, recall20の順。
        """
        self.model.eval()
        output = {}
        with torch.no_grad():
            mrr, recall1, recall5, recall10, recall20 = 0, 0, 0, 0, 0
            print(f"==========  {split.upper()}  ===========")
            if self.args.cut_environment:
                environments = environments[1 : len(environments)]

            for env in environments:
                eval_dataset = dataloader.reverie_dataset(
                    self.model,
                    self.args,
                    split=split,
                    env=env,
                    num_bbox=16,
                    eval=True,
                    N=int(self.args.N),
                    baseline_dataset=self.args.baseline_dataset,
                )
                if len(eval_dataset) == 0:
                    print(f"skip env {env} because there's no sample.")
                    continue
                eval_dataloader = DataLoader(eval_dataset, batch_size=1)

                n_ex = 0
                env_mrr, env_recall1, env_recall5, env_recall10, env_recall20 = 0, 0, 0, 0, 0
                env_output = []

                for (
                    all_bbox_image_features,
                    all_image_features,
                    tokenized_instruction_clip,
                    tokenized_np_clip,
                    all_left_image_features,
                    all_right_image_features,
                    raw_instruction,
                    gt_img_id,
                    imageId_list,
                    instId,
                    gt_bbox,
                ) in tqdm(eval_dataloader):
                    all_tokenized_instruction_clip = tokenized_instruction_clip.to(self.device).repeat(
                        all_bbox_image_features.shape[1], 1
                    )
                    all_tokenized_np_clip = tokenized_np_clip.to(self.device).repeat(
                        all_bbox_image_features.shape[1], 1, 1
                    )

                    logits_per_text, _, _ = self.model.calc_logits(
                        all_image_features.to(self.device).squeeze(0),
                        all_tokenized_instruction_clip,
                        all_tokenized_np_clip,
                        all_left_image_features.to(self.device).squeeze(0),
                        all_right_image_features.to(self.device).squeeze(0),
                        all_bbox_image_features.to(self.device).squeeze(0),
                        eval=True,
                    )

                    _mrr, _recall1, _recall5, _recall10, _recall20, ranks, top20 = self.calc_score(
                        np.diag(logits_per_text.cpu().numpy()), gt_img_id, imageId_list
                    )

                    if file_output:
                        dump = {}

                        dump["instruction_id"] = instId[0]
                        dump["instruction"] = raw_instruction[0]
                        dump["gt_image_id"] = [x[0] for x in gt_img_id]
                        dump["mrr"] = str(_mrr)
                        dump["ranks"] = [str(x) for x in ranks]
                        dump["top20"] = top20

                        tmp_bboxes = []
                        for xy in gt_bbox:
                            tmp_bbox = []
                            for x in xy:
                                tmp_bbox.append(x.item())
                            tmp_bboxes.append(tmp_bbox)
                        dump["gt_bbox"] = tmp_bboxes
                        env_output.append(dump)
                        # env_output.append(dump)

                    n_ex += 1
                    env_mrr += _mrr
                    env_recall1 += _recall1
                    env_recall5 += _recall5
                    env_recall10 += _recall10
                    env_recall20 += _recall20

                mrr += env_mrr / n_ex
                recall1 += env_recall1 / n_ex
                recall5 += env_recall5 / n_ex
                recall10 += env_recall10 / n_ex
                recall20 += env_recall20 / n_ex

                print(
                    ", ".join(
                        [
                            f"num_inst : {n_ex}",
                            f"num_img : {len(imageId_list)} ... {env_mrr/n_ex:.2f}",
                            f"{env_recall1/n_ex:.2f}",
                            f"{env_recall5/n_ex:.2f}",
                            f"{env_recall10/n_ex:.2f}",
                            f"{env_recall20/n_ex:.2f}",
                        ]
                    )
                )

                if file_output:
                    output[env] = env_output

            if file_output:
                with open(f"{file_output}_{split}.json", "w") as wf:
                    json.dump(output, wf, indent=2, ensure_ascii=False)

        n_envs = len(environments)
        return (mrr / n_envs, recall1 / n_envs, recall5 / n_envs, recall10 / n_envs, recall20 / n_envs)


def tir_main():
    """Text-Image Retrievalを実行する"""
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.log_wandb:
        wandb.init(project="clip-reverie", name=args.wandb_name)

    with performance_timer.get_timer("build model", args.profiling):
        tir = TextImageRetrievalMain(args)

    if args.mode == "train":
        tir.train_model()
    elif args.mode == "test":
        tir.load_model(args.infer_model_path)
        tir.test_model()
    elif args.mode == "real_test":
        tir.load_model(args.infer_model_path)
        tir.test_model_real()
    elif args.mode == "oneshot":
        with performance_timer.get_timer("load model", args.profiling):
            tir.load_model(args.infer_model_path)

        sample_data_prefix = os.path.join("scripts", "sample_data")
        posi_sample_img = os.path.join(sample_data_prefix, "posi_center.jpg")
        neg_sample_img = os.path.join(sample_data_prefix, "neg_center.jpg")
        instruction_path = os.path.join(sample_data_prefix, "instruction.json")

        result_str = ""
        for img_path, mode in itertools.product([posi_sample_img, neg_sample_img], ["<target>", "<destination>"]):
            with open(img_path, "rb") as img, open(instruction_path, "r", encoding="utf-8") as inst:
                with performance_timer.get_timer("predict_oneshot", args.profiling):
                    with performance_timer.get_timer("embeding image", args.profiling):
                        embedded_image = tir.embed_image(img)
                    score = tir.predict_oneshot(embedded_image, inst, mode)
                result_str += "\n{0:30}: {1:.04f}".format(os.path.basename(img_path) + "," + mode, score)

        print("=" * 20 + " RESULT " + "=" * 20 + result_str)

    elif args.mode == "start_server":
        tir.load_model(args.infer_model_path)
        with open(args.server_config, "r", encoding="utf-8") as server_conf:
            conf = json.load(server_conf)
        callback_server.start(conf, tir.predict_oneshot, tir.embed_image)

    else:
        raise RuntimeError(f"unknown mode of [{args.mode}]")


if __name__ == "__main__":
    tir_main()
