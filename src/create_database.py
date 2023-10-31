"""学習データとなるデータベースを作成"""
import json
import os

from tqdm import tqdm


def check_bbox(bbox):
    """Return 1 when bbox is acceptable, 0 when else"""
    width = bbox[2]
    height = bbox[3]
    area = width * height
    # unaccepatable when bbox is too small
    if width < 20 or height < 20 or area < 200:
        return 0
    # unacceptable when object is shown in the corner
    x2 = bbox[0] + width
    y2 = bbox[1] + height
    if bbox[0] == 0 or bbox[1] == 0 or x2 == 640 or y2 == 480:
        return 0
    return 1


def create_database(split, dataset_environment):
    """
    Text-Image Retrieval向けのデータセットを作成

    REVERIE_train から REVERIE-retrieval_train を作成
    REVERIE_val_unseen から REVERIE-retrieval_val_unseen, REVERIE-retrieval_val_test を作成
    """
    env_list = {}
    img_dir = "./data/EXTRACTED_IMGS_/"
    reverie_dataset_dir = "./data/REVERIE_dataset"
    bbox_file_path = "./data/REVERIE_dataset/BBoxes_v2.json"
    output_path = f"./data/reverie_database/reverie_retrieval_database_{split}.json"

    if split == "test":
        inst_file = json.load(open(f"{reverie_dataset_dir}/REVERIE_val_unseen.json", "r"))
    else:
        inst_file = json.load(open(f"{reverie_dataset_dir}/REVERIE_{split}.json", "r"))

    bbox_file = json.load(open(bbox_file_path, "r"))

    # create tmp_db
    tmp_db = []
    for target in tqdm(inst_file):
        tmp = {}
        tmp["environment"] = target["scan"]

        # count number of target_object and bbox for each environment
        if tmp["environment"] not in env_list.keys():
            env_list[tmp["environment"]] = {}
            env_list[tmp["environment"]]["target"] = 0
            env_list[tmp["environment"]]["bbox"] = 0
        env_list[tmp["environment"]]["target"] += 1

        tmp["waypoint"] = target["path"][-1]
        tmp["objId"] = f'target_{tmp["environment"]}_{target["objId"]}'
        tmp["instructions"] = target["instructions"]

        bbox_data = bbox_file[f'{tmp["environment"]}_{tmp["waypoint"]}'][str(target["objId"])]
        tmp["target_object"] = bbox_data["name"]

        tmp["image"] = []
        for i, pos in enumerate(bbox_data["visible_pos"]):
            if check_bbox(bbox_data["bbox2d"][i]):
                tmp2 = {}
                tmp2["image_path"] = (
                    img_dir + target["scan"] + "/" + tmp["waypoint"] + "/id" + format(pos, "02d") + ".jpg"
                )
                tmp2["bbox"] = bbox_data["bbox2d"][i]
                tmp["image"].append(tmp2)
                env_list[tmp["environment"]]["bbox"] += 1

        tmp_db.append(tmp)

    # create database
    db = {}
    for data in tmp_db:
        if split != "train":
            if data["environment"] not in dataset_environment:
                continue

        if data["environment"] not in db.keys():
            db[data["environment"]] = {}

        if data["waypoint"] not in db[data["environment"]].keys():
            db[data["environment"]][data["waypoint"]] = {}

        for img in data["image"]:
            if check_bbox(img["bbox"]):
                view = img["image_path"][-6:-4]

                if view not in db[data["environment"]][data["waypoint"]].keys():
                    db[data["environment"]][data["waypoint"]][view] = {}

                tmp = {}
                tmp["bbox"] = img["bbox"]
                tmp["instructions"] = []
                for inst in data["instructions"]:
                    tmp["instructions"].append(inst)
                tmp["target_object"] = data["target_object"]

                db[data["environment"]][data["waypoint"]][view][data["objId"]] = tmp

    delete_keys = []
    for env, data in db.items():
        delete_keys = []
        for waypoint, data2 in data.items():
            if len(data2.keys()) == 0:
                delete_keys.append(waypoint)
        for k in delete_keys:
            db[env].pop(k)

    os.makedirs("./data/reverie_database/", exist_ok=True)
    with open(output_path, "w") as wwf:
        json.dump(db, wwf, indent=2)

    return


if __name__ == "__main__":
    for split in ["train", "val_seen", "test", "val_unseen"]:
        if split == "val_unseen":
            dataset_environment = ["EU6Fwq7SyZv", "x8F5xyUWy9e", "zsNo4HB9uLZ", "oLBMNvg9in8"]
        elif split == "test":
            dataset_environment = ["2azQ1b91cZZ", "QUCTc6BB5sX", "X7HyMhZNoso", "TbHJrupSAjP"]
        elif split == "val_seen":
            dataset_environment = ["8WUmhLawc2A", "1pXnuDYAj8r", "VzqfbhrpDEA", "ac26ZMwG7aT", "rPc6DW4iMge"]
        else:
            dataset_environment = []
        create_database(split, dataset_environment)
