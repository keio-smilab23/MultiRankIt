# PAPER TITLE

[[paper]()]

Kanta Kaneda, Shunya Nagashima, Ryosuke Korekata, Motonari Kambara and Komei Sugiura

Domestic service robots offer a solution to the
increasing demand for daily care and support. A human-
in-the-loop approach that combines automation and operator
intervention is considered to be a realistic approach to their
use in society. Therefore, we focus on the task of retrieving
target objects from open-vocabulary user instructions in a
human-in-the-loop setting, which we define as the learning-
to-rank physical objects (LTRPO) task. For example, given
the instruction "Please go to the dining room which has a
round table. Pick up the bottle on it," the model is required
to output a ranked list of target objects that the operator/user
can select. In this paper, we propose MultiRankIt, which is a
novel approach for the LTRPO task. MultiRankIt introduces
the Crossmodal Noun Phrase Encoder to model the relationship
between phrases that contain referring expressions and the
target bounding box, and the Crossmodal Region Feature
Encoder to model the relationship between the target object
and multiple images of its surrounding contextual environment.
Additionally, we built a new dataset for the LTRPO task
that consists of instructions with complex referring expressions
accompanied by real indoor environmental images that feature
various target objects. We validated our model on the dataset
and it outperformed the baseline method in terms of the mean
reciprocal rank and recall@k. Furthermore, we conducted
physical experiments in a setting where a domestic service
robot retrieved everyday objects in a standardized domestic
environment, based on users’ instruction in a human–in–the–
loop setting. The experimental results demonstrate that the
success rate for object retrieval achieved 80%.


## Setup
```bash
git clone XXX
cd reverie_retrieval
./scripts/build_docker.sh
```

Then install stanford parser from [here](https://nlp.stanford.edu/software/lex-parser.shtml).
We expect the directory structure to be the following:
```
./module
└── stanford_parser
    ├── stanford-corenlp-4.2.0-models-english.jar
    └── stanford-parser-full-2020-11-17
        ├── ...
        └── stanford-parser.jar
```
1. [stanford-parser-full](https://nlp.stanford.edu/software/stanford-parser-4.2.0.zip)
1. [stanford-corenlp](https://nlp.stanford.edu/software/stanford-corenlp-4.2.0-models-english.jar)


## Launch and Attach to docker container
1. launch container
    ```bash
    ./scripts/launch_container.sh
    ```

2. attach to container
    ```bash
    source ./config/set_constants.sh
    docker exec -it $docker_container_name bash
    ```


## Prepare Dataset
Download the necessary files from [here](https://drive.google.com/drive/folders/1fzhT74tiJhu8qDJr_604X5P2OopcDjMW).
We expect the directory structure to be the following:
```
./data
├── REVERIE_dataset
|   ├── BBoxes_v2.json
│   └── REVERIE_[train/val_seen/val_unseen/test].jsonBBoxes_v2.json
|   `
└── EXTRACTED_IMGS_
    ├── 17DRP5sb8fy
    ├── 1LXtFkjw3qL
    ├── ...
    └── zsNo4HB9uLZ
```

Create dataset:
```bash
# Inside docker
poetry run python src/create_database.py 
poetry run python src/create_dataset.py 
```


## Train
```sh
# Inside docker
poetry run python src/main.py train --bbox --eval_unseen --epoch 30 --lr 5e-5 --bs 128
```


## Evaluation
```sh
# Inside docker
poetry run python src/main.py test --infer_model_path ${MODEL_PATH}_best.pth
```

Expected results are as follow:
|    | MRR | Recall@1 | Recall@5 |  Recall@10 |  Recall@20 |
| ---- | ---- | ---- | ---- | ---- | ---- |
|  Val  | 47.16 | 16.44 | 52.62 | 71.43 | 84.39 | 
|  Test  |  50.57 |18.16 |  54.68 | 73.22 | 86.32 | 

