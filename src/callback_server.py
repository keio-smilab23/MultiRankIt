"""
コールバックサーバを定義するモジュール
"""
import base64
import io
import json

import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse


def start(conf, callback, cb_embed_image):
    """
    コールバックサーバを起動する
    """
    fapi = FastAPI()

    @fapi.post("/oneshot/all", responses={200: {"content": {"application/json": {"example": {}}}}})
    def execute_oneshot(
        image: UploadFile = File(..., description="判定対象とする画像"),
        instruction: UploadFile = File(..., description="命令文を含むJSONファイル"),
    ):
        """入力された画像とinstructionのsimilarity scoreをTIRで演算して返すAPI

        出力形式
            JSON形式
            {
                "score": similarity_score
            }
        """
        input_json = json.load(instruction.file)
        embedded_img = cb_embed_image(image.file)
        ret = callback(embedded_img, input_json["instruction"], input_json["mode"])
        return JSONResponse(content={"score": ret})

    @fapi.post("/embedding/image", responses={200: {"content": {"application/json": {"example": {}}}}})
    def execute_embedding_image(
        image: UploadFile = File(..., description="embeddingする画像"),
    ):
        """入力された画像をTIRの画像エンコーダーでエンコードしたベクトルを返すAPI

        出力形式
            JSON形式
            {
                "embeddings": "base64エンコードされたnpyファイル"
            }
        """
        embedded_img = cb_embed_image(image.file, np_return=True)
        buf = io.BytesIO()
        np.save(buf, embedded_img)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        return JSONResponse(content={"embeddings": b64})

    host_name = conf["hostname"]
    port_num = int(conf["port"])

    uvicorn.run(fapi, host=host_name, port=port_num)
