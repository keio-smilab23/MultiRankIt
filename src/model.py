"""
Text-Image Retrieval のモデル定義
"""
import copy
import math

import clip
import torch
import torch.nn.functional as F


class ClipReverie(torch.nn.Module):
    """
    Baseline : CLIP ViT-L/14 を用いて評価
    """

    def __init__(self, clip_base, device, bbox=True, N=30, with_chat_gpt_=True):
        super(ClipReverie, self).__init__()
        self.clip_model, self.preprocess_clip = clip.load(clip_base, device=device)
        for params in self.clip_model.parameters():
            params.requires_grad = False
        self.with_chat_gpt = with_chat_gpt_
        self.temperature = 1
        self.bbox = bbox
        self.N = N

        # id 5003 txt transformer
        h_txt = 4
        d_ff_txt = 768 * 2
        dropout_txt = 0.4
        d_model_txt = 768
        N_txt = 5
        c = copy.deepcopy
        attn_txt = MultiHeadedAttention(h_txt, d_model_txt)
        ff_txt = PositionwiseFeedForward(d_model_txt, d_ff_txt, dropout_txt)
        self.encoder_txt = Encoder(N_txt, EncoderLayer(d_model_txt, c(attn_txt), c(ff_txt), dropout=dropout_txt))

        h_img = 4
        d_ff_img = 768*2
        dropout_img = 0.4
        d_model_img = 768
        N_img = 5
        c = copy.deepcopy
        attn_img = MultiHeadedAttention(h_img, d_model_img)
        ff_img = PositionwiseFeedForward(d_model_img, d_ff_img, dropout_img)
        self.encoder_img = Encoder(N_img, EncoderLayer(
            d_model_img, c(attn_img), c(ff_img), dropout=dropout_img))


        # mlp
        self.fc1 = torch.nn.Linear(768*2, 1000)
        self.fc2 = torch.nn.Linear(1000, 768)
        self.bn = torch.nn.BatchNorm1d(1000)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.4)


    def image_encoder(
        self,
        image,
        left_image_feature,
        right_image_feature,
        bbox_image_feature,
        eval,
    ):
        # id 8013
        left_image_embeddings = left_image_feature
        right_image_embeddings = right_image_feature

        lr_img = torch.cat([bbox_image_feature, image, left_image_embeddings, right_image_embeddings], dim=1)
        image_embeddings = self.encoder_img(lr_img)
        image_embeddings = image_embeddings[:, 0, :].squeeze(1)
        image_embeddings = image_embeddings + bbox_image_feature.squeeze(1) + image.squeeze(1)

        return image_embeddings.half()

    def text_encoder(
        self,
        tokenized_text_clip,
        tokenized_np_clip,
        bbox_image_feature=None,
    ):
        text_embeddings = self.clip_model.encode_text(tokenized_text_clip).float()
        n_np_embeddings = self.clip_model.encode_text(tokenized_np_clip[:,0,:]).float().unsqueeze(1)
        for n in range(1, self.N):
            n_np_embeddings = torch.cat(
                [n_np_embeddings, self.clip_model.encode_text(tokenized_np_clip[:, n, :]).float().unsqueeze(1)],
                dim=1,
            )

        embeddings = torch.cat(
            [
                n_np_embeddings,
                bbox_image_feature,
            ],
            dim=1,
        )

        np_embeddings = self.encoder_txt(embeddings)[:, 0, :].squeeze(1)  # [bs, 768]

        embeddings = torch.cat(
            [
                text_embeddings,
                np_embeddings,
            ],
            dim=1,
        )
        x = self.dropout(self.bn(self.relu(self.fc1(embeddings))))
        text_embeddings = self.fc2(x)

        return text_embeddings.half()

    def calc_logits(
        self,
        image,
        text_clip,
        tokenized_np_clip,
        left_image_feature,
        right_image_feature,
        bbox_image_feature,
        eval=False,
    ):
        # print("tokenized_np.shape : ", tokenized_np_clip.shape)
        image_embeddings = self.image_encoder(
            image,
            left_image_feature,
            right_image_feature,
            bbox_image_feature,
            eval,
        )
        text_embeddings = self.text_encoder(
            text_clip,
            tokenized_np_clip,
            bbox_image_feature,
        )

        logits = (text_embeddings @ image_embeddings.T) / self.temperature  # [128,128]

        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T

        return logits, images_similarity, texts_similarity

    def forward(
        self,
        bbox_image_feature,
        entire_image_feature,
        tokenized_instruction_clip,
        tokenized_np_clip,
        left_image_feature,
        right_image_feature
    ):
        logits, images_similarity, texts_similarity = self.calc_logits(
            entire_image_feature,
            tokenized_instruction_clip,
            tokenized_np_clip,
            left_image_feature,
            right_image_feature,
            bbox_image_feature,
        )
        targets = F.softmax((images_similarity + texts_similarity) / 2 * self.temperature, dim=-1)

        return logits, targets

    def preprocess(self, x):
        return self.preprocess_clip(x)


class Encoder(torch.nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, N, layer):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


def clones(module, N):
    """Produce N identical layers."""
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class EncoderLayer(torch.nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x):
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)


class SublayerConnection(torch.nn.Module):
    """
    A residual connection followed by a layer norm.

    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(torch.nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = torch.nn.Parameter(torch.ones(features))
        self.b_2 = torch.nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        """Definition of forward propagation"""
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(torch.nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(d_model, d_ff)
        self.w_2 = torch.nn.Linear(d_ff, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        """Definition of forward propagation"""
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class MultiHeadedAttention(torch.nn.Module):
    """Take in model size and number of heads."""

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(torch.nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """Implement Figure 2"""
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l_fn(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l_fn, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class CrossAttentionEncoderLayer(torch.nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, cross_attn, feed_forward, dropout):
        super(CrossAttentionEncoderLayer, self).__init__()
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, kv):
        """Follow Figure 1 (left) for connections."""
        print("hello")
        x = self.sublayer[0](x, lambda x: self.cross_attn(x, kv, kv))
        return self.sublayer[1](x, self.feed_forward)


def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
