# coding=utf-8
import os
import sys
import argparse
import json
import numpy as np
from PIL import Image
import cv2
import requests
import random

import flask
from flask_cors import CORS

import torch
import torchvision.transforms as transforms
from mmcv import Config
from mmcv.runner import load_checkpoint

from mmfashion.core import ClothesRetriever
from mmfashion.datasets import build_dataloader, build_dataset
from mmfashion.models import build_retriever
from mmfashion.utils import get_img_tensor

from utils import conv_base64_to_pillow, conv_pillow_to_base64, conv_base64_to_cv, conv_tensor_to_pillow, create_binary_mask

#======================
# parameter
#======================
args = None

#-----------------
# flask
#-----------------
app = flask.Flask(__name__)

# Access-Control-Allow-Origin
CORS(app, resources={r"*": {"origins": "*"}}, methods=['POST', 'GET'])

app.config['JSON_AS_ASCII'] = False
app.config["JSON_SORT_KEYS"] = False

#-----------------
# global parameter
#-----------------
device = None
model = None
cfg = None
gallery_set = None
gallery_embeds = None
retriever = None

#================================================================
# "http://host_ip:5012/mmfashion"
#================================================================
@app.route('/mmfashion', methods=['POST'])
def responce():
    print('MMFashion')
    if( app.debug ):
        print( "flask.request.method : ", flask.request.method )
        print( "flask.request.headers \n: ", flask.request.headers )

    if( flask.request.headers["User-Agent"].split("/")[0] in "python-requests" ):
        json_data = json.loads(flask.request.json)
    else:
        json_data = flask.request.get_json()

    img_path = args.input

    global model
    if args.use_cuda:
        model = model.cuda()

    #------------------------------------------
    # ブラウザから送信された画像データの変換
    #------------------------------------------
    img_cv = conv_base64_to_cv( json_data["img_base64"] )
    cv2.imwrite( img_path, img_cv )

    img_tensor = get_img_tensor(args.input, args.use_cuda)

    query_feat = model(img_tensor, landmark=None, return_loss=False)
    query_feat = query_feat.data.cpu().numpy()

    retrieved_paths = retriever.show_retrieved_images(query_feat, gallery_embeds)

    retrieved_imgs = []
    for retrieved_path in retrieved_paths:
        retrieved_path = retrieved_path.replace('data/In-shop/', '', 1)
        retrieved_img = Image.open(retrieved_path)
        retrieved_img_base64 = conv_pillow_to_base64(retrieved_img)
        retrieved_imgs.append(retrieved_img_base64)

    #------------------------------------------
    # json 形式のレスポンスメッセージを作成
    #------------------------------------------
    #torch.cuda.empty_cache()
    http_status_code = 200
    response = flask.jsonify(
        {
            'status':'OK',
            'origin_img': conv_pillow_to_base64(Image.open(img_path)),
            'retrieved_imgs': retrieved_imgs
        }
    )

    # Access-Control-Allow-Origin
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    if( app.debug ):
        print( "response.headers : \n", response.headers )

    # release gpu
    del img_tensor, query_feat, retrieved_paths
    if args.use_cuda:
        model = model.cpu()
        torch.cuda.empty_cache()

    return response, http_status_code

def _process_embeds(dataset, model, cfg, use_cuda=True):
    data_loader = build_dataloader(
        dataset,
        cfg.data.imgs_per_gpu,
        cfg.data.workers_per_gpu,
        len(cfg.gpus.test),
        dist=False,
        shuffle=False)

    embeds = []
    with torch.no_grad():
        for data in data_loader:
            img = data['img']
            if use_cuda:
                img = img.cuda()
            embed = model(img, landmark=data['landmark'], return_loss=False)
            embeds.append(embed)

    embeds = torch.cat(embeds)
    embeds = embeds.data.cpu().numpy()
    return embeds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default="0.0.0.0", help="IP 0.0.0.0")
    parser.add_argument('--port', type=int, default=5012, help="port number")
    parser.add_argument('--enable_threaded', action='store_true', help="multiple threaded")
    parser.add_argument('--debug', action='store_true', help="debug")
    parser.add_argument('--input', type=str, help='input image path', default='demo/imgs/06_1_front.jpg')
    parser.add_argument('--image_width', type=int, help='width of input image', default=256)
    parser.add_argument('--image_height', type=int, help='height of input image', default=256)
    parser.add_argument('--topk', type=int, default=5, help='retrieve topk items')
    parser.add_argument('--config', help='train config file path', default='configs/retriever_in_shop/global_retriever_vgg_loss_id.py')
    parser.add_argument('--checkpoint', type=str, default='checkpoint/Retrieve/vgg/global/epoch_100.pth', help='the checkpoint file to resume from')
    parser.add_argument('--use_cuda', type=bool, default=True, help='use gpu or not')
    args = parser.parse_args()
    
    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg = Config.fromfile(args.config)

    model = build_retriever(cfg.model)
    load_checkpoint(model, args.checkpoint)
    print('load checkpoint from {}'.format(args.checkpoint))

    if args.use_cuda:
        model = model.cuda()
    model.eval()

    gallery_set = build_dataset(cfg.data.gallery)
    gallery_embeds = _process_embeds(gallery_set, model, cfg)

    retriever = ClothesRetriever(cfg.data.gallery.img_file, cfg.data_root,
                                 cfg.data.gallery.img_path)

    model = model.cpu()
    torch.cuda.empty_cache()

    # Flask
    #------------------------------------
    print("MMFashion server started!")
    app.debug = args.debug
    if( args.enable_threaded ):
        app.run( host=args.host, port=args.port, threaded=False )
    else:
        app.run( host=args.host, port=args.port, threaded=True )

