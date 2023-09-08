import numpy as np 
from zoedepth.utils.geometry import depth_to_points, create_triangles
from zoedepth.utils.misc import get_image_from_url, colorize
import torch 
from PIL import Image 
import matplotlib.pyplot as plt
import tempfile
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

import matplotlib.pyplot as plt
from bottle import route, run, template, request, static_file, url, get, post, response, error, abort, redirect, os
import argparse
import datetime
import uuid


t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, 'JST')

@get('/')
def upload():
    return '''
        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="submit" value="Upload"></br>
            <input type="file" name="image"></br>
        </form>
    '''
@route('/assets/<filepath:path>', name='assets')
def server_static(filepath):
    return static_file(filepath, root=base_shape_dir)

@route('/upload', method='POST')
def do_upload():
    upload = request.files.get('image', '')
    if not upload.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return 'File extension not allowed!'
    
    now = datetime.datetime.now(JST)
    d = '{:%Y%m%d%H%M%S}'.format(now)
    short_id = str(uuid.uuid4())[:8]
    dir_name = f"{d}_{short_id}"
    shape_dir = os.path.join(base_shape_dir, dir_name)
    os.makedirs(shape_dir, exist_ok=True)
    print(shape_dir)

    filename = upload.filename.lower()
    # root, ext = os.path.splitext(filename)
    save_path = os.path.join(shape_dir, filename)
    upload.save(save_path, overwrite=True)

    img = Image.open(save_path).convert('RGB')

    depth = model.infer_pil(img)
    colored_depth = colorize(depth)
    colored_depth = Image.fromarray(colored_depth)

    # save both depth map and corresponding colored image
    output_img_path = os.path.join(shape_dir, "depth.png")
    output_depth_path = os.path.join(shape_dir, "depth.npy")
    np.save(output_depth_path, depth)
    colored_depth.save(output_img_path)
    print(f"output saved to {shape_dir}")


    # return {"status": "success", "saved_paths": saved_paths}
    body = {"status": 0, "data": f"http://20.168.237.190:8000/assets/{dir_name}/result.png"}
    return body

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--gpu', type=int, default=1, help='Use GPU or not')
    args = parser.parse_args()

    base_shape_dir = f"./exp/web"
    os.makedirs(base_shape_dir, exist_ok=True)

    conf = get_config("zoedepth_nk", "infer")
    model= build_model(conf)
    device = "cuda" if args.gpu else "cpu"
    model = model.to(device)

    print("run")
    run(host="0.0.0.0", port=8000, debug=True)
