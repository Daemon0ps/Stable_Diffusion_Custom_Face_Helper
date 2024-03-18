
from __future__ import annotations
import os
import re
import cv2
import nltk
import string
import webuiapi
import numpy as np
import multiprocessing
from tqdm import tqdm
from enum import Enum
from random import randint,shuffle
from unidecode import unidecode
from dataclasses import dataclass
from nltk.corpus import stopwords
from types import SimpleNamespace
from datetime import datetime
from nltk.tokenize import word_tokenize
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image,ImageFile, ImageDraw2
from random import randint


Image.MAX_IMAGE_PIXELS = None
ImageDraw2.Font
ImageFile.LOAD_TRUNCATED_IMAGES = True
nltk.download("punkt",quiet=True)
nltk.download("stopwords",quiet=True)
nltk.download("tagsets",quiet=True)
sw = set(stopwords.words("english"))

YUNET_ONNX_PATH = "E:/opencv_zoo/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

LORA_DIR = "K:/SD/LORA/"

FILE_LIST : list[str] = []
LORAS:dict = {}
TAG_DICT:dict = {}
KW_LIST:list[str] = []
TAG_DICT:dict = {}
COMBI_DICT:dict={}
WILDCARDS:dict = {}
pl:list[str] = []

BACKEND_TARGET_PAIRS = [
    [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],           #0
    [cv2.dnn.DNN_BACKEND_CUDA,   cv2.dnn.DNN_TARGET_CUDA],          #1
    [cv2.dnn.DNN_BACKEND_CUDA,   cv2.dnn.DNN_TARGET_CUDA_FP16],     #2
    [cv2.dnn.DNN_BACKEND_TIMVX,  cv2.dnn.DNN_TARGET_NPU],           #3
    [cv2.dnn.DNN_BACKEND_CANN,   cv2.dnn.DNN_TARGET_NPU]            #4
]
backend_id = BACKEND_TARGET_PAIRS[2][0]
target_id = BACKEND_TARGET_PAIRS[2][1]

SAVE_DIR = "F:/sd/outputs/webui_i2i/"

IMG_TYPES = [
            'blp', 'bmp', 'dib', 'bufr', 'cur', 'pcx', 'dcx', 'dds', 'ps', 'eps'
            , 'fit', 'fits', 'fli', 'flc', 'ftc', 'ftu', 'gbr', 'gif', 'grib', 'h5'
            , 'hdf', 'png', 'apng', 'jp2', 'j2k', 'jpc', 'jpf', 'jpx', 'j2c', 'icns'
            , 'ico', 'im', 'iim', 'tif', 'tiff', 'jfif', 'jpe', 'jpg', 'jpeg', 'mpg'
            , 'mpeg', 'mpo', 'msp', 'palm', 'pcd', 'pxr', 'pbm', 'pgm', 'ppm', 'pnm'
            , 'psd', 'bw', 'rgb', 'rgba', 'sgi', 'ras', 'tga', 'icb', 'vda', 'vst'
            , 'webp', 'wmf', 'emf', 'xbm', 'xpm','nef'
            ]

replace_list:dict = {}
remove_list:list[str] = []


def dns(d:dict) -> SimpleNamespace:
    return SimpleNamespace(**d)

def im_ratio(im:np.uint8, dw:int, dh:int)->tuple(int,int):
    im_w = np.array(im).shape[1]
    im_h = np.array(im).shape[0]
    ir = float(im_w / im_h)
    if im_w<im_h:
        return (int(abs(dw*ir)),int(abs(dh)))
    elif im_w>im_h:
        return (int(abs(dw)),int(abs(dh//ir)))
    elif im_w==im_h:
        return (dw,dh)


def f_split(f: str) -> list:
    return list([f[:len(f)-(f[::-1].find(chr(47))):],
                f[len(f)-(f[::-1].find(chr(47))):(len(f))-1-len(f[-(f[::-1].find(".")):])],
                f[-(f[::-1].find(".")):],]
                for f in[f.replace(chr(92),chr(47))])[0]

__SB__ = lambda t,d:tqdm(
    total=t, desc=d,bar_format='{desc}: {percentage:3.0f}%|'+
    '| {n_fmt}/{total_fmt} [elapsed: {elapsed} / Remaining: {remaining}] '+
    '{rate_fmt}{postfix}]')

_f = lambda f:[
    str(f)[: len(f)-(str(f)[::-1].find("/")):].lower(),
    str(f)[len(f)-(str(f)[::-1].find("/")):(len(f))-1-len(f[-(str(f)[::-1].find(".")):])],
    str(f)[-(str(f)[::-1].find(".")):].lower(),]

_imr = lambda x: (
    int(abs(x[2]*(x[0]/x[1]))),
    int(abs(x[3]))) if x[0]<x[1] \
        else (int(abs(x[2])),
    int(abs(x[2]//(x[0]/x[1])))) if x[0]>x[1] \
        else (x[2],
    x[3])

_chk_bgra=lambda i: np.uint8(i[::,::,:-1:]) if np.uint8(i).shape[2]==4 else np.uint8(i)

_GET_LIST_ = lambda fp,exts: [
    fp+f for f in os.listdir(fp[:-1:]) 
    if os.path.isfile(fp+f) 
    and str(f[-(f[::-1].find('.')):]).lower() in exts]

_ffn = lambda s: str(s).replace(chr(92),chr(47)).replace(chr(34),'') if str(s)[-1]==chr(47) else str(s).replace(chr(92),chr(47)).replace(chr(34),'')+chr(47) 

def _unique(l):
    s = set()
    n = 0
    for x in l:
        if x not in s:
            s.add(x)
            l[n] = x
            n += 1
    del l[n:]
    return l


def read_metadata_from_safetensors(filename):
    import json

    with open(filename, mode="rb") as file:
        metadata_len = file.read(8)
        metadata_len = int.from_bytes(metadata_len, "little")
        json_start = file.read(2)

        assert metadata_len > 2 and json_start in (b'{"', b"{'"), f"{filename} is not a safetensors file"
        json_data = json_start + file.read(metadata_len-2)
        json_obj = json.loads(json_data)

        res = {}
        for k, v in json_obj.get("__metadata__", {}).items():
            res[k] = v
            if isinstance(v, str) and v[0:1] == '{':
                try:
                    res[k] = json.loads(v)
                except Exception:
                    pass

        return res
    
def get_metadata_skeleton():
    metadata = {
        "id": "",
        "modelId": "",
        "name": "",
        "trainedWords": [],
        "baseModel": "Unknown",
        "description": "",
        "model": {
            "name": "",
            "type": "",
            "nsfw": "",
            "poi": ""
        },
        "files": [
            {
                "name": "",
                "sizeKB": 0,
                "type": "Model",
                "hashes": {
                    "AutoV2": "",
                    "SHA256": ""
                }
            }
        ],
        "tags": [],
        "downloadUrl": "",
        "skeleton_file": True,
        "ss_tag_frequency":{}
    }

    return metadata

def dummy_model_info(path):
    model_info = get_metadata_skeleton()
    filename = os.path.basename(path)
    filesize = os.path.getsize(path) // 1024
    model_metadata = model_info["model"]
    file_metadata = model_info["files"][0]
    model_metadata["name"] = filename
    file_metadata["name"] = filename
    file_metadata["sizeKB"] = filesize
    trained_words = model_info["trainedWords"]
    tags = model_info["tags"]

    try:
        file_metadata = read_metadata_from_safetensors(path)
    except AssertionError:
        pass

    tag_frequency = file_metadata.get("ss_tag_frequency", {})

    for trained_word in tag_frequency.keys():
        word = re.sub(r"^\d+_", "", trained_word)
        trained_words.append(word)
        for tag in tag_frequency[trained_word].keys():
            tag = tag.replace(",", "").strip()
            if tag == "":
                continue
            tags.append(tag)
    model_info['ss_tag_frequency'] = file_metadata.get("ss_tag_frequency", {})
    return model_info


class YuNet:
    import cv2
    def __init__(self, modelPath, inputSize=[320, 320], confThreshold=0.6, nmsThreshold=0.3, topK=5000, backendId=0, targetId=0):
        self._modelPath = modelPath
        self._inputSize = tuple(inputSize) # [w, h]
        self._confThreshold = confThreshold
        self._nmsThreshold = nmsThreshold
        self._topK = topK
        self._backendId = backendId
        self._targetId = targetId
        self._model = cv2.FaceDetectorYN.create(
            model=self._modelPath,
            config="",
            input_size=self._inputSize,
            score_threshold=self._confThreshold,
            nms_threshold=self._nmsThreshold,
            top_k=self._topK,
            backend_id=self._backendId,
            target_id=self._targetId)

    @property
    def name(self):
        return self.__class__.__name__

    def setBackendAndTarget(self, backendId, targetId):
        self._backendId = backendId
        self._targetId = targetId
        self._model = cv2.FaceDetectorYN.create(
            model=self._modelPath,
            config="",
            input_size=self._inputSize,
            score_threshold=self._confThreshold,
            nms_threshold=self._nmsThreshold,
            top_k=self._topK,
            backend_id=self._backendId,
            target_id=self._targetId)

    def setInputSize(self, input_size):
        self._model.setInputSize(tuple(input_size))

    def infer(self, image):
        faces = self._model.detect(image)
        return faces[1]


class Upscaler(str, Enum):
    none = "None"
    Lanczos = "Lanczos"
    Nearest = "Nearest"
    LDSR = "LDSR"
    BSRGAN = "BSRGAN"
    ESRGAN_4x = "ESRGAN_4x"
    R_ESRGAN_General_4xV3 = "R-ESRGAN General 4xV3"
    ScuNET_GAN = "ScuNET GAN"
    ScuNET_PSNR = "ScuNET PSNR"
    SwinIR_4x = "SwinIR 4x"


class HiResUpscaler(str, Enum):
    none = "None"
    Latent = "Latent"
    LatentAntialiased = "Latent (antialiased)"
    LatentBicubic = "Latent (bicubic)"
    LatentBicubicAntialiased = "Latent (bicubic antialiased)"
    LatentNearest = "Latent (nearist)"
    LatentNearestExact = "Latent (nearist-exact)"
    Lanczos = "Lanczos"
    Nearest = "Nearest"
    ESRGAN_4x = "ESRGAN_4x"
    LDSR = "LDSR"
    ScuNET_GAN = "ScuNET GAN"
    ScuNET_PSNR = "ScuNET PSNR"
    SwinIR_4x = "SwinIR 4x"


@dataclass
class sd:
    api : webuiapi.WebUIApi = None
    alwayson_scripts = {}
    batch_size : int = 1
    bsrgan : str = "BSRGAN"
    cfg_scale : float = 7.0
    codeformer_visibility : float = 0
    codeformer_weight : float = 0
    controlnet_units = []
    denoising_strength : float = 0.29
    do_not_save_grid : bool = False
    do_not_save_samples : bool = False
    enable_hr : bool = False
    esrgan_4x : str = "ESRGAN_4x"
    eta : float = 1.0
    extras_upscaler_2_visibility : float = 0
    firstphase_height : int = 0
    firstphase_width : int = 0
    gfpgan_visibility : float = 0
    height : int = 512
    hr_resize_x : int = 0
    hr_resize_y : int = 0
    hr_scale : float = 2
    hr_second_pass_steps : int = 22
    hr_upscaler : str = "None"
    hr_upscaler_index = []
    image : Image.Image = None
    image_cfg_scale : float = 1.5
    images = []
    include_init_images : bool = False
    initial_noise_multiplier : float = 1
    inpaint_full_res : bool = True
    inpaint_full_res_padding : int = 0
    inpainting_fill : int = 0
    inpainting_mask_invert : int = 0
    lanczos : str = "Lanczos"
    latent : str = "Latent"
    latentantialiased : str = "Latent (antialiased)"
    latentbicubic : str = "Latent (bicubic)"
    latentbicubicantialiased : str = "Latent (bicubic antialiased)"
    latentnearest : str = "Latent (nearist)"
    latentnearestexact : str = "Latent (nearist-exact)"
    ldsr : str = "LDSR"
    mask_blur : int = 4
    mask_image : Image = None
    model_index = {}
    n_iter : int = 1
    name_list = []
    nearest : str = "Nearest"
    negative_prompt : str = ""
    none : str = "None"
    override_settings = {}
    override_settings_restore_afterwards : bool = True
    T2I_PROMPT : str = ""
    I2I_PROMPT : str = ""
    prompt : str = ""
    r_esrgan_general_4xv3 : str = "R-ESRGAN General 4xV3"
    settings_info = ""
    resize_mode : int = 0
    restore_faces : bool = False
    s_churn : float = 0
    s_noise : float = 1
    s_tmax : float = 0
    s_tmin : float = 0
    sampler_dicts = []
    sampler_index = []
    sampler_list = []
    sampler_name : str = "Euler a"
    save_images : bool = False
    script_args = []
    script_name : str = ""
    scunet_gan : str = "ScuNET GAN"
    scunet_psnr : str = "ScuNET PSNR"
    seed : int = -1
    seed_resize_from_h : int = 0
    seed_resize_from_w : int = 0
    send_images : bool = True
    show_extras_results : bool = True
    steps : int = 33
    styles = []
    subseed : int = -1
    subseed_strength : float = 0
    swinir_4x : str = "SwinIR 4x"
    tiling : bool = False
    upscale_first : bool = False
    upscaler_1 : str = None
    upscaler_2 : str = None
    upscaling_crop : bool = True
    upscaling_resize : int = 2
    upscaling_resize_h : int = 512
    upscaling_resize_w : int = 512
    use_async : bool = False
    use_deprecated_controlnet : bool = False
    width : int = 512
    lw_kern  =  np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    hw_kern = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    mh_kern = np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]])
    _model_id_ep = lambda x: str(f'https://civitai.com/api/v1/models/{x}')
    _model_vers_id_ep = lambda x: str(f'https://civitai.com/api/v1/model-versions/{x}')
    _model_by_hash_ep = lambda x: str(f'https://civitai.com/api/v1/model-versions/by-hash/{x}')
    _ld = lambda x: { k:v for k, v in dict(x).items() }
    _lh = lambda x: list([dict(x)[k]['url'] for k, v in dict(x).items()])
    now = lambda: datetime.strftime(datetime.now(), r"%Y%m%d%H%M%S")
    yyyymmddhhmmssffffff = lambda: str(datetime.strftime(datetime.now(), r"%Y%m%d%H%M%S%f"))
    yyyymmdd = lambda: str(datetime.strftime(datetime.now(), r"%Y%m%d"))
    _0x0C = lambda x: str(''.join(l for l in [x for x in str(x).replace(chr(32),chr(95))] if l in set('ABCDEF0123456789')))
    fn_rm = lambda s: str(''.join(l for l in [x for x in str(s).replace(chr(32),chr(95))] if l in set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.')))
    np_k2_ord_srt = lambda npl: [ 
                        x for x in [np.array(c)[0:].reshape(-1) for c in [ np.array(f[np.argsort(g)]).astype('str').tolist()
                        for f,g in [np.unique(np.array([[str(x[0]).split(chr(32))
                        for x in sorted([[s,npl.count(s)] 
                        for s in [ np.array(y[np.argsort(z)]).astype('str').tolist()[0]
                        for y,z in [np.unique(np.array(npl),return_index=True)]]],
                        key=lambda x: x[1], reverse=False)]]),return_index=True)]]]]
    rm_sp = lambda s: str(re.sub('\s+', ' ', (re.sub('_', '', (re.sub('[^a-zA-z0-9\s]', '', unidecode(s)))))).strip().lower())
    rm_aw = lambda s: "".join(str(s).split())
    rm_sw = lambda s: str(' '.join(w for w in word_tokenize(s) if w not in sw))
    rm_wh = lambda s: " ".join(str(s).split())
    rm_pu = lambda s: str(s).translate(str.maketrans('', '', string.punctuation))

    def __post_init__(self):
        self.alwayson_scripts = sd.alwayson_scripts
        self.api = sd.api
        self.backend_target_pairs = sd.backend_target_pairs
        self.batch_size = sd.batch_size
        self.bsrgan = sd.bsrgan
        self.cfg_scale = sd.cfg_scale
        self.codeformer_visibility = sd.codeformer_visibility
        self.codeformer_weight = sd.codeformer_weight
        self.controlnet_units = sd.controlnet_units
        self.denoising_strength = sd.denoising_strength
        self.do_not_save_grid = sd.do_not_save_grid
        self.do_not_save_samples = sd.do_not_save_samples
        self.enable_hr = sd.enable_hr
        self.esrgan_4x = sd.esrgan_4x
        self.eta = sd.eta
        self.extras_upscaler_2_visibility = sd.extras_upscaler_2_visibility
        self.firstphase_height = sd.firstphase_height
        self.firstphase_width = sd.firstphase_width
        self.gfpgan_visibility = sd.gfpgan_visibility
        self.height = sd.height
        self.hr_resize_x = sd.hr_resize_x
        self.hr_resize_y = sd.hr_resize_y
        self.hr_scale = sd.hr_scale
        self.hr_second_pass_steps = sd.hr_second_pass_steps
        self.hr_upscaler = sd.hr_upscaler
        self.image = sd.image
        self.image_cfg_scale = sd.image_cfg_scale
        self.images = sd.images
        self.images = sd.images
        self.include_init_images = sd.include_init_images
        self.initial_noise_multiplier = sd.initial_noise_multiplier
        self.inpaint_full_res = sd.inpaint_full_res
        self.inpaint_full_res_padding = sd.inpaint_full_res_padding
        self.inpainting_fill = sd.inpainting_fill
        self.inpainting_mask_invert = sd.inpainting_mask_invert
        self.lanczos = sd.lanczos
        self.latent = sd.latent
        self.latentantialiased = sd.latentantialiased
        self.latentbicubic = sd.latentbicubic
        self.latentbicubicantialiased = sd.latentbicubicantialiased
        self.latentnearest = sd.latentnearest
        self.latentnearestexact = sd.latentnearestexact
        self.ldsr = sd.ldsr
        self.mask_blur = sd.mask_blur
        self.mask_image = sd.mask_image
        self.model_index = sd.model_index
        self.n_iter = sd.n_iter
        self.name_list = sd.name_list
        self.nearest = sd.nearest
        self.negative_prompt = sd.negative_prompt
        self.none = sd.none
        self.none = sd.none
        self.override_settings = sd.override_settings
        self.override_settings_restore_afterwards = sd.override_settings_restore_afterwards
        self.T2I_PROMPT = sd.T2I_PROMPT
        self.I2I_PROMPT = sd.I2I_PROMPT
        self.prompt = sd.prompt
        self.r_esrgan_general_4xv3 = sd.r_esrgan_general_4xv3
        self.resize_mode = sd.resize_mode
        self.restore_faces = sd.restore_faces
        self.s_churn = sd.s_churn
        self.s_noise = sd.s_noise
        self.s_tmax = sd.s_tmax
        self.s_tmin = sd.s_tmin
        self.sampler_index = sd.sampler_index
        self.sampler_list = sd.sampler_list
        self.sampler_name = sd.sampler_name
        self.save_images = sd.save_images
        self.script_args = sd.script_args
        self.script_name = sd.script_name
        self.scunet_gan = sd.scunet_gan
        self.scunet_psnr = sd.scunet_psnr
        self.seed = sd.seed
        self.seed_resize_from_h = sd.seed_resize_from_h
        self.seed_resize_from_w = sd.seed_resize_from_w
        self.send_images = sd.send_images
        self.show_extras_results = sd.show_extras_results
        self.steps = sd.steps
        self.styles = sd.styles
        self.subseed = sd.subseed
        self.subseed_strength = sd.subseed_strength
        self.swinir_4x = sd.swinir_4x
        self.tiling = sd.tiling
        self.upscale_first = sd.upscale_first
        self.upscaler_1 = sd.upscaler_1
        self.upscaler_2 = sd.upscaler_2
        self.upscaling_crop = sd.upscaling_crop
        self.upscaling_resize = sd.upscaling_resize
        self.upscaling_resize_h = sd.upscaling_resize_h
        self.upscaling_resize_w = sd.upscaling_resize_w
        self.use_async = sd.use_async
        self.use_deprecated_controlnet = sd.use_deprecated_controlnet
        self.width = sd.width
        self.lw_kern  = sd.lw_kern
        self.hw_kern = sd.hw_kern
        self.mh_kern = sd.mh_kern
        super().__setattr__('attr_name', self)

sd.api = webuiapi.WebUIApi(
            host="127.0.0.1",
            port=7860,
            baseurl=None,
            sampler="Euler a",
            steps=33,
            use_https=False,
            username=None,
            password=None,    
    )

sampler_list = [
    'DPM++ 2M Karras','DPM++ SDE Karras','DPM++ 2M SDE Exponential','DPM++ 2M SDE Karras',
    'Euler a','Euler','LMS','Heun','DPM2','DPM2 a','DPM++ 2S a','DPM++ 2M','DPM++ SDE',
    'DPM++ 2M SDE','DPM++ 2M SDE Heun','DPM++ 2M SDE Heun Karras','DPM++ 2M SDE Heun Exponential',
    'DPM++ 3M SDE','DPM++ 3M SDE Karras','DPM++ 3M SDE Exponential','DPM fast','DPM adaptive',
    'LMS Karras','DPM2 Karras','DPM2 a Karras','DPM++ 2S a Karras','Restart','DDIM','PLMS','UniPC',
]

sd.hr_upscaler_index = [          
                    "Latent",
                    "Latent (antialiased)",
                    "Latent (bicubic)",
                    "Latent (bicubic antialiased)",
                    "Latent (nearest)",
                    "Latent (nearest-exact)",
                    "None",
                    "Lanczos",
                    "Nearest",
                    "4x_fatal_Anime_500000_G",
                    "ESRGAN_4x",
                    "LDSR",
                    "R-ESRGAN 4x+",
                    "R-ESRGAN 4x+ Anime6B",
                    "ScuNET GAN",
                    "ScuNET PSNR",
                    "SwinIR_4x"
]


def IMG2IMG() -> list[Image.Image]:
    
    return sd.api.img2img(
        steps=sd.steps,
        images = sd.images,
        resize_mode = sd.resize_mode,
        denoising_strength = sd.denoising_strength,
        image_cfg_scale = sd.image_cfg_scale,
        mask_image = sd.mask_image,
        mask_blur = sd.mask_blur,
        inpainting_fill = sd.inpainting_fill,
        inpaint_full_res = sd.inpaint_full_res,
        inpaint_full_res_padding = sd.inpaint_full_res_padding,
        inpainting_mask_invert = sd.inpainting_mask_invert,
        initial_noise_multiplier = sd.initial_noise_multiplier,
        prompt = sd.prompt,
        styles = sd.styles,
        seed = sd.seed,
        subseed = sd.subseed,
        subseed_strength = sd.subseed_strength,
        seed_resize_from_h = sd.seed_resize_from_h,
        seed_resize_from_w = sd.seed_resize_from_w,
        sampler_name = sd.sampler_name,
        batch_size = sd.batch_size,
        n_iter = sd.n_iter,
        cfg_scale = sd.cfg_scale,
        width = sd.width,
        height = sd.height,
        restore_faces = sd.restore_faces,
        tiling = sd.tiling,
        do_not_save_samples = sd.do_not_save_samples,
        do_not_save_grid = sd.do_not_save_grid,
        negative_prompt = sd.negative_prompt,
        eta = sd.eta,
        s_churn = sd.s_churn,
        s_tmax = sd.s_tmax,
        s_tmin = sd.s_tmin,
        s_noise = sd.s_noise,
        override_settings = sd.override_settings,
        override_settings_restore_afterwards = sd.override_settings_restore_afterwards,
        script_args = sd.script_args,
        include_init_images = sd.include_init_images,
        script_name = sd.script_name,
        send_images = sd.send_images,
        save_images = sd.save_images,
        alwayson_scripts = sd.alwayson_scripts,
        controlnet_units = sd.controlnet_units,
        use_deprecated_controlnet = sd.use_deprecated_controlnet,
        use_async = sd.use_async,
    )

def ESI():
    return sd.api.extra_single_image(
        image = sd.image,
        resize_mode = sd.resize_mode,
        show_extras_results = sd.show_extras_results,
        gfpgan_visibility = sd.gfpgan_visibility,
        codeformer_visibility = sd.codeformer_visibility,
        codeformer_weight = sd.codeformer_weight,
        upscaling_resize = sd.upscaling_resize,
        upscaling_resize_w = sd.upscaling_resize_w,
        upscaling_resize_h = sd.upscaling_resize_h,
        upscaling_crop = sd.upscaling_crop,
        upscaler_1 = sd.upscaler_1,
        upscaler_2 = sd.upscaler_2,
        extras_upscaler_2_visibility = sd.extras_upscaler_2_visibility,
        upscale_first = sd.upscale_first,
        use_async = sd.use_async,
    )


def EBI():
    return sd.api.extra_batch_images(
        images = sd.images,
        name_list = sd.name_list,
        resize_mode = sd.resize_mode,
        show_extras_results = sd.show_extras_results,
        gfpgan_visibility = sd.gfpgan_visibility,
        codeformer_visibility = sd.codeformer_visibility,
        codeformer_weight = sd.codeformer_weight,
        upscaling_resize = sd.upscaling_resize,
        upscaling_resize_w = sd.upscaling_resize_w,
        upscaling_resize_h = sd.upscaling_resize_h,
        upscaling_crop = sd.upscaling_crop,
        upscaler_1 = sd.upscaler_1,
        upscaler_2 = sd.upscaler_2,
        extras_upscaler_2_visibility = sd.extras_upscaler_2_visibility,
        upscale_first = sd.upscale_first,
        use_async = sd.use_async,

    )


def TXT2IMG() -> Image.Image:
    return sd.api.txt2img(
        enable_hr = sd.enable_hr,
        denoising_strength = sd.denoising_strength,
        firstphase_width = sd.firstphase_width,
        firstphase_height = sd.firstphase_height,
        hr_scale = sd.hr_scale,
        hr_upscaler = sd.hr_upscaler,
        hr_second_pass_steps = sd.hr_second_pass_steps,
        hr_resize_x = sd.hr_resize_x,
        hr_resize_y = sd.hr_resize_y,
        prompt = sd.prompt,
        styles = sd.styles,
        seed = sd.seed,
        subseed = sd.subseed,
        subseed_strength = sd.subseed_strength,
        seed_resize_from_h = sd.seed_resize_from_h,
        seed_resize_from_w = sd.seed_resize_from_w,
        sampler_name = sd.sampler_name,
        batch_size = sd.batch_size,
        n_iter = sd.n_iter,
        steps = sd.steps,
        cfg_scale = sd.cfg_scale,
        width = sd.width,
        height = sd.height,
        restore_faces = sd.restore_faces,
        tiling = sd.tiling,
        do_not_save_samples = sd.do_not_save_samples,
        do_not_save_grid = sd.do_not_save_grid,
        negative_prompt = sd.negative_prompt,
        eta = sd.eta,
        s_churn = sd.s_churn,
        s_tmax = sd.s_tmax,
        s_tmin = sd.s_tmin,
        s_noise = sd.s_noise,
        override_settings = sd.override_settings,
        override_settings_restore_afterwards = sd.override_settings_restore_afterwards,
        script_args = sd.script_args,
        script_name = sd.script_name,
        send_images = sd.send_images,
        save_images = sd.save_images,
        alwayson_scripts = sd.alwayson_scripts,
        sampler_index = "Euler a",
        use_deprecated_controlnet = False,
        use_async = False
    )


def _js_sequence(seq:list[float]):
    js=[
        f"CUSTOM_WAIFU_name",
        f"<lora:CUSTOM_WAIFU_LORA-000032:{str(seq[0])}>",
        f"<lora:CUSTOM_WAIFU_LORA202403-000015:{str(seq[0])}>",
        f"<lora:last-step00002200_CUSTOM_WAIFU_LORA:{str(seq[0])}>"
    ]
    return js


def gen_proc(prompt_text:str,js_seq:list[float]):

    def im_ratio(im:np.uint8, dw:int, dh:int)->tuple:
        im_w = im.shape[1]
        im_h = im.shape[0]
        ir = float(im_w / im_h)
        if im_w<im_h:
            return (int(abs(dw*ir)),int(abs(dh)))
        elif im_w>im_h:
            return (int(abs(dw)),int(abs(dh//ir)))
        elif im_w==im_h:
            return (dw,dh)
        
    YN_MODEL = YuNet(modelPath=YUNET_ONNX_PATH,
                    inputSize=[320, 320],
                    confThreshold=0.8,
                    nmsThreshold=0.3,
                    topK=5000,
                    backendId=backend_id,
                    targetId=target_id)

    def TXT2IMG() -> Image.Image:
        return sd.api.txt2img(
            enable_hr = sd.enable_hr,
            denoising_strength = sd.denoising_strength,
            firstphase_width = sd.firstphase_width,
            firstphase_height = sd.firstphase_height,
            hr_scale = sd.hr_scale,
            hr_upscaler = sd.hr_upscaler,
            hr_second_pass_steps = sd.hr_second_pass_steps,
            hr_resize_x = sd.hr_resize_x,
            hr_resize_y = sd.hr_resize_y,
            prompt = sd.prompt,
            styles = sd.styles,
            seed = sd.seed,
            subseed = sd.subseed,
            subseed_strength = sd.subseed_strength,
            seed_resize_from_h = sd.seed_resize_from_h,
            seed_resize_from_w = sd.seed_resize_from_w,
            sampler_name = sd.sampler_name,
            batch_size = sd.batch_size,
            n_iter = sd.n_iter,
            steps = 21,
            cfg_scale = sd.cfg_scale,
            width = sd.width,
            height = sd.height,
            restore_faces = sd.restore_faces,
            tiling = sd.tiling,
            do_not_save_samples = sd.do_not_save_samples,
            do_not_save_grid = sd.do_not_save_grid,
            negative_prompt = sd.negative_prompt,
            eta = sd.eta,
            s_churn = sd.s_churn,
            s_tmax = sd.s_tmax,
            s_tmin = sd.s_tmin,
            s_noise = sd.s_noise,
            override_settings = sd.override_settings,
            override_settings_restore_afterwards = sd.override_settings_restore_afterwards,
            script_args = sd.script_args,
            script_name = sd.script_name,
            send_images = sd.send_images,
            save_images = sd.save_images,
            alwayson_scripts = sd.alwayson_scripts,
            sampler_index = sd.sampler_index,
            use_deprecated_controlnet = False,
            use_async = False
        )
    sd.sampler_index = 'DPM++ 2M'
    strnow = str(datetime.strftime(datetime.now(), r"%Y%m%d%H%M%S%f"))
    imghw = [512,576,640,704,768]
    img_hw_l = lambda: imghw[randint(0,4)]
    sd.width = img_hw_l()
    sd.height = img_hw_l()
    sd.steps = 33
    _prompt = sd.prompt
    sd.prompt = _prompt + chr(44).join(j for j in _js_sequence(js_seq[0]))
    image = np.array(TXT2IMG().image).astype('uint8')[:,:,::-1].copy()
    cv2.imwrite(str(f"{SAVE_DIR}grids/{strnow}_00.jpg"),image)
    rs_w, rs_h = im_ratio(np.uint8(image),1664,1664)
    fc_img = cv2.resize(src=image,dsize=(rs_w, rs_h),interpolation=cv2.INTER_LANCZOS4)
    YN_MODEL = YuNet(modelPath=YUNET_ONNX_PATH,
                        inputSize=fc_img.shape[:2][::-1],
                        confThreshold=0.8,
                        nmsThreshold=0.3,
                        topK=5000,
                        backendId=backend_id,
                        targetId=target_id)
    YN_MODEL.setInputSize([fc_img.shape[1], fc_img.shape[0]])
    fd_res = YN_MODEL.infer(fc_img)
    if fd_res is not None:
        for res in fd_res:
            try:
                    bbox = res[0:4].astype(np.int32)
                    x,y,w,h = (bbox[0], bbox[1], bbox[2], bbox[3])
                    fc = fc_img[y:y+h, x:x+w]
                    fc_w, fc_h = (fc.shape[1],fc.shape[0])
                    imr_w,imr_h = im_ratio(np.uint8(fc), 640,640)
                    rs_fc = cv2.resize(fc.copy(),(imr_w,imr_h),interpolation=cv2.INTER_LANCZOS4)
                    sd.sampler_name = "Euler a"
                    sd.image_cfg_scale = 1.0
                    sd.denoising_strength = 0.39
                    sd.width = imr_w
                    sd.height = imr_h
                    sd.steps = 5
                    sd.eta = 1.0
                    sd.resize_mode = 0
                    sd.images=[Image.fromarray(np.array(rs_fc).astype('uint8')[:,:,::-1])]
                    prompt_list:list[str] = _unique([str(x).strip() for x in _prompt.split(chr(44))])
                    _pr = chr(44).join(t for t in prompt_list if str(t).find('<')==-1)
                    sd.prompt = _pr + chr(44).join(j for j in _js_sequence(js_seq[1]))
                    ret_i2i = IMG2IMG()
                    cv2.imwrite(str(f"{SAVE_DIR}grids/{strnow}_01.jpg"),np.array(ret_i2i.image).astype('uint8')[:,:,::-1])
                    img_p2 = cv2.resize(np.array(ret_i2i.image).astype('uint8'),(imr_w,imr_h),interpolation=cv2.INTER_LANCZOS4)
                    sd.images=[Image.fromarray(np.array(img_p2).astype('uint8'))]    
                    sd.image_cfg_scale = 1.0
                    sd.denoising_strength = 0.29
                    sd.width = imr_w
                    sd.height = imr_h
                    sd.steps = 5
                    sd.eta = 1.0
                    sd.resize_mode = 0
                    prompt_list:list[str] = _unique([str(x).strip() for x in _prompt.split(chr(44))])
                    _pr = chr(44).join(t for t in prompt_list if str(t).find('<')==-1)
                    sd.prompt = _pr + chr(44).join(j for j in _js_sequence(js_seq[2]))
                    ret2_i2i = IMG2IMG()
                    cv2.imwrite(str(f"{SAVE_DIR}grids/{strnow}_02.jpg"),np.array(ret2_i2i.image).astype('uint8')[:,:,::-1])
                    fc_img[y:y+h, x:x+w] = cv2.resize(np.array(ret2_i2i.image).astype('uint8')[:,:,::-1],(fc_w, fc_h),interpolation=cv2.INTER_LANCZOS4)
                    cv2.imwrite(str(f"{SAVE_DIR}grids/{strnow}_03.jpg"),fc_img)
            except Exception:
                pass
    rf_w, rf_h = (int(fc_img.shape[1]),int(fc_img.shape[0]))
    sd.images = [Image.fromarray(np.uint8(np.array(fc_img).astype('uint8')[:,:,::-1].copy()))]
    sd.image_cfg_scale = 1
    sd.resize_mode = 0
    sd.sampler_name = "Euler a"
    sd.width, sd.height = rs_w, rs_h
    sd.denoising_strength = 0.19
    sd.steps = 5
    sd.eta = 1.0
    sd.prompt = _prompt + chr(44).join(j for j in _js_sequence(js_seq[3]))
    i2i_00 = IMG2IMG()
    cv2.imwrite(str(f"{SAVE_DIR}grids/{strnow}_04.jpg"),np.array(i2i_00.image).astype('uint8')[:,:,::-1])
    sd.images = [i2i_00.image]
    sd.width, sd.height = rs_w, rs_h
    sd.denoising_strength = 0.19
    sd.steps = 5
    sd.eta = 1.0
    sd.prompt = _prompt + chr(44).join(j for j in _js_sequence(js_seq[4]))
    i2i_01 = IMG2IMG()
    cv2.imwrite(str(f"{SAVE_DIR}grids/{strnow}_05.jpg"),np.array(i2i_01.image).astype('uint8')[:,:,::-1])
    image = np.array(i2i_01.image).astype('uint8')[:,:,::-1].copy()
    YN_MODEL = YuNet(modelPath=YUNET_ONNX_PATH,
                        inputSize=image.shape[:2][::-1],
                        confThreshold=0.8,
                        nmsThreshold=0.3,
                        topK=5000,
                        backendId=backend_id,
                        targetId=target_id)
    YN_MODEL.setInputSize([image.shape[1], image.shape[0]])
    fd_res = YN_MODEL.infer(image)
    if fd_res is not None:
        for res in fd_res:
                bbox = res[0:4].astype(np.int32)
                x,y,w,h = (bbox[0], bbox[1], bbox[2], bbox[3])
                fc = image[y:y+h, x:x+w]
                fc_w, fc_h = (fc.shape[1],fc.shape[0])
                imr_w,imr_h = im_ratio(np.uint8(fc),640,640)
                rs_fc = cv2.resize(fc.copy(),(imr_w,imr_h),interpolation=cv2.INTER_LANCZOS4)
                sd.sampler_index = 'DPM++ 2M'
                sd.image_cfg_scale = 1.0
                sd.denoising_strength = 0.29
                sd.width = imr_w
                sd.height = imr_h
                sd.steps = 5
                sd.eta = 1.0
                sd.resize_mode = 0
                sd.images=[Image.fromarray(np.array(rs_fc).astype('uint8')[:,:,::-1])]
                prompt_list:list[str] = _unique([str(x).strip() for x in _prompt.split(chr(44))])
                _pr = chr(44).join(t for t in prompt_list if str(t).find('<')==-1)
                sd.prompt = _prompt + chr(44).join(j for j in _js_sequence(js_seq[5]))
                ret_i2i = IMG2IMG()
                cv2.imwrite(str(f"{SAVE_DIR}grids/{strnow}_06.jpg"),np.array(ret_i2i.image).astype('uint8')[:,:,::-1])
                sd.images=[ret_i2i.image]    
                sd.image_cfg_scale = 1.0
                sd.denoising_strength = 0.29
                sd.width = imr_w
                sd.height = imr_h
                sd.steps = 5
                sd.eta = 1.0
                sd.resize_mode = 0
                prompt_list:list[str] = _unique([str(x).strip() for x in _prompt.split(chr(44))])
                _pr = chr(44).join(t for t in prompt_list if str(t).find('<')==-1)
                sd.prompt = _prompt + chr(44).join(j for j in _js_sequence(js_seq[6]))
                ret2_i2i = IMG2IMG()
                cv2.imwrite(str(f"{SAVE_DIR}grids/{strnow}_07.jpg"),np.array(ret2_i2i.image).astype('uint8')[:,:,::-1])
                image = np.array(image).astype('uint8').copy()
                image[y:y+h, x:x+w] = cv2.resize(np.array(ret2_i2i.image).astype('uint8')[:,:,::-1],(fc_w, fc_h),interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(str(f"{SAVE_DIR}grids/{strnow}_08.jpg"),image)
                sd.images = [Image.fromarray(image[:,:,::-1])]
                sd.sampler_name = "Euler a"
    sd.width, sd.height = rs_w, rs_h
    sd.denoising_strength = 0.21
    sd.steps = 15
    sd.eta = 1.0
    sd.prompt = _prompt + chr(44).join(j for j in _js_sequence(js_seq[7]))
    i2i_02 = IMG2IMG()
    img_fn = str(f"{SAVE_DIR}{strnow}.jpg")
    i2i_02.image.save(fp=img_fn, format="JPEG")
    cv2.imwrite(str(f"{SAVE_DIR}grids/{strnow}_09.jpg"),np.array(i2i_02.image).astype('uint8')[:,:,::-1])
    with open(str(f"{SAVE_DIR}grids/{strnow}.txt"),'wt') as fi:
        for k,v in dict(i2i_02.info).items():
            fi.write(str(f"{k}: {v}{chr(10)}"))


def _lora_info(file:str):
    f = _f(file)
    f_str = f'{f[1]}'
    info = dummy_model_info(file)
    LORAS[f[1]] = {}
    LORAS[f[1]]['lora'] = f_str
    LORAS[f[1]]['tags']  = info['tags']
    LORAS[f[1]]['ss_tag_frequency']  = info['ss_tag_frequency']
    tagf = LORAS[f[1]]['ss_tag_frequency']
    tagsort:list = []
    try: tagsort = sorted([[k,v] for k,v in tagf[[_ for _ in tagf.keys()][0]].items()],key=lambda x:x[1],reverse=True)
    except: pass
    LORAS[f[1]]['tagsort'] = []
    LORAS[f[1]]['tagsort'] = tagsort

def _lora_list_gen():
    file_list = _GET_LIST_(LORA_DIR,['safetensors'])
    with ThreadPoolExecutor(8) as executor:
        status_bar = tqdm(desc=r'Lora_Model_Info')
        futures = [executor.submit(_lora_info, file) for file in file_list]
        for _ in as_completed(futures): status_bar.update(n=1)

def _lora_tags(lora:str):
    try:
        taglist:list=[]
        taglist = LORAS[lora]['tagsort']
        keywords = [x[0] for x in taglist[:10]]
        for keyword in keywords:
            if keyword not in [kw for kw in TAG_DICT.keys()]:
                TAG_DICT[keyword] = []
                TAG_DICT[keyword].append(LORAS[lora]['lora'])
            else:
                TAG_DICT[keyword].append(LORAS[lora]['lora'])
    except Exception as e:
        print(e)
        pass

def _lora_taglist_gen():
    lora_list:list = [k for k in LORAS.keys()]
    print(len(lora_list))
    with ThreadPoolExecutor(8) as executor:
        status_bar = tqdm(desc=r'Lora_TagsList')
        futures = [executor.submit(_lora_tags, k) for k in lora_list]
        for _ in as_completed(futures): 
            status_bar.update(n=1)




def txt_proc():
    txt_list =  _GET_LIST_(_ffn(fr"T:\wat\test/"),['txt'])#_GET_LIST_(_ffn(fr"T:\wat\unzip/"),['txt']) + _GET_LIST_(_ffn(fr"T:\wat\unzip2/"),['txt']) + _GET_LIST_(_ffn(fr"T:\wat\test2/"),['txt'])
    print(len(txt_list))
    for file in txt_list:
        txt_data:str=''
        with open(file,'rt')as fi:
            txt_data = fi.read().lower()
        for rm in remove_list:
            txt_data.replace(rm,'')
        for k,v in replace_list.items():
            txt_data.replace(k,v)
        if len(txt_data)!='':
            pl.append(txt_data)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    _lora_list_gen()
    _lora_taglist_gen()
    txt_proc()


replace_list:dict = {
"blue eyes":"brown eyes",
"black eyes":"brown eyes",
"closed eyes":"brown eyes",
"green eyes":"brown eyes",
"grey eyes":"brown eyes",
"grey hair":"brown eyes",
"orange eyes":"brown eyes",
"pink eyes":"brown eyes",
"purple eyes":"brown eyes",
"red eyes":"brown eyes",
"white eyes":"brown eyes",
"yellow eyes":"brown eyes",
"huge breasts":"natural breasts, breasts apart",
"flat chest":"natural breasts, breasts apart",
}

remove_list:list[str] = [
"eyes visible through hair",
"english text",
"half-closed eyes",
"signature",
"simple background ",
"web address",
"watermark",
"patreon logo",
"patreon username",
"official alternate costume",
"mole",
"mole above mouth",
"mole on breast",
"mole under eye",
"mole under mouth",
"facial mark",
"dutch angle",
"loli",
"dirty",
"blurry",
"blurry background",
"artist name",
"closed eyes",
"abs",
"age difference",
"alternate breast size",
"alternate costume",
"alternate hair length",
"alternate hairstyle",
"one eye closed",
"used condom",
"symbol-shaped pupils",
"sleep molestation",
"(",
")",
"pillarboxed",
]


sd.negative_prompt = "tongue, bad-hands-5,  (EasyNegative:0.8), (worst quality, low quality:1.2),(watermark, white border, monochrome, greyscale, text, speech bubble, artist name, signature),  epiCPhotoGasm-softPhoto-neg,  FastNegativeV2 nipplegeddon" 

EXTRA_KW = "chubby, natural breasts, thick thighs, fat, big belly,"

if __name__ == '__main__':
    multiprocessing.freeze_support()
    shuffle(pl)
    KW_LIST = [k for k in TAG_DICT.keys()]
    repl_keys = [k for k in replace_list.keys()]
    for prompt in tqdm(pl):
        for i in range(3):
            try:
                prompt_loras:list = []
                unq_list:list=[]
                w_loras:list=[]
                cnt_list:list = []
                prompt_list:list[str]=[]
                prompt_list = _unique([replace_list[str(x).lower().strip()] if str(x).strip().lower() in repl_keys else str(x).lower().strip() for x in str(EXTRA_KW+prompt).split(chr(44)) if str(x).strip().lower() not in remove_list])
                shuffle(prompt_list)
                prompt_list = prompt_list[:-5:]
                for w in prompt_list:
                    if w in KW_LIST:
                        w_list:list=[]
                        w_list = [lora for lora in TAG_DICT[w]]
                        len(list(map(lambda x: w_loras.append(x),[x for x in w_list])))
                cnt_list = _unique([ t[0] for t in sorted([ [x,w_loras.count(x)] for x in w_loras if w_loras.count(x)>4 ], key = lambda x: x[1], reverse = True)[:50]])
                lora_strength = 0.25
                len(list(map(lambda x: prompt_loras.append(x),[x for x in cnt_list])))
                prompt_loras = [str('<' + wk + f':{lora_strength}:{lora_strength}>') for wk in cnt_list]
                shuffle(prompt_loras)
                sd.prompt = chr(44).join(t for t in prompt_list) + chr(44).join(t for t in prompt_loras[:10])
                print(sd.prompt)
                _seq = [
                    [0.15, 0.15, 0.15],
                    [0.45, 0.65, 0.35],#face
                    [0.45, 0.65, 0.35],#face
                    [0.25, 0.25, 0.25],
                    [0.35, 0.35, 0.35],
                    [0.45, 0.65, 0.35],#face
                    [0.45, 0.65, 0.35],#face
                    [0.45, 0.65, 0.45],
                ]
                gen_proc(sd.prompt,js_seq=_seq)
            except: pass







































































