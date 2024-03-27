from __future__ import annotations
import codecs
import copy
import cv2
import json
import multiprocessing
import nltk
import numpy as np
import os
import io
import re
import requests
import string
import unicodedata
import webuiapi
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from PIL import Image, ImageDraw2, ImageFile
from random import randint, shuffle
from tqdm import tqdm
from traceback_with_variables import (
    Format,
    ColorSchemes,
    default_format as defaults,
    prints_exc,
    printing_exc,
)
from types import SimpleNamespace
from unidecode import unidecode


Image.MAX_IMAGE_PIXELS = None
ImageDraw2.Font
ImageFile.LOAD_TRUNCATED_IMAGES = True

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("tagsets", quiet=True)
sw = set(stopwords.words("english"))


defaults.max_value_str_len = 10000

fmt1 = Format(
    before=3,
    after=1,
    max_value_str_len=100,
    max_exc_str_len=100,
    ellipsis_="...",
    color_scheme=ColorSchemes.synthwave,
    custom_var_printers=[
        (list, lambda v: f"list{v}"),
    ],
)


DL_PATH = (
    __file__[: len(__file__) - str(__file__)[::-1].find(chr(92)) :].replace(
        chr(92), chr(47)
    )
    if __file__.count(chr(92)) > 1
    else __file__[: len(__file__) - str(__file__)[::-1].find(chr(47)) :]
)

print(DL_PATH)

YUNET_ONNX_PATH = ""

if not os.path.isfile(YUNET_ONNX_PATH):
    with requests.Session() as s:
        fb = io.BytesIO(
            s.get(
                "https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/"
                + "face_detection_yunet_2023mar.onnx"
            ).content
        )
        with open(DL_PATH + "face_detection_yunet_2023mar.onnx", "wb") as fi:
            fi.write(fb.read())

assert os.path.isfile(DL_PATH + "face_detection_yunet_2023mar.onnx")

IMG_TYPES = [
    "blp",
    "bmp",
    "dib",
    "bufr",
    "cur",
    "pcx",
    "dcx",
    "dds",
    "ps",
    "eps",
    "fit",
    "fits",
    "fli",
    "flc",
    "ftc",
    "ftu",
    "gbr",
    "gif",
    "grib",
    "h5",
    "hdf",
    "png",
    "apng",
    "jp2",
    "j2k",
    "jpc",
    "jpf",
    "jpx",
    "j2c",
    "icns",
    "ico",
    "im",
    "iim",
    "tif",
    "tiff",
    "jfif",
    "jpe",
    "jpg",
    "jpeg",
    "mpg",
    "mpeg",
    "mpo",
    "msp",
    "palm",
    "pcd",
    "pxr",
    "pbm",
    "pgm",
    "ppm",
    "pnm",
    "psd",
    "bw",
    "rgb",
    "rgba",
    "sgi",
    "ras",
    "tga",
    "icb",
    "vda",
    "vst",
    "webp",
    "wmf",
    "emf",
    "xbm",
    "xpm",
    "nef",
]

LORA_DIR = "/mnt/h/sd/lora/"

FILE_LIST: list[str] = []
LORAS: dict = {}
TAG_DICT: dict = {}
KW_LIST: list[str] = []
COMBI_DICT: dict = {}
WILDCARDS: dict = {}
pl: list[str] = []
K_DICT: dict = {}
KW_DICT: dict = {}
replace_list: dict = {}
remove_list: list[str] = []
prompt_loras: list = []
unq_list: list = []
w_loras: list = []
cnt_list: list = []
prompt_list: list[str] = []
repl_keys = []
KW_LIST = []


BACKEND_TARGET_PAIRS = [
    [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],  # 0
    [cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA],  # 1
    [cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA_FP16],  # 2
    [cv2.dnn.DNN_BACKEND_TIMVX, cv2.dnn.DNN_TARGET_NPU],  # 3
    [cv2.dnn.DNN_BACKEND_CANN, cv2.dnn.DNN_TARGET_NPU],  # 4
]
backend_id = BACKEND_TARGET_PAIRS[0][0]
target_id = BACKEND_TARGET_PAIRS[0][1]

SAVE_DIR = "/mnt/r/sd/outputs/webui_i2i/"


replace_list: dict = {}
remove_list: list[str] = []

replace_list: dict = {"": ""}

# undesirable keywords / strings
remove_list: list[str] = [""]


face_hd = [
    "High Quality",
    "4k",
    "Closeup",
    "Absurdres",
    "Highres",
    "Ultra Detailed",
    "Intricate",
    "Solo",
    "Realistic",
    "Caustics",
    "Absurdres",
    "Eye Focus",
    "(Beautiful)",
    "Studio Quality Post Processing",
    "Close-Up",
    "Photorealistic",
]

img_hd_kw = [
    "High Quality",
    "4k",
    "Absurdres",
    "Highres",
    "Ultra Detailed",
    "Intricate",
]

# WIP - for choosing what keywords to not use as weighted modifiers, need to evaluate with NLTK
un_weight = []

sampler_list = [
    "DPM++ 2M Karras",
    "DPM++ SDE Karras",
    "DPM++ 2M SDE Exponential",
    "DPM++ 2M SDE Karras",
    "Euler a",
    "Euler",
    "LMS",
    "Heun",
    "DPM2",
    "DPM2 a",
    "DPM++ 2S a",
    "DPM++ 2M",
    "DPM++ SDE",
    "DPM++ 2M SDE",
    "DPM++ 2M SDE Heun",
    "DPM++ 2M SDE Heun Karras",
    "DPM++ 2M SDE Heun Exponential",
    "DPM++ 3M SDE",
    "DPM++ 3M SDE Karras",
    "DPM++ 3M SDE Exponential",
    "DPM fast",
    "DPM adaptive",
    "LMS Karras",
    "DPM2 Karras",
    "DPM2 a Karras",
    "DPM++ 2S a Karras",
    "Restart",
    "DDIM",
    "PLMS",
    "UniPC",
]

hr_upscaler_index = [
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
    "SwinIR_4x",
]


class YuNet:
    import cv2

    def __init__(
        self,
        modelPath,
        inputSize=[320, 320],
        confThreshold=0.6,
        nmsThreshold=0.3,
        topK=5000,
        backendId=0,
        targetId=0,
    ):
        self._modelPath = modelPath
        self._inputSize = tuple(inputSize)  # [w, h]
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
            target_id=self._targetId,
        )

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
            target_id=self._targetId,
        )

    def setInputSize(self, input_size):
        self._model.setInputSize(tuple(input_size))

    def infer(self, image):
        faces = self._model.detect(image)
        return faces[1] if faces[1] is not None else None


class Upscaler(str, Enum):
    BSRGAN = "BSRGAN"
    ESRGAN_4x = "ESRGAN_4x"
    Lanczos = "Lanczos"
    LDSR = "LDSR"
    Nearest = "Nearest"
    none = "None"
    R_ESRGAN_General_4xV3 = "R-ESRGAN General 4xV3"
    ScuNET_GAN = "ScuNET GAN"
    ScuNET_PSNR = "ScuNET PSNR"
    SwinIR_4x = "SwinIR 4x"


class HiResUpscaler(str, Enum):
    ESRGAN_4x = "ESRGAN_4x"
    Lanczos = "Lanczos"
    Latent = "Latent"
    LatentAntialiased = "Latent (antialiased)"
    LatentBicubic = "Latent (bicubic)"
    LatentBicubicAntialiased = "Latent (bicubic antialiased)"
    LatentNearest = "Latent (nearist)"
    LatentNearestExact = "Latent (nearist-exact)"
    LDSR = "LDSR"
    Nearest = "Nearest"
    none = "None"
    ScuNET_GAN = "ScuNET GAN"
    ScuNET_PSNR = "ScuNET PSNR"
    SwinIR_4x = "SwinIR 4x"


@dataclass
class sd:
    alwayson_scripts = {}
    api: webuiapi.WebUIApi = None
    batch_gen: int = 1
    batch_size: int = 1
    bsrgan: str = "BSRGAN"
    cfg_scale: float = 7.0
    codeformer_visibility: float = 0
    codeformer_weight: float = 0
    controlnet_units = []
    CUSTOM_MODELS = []
    denoising_strength: float = 0.29
    do_not_save_grid: bool = False
    do_not_save_samples: bool = False
    enable_hr: bool = False
    esrgan_4x: str = "ESRGAN_4x"
    eta: float = 1.0
    EXTRA_KW: str = ""
    extras_upscaler_2_visibility: float = 0
    fd_res: np.array = None
    firstphase_height: int = 0
    firstphase_width: int = 0
    folder: str = ""
    fp_image: np.uint8 = None
    gfpgan_visibility: float = 0.0
    height: int = 512
    hr_resize_x: int = 0
    hr_resize_y: int = 0
    hr_scale: float = 2.0
    hr_second_pass_steps: int = 22
    hr_upscaler: str = "None"
    hr_upscaler_index = []
    I2I_PROMPT: str = ""
    image: Image.Image = None
    image_cfg_scale: float = 1.5
    images = []
    include_init_images: bool = False
    initial_noise_multiplier: float = 1.0
    inpaint_full_res: bool = True
    inpaint_full_res_padding: int = 0
    inpainting_fill: int = 0
    inpainting_mask_invert: int = 0
    lanczos: str = "Lanczos"
    latent: str = "Latent"
    latentantialiased: str = "Latent (antialiased)"
    latentbicubic: str = "Latent (bicubic)"
    latentbicubicantialiased: str = "Latent (bicubic antialiased)"
    latentnearest: str = "Latent (nearist)"
    latentnearestexact: str = "Latent (nearist-exact)"
    ldsr: str = "LDSR"
    mask_blur: int = 4
    mask_image: Image = None
    model_index = {}
    n_iter: int = 1
    name_list = []
    nearest: str = "Nearest"
    negative_prompt: str = ""
    none: str = "None"
    override_settings = {}
    override_settings_restore_afterwards: bool = True
    proc_seq = []
    prompt: str = ""
    r_esrgan_general_4xv3: str = "R-ESRGAN General 4xV3"
    resize_mode: int = 0
    restore_faces: bool = False
    s_churn: float = 0.0
    s_noise: float = 1.0
    s_tmax: float = 0.0
    s_tmin: float = 0.0
    sampler_dicts = []
    sampler_list = []
    sampler_name = []
    sampler_name: str = "Euler a"
    save_images: bool = False
    script_args = []
    script_name: str = ""
    scunet_gan: str = "ScuNET GAN"
    scunet_psnr: str = "ScuNET PSNR"
    seed: int = -1
    seed_resize_from_h: int = 0
    seed_resize_from_w: int = 0
    send_images: bool = True
    seq = []
    settings_info = ""
    show_extras_results: bool = True
    steps: int = 33
    strnow: str = ""
    styles = []
    subseed: int = -1
    subseed_strength: float = 0.0
    swinir_4x: str = "SwinIR 4x"
    T2I_PROMPT: str = ""
    THREADS: int = 8
    tiling: bool = False
    upscale_first: bool = False
    upscaler_1: str = None
    upscaler_2: str = None
    upscaling_crop: bool = True
    upscaling_resize: int = 2
    upscaling_resize_h: int = 512
    upscaling_resize_w: int = 512
    use_async: bool = False
    use_deprecated_controlnet: bool = False
    w_seq = []

    weight_seq = [
        [0.15, 0.15, 0.15],
        [0.45, 0.65, 0.35],  # face
        [0.45, 0.65, 0.35],  # face
        [0.25, 0.25, 0.25],
        [0.35, 0.35, 0.35],
        [0.45, 0.65, 0.35],  # face
        [0.45, 0.65, 0.35],  # face
        [0.45, 0.65, 0.35],
    ]

    width: int = 512
    width: int = 512
    weight_calc: float = 0.0
    lw_kern = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    hw_kern = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    mh_kern = np.array(
        [
            [0, 0, -1, 0, 0],
            [0, -1, -2, -1, 0],
            [-1, -2, 16, -2, -1],
            [0, -1, -2, -1, 0],
            [0, 0, -1, 0, 0],
        ]
    )
    _model_id_ep = lambda x: str(f"https://civitai.com/api/v1/models/{x}")
    _model_vers_id_ep = lambda x: str(f"https://civitai.com/api/v1/model-versions/{x}")
    _model_by_hash_ep = lambda x: str(
        f"https://civitai.com/api/v1/model-versions/by-hash/{x}"
    )
    _ld = lambda x: {k: v for k, v in dict(x).items()}
    _lh = lambda x: list([dict(x)[k]["url"] for k, v in dict(x).items()])
    now = lambda: datetime.strftime(datetime.now(), r"%Y%m%d%H%M%S")
    yyyymmddhhmmssffffff = lambda: str(
        datetime.strftime(datetime.now(), r"%Y%m%d%H%M%S%f")
    )
    yyyymmdd = lambda: str(datetime.strftime(datetime.now(), r"%Y%m%d"))
    _hex_check = lambda x: str(
        "".join(
            l
            for l in [x for x in str(x).replace(chr(32), chr(95))]
            if l in set("ABCDEF0123456789")
        )
    )
    fn_rm = lambda s: str(
        "".join(
            l
            for l in [x for x in str(s).replace(chr(32), chr(95))]
            if l
            in set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")
        )
    )
    rm_sp = lambda s: str(
        re.sub(
            "\s+", " ", (re.sub("_", "", (re.sub("[^a-zA-z0-9\s]", "", unidecode(s)))))
        )
        .strip()
        .lower()
    )
    rm_aw = lambda s: "".join(str(s).split())
    rm_sw = lambda s: str(" ".join(w for w in word_tokenize(s) if w not in sw))
    rm_wh = lambda s: " ".join(str(s).split())
    rm_pu = lambda s: str(s).translate(str.maketrans("", "", string.punctuation))

    def __post_init__(self):
        self.alwayson_scripts = (sd.alwayson_scripts,)
        self.api = (sd.api,)
        self.backend_target_pairs = (sd.backend_target_pairs,)
        self.batch_gen = (sd.batch_gen,)
        self.batch_size = (sd.batch_size,)
        self.bsrgan = (sd.bsrgan,)
        self.cfg_scale = (sd.cfg_scale,)
        self.codeformer_visibility = (sd.codeformer_visibility,)
        self.codeformer_weight = (sd.codeformer_weight,)
        self.controlnet_units = (sd.controlnet_units,)
        self.CUSTOM_MODELS = (sd.CUSTOM_MODELS,)
        self.denoising_strength = (sd.denoising_strength,)
        self.do_not_save_grid = (sd.do_not_save_grid,)
        self.do_not_save_samples = (sd.do_not_save_samples,)
        self.enable_hr = (sd.enable_hr,)
        self.esrgan_4x = (sd.esrgan_4x,)
        self.eta = (sd.eta,)
        self.EXTRA_KW = (sd.EXTRA_KW,)
        self.extras_upscaler_2_visibility = (sd.extras_upscaler_2_visibility,)
        self.fd_res = (sd.fd_res,)
        self.firstphase_height = (sd.firstphase_height,)
        self.firstphase_width = (sd.firstphase_width,)
        self.folder = (sd.folder,)
        self.fp_image = (sd.fp_image,)
        self.gfpgan_visibility = (sd.gfpgan_visibility,)
        self.height = (sd.height,)
        self.hr_resize_x = (sd.hr_resize_x,)
        self.hr_resize_y = (sd.hr_resize_y,)
        self.hr_scale = (sd.hr_scale,)
        self.hr_second_pass_steps = (sd.hr_second_pass_steps,)
        self.hr_upscaler = (sd.hr_upscaler,)
        self.hw_kern = (sd.hw_kern,)
        self.I2I_PROMPT = (sd.I2I_PROMPT,)
        self.image = (sd.image,)
        self.image_cfg_scale = (sd.image_cfg_scale,)
        self.images = (sd.images,)
        self.images = (sd.images,)
        self.include_init_images = (sd.include_init_images,)
        self.initial_noise_multiplier = (sd.initial_noise_multiplier,)
        self.inpaint_full_res = (sd.inpaint_full_res,)
        self.inpaint_full_res_padding = (sd.inpaint_full_res_padding,)
        self.inpainting_fill = (sd.inpainting_fill,)
        self.inpainting_mask_invert = (sd.inpainting_mask_invert,)
        self.lanczos = (sd.lanczos,)
        self.latent = (sd.latent,)
        self.latentantialiased = (sd.latentantialiased,)
        self.latentbicubic = (sd.latentbicubic,)
        self.latentbicubicantialiased = (sd.latentbicubicantialiased,)
        self.latentnearest = (sd.latentnearest,)
        self.latentnearestexact = (sd.latentnearestexact,)
        self.ldsr = (sd.ldsr,)
        self.lw_kern = (sd.lw_kern,)
        self.mask_blur = (sd.mask_blur,)
        self.mask_image = (sd.mask_image,)
        self.mh_kern = (sd.mh_kern,)
        self.model_index = (sd.model_index,)
        self.n_iter = (sd.n_iter,)
        self.name_list = (sd.name_list,)
        self.nearest = (sd.nearest,)
        self.negative_prompt = (sd.negative_prompt,)
        self.none = (sd.none,)
        self.none = (sd.none,)
        self.override_settings = (sd.override_settings,)
        self.override_settings_restore_afterwards = (
            (sd.override_settings_restore_afterwards),
        )
        self.proc_seq = (sd.proc_seq,)
        self.prompt = (sd.prompt,)
        self.r_esrgan_general_4xv3 = (sd.r_esrgan_general_4xv3,)
        self.resize_mode = (sd.resize_mode,)
        self.restore_faces = (sd.restore_faces,)
        self.s_churn = (sd.s_churn,)
        self.s_noise = (sd.s_noise,)
        self.s_tmax = (sd.s_tmax,)
        self.s_tmin = (sd.s_tmin,)
        self.sampler_list = (sd.sampler_list,)
        self.sampler_name = (sd.sampler_name,)
        self.sampler_name = (sd.sampler_name,)
        self.save_images = (sd.save_images,)
        self.script_args = (sd.script_args,)
        self.script_name = (sd.script_name,)
        self.scunet_gan = (sd.scunet_gan,)
        self.scunet_psnr = (sd.scunet_psnr,)
        self.seed = (sd.seed,)
        self.seed_resize_from_h = (sd.seed_resize_from_h,)
        self.seed_resize_from_w = (sd.seed_resize_from_w,)
        self.send_images = (sd.send_images,)
        self.seq = (sd.seq,)
        self.show_extras_results = (sd.show_extras_results,)
        self.steps = (sd.steps,)
        self.strnow = (sd.strnow,)
        self.styles = (sd.styles,)
        self.subseed = (sd.subseed,)
        self.subseed_strength = (sd.subseed_strength,)
        self.swinir_4x = (sd.swinir_4x,)
        self.T2I_PROMPT = (sd.T2I_PROMPT,)
        self.THREADS = (sd.THREADS,)
        self.tiling = (sd.tiling,)
        self.upscale_first = (sd.upscale_first,)
        self.upscaler_1 = (sd.upscaler_1,)
        self.upscaler_2 = (sd.upscaler_2,)
        self.upscaling_crop = (sd.upscaling_crop,)
        self.upscaling_resize = (sd.upscaling_resize,)
        self.upscaling_resize_h = (sd.upscaling_resize_h,)
        self.upscaling_resize_w = (sd.upscaling_resize_w,)
        self.use_async = (sd.use_async,)
        self.use_deprecated_controlnet = (sd.use_deprecated_controlnet,)
        self.w_seq = (sd.w_seq,)
        self.weight_calc = (sd.weight_calc,)
        self.weight_seq = (sd.weight_seq,)
        self.width = (sd.width,)
        super().__setattr__("attr_name", self)


def dns(d: dict) -> SimpleNamespace:
    return SimpleNamespace(**d)


def im_ratio(im: np.uint8, dw: int, dh: int) -> tuple[int, int]:
    im_w = np.array(im).shape[1]
    im_h = np.array(im).shape[0]
    ir = float(im_w / im_h)
    if im_w < im_h:
        return (int(abs(dw * ir)), int(abs(dh)))
    elif im_w > im_h:
        return (int(abs(dw)), int(abs(dh // ir)))
    elif im_w == im_h:
        return (dw, dh)


# same as im_ratio:
_imr = (
    lambda x: (int(abs(x[2] * (x[0] / x[1]))), int(abs(x[3])))
    if x[0] < x[1]
    else (int(abs(x[2])), int(abs(x[2] // (x[0] / x[1]))))
    if x[0] > x[1]
    else (x[2], x[3])
)


__SB__ = lambda t, d: tqdm(
    total=t,
    desc=d,
    bar_format="{desc}: {percentage:3.0f}%|"
    + "| {n_fmt}/{total_fmt} [elapsed: {elapsed} / Remaining: {remaining}] "
    + "{rate_fmt}{postfix}]",
)

# 0 - basedir  ,  1 - filename  ,  2 - extension
# 0 - '/mnt/z/sd/'  ,  'file_name'  ,  'jpg'
_f = lambda f: [
    str(f)[: len(f) - (str(f)[::-1].find("/")) :].lower(),
    str(f)[
        len(f)
        - (str(f)[::-1].find("/")) : (len(f))
        - 1
        - len(f[-(str(f)[::-1].find(".")) :])
    ],
    str(f)[-(str(f)[::-1].find(".")) :].lower(),
]


# check for alpha channel
_chk_bgra = (
    lambda i: np.uint8(i[::, ::, :-1:]) if np.uint8(i).shape[2] == 4 else np.uint8(i)
)

# list generator
_GET_LIST_ = lambda fp, exts: [
    fp + f
    for f in os.listdir(fp[:-1:])
    if os.path.isfile(fp + f) and str(f[-(f[::-1].find(".")) :]).lower() in exts
]

# to be able to paste Windows directors using fr"C:\WINDOWS\" without error
_ffn = (
    lambda s: str(s).replace(chr(92), chr(47)).replace(chr(34), "")
    if os.path.isfile(str(s).replace(chr(92), chr(47)).replace(chr(34), ""))
    else str(s + chr(47)).replace(chr(92), chr(47)).replace(chr(34), "")
    if os.path.isdir(str(s).replace(chr(92), chr(47)).replace(chr(34), ""))
    else None
)


# e-z unique check
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

        assert metadata_len > 2 and json_start in (
            b'{"',
            b"{'",
        ), f"{filename} is not a safetensors file"
        json_data = json_start + file.read(metadata_len - 2)
        json_obj = json.loads(json_data)

        res = {}
        for k, v in json_obj.get("__metadata__", {}).items():
            res[k] = v
            if isinstance(v, str) and v[0:1] == "{":
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
        "model": {"name": "", "type": "", "nsfw": "", "poi": ""},
        "files": [
            {
                "name": "",
                "sizeKB": 0,
                "type": "Model",
                "hashes": {"AutoV2": "", "SHA256": ""},
            }
        ],
        "tags": [],
        "downloadUrl": "",
        "skeleton_file": True,
        "ss_tag_frequency": {},
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
    model_info["ss_tag_frequency"] = file_metadata.get("ss_tag_frequency", {})
    return model_info


# initialize sdwebuiapi
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

sd.model_index = {}


# generate model index
len(
    list(
        map(
            lambda x: sd.model_index.update(x),
            list(
                map(
                    lambda x: {
                        str(x[0]): {
                            "file": str(
                                x[1][: len(x[1]) - x[1][::-1].find(chr(32))]
                            ).strip(),
                            "hash": str(
                                x[1][-x[1][::-1].find(chr(32)) :][:-1:][1::]
                            ).strip(),
                        }
                    }
                    if x[1][-1:] == "]"
                    else {
                        str(x[0]): {
                            "file": str(
                                x[1][: len(x[1]) - x[1][::-1].find(chr(32))]
                            ).strip(),
                            "hash": None,
                        }
                    },
                    [[i, x] for i, x in enumerate(sd.api.util_get_model_names())],
                )
            ),
        )
    )
)

# generate sampler names
sd.sampler_dicts = [dict(x) for x in sd.api.get_samplers()]

sampler_list = [x["name"] for x in sd.sampler_dicts]

sd.hr_upscaler_index = hr_upscaler_index


def IMG2IMG() -> list[Image.Image]:
    return sd.api.img2img(
        steps=sd.steps,
        images=sd.images,
        resize_mode=sd.resize_mode,
        denoising_strength=sd.denoising_strength,
        image_cfg_scale=sd.image_cfg_scale,
        mask_image=sd.mask_image,
        mask_blur=sd.mask_blur,
        inpainting_fill=sd.inpainting_fill,
        inpaint_full_res=sd.inpaint_full_res,
        inpaint_full_res_padding=sd.inpaint_full_res_padding,
        inpainting_mask_invert=sd.inpainting_mask_invert,
        initial_noise_multiplier=sd.initial_noise_multiplier,
        prompt=sd.prompt,
        styles=sd.styles,
        seed=sd.seed,
        subseed=sd.subseed,
        subseed_strength=sd.subseed_strength,
        seed_resize_from_h=sd.seed_resize_from_h,
        seed_resize_from_w=sd.seed_resize_from_w,
        sampler_name=sd.sampler_name,
        batch_size=sd.batch_size,
        n_iter=sd.n_iter,
        cfg_scale=sd.cfg_scale,
        width=sd.width,
        height=sd.height,
        restore_faces=sd.restore_faces,
        tiling=sd.tiling,
        do_not_save_samples=sd.do_not_save_samples,
        do_not_save_grid=sd.do_not_save_grid,
        negative_prompt=sd.negative_prompt,
        eta=sd.eta,
        s_churn=sd.s_churn,
        s_tmax=sd.s_tmax,
        s_tmin=sd.s_tmin,
        s_noise=sd.s_noise,
        override_settings=sd.override_settings,
        override_settings_restore_afterwards=sd.override_settings_restore_afterwards,
        script_args=sd.script_args,
        include_init_images=sd.include_init_images,
        script_name=sd.script_name,
        send_images=sd.send_images,
        save_images=sd.save_images,
        alwayson_scripts=sd.alwayson_scripts,
        controlnet_units=sd.controlnet_units,
        use_deprecated_controlnet=sd.use_deprecated_controlnet,
        use_async=sd.use_async,
    )


def ESI():
    return sd.api.extra_single_image(
        image=sd.image,
        resize_mode=sd.resize_mode,
        show_extras_results=sd.show_extras_results,
        gfpgan_visibility=sd.gfpgan_visibility,
        codeformer_visibility=sd.codeformer_visibility,
        codeformer_weight=sd.codeformer_weight,
        upscaling_resize=sd.upscaling_resize,
        upscaling_resize_w=sd.upscaling_resize_w,
        upscaling_resize_h=sd.upscaling_resize_h,
        upscaling_crop=sd.upscaling_crop,
        upscaler_1=sd.upscaler_1,
        upscaler_2=sd.upscaler_2,
        extras_upscaler_2_visibility=sd.extras_upscaler_2_visibility,
        upscale_first=sd.upscale_first,
        use_async=sd.use_async,
    )


def EBI():
    return sd.api.extra_batch_images(
        images=sd.images,
        name_list=sd.name_list,
        resize_mode=sd.resize_mode,
        show_extras_results=sd.show_extras_results,
        gfpgan_visibility=sd.gfpgan_visibility,
        codeformer_visibility=sd.codeformer_visibility,
        codeformer_weight=sd.codeformer_weight,
        upscaling_resize=sd.upscaling_resize,
        upscaling_resize_w=sd.upscaling_resize_w,
        upscaling_resize_h=sd.upscaling_resize_h,
        upscaling_crop=sd.upscaling_crop,
        upscaler_1=sd.upscaler_1,
        upscaler_2=sd.upscaler_2,
        extras_upscaler_2_visibility=sd.extras_upscaler_2_visibility,
        upscale_first=sd.upscale_first,
        use_async=sd.use_async,
    )


def TXT2IMG() -> Image.Image:
    return sd.api.txt2img(
        enable_hr=sd.enable_hr,
        denoising_strength=sd.denoising_strength,
        firstphase_width=sd.firstphase_width,
        firstphase_height=sd.firstphase_height,
        hr_scale=sd.hr_scale,
        hr_upscaler=sd.hr_upscaler,
        hr_second_pass_steps=sd.hr_second_pass_steps,
        hr_resize_x=sd.hr_resize_x,
        hr_resize_y=sd.hr_resize_y,
        prompt=sd.prompt,
        styles=sd.styles,
        seed=sd.seed,
        subseed=sd.subseed,
        subseed_strength=sd.subseed_strength,
        seed_resize_from_h=sd.seed_resize_from_h,
        seed_resize_from_w=sd.seed_resize_from_w,
        sampler_name=sd.sampler_name,
        batch_size=sd.batch_size,
        n_iter=sd.n_iter,
        steps=sd.steps,
        cfg_scale=sd.cfg_scale,
        width=sd.width,
        height=sd.height,
        restore_faces=sd.restore_faces,
        tiling=sd.tiling,
        do_not_save_samples=sd.do_not_save_samples,
        do_not_save_grid=sd.do_not_save_grid,
        negative_prompt=sd.negative_prompt,
        eta=sd.eta,
        s_churn=sd.s_churn,
        s_tmax=sd.s_tmax,
        s_tmin=sd.s_tmin,
        s_noise=sd.s_noise,
        override_settings=sd.override_settings,
        override_settings_restore_afterwards=sd.override_settings_restore_afterwards,
        script_args=sd.script_args,
        script_name=sd.script_name,
        send_images=sd.send_images,
        save_images=sd.save_images,
        alwayson_scripts=sd.alwayson_scripts,
        sampler_index="Euler a",
        use_deprecated_controlnet=False,
        use_async=False,
    )


banned_lora_list: list[str] = [
    # put banned / excluded LORA names here
]


def _lora_info(file: str):
    f = _f(file)
    f_str = f"{f[1]}"
    info = dummy_model_info(file)
    LORAS[f[1]] = {}
    LORAS[f[1]]["lora"] = f_str
    LORAS[f[1]]["ss_tag_frequency"] = info["ss_tag_frequency"]


def _proc_weights(key: str):
    ssft = LORAS[key]["ss_tag_frequency"]
    fname = LORAS[key]["lora"]
    for tk, tv in ssft.items():
        for k, v in tv.items():
            KW_LIST.append(k)
            if k not in [kw for kw in K_DICT.keys()]:
                K_DICT[k] = []
                K_DICT[k].append(
                    [
                        fname,
                        float(2)
                        if ssft[tk][k] > 20
                        else float(1)
                        if ssft[tk][k] > 10
                        else 0.01,
                    ]
                )
            else:
                K_DICT[k].append(
                    [
                        fname,
                        float(2)
                        if ssft[tk][k] > 20
                        else float(1)
                        if ssft[tk][k] > 10
                        else 0.01,
                    ]
                )


def _lora_list_gen():
    file_list: list[str] = [
        x
        for x in _GET_LIST_(LORA_DIR, ["safetensors", "ckpt", "pt"])
        if x not in banned_lora_list
    ]
    with ThreadPoolExecutor(sd.THREADS) as executor:
        status_bar = tqdm(total=len(file_list), desc=r"Lora_Model_Info")
        futures = [executor.submit(_lora_info, file) for file in file_list]
        for _ in as_completed(futures):
            status_bar.update(n=1)


def _key_weight_gen():
    key_list = [k for k in LORAS.keys()]
    with ThreadPoolExecutor(sd.THREADS) as executor:
        status_bar = tqdm(total=len(key_list), desc=r"Process_Tag_Weights")
        futures = [executor.submit(_proc_weights, key) for key in key_list]
        for _ in as_completed(futures):
            status_bar.update(n=1)


def txt_proc(folder: str):
    folder_prompts: list[str] = []
    txt_list = _GET_LIST_(_ffn(folder), ["txt"])
    print(len(txt_list))
    for file in tqdm(txt_list):
        txt_data: str = ""
        with open(file, "rt") as fi:
            txt_data = fi.read().lower()
        for rm in remove_list:
            txt_data.replace(rm, "")
        for k, v in replace_list.items():
            txt_data.replace(k, v)
        if len(txt_data) != "":
            folder_prompts.append(txt_data)
    return folder_prompts


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if not os.path.isfile(f"{LORA_DIR}LORAS.JSON"):
        _lora_list_gen()
        _key_weight_gen()
        for key in K_DICT.keys():
            KW_DICT[key] = sorted(
                [x for x in K_DICT[key]], key=lambda x: x[1], reverse=True
            )
        KW_LIST = [k for k in KW_DICT.keys()]
        repl_keys = [k for k in replace_list.keys()]
        cust_workflow: dict = {
            "LORAS": LORAS,
            "K_DICT": K_DICT,
            "KW_DICT": KW_DICT,
            "REPLACE_LIST": replace_list,
            "REMOVE_LIST": remove_list,
            "UN_WEIGHT": un_weight,
        }
        j_lora_dicts = json.dumps(
            cust_workflow, sort_keys=True, indent=4, separators=(",", ": ")
        )
        with open(f"{LORA_DIR}LORAS.JSON", "wt") as fi:
            fi.write(j_lora_dicts)
    else:
        j_lora_dict = None
        with open(f"{LORA_DIR}LORAS.JSON", "rt") as fi:
            j_lora_dict = json.loads(fi.read())
        print(f"settings loaded")
        LORAS: dict = j_lora_dict["LORAS"]
        K_DICT: dict = j_lora_dict["K_DICT"]
        KW_DICT: dict = j_lora_dict["KW_DICT"]
        replace_list: dict = j_lora_dict["REPLACE_LIST"]
        remove_list: list = j_lora_dict["REMOVE_LIST"]
        un_weight: list = j_lora_dict["UN_WEIGHT"]
    KW_LIST = [k for k in KW_DICT.keys()]
    repl_keys = [k for k in replace_list.keys()]


def _lw_sequence(seq: list[str]):
    js: list[str] = []
    js.append(sd.EXTRA_KW)
    for i, extra_net in enumerate(sd.CUSTOM_MODELS):
        js.append(f"<lora:{extra_net}:{str(seq[i])}>")
    return js


def img_save(image: np.uint8 = sd.fp_image):
    strms = str(datetime.strftime(datetime.now(), r"%H%M%S%f"))
    cv2.imwrite(str(f"{SAVE_DIR}grids/{sd.strnow}_{strms}.jpg"), image)


def weight_adjust(sdprompt: str, weight: str = "0.25") -> str:
    with printing_exc(fmt=fmt1):
        _lr = lambda s: str(
            str(s).replace("<lora:", "")[: str(s).replace("<lora:", "").find(":") :]
        )
        prompt_loras: list[str] = []
        prompt_list: list[str] = []
        prompt_list_str: str = ""
        prompt_loras_str: str = ""
        prompt_list = [
            str(x).strip()
            for x in str(sdprompt).split(chr(44))
            if len(str(x).strip()) > 1 and str(x).find("<lora:") == -1
        ]
        prompt_loras = [
            str(t).strip()
            for t in str(sdprompt).split(chr(44))
            if str(t).find("<lora:") != -1
        ]
        prompt_loras = [str(f"<lora:{_lr(t)}:{weight}>") for t in prompt_loras]
        prompt_list_str = str(
            chr(44).join(
                str(t).strip().lower()
                for t in prompt_list
                if str(t).strip().lower() not in remove_list
            )
        )
        prompt_loras_str = str(
            chr(44).join(str(t).strip().lower() for t in prompt_loras)
        )
        return str(prompt_list_str + prompt_loras_str)


def t2i(size: list[int, int]) -> tuple[Image.Image, dict]:
    imghw: list[int] = [512, 576, 640, 704, 768, 832]
    img_hw_l = lambda: imghw[randint(0, 5)]
    sd.width = img_hw_l()
    sd.height = img_hw_l()
    sd.steps = 21
    sd.eta = 1.0
    sd.sampler_name = "DPM++ 2M"
    t2i_image = TXT2IMG()
    image: Image.Image = t2i_image.image
    info: dict = t2i_image.info
    sd.fp_image = np.uint8(t2i_image.image)[:, :, ::-1].copy()
    strms = str(datetime.strftime(datetime.now(), r"%H%M%S%f"))
    cv2.imwrite(str(f"{SAVE_DIR}grids/{sd.strnow}_{strms}.jpg"), sd.fp_image)
    rs_w, rs_h = im_ratio(np.uint8(sd.fp_image), 1664, 1664)
    sd.fp_image = cv2.resize(
        src=sd.fp_image, dsize=(rs_w, rs_h), interpolation=cv2.INTER_LANCZOS4
    )
    sd.width = sd.fp_image.shape[1]
    sd.height = sd.fp_image.shape[0]
    return (image, info)


def i2i(
    w_seq: int,
    steps: int = sd.steps,
    eta: float = sd.eta,
    denoising_strength=sd.denoising_strength,
    sampler_name: str = sd.sampler_name,
    prompt: str = sd.prompt,
    image_cfg_scale=1,
    resize_mode=0,
    strnow: str = sd.strnow,
    height: int = sd.height,
    width: int = sd.width,
    image: np.uint8 = sd.fp_image,
) -> tuple[Image.Image, dict]:
    sd.strnow = strnow
    sd.image_cfg_scale = image_cfg_scale
    sd.resize_mode = resize_mode
    sd.width = width
    sd.height = height
    sd.sampler_name = sampler_name
    sd.denoising_strength = denoising_strength
    sd.steps = steps
    sd.eta = eta
    sd.prompt = prompt + chr(44).join(j for j in _lw_sequence(seq=w_seq))
    sd.images = [
        Image.fromarray(
            np.uint8(np.array(sd.fp_image).astype("uint8")[:, :, ::-1].copy())
        )
    ]
    i2i_image = IMG2IMG()
    image: Image.Image = i2i_image.image
    info: dict = i2i_image.info
    sd.fp_image = np.uint8(np.array(image).astype("uint8")[:, :, ::-1].copy())
    strms = str(datetime.strftime(datetime.now(), r"%H%M%S%f"))
    cv2.imwrite(str(f"{SAVE_DIR}grids/{sd.strnow}_{strms}.jpg"), sd.fp_image)
    return (image, info)


def yu_i2i():
    YN_MODEL = YuNet(
        modelPath=YUNET_ONNX_PATH,
        inputSize=sd.fp_image.shape[:2][::-1],
        confThreshold=0.75,
        nmsThreshold=0.35,
        topK=2,
        backendId=backend_id,
        targetId=target_id,
    )
    YN_MODEL.setInputSize(sd.fp_image.shape[:2][::-1])
    fd_res = YN_MODEL.infer(sd.fp_image)
    if fd_res is not None:
        sd.fd_res = fd_res
        sd.fp_image = sd.fp_image


def face_pass(__prompt__: str, w_seq: list[float]) -> np.uint8:
    fp_image = copy.copy(sd.fp_image)

    img_save(fp_image)

    _prompt_list: list[str] = _unique(
        [str(x).strip() for x in __prompt__.split(chr(44)) if str(x).find("<") == -1]
    )
    _prompt_loras: list[str] = _unique(
        [t for t in __prompt__.split(chr(44)) if str(t).find("<") != -1]
    )
    _prompt: str = chr(44).join(
        str(t).strip().lower() + chr(10)
        for t in _prompt_list + _prompt_loras
        if str(t).strip().lower() not in remove_list
    )
    _prompt: str = weight_adjust(
        chr(44).join(
            str(t).strip().lower() + chr(10)
            for t in face_hd + _prompt_list + _prompt_loras
            if str(t).strip().lower() not in remove_list
        ),
        "0.125:0.125",
    )
    for res in sd.fd_res:
        bbox = res[0:4].astype(np.int32)
        x, y, w, h = (bbox[0], bbox[1], bbox[2], bbox[3])
        fc = np.uint8(fp_image[y - 64 : y + h + 64, x - 64 : x + w + 64])
        fc_w, fc_h = (fc.shape[1], fc.shape[0])
        imr_w, imr_h = im_ratio(np.uint8(fc), 640, 640)
        sd.fp_image = cv2.resize(
            fc.copy(), (imr_w, imr_h), interpolation=cv2.INTER_LANCZOS4
        )

        # WIP - interrogate via DeepDanBooru to check for proper subject matter
        # ddb_check = sd.api.interrogate(Image.fromarray(sd.fp_image[:,:,::-1]),"deepdanbooru")
        # ddb_list = [str(x).strip().lower() for x in ddb_check.info.split(chr(44))]
        # print(ddb_list)
        # if '1girl' in ddb_list:
        #     print('1GIRL')
        #     print('1GIRL')
        #     print('1GIRL')
        #     print('1GIRL')

        hw = 832

        sd.prompt = _prompt + chr(44).join(j for j in _lw_sequence(seq=w_seq[0]))
        print(sd.prompt)
        i2i(
            sd.weight_seq[3],
            image_cfg_scale=1,
            resize_mode=0,
            sampler_name="Euler a",
            width=hw,
            height=hw,
            denoising_strength=0.29,
            steps=15,
            eta=1.0,
        )

        sd.prompt = _prompt + chr(44).join(j for j in _lw_sequence(seq=w_seq[1]))
        print(sd.prompt)
        i2i(
            sd.weight_seq[3],
            image_cfg_scale=1,
            resize_mode=0,
            sampler_name="Euler a",
            width=hw,
            height=hw,
            denoising_strength=0.19,
            steps=15,
            eta=1.0,
        )

        fc = sd.fp_image.copy()
        inter_img = copy.copy(fp_image)
        fp_image[y - 64 : y + h + 64, x - 64 : x + w + 64] = cv2.resize(
            np.array(fc).astype("uint8"), (fc_w, fc_h), interpolation=cv2.INTER_LANCZOS4
        )

        fp_image = cv2.addWeighted(inter_img, 0.3, fp_image, 0.7, 0)

        sd.fp_image = copy.copy(fp_image)

        img_save(sd.fp_image)

        sd.width = fp_image.shape[1]
        sd.height = fp_image.shape[0]


def gen_proc():
    with printing_exc(fmt=fmt1):
        strnow = lambda: str(datetime.strftime(datetime.now(), r"%Y%m%d%H%M%S%f"))
        sd.strnow = str(strnow())

        __prompt__: str = copy.copy(sd.prompt)
        _prompt_list: list[str] = _unique(
            [
                str(x).strip()
                for x in __prompt__.split(chr(44))
                if str(x).find("<") == -1
            ]
        )
        _prompt_loras: list[str] = _unique(
            [t for t in __prompt__.split(chr(44)) if str(t).find("<") != -1]
        )
        _prompt: str = chr(44).join(
            str(t).strip().lower() + chr(10)
            for t in _prompt_list + _prompt_loras
            if str(t).strip().lower() not in remove_list
        )

        image, info = t2i([1664, 1664])

        model_no = 2
        sd.api.util_set_model(
            str(f"{sd.model_index[f'{model_no}']['file']}"), find_closest=True
        )

        sd.prompt = _prompt + chr(44).join(
            j for j in _lw_sequence(seq=sd.weight_seq[0])
        )

        yu_i2i()

        face_pass(__prompt__=__prompt__, w_seq=[sd.weight_seq[1], sd.weight_seq[2]])

        sd.prompt = _prompt + chr(44).join(
            j for j in _lw_sequence(seq=sd.weight_seq[3])
        )

        i2i(
            sd.weight_seq[3],
            image_cfg_scale=1,
            resize_mode=0,
            sampler_name="DPM++ 2M",
            width=sd.width,
            height=sd.height,
            denoising_strength=0.29,
            steps=5,
            eta=1.0,
        )

        yu_i2i()

        face_pass(__prompt__=__prompt__, w_seq=[sd.weight_seq[1], sd.weight_seq[2]])

        sd.prompt = _prompt + chr(44).join(
            j for j in _lw_sequence(seq=sd.weight_seq[4])
        )

        i2i(
            sd.weight_seq[3],
            image_cfg_scale=1,
            resize_mode=0,
            sampler_name="Euler a",
            width=sd.width,
            height=sd.height,
            denoising_strength=0.19,
            steps=5,
            eta=1.0,
        )

        sd.prompt = _prompt + chr(44).join(
            j for j in _lw_sequence(seq=sd.weight_seq[7])
        )

        i2i(
            prompt=__prompt__,
            w_seq=sd.weight_seq[3],
            image_cfg_scale=1,
            resize_mode=0,
            sampler_name="DPM++ 2M",
            width=sd.width,
            height=sd.height,
            denoising_strength=0.29,
            steps=5,
            eta=1.0,
        )
        final_img, final_info = i2i(
            sd.weight_seq[3],
            image_cfg_scale=1,
            resize_mode=0,
            sampler_name="Euler a",
            width=sd.width,
            height=sd.height,
            denoising_strength=0.19,
            steps=10,
            eta=1.0,
        )

        with open(str(f"{SAVE_DIR}grids/{sd.strnow}.txt"), "wt") as fi:
            fi.write(str(sd.weight_seq) + chr(10))
            for k, v in dict(final_info).items():
                fi.write(str(f"{k}: {v}{chr(10)}"))
        img_fn = str(f"{SAVE_DIR}{sd.strnow}.jpg")
        final_img.save(fp=img_fn, format="JPEG")
        img_save(np.uint8(final_img)[:, :, ::-1])


# generate pictures based on reading .txt tag files from a folder
@prints_exc(fmt=fmt1)
def _gen_folder_lora_matching():
    folder_prompts = txt_proc(sd.folder)
    repl_keys = [k for k in replace_list.keys()]
    shuffle(pl)
    for prompt in folder_prompts:
        for i in range(sd.batch_gen):
            strt1 = datetime.now()
            try:
                prompt_loras: list = []
                prompt_list: list[str] = []
                top_k: list = []
                w_lora_topk: dict = {}
                w_loras: list = []
                prompt_list = _unique(
                    [
                        replace_list[str(x).lower().strip()]
                        if str(x).strip().lower() in repl_keys
                        else str(x).lower().strip()
                        for x in str(sd.EXTRA_KW + prompt)
                        .replace(chr(47), "")
                        .replace(chr(92), "")
                        .replace(")", "")
                        .replace("(", "")
                        .split(chr(44))
                        if str(x).strip().lower() not in remove_list
                    ]
                )
                for w in prompt_list:
                    if w in KW_LIST and w not in un_weight:
                        w_list: list = []
                        for lora in KW_DICT[w]:
                            if lora[0] not in w_lora_topk.keys():
                                w_lora_topk[lora[0]] = lora[1]
                            else:
                                w_lora_topk[lora[0]] = w_lora_topk[lora[0]] + lora[1]
                        len(list(map(lambda x: w_loras.append(x), [x for x in w_list])))
                top_k = sorted(
                    [[k, round(v, ndigits=1)] for k, v in w_lora_topk.items()],
                    key=lambda x: x[1],
                    reverse=True,
                )[:100]
                prompt_loras = [tk for tk in top_k]
                shuffle(prompt_loras)
                prompt_loras = prompt_loras[:10]
                shuffle(prompt_loras)
                kw_weight: float = 0.0
                kw_weight = float(round(1.0 / len(prompt_loras), ndigits=3))
                sd.weight_calc = kw_weight
                prompt_list = prompt_list + img_hd_kw
                shuffle(prompt_list)
                sd.prompt = chr(44).join(t for t in prompt_list) + chr(44).join(
                    f"<lora:{t[0]}:{kw_weight}>" for t in prompt_loras
                )
                print(sd.prompt)
                gen_proc()
            except:
                pass
            strt2 = datetime.now()
            strt_tot = strt2 - strt1
            print("")
            print(str(f"Total Time: {strt_tot.total_seconds()}"))
            print("")


# matches existing LORAs based on keywords
@prints_exc(fmt=fmt1)
def _gen_lora_matching(kw_search: str):
    for i in range(sd.batch_gen):
        strt1 = datetime.now()
        try:
            prompt_loras: list = []
            prompt_list: list[str] = []
            top_k: list = []
            w_lora_topk: dict = {}
            w_loras: list = []
            prompt_list = _unique(
                [
                    replace_list[str(x).lower().strip()]
                    if str(x).strip().lower() in repl_keys
                    else str(x).lower().strip()
                    for x in str(sd.EXTRA_KW + kw_search)
                    .replace(chr(47), "")
                    .replace(chr(92), "")
                    .replace(")", "")
                    .replace("(", "")
                    .split(chr(44))
                    if str(x).strip().lower() not in remove_list
                ]
            )
            for w in prompt_list:
                if w in KW_LIST and w not in un_weight:
                    w_list: list = []
                    for lora in KW_DICT[w]:
                        if lora[0] not in w_lora_topk.keys():
                            w_lora_topk[lora[0]] = lora[1]
                        else:
                            w_lora_topk[lora[0]] = w_lora_topk[lora[0]] + lora[1]
                    len(list(map(lambda x: w_loras.append(x), [x for x in w_list])))
            top_k = sorted(
                [[k, round(v, ndigits=1)] for k, v in w_lora_topk.items()],
                key=lambda x: x[1],
                reverse=True,
            )[:100]
            prompt_loras = [tk for tk in top_k]
            shuffle(prompt_loras)
            prompt_loras = prompt_loras[:10]
            shuffle(prompt_loras)
            kw_weight: float = 0.0
            kw_weight = float(round(3.0 / len(prompt_loras), ndigits=3))
            sd.weight_calc = kw_weight
            prompt_list = prompt_list + img_hd_kw
            shuffle(prompt_list)
            sd.prompt = chr(44).join(t for t in prompt_list) + chr(44).join(
                f"<lora:{t[0]}:{kw_weight}>" for t in prompt_loras
            )
            print(sd.prompt)
            gen_proc()
        except:
            pass
        strt2 = datetime.now()
        strt_tot = strt2 - strt1
        print("")
        print(str(f"Total Time: {strt_tot.total_seconds()}"))
        print("")


@prints_exc(fmt=fmt1)
def _gen_img_facepass():
    for i in range(sd.batch_gen):
        strt1 = datetime.now()
        try:
            gen_proc()
        except:
            pass
        strt2 = datetime.now()
        strt_tot = strt2 - strt1
        print("")
        print(strt_tot.total_seconds())
        print("")


_rp = lambda x: x[randint(0, len(x) - 1)]


@prints_exc(fmt=fmt1)
def _gen_lora_matching(kw_search: str):
    pre_prompt: str = str(sd.prompt)
    for i in range(sd.batch_gen):
        strt1 = datetime.now()
        try:
            prompt_loras: list = []
            prompt_list: list[str] = []
            top_k: list = []
            w_lora_topk: dict = {}
            w_loras: list = []
            prompt_list = _unique(
                [
                    replace_list[str(x).lower().strip()]
                    if str(x).strip().lower() in repl_keys
                    else str(x).lower().strip()
                    for x in str(sd.EXTRA_KW + kw_search)
                    .replace(chr(47), "")
                    .replace(chr(92), "")
                    .replace(")", "")
                    .replace("(", "")
                    .split(chr(44))
                    if str(x).strip().lower() not in remove_list
                ]
            )
            for w in prompt_list:
                if w in KW_LIST and w not in un_weight:
                    w_list: list = []
                    for lora in KW_DICT[w]:
                        if lora[0] not in w_lora_topk.keys():
                            w_lora_topk[lora[0]] = lora[1]
                        else:
                            w_lora_topk[lora[0]] = w_lora_topk[lora[0]] + lora[1]
                    len(list(map(lambda x: w_loras.append(x), [x for x in w_list])))
            top_k = sorted(
                [[k, round(v, ndigits=1)] for k, v in w_lora_topk.items()],
                key=lambda x: x[1],
                reverse=True,
            )[:100]
            prompt_loras = [tk for tk in top_k]
            shuffle(prompt_loras)
            prompt_loras = prompt_loras[:10]
            shuffle(prompt_loras)
            kw_weight: float = 0.0
            kw_weight = float(round(3.0 / len(prompt_loras), ndigits=3))
            sd.weight_calc = kw_weight
            prompt_list = prompt_list + img_hd_kw
            shuffle(prompt_list)
            lora_prompt = (
                kw_search
                + chr(44)
                + chr(44).join(t for t in prompt_list)
                + chr(44).join(f"<lora:{t[0]}:{kw_weight}>" for t in prompt_loras)
            )
            sd.prompt = pre_prompt + lora_prompt
            print(sd.prompt)
            gen_proc()
        except:
            pass
        strt2 = datetime.now()
        strt_tot = strt2 - strt1
        print("")
        print(str(f"Total Time: {strt_tot.total_seconds()}"))
        print("")


sd.negative_prompt = "bad-hands-5,  EasyNegative, worst quality, low quality ,(watermark, white border, monochrome, greyscale, text, speech bubble, artist name, signature),  epiCPhotoGasm-softPhoto-neg,  FastNegativeV2"

sd.CUSTOM_MODELS = [
    "SUBJECT_FACE_LORA1",
    "SUBJECT_FACE_LORA2",
    "SUBJECT_FACE_LORA3",
    "SUBJECT_FACE_LORA14",
]

sd.EXTRA_KW = "SUBJECT_ACTIVATION_KW,"

sd.folder = rf"F:/wat/test2/"

sd.weight_seq = [
    ["0.15:0.15", "0.15:0.15", "0.15:0.15", "0.15:0.15"],
    ["0.75:0.45", "0.75:0.65", "0.75:0.35", "0.75:0.35"],  # face
    ["0.75:0.45", "0.75:0.65", "0.75:0.35", "0.75:0.35"],  # face
    ["0.15:0.25", "0.15:0.25", "0.15:0.25", "0.15:0.15"],
    ["0.15:0.35", "0.15:0.35", "0.15:0.35", "0.15:0.15"],
    ["0.75:0.45", "0.75:0.65", "0.75:0.35", "0.75:0.35"],  # face
    ["0.75:0.45", "0.75:0.65", "0.75:0.35", "0.75:0.35"],  # face
    ["0.15:0.45", "0.15:0.65", "0.15:0.35", "0.15:0.15"],
]


def model_rotate(model_no: int):
    sd.api.util_set_model(
        str(
            f"{str(sd.model_index[f'{model_no}']['file'])} [{str(sd.model_index[f'{model_no}']['hash'])}]"
        ),
        find_closest=False,
    )

    sd.weight_seq = [
        ["0.15:0.15", "0.15:0.65", "0.15:0.15", "0.15:0.15"],
        ["0.15:0.45", "0.65", "0.35", "0.35"],  # face
        ["0.45", "0.65", "0.35", "0.35"],  # face
        ["0.15:0.25", "0.15:0.65", "0.15:0.25", "0.15:0.15"],
        ["0.15:0.35", "0.15:0.65", "0.15:0.35", "0.15:0.15"],
        ["0.45", "0.65", "0.35", "0.35"],  # face
        ["0.45", "0.65", "0.35", "0.35"],  # face
        ["0.15:0.45", "0.15:0.65", "0.15:0.35", "0.15:0.15"],
    ]

    kw_search = rf"\
weight_emphasis_keyword_string1,\
weight_emphasis_keyword_string2,\
weight_emphasis_keyword_string3"

    sd.prompt = ""
    _gen_img_facepass()


sd.batch_gen = 1
if __name__ == "__main__":
    multiprocessing.freeze_support()
    model_list: list[int] = [16]
    # shuffle(model_list)
    for _model in model_list:
        for i in range(100):
            model_no: int = _model
            sd.api.util_set_model(
                str(f"{sd.model_index[f'{model_no}']['file']}"), find_closest=True
            )
            model_rotate(_model)
