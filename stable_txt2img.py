import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
import PIL
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid, save_image
import time
from pytorch_lightning import seed_everything
#from torch import autocast
from torch.cuda.amp import autocast as autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

import torch.nn.functional as F

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    keys = list(sd.keys())
    #print("\n".join(keys))
    ####
    for k in keys:
        if k.find("input_blocks.0.0.weight")>=0 and sd[k].shape[1] == 4:
            w = torch.zeros([320, 8, 3, 3], dtype=torch.float)
            w[:,:4,:,:] = sd[k]
            sd[k] = w
            sd["input_blocks.0.1.weight"] = torch.ones([320, 320, 11, 11], dtype=torch.float)
    ####
    model = instantiate_from_config(config.model)
    #print(model.cond_stage_model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def center_crop(img):
    crop = min(img.shape[0], img.shape[1])
    h, w, = img.shape[0], img.shape[1]
    img = img[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]
    return img

def load_img(path, size):
    image = Image.open(path).convert("RGB")
    #image = Image.fromarray(center_crop(np.array(image)))
    if True:
        W, H = size
        w, h = image.size
        if w <= h:
            h = h*W//w
            w = W
        else:
            w = w*W//h
            h = W
        w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32
    else:
        w,h = size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1., (w,h)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--init_img",
        type=str,
        nargs="?",
        help="path to the input image"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="",
        help="the prompt to render"
    )
    # https://www.assemblyai.com/blog/stable-diffusion-1-vs-2-what-you-need-to-know/
    # ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face,
    # out of frame,
    # mutation, mutated, extra limbs, extra legs, extra arms,
    # disfigured, deformed, cross-eye, body out of frame, blurry,
    # bad art, bad anatomy, blurred, text, watermark, grainy
    parser.add_argument(
        "--negprompt",
        type=str,
        nargs="?",
        default="",
        help="the negative prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--iscale",
        type=float,
        default=1.,
        help="image guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--blendmodel",
        action='store_true',
        default=False,
        help="blend with original model, check ldm/models/diffusion/ddim.py",
    )
    parser.add_argument(
        "--blendpos",
        type=int,
        default=-1,
        help="blend with original model, check ldm/models/diffusion/ddim.py",
    )
    parser.add_argument(
        "--t0",
        type=float,
        default=1.,
        help="t0 for SDEdit mentioned in paper",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference_instruct.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )


    parser.add_argument(
        "--embedding_path", 
        type=str, 
        help="Path to a pre-trained embedding manager checkpoint")

    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    #model.embedding_manager.load(opt.embedding_path)
    print(model)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    blendmodel = None

    if opt.blendmodel:
        blendmodel = load_model_from_config(config, f"./sd-v1-4-full-ema.ckpt")
        blendmodel = blendmodel.to(device)

        if opt.blendpos < 0:
            opt.blendpos = opt.ddim_steps // 2

        if opt.ddim_steps-opt.blendpos < int(opt.ddim_steps * (1.-opt.t0)):
            opt.blendpos = opt.ddim_steps-int(opt.ddim_steps * (1.-opt.t0))

    #token = opt.prompt.split(' ')[0] # assume token pos
    #editprompt = opt.prompt[min(len(token)+1,len(opt.prompt)):]
    #print("token: {}, edit prompt: {}".format(token, editprompt))

    if opt.plms:
        sampler = PLMSSampler(model, blendmodel, opt.blendpos)
    else:
        sampler = DDIMSampler(model, blendmodel, opt.blendpos)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples

    init_latent = None
    W,H = (opt.W,opt.H)

    if isinstance(opt.init_img, str) and os.path.exists(opt.init_img):
        assert os.path.isfile(opt.init_img)
        init_image,size = load_img(opt.init_img, (opt.W,opt.H))
        init_image = init_image.to(device)
        W,H = size
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, H // opt.f, W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        #with precision_scope("cuda"):
        with precision_scope():
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0: 
                            uc = model.get_learned_conditioning(batch_size * [opt.negprompt])
                            #uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        #c0 = model.get_learned_conditioning(batch_size * [token])
                        #cedit = model.get_learned_conditioning(batch_size * [editprompt])
                        c0=cedit=c
                        shape = [opt.C, H // opt.f, W // opt.f]
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                         conditioning=c,
                                                         cedit=cedit,
                                                         conditioning0=c0,
                                                         t0=opt.t0,
                                                         batch_size=opt.n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt.ddim_eta,
                                                         x_T=start_code,
                                                         imagec=init_latent,
                                                         image_guidance_scale=opt.iscale)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                        if not opt.skip_save:
                            for x_sample in x_samples_ddim:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                Image.fromarray(x_sample.astype(np.uint8)).save(
                                    os.path.join(sample_path, f"{base_count:05}.jpg"))
                                base_count += 1

                        if not opt.skip_grid:
                            all_samples.append(x_samples_ddim)

                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    
                    for i in range(grid.size(0)):
                        save_image(grid[i, :, :, :], os.path.join(outpath,opt.prompt+'_{}.png'.format(i)))
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    if opt.negprompt != '':
                        Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'{prompt.replace(" ", "-")}-{grid_count:04}-{opt.negprompt.replace(" ", "-")}.jpg'))
                    else:
                        Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'{prompt.replace(" ", "-")}-{grid_count:04}.jpg'))
                    grid_count += 1
                    
                    

                toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
