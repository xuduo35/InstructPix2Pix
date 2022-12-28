import os
import random
from typing import OrderedDict
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from captionizer import caption_from_path, generic_captions_from_path
from captionizer import find_images

def shuffle_list(*ls):
    l = list(zip(*ls))
    random.shuffle(l)
    return zip(*l)

class PersonalizedInstruct(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 repeats=100,
                 interpolation="bicubic",
                 flip_p=0.5,
                 set="train",
                 placeholder_token="instruct",
                 per_image_tokens=False,
                 center_crop=False,
                 mixing_prob=0.25,
                 coarse_class_text=None,
                 reg=False
                 ):

        self.data_root = data_root

        f = open(os.path.join(self.data_root, "prompts.txt"))
        lines = [line.strip() for line in f.readlines()]
        f.close()

        self.image_paths = []
        self.prompt_prompts = []
        self.prompt_instructs = []
        self.prompt_completions = []

        for line in lines:
            items = line.split('|')

            self.image_paths.append(os.path.join(self.data_root, 'A', items[0]))
            self.prompt_prompts.append(items[1])
            self.prompt_instructs.append(items[2])
            self.prompt_completions.append(items[3])

        self.image_paths, self.prompt_prompts, self.prompt_instructs, self.prompt_completions = \
                shuffle_list(self.image_paths, self.prompt_prompts, self.prompt_instructs, self.prompt_completions)

        # self._length = len(self.image_paths)
        self.num_images = len(self.image_paths)
        self._length = self.num_images

        self.placeholder_token = placeholder_token
        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob

        self.coarse_class_text = coarse_class_text

        if set == "train":
            self._length = self.num_images * repeats
        else: # shorten val/test/..., actually no special data for them
            self._length = 256

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

        print(f"Total {set} data length: {self._length}")

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image_path = self.image_paths[i % self.num_images]
        image = Image.open(image_path.replace('/A/', '/B/')).convert("RGB")
        condi = Image.open(image_path).convert("RGB")

        prompt = self.prompt_prompts[i % self.num_images]
        instruct = self.prompt_instructs[i % self.num_images]
        completion = self.prompt_completions[i % self.num_images]
        example["caption"] = caption_from_path( \
                image_path, self.data_root, self.coarse_class_text, instruct)

        p = random.random()

        if p < 0.10:
            example["caption"] = caption_from_path( \
                image_path, self.data_root, self.coarse_class_text, '')
            condi = image.copy()
        elif p < 0.20:
            if prompt != '':
                example["caption"] = caption_from_path( \
                    image_path, self.data_root, self.coarse_class_text, prompt)
                image = condi.copy()
                condi = Image.new(mode ="RGB", size=image.size)
        elif p < 0.30:
            if prompt != '':
                example["caption"] = caption_from_path( \
                    image_path, self.data_root, self.coarse_class_text, completion)
                condi = Image.new(mode ="RGB", size=image.size)

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        cnd = np.array(condi).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]
            cnd = cnd[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        condi = Image.fromarray(cnd)
        '''
        if random.random() < 0.50:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            condi = condi.transpose(Image.FLIP_LEFT_RIGHT)
        '''
        if self.size is not None:
            #image = image.resize((self.size, self.size), resample=self.interpolation)
            #condi = condi.resize((self.size, self.size), resample=self.interpolation)

            w,h = image.size

            if w <= h:
                h = h*(self.size+self.size//8)//w
                w = self.size+self.size//8
            else:
                w = w*(self.size+self.size//8)//h
                h = self.size+self.size//8

            image = image.resize((w,h), resample=self.interpolation)
            condi = condi.resize((w,h), resample=self.interpolation)

            x = random.randint(0,w-self.size)
            y = random.randint(0,h-self.size)

            image = image.crop((x,y,x+self.size,y+self.size))
            condi = condi.crop((x,y,x+self.size,y+self.size))

        image = np.array(image).astype(np.uint8)
        condi = np.array(condi).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        example["condi"] = (condi / 127.5 - 1.0).astype(np.float32)
        return example
