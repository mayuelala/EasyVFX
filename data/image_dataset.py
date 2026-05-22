# import kornia
import os
import glob
import numpy as np

import torch
from PIL import Image
from datasets import load_dataset
import random
from transformers import CLIPTokenizer, CLIPImageProcessor
import cv2
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode
import torchvision.transforms as TT
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer
from typing import Union, Optional, List




class BrushNetDataset():
    def __init__(self, resolution, data_dir, batch_size, num_workers, tokenizer, max_sequence_length):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_len = len(os.listdir(self.data_dir))

        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.resolution = resolution

    def train_dataset(self):
        file_pattern = os.path.join(self.data_dir, "*.tar")
        file_list = glob.glob(file_pattern)
        random.shuffle(file_list)
        train_dataset = load_dataset("webdataset", 
                        data_files={"train": file_list}, 
                        split="train", 
                        streaming=True,
                        )
        return train_dataset
    
    def train_dataloader(self):
        file_pattern = os.path.join(self.data_dir, "*.tar")
        file_list = glob.glob(file_pattern)
        random.shuffle(file_list)
        train_dataset = load_dataset("webdataset", 
                        data_files={"train": file_list}, 
                        split="train", 
                        streaming=True,
                        )
        train_dataset_len= len(os.listdir(self.data_dir))
        self.dataset_len = train_dataset_len
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=False,
            collate_fn=MyWebDataset(
                resolution = self.resolution, 
                tokenizer=self.tokenizer,
                max_sequence_length=self.max_sequence_length,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        return train_dataloader


class MyWebDataset():
    def __init__(
            self, 
            resolution,
            t_drop_rate=0.05, 
            i_drop_rate=0.05, 
            ti_drop_rate=0.05, 
            # for CogVideo
            height=480,
            width=720,
            video_reshape_mode: str = "center",
            tokenizer=None,
            max_sequence_length=266,
        ):
        self.resolution = resolution
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.clip_image_processor = CLIPImageProcessor()

        self.height = height
        self.width = width
        self.video_reshape_mode = video_reshape_mode
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
    
    def _resize_for_rectangle_crop(self, arr):
        image_size = self.height, self.width
        reshape_mode = self.video_reshape_mode
        if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
            arr = resize(
                arr,
                size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            arr = resize(
                arr,
                size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                interpolation=InterpolationMode.BICUBIC,
            )

        h, w = arr.shape[2], arr.shape[3]
        arr = arr.squeeze(0)

        delta_h = h - image_size[0]
        delta_w = w - image_size[1]

        if reshape_mode == "random" or reshape_mode == "none":
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif reshape_mode == "center":
            top, left = delta_h // 2, delta_w // 2
        else:
            raise NotImplementedError
        arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
        return arr
    
    def call_single_example(self, example, pixel_values, input_ids, image_values, captions):
        if random.random()>0.1 and 'qwen_caption' in example.keys():
            caption=example["qwen_caption"].decode('utf-8')
        else:
            caption=example["caption"].decode('utf-8')
        image = cv2.imdecode(np.asarray(bytearray(example["image"]), dtype="uint8"), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 将图像转换为 PyTorch 的 Tensor
        image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)
        image = self._resize_for_rectangle_crop(image).unsqueeze(0).unsqueeze(0).float() / 127.5 - 1.0
        if image.max() > 1.0 or image.min() < -1.0:
            print(image.max(), image.min())
            raise ValueError("Image pixel values should be in the range [-1, 1]")
        pixel_values.append(image)
        image_values.append(image)
        captions.append(caption)
        # drop
        rand_num = random.random()
        if rand_num < 2 * self.t_drop_rate:
            caption = ""
        
        input_ids.append(
            compute_t5_ids(
                self.tokenizer,
                caption,
                self.max_sequence_length,
                requires_grad=False,
            )
        )
        return pixel_values, input_ids, image_values, captions


    def __call__(self,examples):
        images=[]
        input_ids=[]
        
        for example in examples:
            if random.random()>0.1 and 'qwen_caption' in example.keys():
                caption=example["qwen_caption"].decode('utf-8')
            else:
                caption=example["caption"].decode('utf-8')
            image = cv2.imdecode(np.asarray(bytearray(example["image"]), dtype="uint8"), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 将图像转换为 PyTorch 的 Tensor
            pixel_values = torch.tensor(image).permute(2, 0, 1).unsqueeze(0) # T C H W
            pixel_values = self._resize_for_rectangle_crop(pixel_values).unsqueeze(0).unsqueeze(0).float() / 127.5 - 1.0
            images.append(pixel_values)

            # drop
            rand_num = random.random()
            if rand_num < 2 * self.t_drop_rate:
                caption = ""

            input_ids.append(
                compute_t5_ids(
                    self.tokenizer,
                    caption,
                    self.max_sequence_length,
                    requires_grad=False,
                )
            )

        input_ids = torch.cat(input_ids)
        images = torch.cat(images, dim=0) # B T C H W

        return {
            "image": images,
            "pixel_values": images,
            "prompts": input_ids,
            "input_ids": input_ids,
        }


def _get_t5_input_ids(
    tokenizer: T5Tokenizer,
    prompt: Union[str, List[str]],
    max_sequence_length: int = 226,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")

    return text_input_ids


def get_input_ids(
    tokenizer: T5Tokenizer,
    prompt: Union[str, List[str]],
    max_sequence_length: int = 226,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    text_input_ids = _get_t5_input_ids(
        tokenizer,
        prompt=prompt,
        max_sequence_length=max_sequence_length,
        text_input_ids=text_input_ids,
    )
    return text_input_ids


def compute_t5_ids(
    tokenizer, prompt, max_sequence_length, requires_grad: bool = False
):
    if requires_grad:
        text_input_ids = get_input_ids(
            tokenizer,
            prompt,
            max_sequence_length=max_sequence_length,
        )
    else:
        with torch.no_grad():
            text_input_ids = get_input_ids(
                tokenizer,
                prompt,
                max_sequence_length=max_sequence_length,
            )
    return text_input_ids