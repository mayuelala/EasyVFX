import random
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import CLIPTokenizer, CLIPImageProcessor
import cv2
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode
import torchvision.transforms as TT
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer
from data.videovo_dataset import MyWebDataset as videovo_collate_fn
from data.videovo_dataset import resize_for_rectangle_crop
from data.image_dataset import MyWebDataset as brushnet_collate_fn
import torch
import numpy as np
import pdb

class CombinedDataLoader:
    def __init__(self, loader1, loader2, prob_loader1=0.3, batch_size=1):
        self.loader1 = loader1
        self.loader2 = loader2
        self.prob_loader1 = prob_loader1
        self.iter_a = iter(self.loader1)
        self.iter_b = iter(self.loader2)
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if random.random() < self.prob_loader1:
            try:
                return next(self.iter_a)
            except StopIteration:
                self.iter_a = iter(self.loader1)
                return next(self.iter_a)
        else:
            try:
                return next(self.iter_b)
            except StopIteration:
                self.iter_b = iter(self.loader2)
                return next(self.iter_b)
            
class CombineDataset(Dataset):
    def __init__(self, dataset1, dataset2, prob_set1=0.5):
        self.dataset1 = dataset1.train_dataset()
        self.dataset2 = dataset2
        self.prob_set1 = prob_set1
        self.dataset_len = dataset2.__len__()
        self.dataset1_iter = iter(self.dataset1)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        if random.random() < self.prob_set1:
            return next(self.dataset1_iter)
        else:
            return self.dataset2[idx]
        
class Combine_Collate_Fn():
    def __init__(self,resolution, tokenizer, max_num_frames, max_sequence_length, proportion_empty_prompts, is_train=True, emptytxt=False, use_different_first_frame=False):

        self.resolution = resolution
        self.tokenizer = tokenizer
        self.max_num_frames = max_num_frames
        self.max_sequence_length = max_sequence_length
        self.proportion_empty_prompts = proportion_empty_prompts
        self.is_train = is_train
        self.emptytxt = emptytxt

        self.videovo_collate_fn = videovo_collate_fn(
            resolution = resolution, 
            tokenizer=tokenizer,
            max_num_frames=max_num_frames,
            max_sequence_length=max_sequence_length,
            proportion_empty_prompts=proportion_empty_prompts,
            is_train=is_train,
            emptytxt=emptytxt,
            use_different_first_frame=use_different_first_frame,
        )

        self.brushnet_collate_fn = brushnet_collate_fn(
            resolution = resolution, 
            tokenizer=tokenizer,
            max_sequence_length=max_sequence_length,
        )

    def __call__(self, examples):
        pixel_values=[]
        input_ids=[]
        image_values=[]
        captions=[]
        for example in examples:
            if "video" in example.keys():
                pixel_values, input_ids, image_values, captions = self.videovo_collate_fn.call_single_example(example, pixel_values, input_ids, image_values, captions)
            else:
                pixel_values, input_ids, image_values, captions = self.brushnet_collate_fn.call_single_example(example, pixel_values, input_ids, image_values, captions)

        input_ids = torch.cat(input_ids)
        pixel_values = torch.cat(pixel_values) # B T C H W
        image_values = torch.cat(image_values) # B 1 C H W
        return {
            "pixel_values": pixel_values.to(memory_format=torch.contiguous_format).float(), 
            "input_ids": input_ids,
            "image_values": image_values,
            "captions": captions,
        }