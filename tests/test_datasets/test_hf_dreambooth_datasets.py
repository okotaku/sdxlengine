import numpy as np
from mmengine.testing import RunnerTestCase
from PIL import Image
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

from diffengine.datasets import (
    HFDreamBoothDataset,
    HFDreamBoothDatasetPreComputeEmbs,
)


class TestHFDreamBoothDataset(RunnerTestCase):

    def test_dataset(self):
        dataset = HFDreamBoothDataset(
            dataset="diffusers/dog-example",
            instance_prompt="a photo of sks dog")
        assert len(dataset) == 5

        data = dataset[0]
        assert data["text"] == "a photo of sks dog"
        assert isinstance(data["img"], Image.Image)
        assert data["img"].width == 1815

    def test_dataset_from_local(self):
        dataset = HFDreamBoothDataset(
            dataset="tests/testdata/dataset_db",
            instance_prompt="a photo of sks dog")
        assert len(dataset) == 1

        data = dataset[0]
        assert data["text"] == "a photo of sks dog"
        assert isinstance(data["img"], Image.Image)
        assert data["img"].width == 400

    def test_dataset_from_local_with_csv(self):
        dataset = HFDreamBoothDataset(
            dataset="tests/testdata/dataset",
            csv="metadata.csv",
            image_column="file_name",
            instance_prompt="a photo of sks dog")
        assert len(dataset) == 1

        data = dataset[0]
        assert data["text"] == "a photo of sks dog"
        assert isinstance(data["img"], Image.Image)
        assert data["img"].width == 400


class TestHFDreamBoothDatasetPreComputeEmbs(RunnerTestCase):

    def test_dataset(self):
        dataset = HFDreamBoothDatasetPreComputeEmbs(
            dataset="diffusers/dog-example",
            instance_prompt="a photo of sks dog",
            model="hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            tokenizer_one=dict(type=AutoTokenizer.from_pretrained,
                            subfolder="tokenizer",
                            use_fast=False),
            tokenizer_two=dict(type=AutoTokenizer.from_pretrained,
                            subfolder="tokenizer_2",
                            use_fast=False),
            text_encoder_one=dict(type=CLIPTextModel.from_pretrained,
                            subfolder="text_encoder"),
            text_encoder_two=dict(type=CLIPTextModelWithProjection.from_pretrained,
                            subfolder="text_encoder_2"))
        assert len(dataset) == 5

        data = dataset[0]
        assert "text" not in data
        assert np.array(data["prompt_embeds"]).shape == (77, 64)
        assert np.array(data["pooled_prompt_embeds"]).shape == (32,)
        assert isinstance(data["img"], Image.Image)
        assert data["img"].width == 1815

    def test_dataset_from_local(self):
        dataset = HFDreamBoothDatasetPreComputeEmbs(
            dataset="tests/testdata/dataset_db",
            instance_prompt="a photo of sks dog",
            model="hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            tokenizer_one=dict(type=AutoTokenizer.from_pretrained,
                            subfolder="tokenizer",
                            use_fast=False),
            tokenizer_two=dict(type=AutoTokenizer.from_pretrained,
                            subfolder="tokenizer_2",
                            use_fast=False),
            text_encoder_one=dict(type=CLIPTextModel.from_pretrained,
                            subfolder="text_encoder"),
            text_encoder_two=dict(type=CLIPTextModelWithProjection.from_pretrained,
                            subfolder="text_encoder_2"))
        assert len(dataset) == 1

        data = dataset[0]
        assert "text" not in data
        assert np.array(data["prompt_embeds"]).shape == (77, 64)
        assert np.array(data["pooled_prompt_embeds"]).shape == (32,)
        assert isinstance(data["img"], Image.Image)
        assert data["img"].width == 400

    def test_dataset_from_local_with_csv(self):
        dataset = HFDreamBoothDatasetPreComputeEmbs(
            dataset="tests/testdata/dataset",
            csv="metadata.csv",
            image_column="file_name",
            instance_prompt="a photo of sks dog",
            model="hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            tokenizer_one=dict(type=AutoTokenizer.from_pretrained,
                            subfolder="tokenizer",
                            use_fast=False),
            tokenizer_two=dict(type=AutoTokenizer.from_pretrained,
                            subfolder="tokenizer_2",
                            use_fast=False),
            text_encoder_one=dict(type=CLIPTextModel.from_pretrained,
                            subfolder="text_encoder"),
            text_encoder_two=dict(type=CLIPTextModelWithProjection.from_pretrained,
                            subfolder="text_encoder_2"))
        assert len(dataset) == 1

        data = dataset[0]
        assert "text" not in data
        assert np.array(data["prompt_embeds"]).shape == (77, 64)
        assert np.array(data["pooled_prompt_embeds"]).shape == (32,)
        assert isinstance(data["img"], Image.Image)
        assert data["img"].width == 400
