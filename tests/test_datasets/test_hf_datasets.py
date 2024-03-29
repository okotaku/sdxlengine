import numpy as np
from mmengine.testing import RunnerTestCase
from PIL import Image
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

from diffengine.datasets import HFDataset, HFDatasetPreComputeEmbs


class TestHFDataset(RunnerTestCase):

    def test_dataset_from_local(self):
        dataset = HFDataset(
            dataset="tests/testdata/dataset", image_column="file_name")
        assert len(dataset) == 1

        data = dataset[0]
        assert data["text"] == "a dog"
        assert isinstance(data["img"], Image.Image)
        assert data["img"].width == 400

        dataset = HFDataset(
            dataset="tests/testdata/dataset",
            image_column="file_name",
            csv="metadata2.csv")
        assert len(dataset) == 1

        data = dataset[0]
        assert data["text"] == "a cat"
        assert isinstance(data["img"], Image.Image)
        assert data["img"].width == 400


class TestHFDatasetPreComputeEmbs(RunnerTestCase):

    def test_dataset_from_local(self):
        dataset = HFDatasetPreComputeEmbs(
            dataset="tests/testdata/dataset", image_column="file_name",
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
                            subfolder="text_encoder_2"),
            device="cpu")
        assert len(dataset) == 1

        data = dataset[0]
        assert "text" not in data
        assert np.array(data["prompt_embeds"]).shape == (77, 64)
        assert np.array(data["pooled_prompt_embeds"]).shape == (32,)
        assert isinstance(data["img"], Image.Image)
        assert data["img"].width == 400
