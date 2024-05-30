import numpy as np
import torch
import rasterio
import onnxruntime
import torchgeo.datasets
from typing import List
import os

MODEL_PATHS = {
    "WATER_SEGMENTATION_MODEL": os.path.abspath(os.path.join(os.path.dirname(__file__), "model_mini_img_ii30.onnx")),
}


class AbstractModel(object):
    _paths_to_images: List[str]
    _path_to_model: str

    def __init__(self):
        super().__init__()
        self._paths_to_images = []
        self._path_to_model = ""

    @property
    def path(self):
        return self._path_to_model

    @path.setter
    def path(self, path):
        self._path_to_model = path

    @property
    def images(self):
        return self._paths_to_images

    @images.setter
    def images(self, paths):
        self._paths_to_images = paths

    def get_result(self):
        return self._predict()

    def _predict(self):
        raise NotImplementedError


class SegmentationModel(AbstractModel):
    def __init__(self):
        super().__init__()

        self.__ort_session = None

    def _predict(self):
        self.__ort_session = onnxruntime.InferenceSession(
            self._path_to_model,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

        for path in self._paths_to_images:
            img = SegmentationModel.convert_image(path)

            model_output = self.__get_model_output(img)

            prediction_mask = torch.sigmoid(torch.from_numpy(model_output))
            mask = (prediction_mask > 0.5) * 1.0

            yield path, mask

    def __get_model_output(self, img):
        if torch.cuda.is_available():
            ort_inputs = {
                self.__ort_session.get_inputs()[0].name: SegmentationModel.to_numpy(
                    torch.tensor(img).to('cuda', dtype=torch.float32).unsqueeze(0)
                )
            }
        else:
            ort_inputs = {
                self.__ort_session.get_inputs()[0].name: SegmentationModel.to_numpy(
                    torch.tensor(img).to('cpu', dtype=torch.float32).unsqueeze(0)
                )
            }

        output = self.__ort_session.run(None, ort_inputs)
        output = torch.squeeze(torch.tensor(output), [1, 2])
        output = SegmentationModel.to_numpy(output)

        return output

    @staticmethod
    def convert_image(path):
        img = torchgeo.datasets.utils.rasterio_loader(path)
        img = np.array(img[:, :, 0:3], dtype='uint8')
        img = np.transpose(img, (2, 0, 1))
        img = img / 255.0

        return img

    @staticmethod
    def save_mask(mask, output_path, reference_image_path):
        with rasterio.open(reference_image_path) as ref_image:
            profile = ref_image.profile.copy()
            profile.update(dtype=rasterio.uint8, count=1)

        scaled_mask = (mask.cpu().numpy() * 255).astype(rasterio.uint8)
        scaled_mask = scaled_mask.squeeze()

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(scaled_mask, 1)

    @staticmethod
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
