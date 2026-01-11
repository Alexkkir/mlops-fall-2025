import base64
import io

import torch
import torchvision.transforms as transforms
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler


class PointwiseHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        super(PointwiseHandler, self).__init__()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

    def preprocess(self, data):
        """
        Process the request data to get the input image tensor.
        Data comes in as a list of dictionaries.
        """
        images = []
        for row in data:
            # Compat with different client requests
            image = row.get("data") or row.get("body")
            if isinstance(image, (bytes, bytearray)):
                image = Image.open(io.BytesIO(image))
            elif isinstance(image, str):
                # Assume base64 string if string
                # Or handle URL if needed
                image = Image.open(io.BytesIO(base64.b64decode(image)))

            image = self.transform(image)
            images.append(image)

        return torch.stack(images).to(self.device)

    def inference(self, data, *args, **kwargs):
        """
        The Inference method is invoked by TorchServe.
        """
        with torch.no_grad():
            results = self.model(data)
        return results

    def postprocess(self, data):
        """
        The post process of TorchServe converts the predicted output response to a client.
        """
        return data.tolist()
