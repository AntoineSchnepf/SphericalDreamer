from typing import Dict, Any, Optional, Union
import torch
import numpy as np
from PIL import Image

from transformers import AutoProcessor, AutoModelForUniversalSegmentation


class SkyMaskDetector:
    """
    Sky mask detector using OneFormer panoptic segmentation.

    Uses label_id = 119 ("sky-other-merged" in COCO).
    Model and processor are lazily instantiated once.
    """

    # -------- class-level cached objects --------
    _processor: Optional[AutoProcessor] = None
    _model: Optional[AutoModelForUniversalSegmentation] = None
    _device: Optional[torch.device] = None

    MODEL_NAME = "shi-labs/oneformer_coco_swin_large"
    SKY_LABEL_ID = 119

    # -------- internal helpers --------
    @staticmethod
    def _ensure_model(device: Optional[Union[str, torch.device]] = None):
        """Instantiate processor and model once."""
        if SkyMaskDetector._processor is None or SkyMaskDetector._model is None:
            SkyMaskDetector._processor = AutoProcessor.from_pretrained(
                SkyMaskDetector.MODEL_NAME
            )
            SkyMaskDetector._model = AutoModelForUniversalSegmentation.from_pretrained(
                SkyMaskDetector.MODEL_NAME
            )

            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device(device)

            SkyMaskDetector._device = device
            SkyMaskDetector._model.to(device)
            SkyMaskDetector._model.eval()

    @staticmethod
    def _to_pil(image: Union[Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
        """Convert input to PIL Image."""
        if isinstance(image, Image.Image):
            return image

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[-1] in (3, 4):
                return Image.fromarray(image.astype(np.uint8))
            raise ValueError("NumPy image must be H×W×3 or H×W×4")

        raise TypeError("image must be PIL.Image, numpy array, or torch tensor")

    @staticmethod
    def _mask_from_label_id(
        panoptic_segmentation: Dict[str, Any],
        target_label_id: int,
    ) -> torch.Tensor:
        """
        Create boolean mask for a given label_id from OneFormer panoptic output.
        """
        seg = panoptic_segmentation["segmentation"]
        segments_info = panoptic_segmentation.get("segments_info", [])

        if not torch.is_tensor(seg):
            seg = torch.as_tensor(seg)

        target_segment_ids = [
            s["id"] for s in segments_info if s.get("label_id") == target_label_id
        ]

        if len(target_segment_ids) == 0:
            return torch.zeros(seg.shape, dtype=torch.bool, device=seg.device)

        ids = torch.tensor(target_segment_ids, device=seg.device, dtype=seg.dtype)
        return (seg[..., None] == ids).any(dim=-1)

    # -------- public API --------
    @staticmethod
    @torch.no_grad()
    def get_mask(
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        *,
        device: Optional[Union[str, torch.device]] = None,
        return_numpy: bool = True,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Compute a sky mask (label_id=119) from an input image.

        Parameters
        ----------
        image : PIL.Image | np.ndarray | torch.Tensor
            RGB image.
        device : optional
            Override device ("cuda", "cpu", torch.device).
        return_numpy : bool
            If True, return NumPy bool array.

        Returns
        -------
        mask : torch.BoolTensor[H,W] or np.ndarray[H,W]
            Boolean sky mask.
        """
        SkyMaskDetector._ensure_model(device)

        image_pil = SkyMaskDetector._to_pil(image)

        inputs = SkyMaskDetector._processor(
            images=image_pil,
            task_inputs=["panoptic"],
            return_tensors="pt",
        )

        inputs = {
            k: v.to(SkyMaskDetector._device) if torch.is_tensor(v) else v
            for k, v in inputs.items()
        }

        outputs = SkyMaskDetector._model(**inputs)

        panoptic = SkyMaskDetector._processor.post_process_panoptic_segmentation(
            outputs,
            target_sizes=[image_pil.size[::-1]],
        )[0]

        mask = SkyMaskDetector._mask_from_label_id(
            panoptic, SkyMaskDetector.SKY_LABEL_ID
        )

        if return_numpy:
            return mask.detach().cpu().numpy().astype(bool)

        return mask
    

if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt
    image = Image.open("/home/a.schnepf/phd/LayerPano3D/img_test_seg.png")

    sky_mask = SkyMaskDetector.get_mask(image)
    plt.imshow(sky_mask, cmap="gray")
    plt.show()  