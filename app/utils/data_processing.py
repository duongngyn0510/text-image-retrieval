import clip
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
)

BICUBIC = InterpolationMode.BICUBIC
INPUT_RESOLUTION = 224


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def preprocess_image(n_px=INPUT_RESOLUTION):
    """
    A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input

    Args:
        n_px : int, default=224
        Input resolution

    Returns:
        preprocess : Callable[[PIL.Image], torch.Tensor]
    """
    return Compose(
        [
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def get_image_embedding(model, device, pil_image):
    image_tensor = preprocess_image()(pil_image).unsqueeze(0).to(device)
    image_embedding = model.encode_image(image_tensor)
    image_embedding = image_embedding.detach().cpu().numpy()[0].tolist()
    return image_embedding


def get_text_embedding(model, device, text_query):
    text_token = clip.tokenize(text_query)
    text_embedding = model.encode_text(text_token.to(device))
    text_embedding = text_embedding.detach().cpu().numpy()[0].tolist()
    return text_embedding
