from clip import clip
import torch

def build_clip_templates(dataset_name):

    CUSTOM_TEMPLATES = {
        "OxfordPets": "a photo of a {}, a type of pet.",
        "OxfordFlowers": "a photo of a {}, a type of flower.",
        "FGVCAircraft": "a photo of a {}, a type of aircraft.",
        "DescribableTextures": "a photo of a {}, a type of texture.",
        "EuroSAT": "a centered satellite photo of {}.",
        "StanfordCars": "a photo of a {}.",
        "Food101": "a photo of {}, a type of food.",
        "SUN397": "a photo of a {}.",
        "Caltech101": "a photo of a {}.",
        "UCF101": "a photo of a person doing {}.",
        "ImageNet": "a photo of a {}.",
        "ImageNetSketch": "a photo of a {}.",
        "ImageNetV2": "a photo of a {}.",
        "ImageNetA": "a photo of a {}.",
        "ImageNetR": "a photo of a {}.",
    }
    return CUSTOM_TEMPLATES[dataset_name]


def build_zsclip(backbone_name):

    # load zero-shot CLIP model
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {"trainer": 'ZeroshotCLIP',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    
    clip_model = clip.build_model(state_dict or model.state_dict(), design_details)
    # clip_model.cuda()

    return clip_model

