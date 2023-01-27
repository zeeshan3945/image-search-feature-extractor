import torch
from torchvision import transforms, models
from PIL import Image
from io import BytesIO
import base64
import json

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
def init():
    global model
    # Load the pretrained VGG16 model
    print("loading to CPU...")
    model = models.vgg16(pretrained=True)

    # Extract the features, pooling, flatten, and classifier layers
    #features = list(model.features)
    features = torch.nn.ModuleList(list(model.features))
    pooling = model.avgpool
    flatten = torch.nn.Flatten()
    fc = model.classifier[0]

    # Create a sequential model with the extracted layers
    model = torch.nn.Sequential(*features, pooling, flatten, fc)

    # conditionally load to GPU
    if device == "cuda:0":
        print("loading to GPU...")
        model.cuda()
        print("done")

    # device = 0 if torch.cuda.is_available() else -1
    # model = pipeline('fill-mask', model='bert-base-uncased', device=device)

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    tr = transforms.Compose(
                [transforms.Resize(256), transforms.ToTensor(), normalize,]
    )
    
    PIL_image = prompt
    image_binary = base64.b64decode(PIL_image)
    PIL_image=Image.open(BytesIO(image_binary))

    img_tensor = tr(PIL_image.convert(
        "RGB"
    ) )
    img_expanded = torch.unsqueeze(img_tensor, dim=0)
    model.eval()
    out = torch.squeeze(model(img_expanded.to(device)).detach().cpu())

    #result = {"output" : out}
    numpy_array = out.to("cpu").numpy()
    # Return the results as a dictionary
    return json.dumps(numpy_array.tolist())
