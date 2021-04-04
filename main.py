import os
import torch
from torch import nn
from PIL import Image
from model import ASLNet
import torch.nn.functional as F
from torchvision import transforms
from flask import Flask, request, jsonify


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'

model = ASLNet().to(device)
model.load_state_dict(torch.load("./chkpt.pth", map_location=torch.device("cpu")))

LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']


def apply_test_transforms(inp):
    out = transforms.functional.resize(inp, [224, 224])
    out = transforms.functional.to_tensor(out)
    return out



def pred_class(image):
    img = Image.open(image)
    img_tensor = apply_test_transforms(img).to(device)
    img_tensor = img_tensor.unsqueeze(0)
    output = model(img_tensor)
    _, preds = torch.max(output.data, 1)

    return preds.item()


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        f = request.files['image']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        predict = pred_class('./static/uploads/' + f.filename)
        
        return LABELS[predict]


if __name__ == "__main__":
    app.run(debug=True)