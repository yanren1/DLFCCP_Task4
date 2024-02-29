import os
from flask import Flask, request, jsonify,render_template
from torchvision import transforms
from PIL import Image
import torch
from backbone.model import MyEfficientNet_B0,MyEfficientnet_v2_m
import io
from utils.coco_inf import multi_label_inf


app = Flask(__name__)

model = MyEfficientNet_B0(num_classes=257)
weights_pth = 'model_saved/t1_2/EfficientNet_b0_2024-02-28-02-07-10/final.pt'
model.load_state_dict(torch.load(weights_pth))
model.cuda().eval()


multi_label_model = MyEfficientnet_v2_m(pretrained=False, num_classes=80).cuda()
weights_pth = 'best.pt'
multi_label_model.load_state_dict(torch.load(weights_pth))
multi_label_model.eval()

def return_img_stream(img_local_path):
    import base64
    img_stream = ''
    with open(img_local_path, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream).decode()
    return img_stream

def read_label():
    cls_dict = {}
    with open('data/caltech256_label.txt','r') as f:
        lines = f.readlines()
        for l in lines:
            cls_dict[l.split(':')[0]] = l.split(':')[1].strip()
    return cls_dict


def makePred(img,model):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.ColorJitter(),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # img = Image.open('data/caltech256/256_ObjectCategories/257.clutter/257_0002.jpg')
    with torch.no_grad():
        img = transform(img).unsqueeze(0)
        img = img.cuda()
        # print(img.shape)
        pred = model(img)
        _, pred = torch.max(pred.data, 1)
        idx = str((pred.item()))

        cls_dict = read_label()
        print(cls_dict[idx])
    return cls_dict[idx]

@app.route('/')
def index():
    return render_template('imgcls_index.html')

@app.route('/predict', methods=['POST'])
def predict():

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        image = Image.open(io.BytesIO(image_file.read()))

        # pred = makePred(image,model)
        pred = multi_label_inf(image,multi_label_model)

        tmp_img_name = os.path.join('templates','tmp_img.jpg')
        image.save(tmp_img_name)

        img_stream = return_img_stream(tmp_img_name)
        return render_template('imgcls_index.html',img_stream = img_stream ,prediction=pred)
        # return jsonify({'prediction': pred})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(port=5001,debug=True)
