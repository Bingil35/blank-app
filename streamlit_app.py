import streamlit as st

st.title("Model CNN biospy!!!")
st.write(
    "This my first app. The application is model CNN biospy."
)

CLASS_NAME = [
    'Dyskeratotic',
    'Koilocytotic',
    'Metaplastic',
    'Parabasal',
    'Superficial-Intermediate'
]

IMG_PATH = "IMG_PATH.ccc"

MODEL_NAME = "model_weights.pth"

def predict(model_name, img_path):
    model = models.resnet152(pretrained=True)
    model.fc = nn.Linear(in_features=2048, out_features=5, bias=True)

    weights = torch.load(model_name, map_locations = 'cpu')
    model.load_state_dict(weights)

    prep_img_mean = [0.485, 0.456, 0.406]
    prep_img_std = [0.229, 0.224, 0.225]
    transform = transform.Compose (
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=prep_img_mean, std=prep_img_std)
        ]
    )

    image = Image.open(img_path)
    preprocessed_image = transforms(image).unsqueeze(0)

    model.eval()
    output = model(preprocessed_image)

    pred_idx = torch.argmax(ouput, dim=1)
    predicted_class = CLASS_NAME(pred_idx)
    return predicted_class
