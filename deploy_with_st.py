import streamlit as st
from skimage import io
from skimage.transform import resize
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image = Image.open('cry.png')
st.image(image, caption="我只看得懂 0~9 嗚嗚嗚")

# 建立模型
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 256) # 完全連接層
        self.act1 = F.relu
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = torch.nn.Linear(256, 10) # 完全連接層


    def forward(self, x):
        # 完全連接層 + dropout + 完全連接層 + dropout + log_softmax
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

# 建立模型物件
model = Net()

# 模型載入

@st.cache(allow_output_mutation=True)
def load_model():
    # 建立模型
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(28 * 28, 256)  # 完全連接層
            self.act1 = F.relu
            self.bn1 = nn.BatchNorm1d(256)
            self.dropout1 = nn.Dropout(0.2)
            self.fc2 = torch.nn.Linear(256, 10)  # 完全連接層

        def forward(self, x):
            # 完全連接層 + dropout + 完全連接層 + dropout + log_softmax
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = self.act1(x)
            x = self.bn1(x)
            x = self.dropout1(x)
            x = self.fc2(x)
            return x

    # 建立模型物件
    model = Net()
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    model = model.to(device)
    return model

model = load_model()

# 標題
st.title("上傳圖片(0~9)辨識")

# 上傳圖檔
uploaded_file = st.file_uploader("上傳圖片(.png)", type="png")
if uploaded_file is not None:
    # 讀取上傳圖檔
    image1 = io.imread(uploaded_file, as_gray=True)

    # 縮為 (28, 28) 大小的影像
    image_resized = resize(image1, (28, 28), anti_aliasing=True)
    X1 = image_resized.reshape(1,28, 28) #/ 255.0

    # 反轉顏色，顏色0為白色，與 RGB 色碼不同，它的 0 為黑色
    X1 = torch.FloatTensor(1-X1).to(device)

    # 預測
    predictions = torch.softmax(model(X1), dim=1)

    # 顯示預測結果
    st.write(f'### 預測結果:{np.argmax(predictions.detach().cpu().numpy())}')

    # 顯示上傳圖檔
    st.image(image1)

