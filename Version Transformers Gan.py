import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import pandas as pd
import numpy as np
import Net
import matplotlib.pyplot as plt
from sklearn import svm
from accelerate import Accelerator


def normalize(array_):
    array_[:, 0] = (array_[:, 0] - 33.958) / (3458.858 - 33.958)
    array_[:, 1] = (array_[:, 1] - 2.248) / (3243.576 - 2.248)
    array_[:, 2] = (array_[:, 2] - 0.462) / (2837.004 - 0.462)
    array_[:, 3] = (array_[:, 3] - 48.266) / (72.261 - 48.266)
    array_[:, 4] = (array_[:, 4] - 69.722) / (77.023 - 69.722)
    array_[:, 5] = (array_[:, 5] - 68.702) / (76.071 - 68.702)
    array_[:, 6] = (array_[:, 6] - 66.837) / (96.819 - 66.837)
    array_[:, 7] = (array_[:, 7] - 41.22) / (62.194 - 41.22)
    array_[:, 8] = (array_[:, 8] - 0) / (90.389 - 0)
    array_[:, 9] = (array_[:, 9] - 64.3) / (2595 - 64.3)
    array_[:, 10] = (array_[:, 10] + 0.612) / (34.079 + 0.612)
    # array_[:, 11] = array_[:, 11] / 5
    return array_

accelerator = Accelerator()
# device = 'cpu'
device = accelerator.device
# ###################Nets####################
# G net                                     #
G = Net.ViT(  #
    image_size=224,  #
    patch_size=16,  #
    num_classes=11,  #
    dim=1024,  #
    depth=8,  #
    heads=16,  #
    mlp_dim=2048,  #
    dropout=0.3,  #
    emb_dropout=0.3  #
).to(device)  #
# D net                                     #
D = Net.conv1().to("cuda")  #
# diagnosis                                 #
diagnosis = Net.diagnosis().to(device)  #
#############################################

# ---------------config--------------
# few short samples
num_sample = 30

# path
ds = pd.read_csv("./test.csv")

# chose and normalize
x_0 = ds[ds["label"] == 0][:num_sample]
x_1 = ds[ds["label"] == 1][:num_sample]
x_2 = ds[ds["label"] == 2][:num_sample]
x_3 = ds[ds["label"] == 3][:num_sample]
x_4 = ds[ds["label"] == 4][:num_sample]
x_5 = ds[ds["label"] == 5][:num_sample]
x_0 = pd.merge(x_0, x_1, "outer")
x_0 = pd.merge(x_0, x_2, "outer")
x_0 = pd.merge(x_0, x_3, "outer")
x_0 = pd.merge(x_0, x_4, "outer")
x_0 = pd.merge(x_0, x_5, "outer")
Few_short_ds = normalize(x_0.values)

x_1 = pd.merge(x_1, x_2, "outer")
x_1 = pd.merge(x_1, x_3, "outer")
x_1 = pd.merge(x_1, x_4, "outer")
x_1 = pd.merge(x_1, x_5, "outer")

# -----------------------train Classifier-------------------------------------------
print("Train Few-short classifier")

model = Net.diagnosis_model().to(device)
optimizer_1 = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
loss_f = nn.CrossEntropyLoss()

# dataset and one hot
Fw_ds = Few_short_ds[:, :-1]
Fw_label = Few_short_ds[:, -1]
# one hot
b = list(map(int, Fw_label))
label_ont_hot = np.eye(6)[b]

train_input = DataLoader(Fw_ds, batch_size=10, num_workers=0, drop_last=True)
train_label = DataLoader(label_ont_hot, batch_size=10, num_workers=0, drop_last=True)
loss_f1 = []
model, optimizer_1, train_input,  train_label = accelerator.prepare(model, optimizer_1, train_input, train_label)
for __ in range(100):
    # print("Now epoch is ", epoch)
    for batch, (data, label) in enumerate(zip(train_input, train_label)):
        optimizer_1.zero_grad()

        x_ = data.to('cuda', dtype=torch.float)
        y_ = label.to('cuda', dtype=torch.float)

        predict = model(x_)
        loss_ = loss_f(predict, y_)

        accelerator.backward(loss_)
        # loss_.backward()
        optimizer_1.step()

        loss_f1.append(loss_.item())
    print(sum(loss_f1) / len(loss_f1))
    loss_f1 = []

# ------------------------------------------------------------------
print("---------------------------------------------------------------")
clf = svm.SVC(C=75)
clf.fit(Fw_ds, Fw_label)
print("Classifier is train done! ----- Now is train GAN!")
print("---------------------------------------------------------------")

# ------------------lord data------------------------------
checkpoint = torch.load("./save/AHU")
diagnosis.load_state_dict(checkpoint['model_dict'])
print("load weights done!")
# -------------------------------------------

lr = 0.001
epoch = 200000

need_class = 4

# train_datasets = ds[ds["label"] == need_class][:num_sample].values[:, :-1]
x_1 = normalize(x_1.values)
train_datasets = x_1[:, :-1]

train_dataloder_input = DataLoader(train_datasets, batch_size=1, num_workers=0, drop_last=True)

optimizer_G = torch.optim.SGD(G.parameters(), lr=lr, momentum=0.9)
optimizer_D = torch.optim.SGD(D.parameters(), lr=lr, momentum=0.9)
# optimizer_co = torch.optim.SGD(co.parameters(), lr=0.001, momentum=0.9)
D, G, optimizer_G, optimizer_D, train_dataloder_input = accelerator.prepare(D, G, optimizer_G, optimizer_D, train_dataloder_input)

mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()

D_loss = []
vitGan_loss = []
epoch1 = []
result = []
result1 = []
vitGan_loss_ = []
for _ in range(epoch):
    print("----------------------epoch:%s-------------------------" % _)
    for data in train_dataloder_input:
        y_ = data.to("cuda", dtype=torch.float)
        x_ = torch.randn(1, 3, 224, 224).to("cuda")
        # x_ = torch.randn(1, 11).to("cuda")
        real_label_ = (torch.randint(low=90, high=100, size=[1, 1]) / 100).to(device="cuda", dtype=torch.float)
        fake_label_ = (torch.randint(low=0, high=10, size=[1, 1]) / 100).to(device="cuda", dtype=torch.float)
        G_out = G(x_)

        # -------------------------------D train------------------------------
        D_out = D(G_out)
        # D_out = D_out.view(-1)
        D_fake_loss = bce_loss(D_out, fake_label_)

        D_out2 = D(y_)
        # D_out2 = D_out2.view(-1)
        D_real_loss = bce_loss(D_out2, real_label_)

        loss_D = D_fake_loss + D_real_loss

        accelerator.backward(loss_D)
        # loss_D.backward()
        optimizer_D.step()

        D_loss.append(loss_D.item())

        optimizer_D.zero_grad()
        optimizer_G.zero_grad()

        G_out3 = G(x_)
        D_out3 = D(G_out3)

        loss_vitGan = bce_loss(D_out3, real_label_)

        err_loss = loss_vitGan

        # err_loss.backward()
        accelerator.backward(err_loss)
        optimizer_G.step()
        vitGan_loss.append(err_loss.item())

        if len(vitGan_loss) % 5 == 0:
            print("D_loss:", sum(D_loss) / len(D_loss))
            D_loss = []
            loss2 = sum(vitGan_loss) / len(vitGan_loss)
            print("vitGan loss:", loss2)
            vitGan_loss_.append(loss2)
            vitGan_loss = []

    # ----------------check----------------------------
    if _ % 5 == 0:
        for i in range(10):
            k_1 = torch.randn(100, 3, 224, 224).to("cuda")
            with torch.no_grad():
                Gene_out = G(k_1)
                # k_input = diagnosis(Gene_out)
                # k_input = model(Gene_out)
                # pre = torch.topk(k_input, 1)[1].squeeze(1)
                # result += pre.tolist()
                Gene_out_ = Gene_out.cpu().numpy()
                label_index = clf.predict(Gene_out_)
                result += list(label_index)
            # print("Now is check epoch %s. Diagnosis's numbers is %s." % (i, result.count(need_class)))

        # print("1's numbers is %s." % (result.count(1)))
        # print("2's numbers is %s." % (result.count(2)))
        # print("3's numbers is %s." % (result.count(3)))
        # print("4's numbers is %s." % (result.count(4)))
        # print("5's numbers is %s." % (result.count(5)))

        # ---------------------save model-----------------------------
        num_ = result.count(need_class)

        if num_ > 50:
            torch.save({
                'netG_state_dict': G.state_dict(),
                'netD_state_dict': D.state_dict(),
                'optimizerD_state_dict': optimizer_D.state_dict(),
                'optimizerG_state_dict': optimizer_G.state_dict()},
                "./save/model%s_%s" % (num_sample, num_))
            print("Model has saved successfully!")
        result = []
        result1 = []

plt.plot(vitGan_loss_, label="loss")
plt.xlabel('Epochs')
plt.ylabel('Train Loss')
plt.legend()
plt.show()

torch.save({
    'netG_state_dict': G.state_dict(),
    'netD_state_dict': D.state_dict(),
    'optimizerD_state_dict': optimizer_D.state_dict(),
    'optimizerG_state_dict': optimizer_G.state_dict()},
    "./save/model")

print("Model has saved successfully!")
