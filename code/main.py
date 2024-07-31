import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
import torch.optim as optim
import numpy as np
import os
import argparse
import time
from model import AE_3D_Dataset, UNet_3D
from train import training, validation, test, simulate
from utils import (
    find_weight,
    load_transfer_learning_UNet_3D,
    save_loss,
    normalize_data,
    MSE,
    plot_training,
)
import warnings
import cv2

"""
Example Usage in colab:
initializing:
```
# Clone the DL-ROM GitHub repository
!git clone https://github.com/Shen-Ming-Hong/DL-ROM.git

# Change directory to the cloned DL-ROM directory
%cd DL-ROM/

# List the contents of the current directory
!ls

# Download the dataset from the specified Google Drive folder into the 'data' directory
!gdown --folder https://drive.google.com/drive/folders/1JI4jTBM1vE9AjkdxYce0GCDG9tCi5FtQ -O data

# List the contents of the 'data' directory to confirm the download
!ls data

# Change directory to the 'code' folder within the DL-ROM directory
%cd code/
```
# For Training:
!python main.py -N 100 -B 16 -d_set 2d_cylinder_CFD --train

# For Testing:
!python main.py --test -test_epoch 100 -d_set 2d_cylinder_CFD

# For Transfer Learning:
!python main.py --transfer -N 100 -B 16 -d_set 2d_cylinder_CFD --train

# For Simulating:
!python main.py --simulate -test_epoch 100 -d_set 2d_cylinder_CFD

"""

if __name__ == "__main__":

    # arguments for num_epochs and batch_size
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", dest="num_epochs", type=int, help="Number of Epochs")
    parser.add_argument("-B", dest="batch_size", type=int, default=16, help="Batch Size")
    parser.add_argument("-d_set", dest="dataset", type=str, default="2d_cylinder_CFD", help="Name of Dataset")
    parser.add_argument("-test_epoch", dest="test_epoch", type=int, default=None, help="Epoch for testing")
    parser.add_argument("--test", dest="testing", action="store_true")
    parser.add_argument("--train", dest="training", action="store_true")
    parser.add_argument("--transfer", dest="transfer", action="store_true")
    parser.add_argument("--simulate", dest="simulate", action="store_true")

    args = parser.parse_args()
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    dataset_name = args.dataset
    test_epoch = args.test_epoch
    transfer_learning = args.transfer

    print(num_epochs, batch_size)

    if not os.path.exists(f"../results"):
        os.mkdir(f"../results")

    if not os.path.exists(f"../results/{dataset_name}"):
        os.mkdir(f"../results/{dataset_name}")

    if not os.path.exists(f"../simulate"):
        os.mkdir(f"../simulate")

    # Making folders to save reconstructed images, input images and weights
    if not os.path.exists(f"../results/{dataset_name}/output/"):
        os.mkdir(f"../results/{dataset_name}/output/")

    if not os.path.exists(f"../results/{dataset_name}/weights/"):
        os.mkdir(f"../results/{dataset_name}/weights/")

    if not os.path.exists(f"../simulate/{dataset_name}"):
        os.mkdir(f"../simulate/{dataset_name}")

    warnings.filterwarnings("ignore")

    # Running the model on CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if dataset_name == "2d_cylinder":
        u = np.load("../data/cylinder_u.npy", allow_pickle=True)[:-1, ...][:, :, 40:-280]
        u = normalize_data(u)
        # v = np.load('../data/cylinder_v.npy', allow_pickle=True)[:-1, ...]

    elif dataset_name == "boussinesq":
        ux = np.load("../data/boussinesq_u.npy", allow_pickle=True)[:-1, ...][:, 50:-80, :]
        u = np.array([cv2.resize(ux[i], (160, 320), interpolation=cv2.INTER_CUBIC) for i in range(ux.shape[0])])
        u = normalize_data(u)
        # v = np.load('../data/boussinesq_v.npy', allow_pickle=True)[:-1, ...]

    elif dataset_name == "SST":
        u = np.load("../data/sea_surface_noaa.npy", allow_pickle=True)[:2000, ...][:, 10:-10, 20:-20]
        u = normalize_data(u)

    elif dataset_name == "2d_cylinder_CFD":
        u_comp = np.load("../data/Vort100.npz", allow_pickle=True)
        # u_comp = np.load('../data/Velocity160.npz', allow_pickle=True)

        u_flat = u_comp["arr_0"]
        u = u_flat.reshape(u_flat.shape[0], 320, 80)
        u = np.transpose(u, (0, 2, 1)).astype(np.float32)
        u = normalize_data(u)

    elif dataset_name == "2d_sq_cyl":
        u_flat = np.load("../data/sq_cyl_vort.npy", allow_pickle=True)  # sq_cyl_vel
        u = u_flat.reshape(u_flat.shape[0], 320, 80)
        u = np.transpose(u, (0, 2, 1)).astype(np.float32)[:2000, ...]  # temporarily reducing dataset size
        u = normalize_data(u)

    elif dataset_name == "channel_flow":
        u = np.load("../data/channel_data_2500.npy", allow_pickle=True).astype(np.float32)
        u = normalize_data(u)

    elif dataset_name == "2d_airfoil":
        u_flat = np.load("../data/airfoil_80x320_data.npy", allow_pickle=True)
        print(u_flat.shape)
        u = u_flat.reshape(u_flat.shape[0], 320, 80)
        u = np.transpose(u, (0, 2, 1))[:, :, 140:-20].astype(np.float32)
        u = normalize_data(u)

    elif dataset_name == "2d_plate":
        u_flat = np.load("../data/platekepsilon.npy", allow_pickle=True)
        print(u_flat.shape)
        u = u_flat.reshape(u_flat.shape[0], 360, 180)
        u = np.transpose(u, (0, 2, 1))[:, :-20, :-40].astype(np.float32)
        u = normalize_data(u)

    else:
        print("Dataset Not Found")

    print(f"Data Loaded in Dataset: {dataset_name} with shape {u.shape[0]}")

    # train/val split
    train_to_val = 0.75
    # rand_array = np.random.permutation(1500)
    # print(rand_array)

    u_train = u[: int(train_to_val * u.shape[0]), ...]
    u_validation = u[int(train_to_val * u.shape[0]) :, ...]

    print(u_train.shape)
    print(u_validation.shape)

    # u = insert_time_channel(u, 10)
    # print(u.shape);

    img_transform = transforms.Compose(
        [
            # transforms.ToPILImage(),
            # transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5])
        ]
    )

    if transfer_learning:
        print("Using Transfer Learning")
        # final_model = LSTM()
        # pretrained = autoencoder()
        # PATH = "../weights/1000.pth"
        # # PATH = "../weights/bous_500.pth"
        # # pdb.set_trace()
        pre_dataset_name = "2d_cylinder_CFD"
        final_dataset_name = dataset_name
        final_model = UNet_3D(name=final_dataset_name)
        pretrained = UNet_3D(name=pre_dataset_name)

        PATH = f"../results/{pre_dataset_name}/weights/100.pth"

        model = load_transfer_learning_UNet_3D(pretrained, final_model, PATH, req_grad=False)
    else:
        model = UNet_3D(name=dataset_name)

    model = model.to(device)

    if args.training:
        # batch_size = 16
        # Train data_loader
        train_dataset = AE_3D_Dataset(u_train, dataset_name, transform=img_transform)
        train_loader_args = dict(batch_size=batch_size, shuffle=True, num_workers=4)
        train_loader = data.DataLoader(train_dataset, **train_loader_args)

        # print(len(train_loader))

        # val data_loader
        validation_dataset = AE_3D_Dataset(u_validation, dataset_name, transform=img_transform)
        val_loader_args = dict(batch_size=1, shuffle=False, num_workers=4)
        val_loader = data.DataLoader(validation_dataset, **val_loader_args)

        # Instances of optimizer, criterion, scheduler

        optimizer = optim.Adam(model.parameters(), lr=0.05)
        criterion = nn.L1Loss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            verbose=False,
            threshold=1e-3,
            threshold_mode="rel",
            cooldown=5,
            min_lr=1e-5,
            eps=1e-08,
        )

        # model.load_state_dict(torch.load(Path))
        # print(optimizer)

        Train_Loss = []
        Dev_Loss = []

        Val_loss = {}
        Train_loss = {}

        validation_freq = 1
        # Epoch loop
        for epoch in range(num_epochs):
            start_time = time.time()
            print("Epoch no: ", epoch)
            train_loss = training(model, train_loader, criterion, optimizer)

            # Saving weights after every 20epochs
            if epoch % validation_freq == 0:  # and epoch !=0:
                val_loss = validation(model, val_loader, criterion)
                Val_loss[epoch] = val_loss
                Dev_Loss.append(val_loss)

                Train_Loss.append(train_loss)
                Train_loss[epoch] = train_loss

            if epoch % 10 == 0:  # and epoch != 0:
                path = f"../results/{dataset_name}/weights/{epoch}.pth"
                torch.save(model.state_dict(), path)
                print(optimizer)

            scheduler.step(train_loss)
            print("Time : ", time.time() - start_time)
            print("=" * 100)
            print()

        # Saving Loss values as dictionaries for later analyses
        save_loss(Train_loss, dataset_name, "train")
        save_loss(Val_loss, dataset_name, "val")
        plot_training(Train_Loss, Dev_Loss)

    if args.testing:

        PATH = find_weight(dataset_name, test_epoch)

        print(PATH)

        model.load_state_dict(torch.load(PATH))

        test_dataset = AE_3D_Dataset(u_validation, dataset_name, transform=img_transform)
        test_loader_args = dict(batch_size=1, shuffle=False, num_workers=4)
        test_loader = data.DataLoader(test_dataset, **test_loader_args)

        labels, preds = test(model, test_loader)
        name = f"../results/{dataset_name}/output/labels.npy"
        np.save(name, labels)

        name = f"../results/{dataset_name}/output/predictions.npy"
        np.save(name, preds)

        MSE(dataset_name, preds, labels)

    if args.simulate:

        PATH = find_weight(dataset_name, test_epoch)

        print(PATH)

        model.load_state_dict(torch.load(PATH))

        labels, preds, mse = simulate(model, u_validation, img_transform)
        name = f"../simulate/{dataset_name}/labels.npy"
        np.save(name, labels)

        name = f"../simulate/{dataset_name}/predictions.npy"
        np.save(name, preds)

        name = f"../simulate/{dataset_name}/mse.npy"
        np.save(name, mse)
