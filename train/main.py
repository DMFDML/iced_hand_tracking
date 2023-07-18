import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import pickle
import os

dir_path = os.path.dirname(os.path.realpath(__file__))


def train_model(name: str = "", m=None):
    print("### " + name + " ###")
    print(m.shape)
    PLR_model = make_pipeline(
        PolynomialFeatures(degree=2), LinearRegression(fit_intercept=False)
    )
    PLR_model.fit(m[:, 0:6], m[:, 6:9])
    predictions = PLR_model.predict(m[:-30, 0:6])
    rmse = np.sqrt(mean_squared_error(m[:-30, 6:9], predictions))
    print("RMSE:", rmse)
    with open(dir_path + "/../tmp/" + name + ".p", "wb") as f:
        pickle.dump(PLR_model, f)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(m[:-30, 6], m[:-30, 7], m[:-30, 8], marker="o")
    ax.scatter(predictions[:, 0], predictions[:, 1], predictions[:, 2], marker="o")
    ax.set_xlabel("X")
    ax.set_xlabel("Y")
    ax.set_xlabel("Z")
    plt.show()


def train_model_two_cameras(name: str = "", m=None):
    print("### " + name + " ###")
    print(m.shape)
    PLR_model = make_pipeline(
        PolynomialFeatures(degree=2), LinearRegression(fit_intercept=False)
    )
    PLR_model.fit(m[:, 0:4], m[:, 4:7])
    predictions = PLR_model.predict(m[:-30, 0:4])
    rmse = np.sqrt(mean_squared_error(m[:-30, 4:7], predictions))
    print("RMSE:", rmse)
    with open(dir_path + "/../tmp/" + name + ".p", "wb") as f:
        pickle.dump(PLR_model, f)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(m[:-30, 4], m[:-30, 5], m[:-30, 6], marker="o")
    ax.scatter(predictions[:, 0], predictions[:, 1], predictions[:, 2], marker="o")
    ax.set_xlabel("X")
    ax.set_xlabel("Y")
    ax.set_xlabel("Z")
    plt.show()


if __name__ == "__main__":
    m = np.genfromtxt(dir_path + "/../tmp/input_tensors.csv", delimiter=",")
    n = np.genfromtxt(dir_path + "/../tmp/output_tensors.csv", delimiter=",")
    m = m / 1000
    m = np.concatenate((m, n), axis=1)

    PLR_model = make_pipeline(
        PolynomialFeatures(degree=2), LinearRegression(fit_intercept=False)
    )

    rows, cols = m.shape

    print(m.shape)
    camera_one_mask = np.isnan(m[:, 0:3]).any(axis=1)
    camera_two_mask = np.isnan(m[:, 3:5]).any(axis=1)
    camera_three_mask = np.isnan(m[:, 4:6]).any(axis=1)

    ### cameras 1 & 2
    mask = np.zeros(rows, dtype=bool)
    for i, b in enumerate(camera_one_mask):
        if b:
            mask[i] = b
    for i, b in enumerate(camera_two_mask):
        if b:
            mask[i] = b
    cameras_12 = []
    for i, b in enumerate(mask):
        if not b:
            cameras_12.append(
                [m[i, 0], m[i, 1], m[i, 2], m[i, 3], m[i, 6], m[i, 7], m[i, 8]]
            )
    cameras_12 = np.asarray(cameras_12)
    train_model_two_cameras("cameras_12", cameras_12)

    ### cameras 2 & 3
    mask = np.zeros(rows, dtype=bool)
    for i, b in enumerate(camera_two_mask):
        if b:
            mask[i] = b
    for i, b in enumerate(camera_three_mask):
        if b:
            mask[i] = b
    cameras_23 = []
    for i, b in enumerate(mask):
        if not b:
            cameras_23.append(
                [m[i, 2], m[i, 3], m[i, 4], m[i, 5], m[i, 6], m[i, 7], m[i, 8]]
            )
    cameras_23 = np.asarray(cameras_23)
    train_model_two_cameras("cameras_23", cameras_23)

    ### cameras 1 & 3
    mask = np.zeros(rows, dtype=bool)
    for i, b in enumerate(camera_one_mask):
        if b:
            mask[i] = b
    for i, b in enumerate(camera_three_mask):
        if b:
            mask[i] = b
    cameras_13 = []
    for i, b in enumerate(mask):
        if not b:
            cameras_13.append(
                [m[i, 0], m[i, 2], m[i, 4], m[i, 5], m[i, 6], m[i, 7], m[i, 8]]
            )
    cameras_13 = np.asarray(cameras_13)
    train_model_two_cameras("cameras_13", cameras_13)

    ### all_camera_model
    mask = np.isnan(m).any(axis=1)
    all_cameras = m[~mask]
    train_model("all_cameras_model", all_cameras)
