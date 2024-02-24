import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from setup import TRAINING_SIZE, TEST_SIZE

# TODO: This needs cleaning up.


def load_dataset():
    df = pd.read_csv(
        "./datasets/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX_filtered.csv"
    )
    # will need to change the pathway when I find the correct file

    df

    df["Label"] = df[" Label"]
    df["Label"].value_counts()

    """### Data sampling
    Due to the space limit of GitHub files, we sample a small-sized subset for model learning using random sampling
    """

    # Randomly sample instances from majority classes
    df_minor = df[
        (df["Label"] == "WebAttack")
        | (df["Label"] == "Bot")
        | (df["Label"] == "Infiltration")
    ]
    df_BENIGN = df[(df["Label"] == "BENIGN")]
    df_BENIGN = df_BENIGN.sample(
        n=None, frac=0.01, replace=False, weights=None, random_state=None, axis=0
    )
    df_DoS = df[(df["Label"] == "DDoS")]
    df_DoS = df_DoS.sample(
        n=None, frac=0.05, replace=False, weights=None, random_state=None, axis=0
    )
    df_PortScan = df[(df["Label"] == "PortScan")]
    df_PortScan = df_PortScan.sample(
        n=None, frac=0.05, replace=False, weights=None, random_state=None, axis=0
    )
    df_BruteForce = df[(df["Label"] == "BruteForce")]
    df_BruteForce = df_BruteForce.sample(
        n=None, frac=0.2, replace=False, weights=None, random_state=None, axis=0
    )

    df_DoS

    df_s = pd.concat([df_BENIGN, df_DoS, df_PortScan, df_BruteForce, df_minor])

    df_s = df_s.sort_index()

    # Save the sampled dataset
    df_s.to_csv("./datasets/MachineLearningCSV.csv", index=0)
    # Will also need to create a new pathway

    """### Preprocessing (normalization and padding values)"""

    # Min-max normalization
    numeric_features = df.dtypes[df.dtypes != "object"].index
    df[numeric_features] = df[numeric_features].apply(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )
    # Fill empty values by 0
    df = df.fillna(0)

    """### split train set and test set"""

    labelencoder = LabelEncoder()
    y = labelencoder.fit_transform(df.iloc[:, -1])
    X = df.drop(["Label", " Label"], axis=1).values
    y = np.ravel(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=0, stratify=y
    )

    return df, X_train, X_test, y_train, y_test


def sample_data(train_data, test_data, num_sets):
    (x_train, y_train), (x_test, y_test) = train_data, test_data
    new_x_train, new_y_train = [], []
    new_x_test, new_y_test = [], []
    for i in range(num_sets):
        x_temp, y_temp = resample(
            x_train, y_train, n_samples=TRAINING_SIZE, random_state=0
        )
        new_x_train.append(x_temp)
        new_y_train.append(y_temp)
        x_temp, y_temp = resample(x_test, y_test, n_samples=TEST_SIZE, random_state=0)
        new_x_test.append(x_temp)
        new_y_test.append(y_temp)
    return (new_x_train, new_y_train), (new_x_test, new_y_test)
