import pandas as pd
from sklearn.preprocessing import LabelEncoder , MinMaxScaler
from sklearn.model_selection import StratifiedKFold

def read_dataset():
    data = pd.read_csv("data.csv")

    data.dropna(axis = 1,inplace = True)

    label = data["diagnosis"]

    data = data.drop(columns = ["id" , "diagnosis"])

    labelencoder = LabelEncoder()
    scaller = MinMaxScaler(feature_range = (0 , 1))

    data = scaller.fit_transform(data)
    label = labelencoder.fit_transform(label)

    return data , label


if __name__ == "__main__":
    data = pd.read_csv("data.csv")

    data.dropna(axis=1, inplace=True)

    label = data["diagnosis"]

    data = data.drop(columns=["id", "diagnosis"])

    labelencoder = LabelEncoder()
    scaller = MinMaxScaler(feature_range=(0, 1))

    data = scaller.fit_transform(data)
    label = labelencoder.fit_transform(label)

    kfold = StratifiedKFold(n_splits = 4)

    for train , test in kfold.split(data , label):
        print(len(train) , len(test))