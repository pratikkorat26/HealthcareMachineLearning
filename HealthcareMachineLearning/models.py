from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier , AdaBoostClassifier
from main import read_dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score , classification_report

def randomforestclassifier():

    data , label = read_dataset()

    models = RandomForestClassifier()
    kfold = StratifiedKFold(n_splits = 4 , shuffle = True)

    accuracy = []
    report = []
    for batchidx , (train , test) in enumerate(kfold.split(data , label)):
        print(f"fold{batchidx} started")
        train_X = data[train]
        train_y = label[train]
        valid_X = data[test]
        valid_y = label[test]

        models.fit(train_X , train_y)
        output = models.predict(valid_X)

        acc = accuracy_score(valid_y , output)
        report1 = classification_report(valid_y , output)
        #df = pd.DataFrame(report1).transpose()
        accuracy.append(acc)
        report.append(report1)

        with open(f"randomforest\ReportRandomForestClassifier{batchidx}.txt" , "w") as f:
            f.write(f"accuracy: {acc*100}%\n")
            f.write(f"report : \n{report1}")
            f.close()

    return accuracy , report


def gradientbooster():
    data, label = read_dataset()

    models = GradientBoostingClassifier()

    kfold = StratifiedKFold(n_splits=4, shuffle=True)

    accuracy = []
    report = []
    for batchidx, (train, test) in enumerate(kfold.split(data, label)):
        print(f"fold{batchidx} started")
        train_X = data[train]
        train_y = label[train]
        valid_X = data[test]
        valid_y = label[test]

        models.fit(train_X, train_y)
        output = models.predict(valid_X)

        acc = accuracy_score(valid_y, output)
        report1 = classification_report(valid_y, output)
        # df = pd.DataFrame(report1).transpose()
        accuracy.append(acc)
        report.append(report1)

        with open(f"GradientBoosting\ReportGradientBoostingClassifier{batchidx}.txt", "w") as f:
            f.write(f"accuracy: {acc * 100}%\n")
            f.write(f"report : \n{report1}")
            f.close()

    return accuracy, report

def adabooster():
    data, label = read_dataset()

    models = AdaBoostClassifier()

    kfold = StratifiedKFold(n_splits=4, shuffle=True)

    accuracy = []
    report = []
    for batchidx, (train, test) in enumerate(kfold.split(data, label)):
        print(f"fold{batchidx} started")
        train_X = data[train]
        train_y = label[train]
        valid_X = data[test]
        valid_y = label[test]

        models.fit(train_X, train_y)
        output = models.predict(valid_X)

        acc = accuracy_score(valid_y, output)
        report1 = classification_report(valid_y, output)
        # df = pd.DataFrame(report1).transpose()
        accuracy.append(acc)
        report.append(report1)

        with open(f"AdaBoosting\ReportAdaBoostingClassifier{batchidx}.txt", "w") as f:
            f.write(f"accuracy: {acc * 100}%\n")
            f.write(f"report : \n{report1}")
            f.close()

    return accuracy, report


if __name__ == "__main__":
    accuracy , report = adabooster()
