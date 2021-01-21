import pandas as pd
import os
import config
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(config.INPUT_PATH, r'train.csv'))

    #shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)

    df["kfold"] = -1

    #KFold object
    kf = model_selection.KFold(n_splits=5)

    #Applying KFold
    for f, (t_, v_) in enumerate(kf.split(X=df)):
        df.loc[v_, "kfold"] = f

    #Shuffling again
    df = df.sample(frac=1).reset_index(drop=True)

#    print(df.head())

    df.to_csv(os.path.join(config.INPUT_PATH, r'train_folds.csv'))