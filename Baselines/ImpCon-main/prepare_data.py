import pandas as pd
import os
import numpy as np
import random
import nlpaug.augmenter.word as naw
import argparse
import nltk 
nltk.download('wordnet')
nltk.download('omw-1.4')

np.random.seed(0)
random.seed(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_dir', default="dataset/implicit-hate-corpus",type=str, help='Enter dataset')
    parser.add_argument('--dataset_name', default="ConvAbuse",type=str, help='dataset name')
    args = parser.parse_args()


    # load implicit hate corpus dataset
    data = pd.read_csv(args.load_dir)
    not_hate = data.groupby('label').get_group(0)
    hate = data.groupby('label').get_group(1)
    pure_set = pd.concat([hate, not_hate], join='outer')
    print(pure_set.head(5))


    #Train-test-split
    train, valid, test = np.split(pure_set.sample(frac=1, random_state=42), [int(.6*len(pure_set)), int(.8*len(pure_set))])
    # save train / valid / test set
    os.makedirs("dataset/pure", exist_ok=True)
    train.to_csv(os.path.join("dataset/pure", str(args.dataset_name)+"_train.csv"), index=False)
    valid.to_csv(os.path.join("dataset/pure", str(args.dataset_name)+"_valid.csv"), index=False)
    test.to_csv(os.path.join("dataset/pure", str(args.dataset_name)+"_test.csv"), index=False)


    # for train set, we include augmented version of posts
    train = pd.read_csv(os.path.join("dataset/pure", str(args.dataset_name)+"_train.csv"))
    aug = naw.SynonymAug(aug_src='wordnet')
    train['aug_sent1_of_post'] = pd.Series(dtype="object")
    train['aug_sent2_of_post'] = pd.Series(dtype="object")

    for i,one_post in enumerate(train["text"]):
        train['aug_sent1_of_post'][i] = aug.augment(one_post)
        train['aug_sent2_of_post'][i] = aug.augment(one_post)

    # save train set with augmented version of posts
    train.to_csv(os.path.join("dataset/pure", str(args.dataset_name)+"_train.csv"), index=False)