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


import pandas as pd
import pickle
import argparse
import numpy as np
import random
import nlpaug.augmenter.word as naw

from transformers import AutoTokenizer

import numpy as np
import random
import os
import warnings
warnings.filterwarnings("ignore")

def aug_creator(load_dir,dataset_name):
    # load implicit hate corpus dataset
    data = pd.read_csv(load_dir)
    not_hate = data.groupby('label').get_group(0)
    hate = data.groupby('label').get_group(1)
    pure_set = pd.concat([hate, not_hate], join='outer')
    # print(pure_set.head(5))


    #Train-test-split
    train, valid, test = np.split(pure_set.sample(frac=1, random_state=42), [int(.6*len(pure_set)), int(.8*len(pure_set))])
    # save train / valid / test set
    os.makedirs("dataset/pure/"+str(dataset_name), exist_ok=True)
    train.to_csv(os.path.join("dataset/pure/"+str(dataset_name)+"/", str(dataset_name)+"_train.csv"), index=False)
    valid.to_csv(os.path.join("dataset/pure/"+str(dataset_name)+"/", str(dataset_name)+"_valid.csv"), index=False)
    test.to_csv(os.path.join("dataset/pure/"+str(dataset_name)+"/", str(dataset_name)+"_test.csv"), index=False)


    # for train set, we include augmented version of posts
    train = pd.read_csv(os.path.join("dataset/pure/"+str(dataset_name)+"/", str(dataset_name)+"_train.csv"))
    aug = naw.SynonymAug(aug_src='wordnet')
    train['aug_sent1_of_post'] = pd.Series(dtype="object")
    train['aug_sent2_of_post'] = pd.Series(dtype="object")

    for i,one_post in enumerate(train["text"]):
        train['aug_sent1_of_post'][i] = aug.augment(str(one_post))
        train['aug_sent2_of_post'][i] = aug.augment(str(one_post))

    # save train set with augmented version of posts
    train.to_csv(os.path.join("dataset/pure/"+str(dataset_name)+"/", str(dataset_name)+"_train.csv"), index=False)


def get_one_hot(emo, class_size):

	targets = np.zeros(class_size)
	emo_list = [int(e) for e in emo.split(",")]
	for e in emo_list:
		targets[e-1] = 1
	return list(targets)

def preprocessor(dataset,tokenizer_type,w_aug,aug_type):
    os.makedirs("preprocessed_data/"+str(dataset), exist_ok=True)
    class2int = {0.0: 0 ,1.0: 1}
    data_dict = {}
    data_home = "C:/Users/psheth5/OneDrive - Arizona State University/HateSpeech Datasets/Baselines/ImpCon-main/dataset/pure/"+str(dataset)+"/"+str(dataset)+"_"

    for datatype in ["train","valid","test"]:

        datafile = data_home + datatype + ".csv"
        print(datafile)
        data = pd.read_csv(datafile)

        label,post = [],[]
        aug_sent1_of_post = []

        for i,one_class in enumerate(data["label"]):
            label.append(class2int[one_class])
            post.append(data["text"][i])
            
        if datatype == "train" and w_aug:
            for i, one_aug_sent in enumerate(data["aug_sent1_of_post"]):
                aug_sent1_of_post.append(one_aug_sent)

            print("Tokenizing data")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
            tokenized_post = tokenizer.batch_encode_plus(post,add_special_tokens=True,truncation=True,padding=True,max_length=512).input_ids
            tokenized_post_augmented = tokenizer(aug_sent1_of_post,add_special_tokens=True,truncation=True,padding=True,max_length=512).input_ids

            tokenized_combined_prompt = [list(i) for i in zip(tokenized_post,tokenized_post_augmented)]
            combined_prompt = [list(i) for i in zip(post,aug_sent1_of_post)]
            combined_label = [list(i) for i in zip(label,label)]

            processed_data = {}

            processed_data["tokenized_post"] = tokenized_combined_prompt
            processed_data["label"] = combined_label
            processed_data["post"] = combined_prompt

            processed_data = pd.DataFrame.from_dict(processed_data)
            data_dict[datatype] = processed_data

        else:
            print("Tokenizing data")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
            tokenized_post =tokenizer.batch_encode_plus(post,add_special_tokens=True,truncation=True,padding=True,max_length=512).input_ids

            processed_data = {}

            processed_data["tokenized_post"] = tokenized_post
            processed_data["label"] = label
            processed_data["post"] = post

            processed_data = pd.DataFrame.from_dict(processed_data)
            data_dict[datatype] = processed_data

    if w_aug:
        with open("./preprocessed_data/"+str(dataset)+"/"+str(dataset)+"_waug_"+aug_type+"_preprocessed_bert.pkl", 'wb') as f:
            pickle.dump(data_dict, f)
        f.close()
    else:
        with open("./preprocessed_data/"+str(dataset)+"/"+str(dataset)+"_preprocessed_bert.pkl", 'wb') as f:
            pickle.dump(data_dict, f)
            f.close()


#files = ['C:/Users/psheth5/OneDrive - Arizona State University/HateSpeech Datasets/ConvAbuseEMNLPfull_Train_Modifiers.csv','C:/Users/psheth5/OneDrive - Arizona State University/HateSpeech Datasets/FoxNews_Train_Modifiers.csv','C:/Users/psheth5/OneDrive - Arizona State University/HateSpeech Datasets/GabHateCorpusannotations_Train_Modifiers.csv','C:/Users/psheth5/OneDrive - Arizona State University/HateSpeech Datasets/ICWSM18SALMINEN_Train_Modifiers.csv','C:/Users/psheth5/OneDrive - Arizona State University/HateSpeech Datasets/implicithatev1stg1posts_Train_Modifiers.csv','C:/Users/psheth5/OneDrive - Arizona State University/HateSpeech Datasets/Reddit_Train_Modifiers.csv','C:/Users/psheth5/OneDrive - Arizona State University/HateSpeech Datasets/DatasetsFinalSplits/WikiDetox_Train.csv',
#files = ['C:/Users/psheth5/OneDrive - Arizona State University/HateSpeech Datasets/ConvAbuseEMNLPfull_Test_Modifiers.csv','C:/Users/psheth5/OneDrive - Arizona State University/HateSpeech Datasets/FoxNews_Test_Modifiers.csv','C:/Users/psheth5/OneDrive - Arizona State University/HateSpeech Datasets/GabHateCorpusannotations_Test_Modifiers.csv','C:/Users/psheth5/OneDrive - Arizona State University/HateSpeech Datasets/ICWSM18SALMINEN_Test_Modifiers.csv','C:/Users/psheth5/OneDrive - Arizona State University/HateSpeech Datasets/implicithatev1stg1posts_Test_Modifiers.csv','C:/Users/psheth5/OneDrive - Arizona State University/HateSpeech Datasets/Reddit_Test_Modifiers.csv','C:/Users/psheth5/OneDrive - Arizona State University/HateSpeech Datasets/WikiDetox_Test_modifiers.csv', 'C:/Users/psheth5/OneDrive - Arizona State University/HateSpeech Datasets/LargeDatasets/Synthetic_test.csv','C:/Users/psheth5/OneDrive - Arizona State University/HateSpeech Datasets/LargeDatasets/Twi-Red-You_test.csv']
# files = ['C:/Users/psheth5/OneDrive - Arizona State University/HateSpeech Datasets/Dataset-01_24/WikiDetox_Train_Modifiers_HB.csv']
files = ['C:/Users/psheth5/OneDrive - Arizona State University/HateSpeech Datasets/Dataset-01_24/HateEval/HateEval_test_HB.csv',
'C:/Users/psheth5/OneDrive - Arizona State University/HateSpeech Datasets/Dataset-01_24/HateEval/migrants/migrants-test_HB.csv',
'C:/Users/psheth5/OneDrive - Arizona State University/HateSpeech Datasets/Dataset-01_24/HateEval/lgbt/lgbt-test_HB.csv']
#'C:/Users/psheth5/OneDrive - Arizona State University/HateSpeech Datasets/LargeDatasets/Synthetic_train.csv',
 #'C:/Users/psheth5/OneDrive - Arizona State University/HateSpeech Datasets/LargeDatasets/Twi-Red-You_train.csv','C:/Users/psheth5/OneDrive - Arizona State University/HateSpeech Datasets/LargeDatasets/WikiFull_train.csv','C:/Users/psheth5/OneDrive - Arizona State University/HateSpeech Datasets/LargeDatasets/WikiFull_test.csv','C:/Users/psheth5/OneDrive - Arizona State University/HateSpeech Datasets/DatasetsFinalSplits/WikiDetox_Train.csv']

for f in files:
    ds = f.split(".")[0].split("/")[-1]
    aug_creator(f,ds)
    preprocessor(ds,"bert-base-uncased",False,"syn")



