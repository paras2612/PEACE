tuning_param  = ["dataset", "load_dir"]
dataset = ["GabHateCorpusannotations_Test_Modifiers","Reddit_Test_Modifiers","WikiDetox_Test_modifiers","Twi-Red-You_test","HateEval_test_HB"]#,"lgbt-test_HB","migrants-test_HB","Reddit_Test_Modifiers","Twi-Red-You_test","GabHateCorpusannotations_Test_Modifiers","WikiDetox_Test_modifiers"] #["ihc_pure", "sbic", "dynahate"] # dataset for evaluation
# dataset = ["GabHateCorpusannotations_Train_Modifiers"]
# dataset = ["ICWSM18SALMINEN_Train_Modifiers"] #Does not work
# dataset = ['implicithatev1stg1posts_Train_Modifiers']

# load_dir = ['C:/Users/psheth5/OneDrive - Arizona State University/HateSpeech Datasets/Baselines/ImpCon-main/save/0/GabHateCorpusannotations_Train_Modifiers/best/impcon/2023_01_24_15_53_24']
# load_dir = ['C:/Users/psheth5/OneDrive - Arizona State University/HateSpeech Datasets/Baselines/ImpCon-main/save/0/Reddit_Train_Modifiers/best/impcon/2023_01_29_08_33_26']
# load_dir = ['C:/Users/psheth5/OneDrive - Arizona State University/HateSpeech Datasets/Baselines/ImpCon-main/save/0/Twi-Red-You_train/best/impcon/2023_01_26_18_59_00']
# load_dir = ['C:/Users/psheth5/OneDrive - Arizona State University/HateSpeech Datasets/Baselines/ImpCon-main/save/0/WikiDetox_Train_Modifiers_HB/best/impcon/2023_01_30_17_32_46']

#load_dir = ['C:/Users/psheth5/Downloads/ImpCon-main/ImpCon-main/save/0/ConvAbuseEMNLPfull_Train_Modifiers/best/impcon/2023_01_24_15_19_01']
#load_dir = ["C:/Users/psheth5/Downloads/ImpCon-main/ImpCon-main/save/0/FoxNews_Train_Modifiers/best/impcon/2023_01_24_15_51_54"]
#load_dir = ["C:/Users/psheth5/Downloads/ImpCon-main/ImpCon-main/save/0/GabHateCorpusannotations_Train_Modifiers/best/impcon/2023_01_24_15_53_24"]
#load_dir = ['C:/Users/psheth5/Downloads/ImpCon-main/ImpCon-main/save/0/implicithatev1stg1posts_Train_Modifiers/best/impcon/2023_01_25_17_09_40']

#load_dir = ['C:/Users/psheth5/Downloads/ImpCon-main/ImpCon-main/save/0/Reddit_Train_Modifiers/best/impcon/2023_01_29_08_33_26']
#load_dir = ['C:/Users/psheth5/Downloads/ImpCon-main/ImpCon-main/save/0/Synthetic_train/best/impcon/2023_01_26_16_38_13']
load_dir = ['C:/Users/psheth5/OneDrive - Arizona State University/HateSpeech Datasets/Baselines/ImpCon-main/save/0/HateEval_train_HB/best/impcon/2023_02_14_07_45_43']
# load_dir = ['C:/Users/psheth5/OneDrive - Arizona State University/HateSpeech Datasets/Baselines/ImpCon-main/save/0/lgbt_train_HB/best/impcon/2023_02_14_07_53_03']
# load_dir = ['C:/Users/psheth5/OneDrive - Arizona State University/HateSpeech Datasets/Baselines/ImpCon-main/save/0/migrants_train_HB/best/impcon/2023_02_14_07_56_51']
# load_dir = ['C:/Users/psheth5/OneDrive - Arizona State University/HateSpeech Datasets/Baselines/ImpCon-main/save/0/lgbt_train_HB/best/impcon/2023_02_14_07_53_03']
# load_dir = ['C:/Users/psheth5/OneDrive - Arizona State University/HateSpeech Datasets/Baselines/ImpCon-main/save/0/migrants_train_HB/best/impcon/2023_02_14_07_56_51']

train_batch_size = 8
eval_batch_size = 8
hidden_size = 768
model_type = "bert-base-uncased"
SEED = 0

param = {"dataset":dataset,"train_batch_size":train_batch_size,"eval_batch_size":eval_batch_size,"hidden_size":hidden_size,"dataset":dataset,"SEED":SEED,"model_type":model_type, "load_dir":load_dir}