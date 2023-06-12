# dataset = ["ihc_pure"]
# dataset = ["ConvAbuseEMNLPfull_Train_Modifiers"]
# dataset = ["GabHateCorpusannotations_Train_Modifiers"]
# dataset = ["ICWSM18SALMINEN_Train_Modifiers"] #Does not work
# dataset = ['implicithatev1stg1posts_Train_Modifiers']
# dataset = ['Reddit_Train_Modifiers']
# dataset = ["ihc_pure_imp"]
# dataset = ['Synthetic_train']
# dataset = ["Twi-Red-You_train"]
# dataset = ["WikiDetox_Train_Modifiers_HB"]
dataset = ["HateEval_train_HB","lgbt_train_HB","migrants_train_HB"]
# dataset = ["sbic"]
# dataset = ["sbic_imp"]
# dataset = ["dynahate"]

tuning_param  = ["lambda_loss", "main_learning_rate","train_batch_size","eval_batch_size","nepoch","temperature","SEED","dataset", "decay"] ## list of possible paramters to be tuned
lambda_loss = [0.25]
temperature = [0.3]
train_batch_size = [8]
eval_batch_size = [8]
decay = [0.0] # default value of AdamW
main_learning_rate = [2e-5]

hidden_size = 768
nepoch = [6]
run_name = "best"
loss_type = "impcon" # only for saving file name
# model_type = "bert-base-uncased"
model_type = "hatebert"

SEED = [0]
w_aug = False
w_double = False
w_separate = False
w_sup = False

save = True
param = {"temperature":temperature,"run_name":run_name,"dataset":dataset,"main_learning_rate":main_learning_rate,"train_batch_size":train_batch_size,"eval_batch_size":eval_batch_size,"hidden_size":hidden_size,"nepoch":nepoch,"dataset":dataset,"lambda_loss":lambda_loss,"loss_type":loss_type,"decay":decay,"SEED":SEED,"model_type":model_type,"w_aug":w_aug, "w_sup":w_sup, "save":save,"w_double":w_double, "w_separate":w_separate}