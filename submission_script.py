import utils.feature_extractors as utils
from utils.evaluation import action_evaluator
import numpy as np
import sys
import os
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import joblib

config = {
    "train_pos_loc":sys.argv[1],
    "train_neg_loc":sys.argv[2],
    "test_pos_loc":sys.argv[3],
    "test_neg_loc":sys.argv[4],
    "model_save_loc":"optimized",
    "ensemble_loc":"ensemble",
    "random_seed":9
}

class SpiderDataGenerator(object):
    ALL_FEAT = ["AAC","DPC","CTD",
                "PAAC","APAAC","RSacid",
                "RSpolar","RSsecond","RScharge",
                "RSDHP"]
    def __init__(self, pos_data_file, neg_data_file,feat_type=None) -> None:
        super(SpiderDataGenerator).__init__()
        self.pos_data_file = pos_data_file
        self.neg_data_file = neg_data_file
        
        assert feat_type in SpiderDataGenerator.ALL_FEAT or feat_type == None
        
        self.feat_type = feat_type
        
        self.pos_data = utils.read_fasta(self.pos_data_file)
        self.neg_data = utils.read_fasta(self.neg_data_file)
        
        self.data = self.pos_data+self.neg_data
        self.targets = np.array([True]*len(self.pos_data)+[False]*len(self.neg_data))
        
        self.raw = [x[1] for x in self.data]
        
        self.feat_AAC = utils.AAC(self.data)[0]
        print("Generating AAC Feature .....")
        self.feat_DPC = utils.DPC(self.data,0)[0]
        print("Generating DPC Feature .....")
        self.feat_CTD = np.hstack((utils.CTDC(self.data)[0], 
                              utils.CTDD(self.data)[0], 
                              utils.CTDT(self.data)[0]))
        print("Generating CTD Feature .....")
        self.feat_PAAC = utils.PAAC(self.data,1)[0]
        print("Generating PAAC Feature .....")
        self.feat_APAAC = utils.APAAC(self.data,1)[0]
        print("Generating APAAC Feature .....")
        self.feat_RSacid = utils.reducedACID(self.data) 
        print("Generating reducedACID Feature .....")
        self.feat_RSpolar = utils.reducedPOLAR(self.data)
        print("Generating reducedPOLAR Feature .....")
        self.feat_RSsecond = utils.reducedSECOND(self.data)
        print("Generating reducedSECOND Feature .....")
        self.feat_RScharge = utils.reducedCHARGE(self.data)
        print("Generating reducedCHARGE Feature .....")
        self.feat_RSDHP = utils.reducedDHP(self.data)
        print("Generating reducedDHP Feature .....")
        
        
        
    
    def get_combination_feature(self,selected:list = None):
        
        all_feat =[self.feat_AAC,self.feat_DPC,self.feat_CTD,
                   self.feat_PAAC,self.feat_APAAC,self.feat_RSacid,
                   self.feat_RSpolar,self.feat_RSsecond,self.feat_RScharge,
                   self.feat_RSDHP]
        
        if selected:
            select_index = sorted([SpiderDataGenerator.ALL_FEAT.index(x) for x in selected])
            all_feat = [all_feat[x] for x in select_index]
            
        return np.concatenate(all_feat,axis=-1)
        
        
        
            
    def __len__(self) -> int:
        return len(self.data)

print("\n\nTraining Data ...")
train_data = SpiderDataGenerator(pos_data_file=config["train_pos_loc"],neg_data_file=config["train_neg_loc"])
print("\n\nTesting Data ...")
test_data = SpiderDataGenerator(pos_data_file=config["test_pos_loc"],neg_data_file=config["test_neg_loc"])


X_train,y_train = train_data.get_combination_feature(["DPC","RSDHP","RSacid","RSpolar","RSsecond","RScharge"]),train_data.targets
X_test,y_test = test_data.get_combination_feature(["DPC","RSDHP","RSacid","RSpolar","RSsecond","RScharge"]),test_data.targets

print(f'Model :- SVC, Features Used :- "DPC","RSDHP","RSacid","RSpolar","RSsecond","RScharge"')
print("Training Stage with 5 fold cross validation")

#train
X,y = shuffle(X_train,y_train,random_state=config["random_seed"])

scaler = StandardScaler()
scaler.fit(X,y)
X = scaler.transform(X)

clf = SVC()
y_pred = cross_val_predict(clf, X, y, cv=5)

result_values = action_evaluator(y_pred,y,class_names=["Not Druggable","Druggable"],show_plot=False,save_outputs=None)

clf.fit(X,y)

print("Validation results: - ",result_values)
print("\n\nTest Stage")
#test
X,y = shuffle(X_test,y_test,random_state=config["random_seed"])
X = scaler.transform(X)


y_pred = clf.predict(X)

result_values = action_evaluator(y_pred,y,class_names=["Not Druggable","Druggable"],show_plot=False,save_outputs=None)
print("Test results: - ",result_values)

#output test results
pos_outputs = [f">{data_point[0]}\n{int(prediction)}" for data_point,prediction in zip(test_data.data,y_pred) if data_point[0].split("_")[0] == "Positive"]
neg_outputs = [f">{data_point[0]}\n{int(prediction)}" for data_point,prediction in zip(test_data.data,y_pred) if data_point[0].split("_")[0] == "Negative"]

with open("predictions_pos.txt","w") as f0:
    f0.write("\n".join(pos_outputs))
    
with open("predictions_neg.txt","w") as f0:
    f0.write("\n".join(neg_outputs))

print(f"Generated Outputs at predictions_pos.txt and predictions_neg")
print("\n\n")
