
from sklearn.metrics import confusion_matrix
import random
from tqdm import tqdm, tqdm_notebook
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from keras.callbacks import Callback


class EarlyStoppingByMetric(Callback):
                              
    def __init__(self, x_val, y_val, patience, batch_size, threshold_searching = 50, metric = "precision", min_samples = 50):

        self.x_val = x_val
        self.y_val = y_val
       
        self.patience = patience
        self.waiting = waiting
        
        self.bs = batch_size
        
        self.thr_search = threshold_searching
        self.ms = min_samples
        self.metric = metric
        
    def on_train_begin(self, logs={}):
        
        self.history={self.metric : 0}
        
        
    def select_metric(self, tn, fp, fn, tp):
        
        if self.metric == "precision":
            score = tp / (tp + fp)
            
        if self.metric == "accuracy":
            score = (tp+tn) / (tp + fp + tn + fn)
            
        return score



    def on_epoch_end(self, epoch, logs={}):
        
        ## Using predict_pn_batch to avoid memory leakage
        
        preds = self.model.predict_on_batch(self.x_val[:self.bs,:,:]).numpy().reshape(-1)
        for i in range(self.bs,self.x_val.shape[0],self.bs):
            predsnp.concatenate([preds,self.model.predict_on_batch(self.x_val[i:i+self.bs,:,:]).numpy().reshape(-1)])

        best = []
        
        for thr in tqdm([random.uniform(np.min(preds),np.max(preds)) for i in range(self.thr_search)]):
            
            tn, fp, fn, tp = confusion_matrix(self.y_val,preds > thr).ravel()
            
            pr = self.select_metric(tn, fp, fn, tp)
            
            if tp > self.ms:
                
                best.append(pr)
            
        best = np.array(best)
        
        best = best[best.argsort()[-5:]]
        mean = np.mean(best)
        
        if mean > self.history[self.metric]:
            self.history[self.metric] = mean
            print(f"New best model:\nVal {self.metric} mean of {mean:.3f} and Maximum of {best[-1]:.3f}")
            self.model.save(f'modelBest{best[-1]:.3f}', overwrite=True)
            self.waiting = 0 
            
        else:
            self.waiting += 1
            if self.waiting == self.patience:
                print("Training was stopped due to early stopping")
                self.model.stop_training = True

        ## unnecassary memory
        del(preds)

        
