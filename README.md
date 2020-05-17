## ESBM - EarlyStopping customized to your own metric (including threshold optimizing)

![](https://github.com/itamargol/Focal_Loss/blob/master/imbalance.jpg)

## Background

**This package will be helping easily you with several critical issues:**

1. You will define your required metric.
2. ESBM will evaulate your model performance on the validation set.
3. It will select the best classsification threshold and inform you about the results.
4. It will save the best evaualted model for future use.
5. It will initate early stopping after a defined period with no metric improvment.


![](https://github.com/itamargol/Focal_Loss/blob/master/focal_loss.png)

## How to use?

**First you got to initialize your earlystopping object.**

                              
    def __init__(self, x_val, y_val, patience, batch_size, threshold_searching = 50, metric = "precision", min_samples = 50):

``` python

from earlyStopping import EarlyStoppingByMetric

ESBM = EarlyStoppingByMetric(x_val, y_val, patience = 5, batch_size = 256)

```     

**Then you just use it in your .fit as a another callback**

``` python

model.fit(Xtr, Ytr, validation_data = (Xv,Yv),epochs=50, batch_size=256, verbose=1,callbacks=[ESBM],shuffle=True)


```     

