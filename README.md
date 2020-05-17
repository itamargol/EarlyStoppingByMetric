## ESBM - EarlyStopping customized to your own metric (including threshold optimizing and best model serailization)

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

                           
``` python

from earlyStopping import EarlyStoppingByMetric

ESBM = EarlyStoppingByMetric(x_val, y_val, patience = 5, batch_size = 256)

```     

**Then you just use it in your .fit as a another callback**

``` python

model.fit(Xtr, Ytr, validation_data = (Xv,Yv),epochs=50, batch_size=256, verbose=1,callbacks=[ESBM],shuffle=True)


```     

## Some nuances:

**There are a few more arguments you are able to pass into the ESBM object in order to recieve your ideal results.**

``` python

ESBM = EarlyStoppingByMetric(x_val, y_val, patience, batch_size, threshold_searching = 50, metric = "precision", min_samples = 50)

```    

**How long would you like to wait before earlystopping initiation?**

``` python

ESBM = EarlyStoppingByMetric(... , patience = 10 , ...)
```   

**How many iterations to perform while looking for your best classification threshold?**

``` python

ESBM = EarlyStoppingByMetric(... , threshold_searching = 50 , ...)
```    


**Which metric which you like to optimize?**

``` python

ESBM = EarlyStoppingByMetric(... , metric = "precision" , ...)

```  

**What is the minimum amount of samples would you like to take into account while optimizing metric on validation set?**

``` python

ESBM = EarlyStoppingByMetric(... , min_samples = 50 , ...)

```    




