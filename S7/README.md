# ERAV1 SESSION 7 ASSIGNMENT

This folder contains 3 python notebook. Each notebook gets model from model.py.

*model.py* contains 3 models whose details are mentioned below:

## Model 1

**Target:** 
```
1. Achieve accuracy with 194k parameters 
2. Get the basic skeleton
```

**Results:**
```
- Parameters: 194,884
- Best Train Accuracy: 98.38
- Best Test Accuracy: 98.79 (14th Epoch)
```
**Analysis:**
```
1. Dropout overcomes overfitting
```


## Model 2

**Target:** 
```
1. Add Data Augmentation
2. Added Dropout and BN
3. Make model lighter
```

**Results:**
```
- Parameters: 13,808
- Best Train Accuracy: 99.22
- Best Test Accuracy: 99.50 (13th Epoch)
```
**Analysis:**
```
1. Decreasing the number of channels makes model lightweight
2. Adding Data Augmentation helps overcome over fitting.
```

## Model 3

**Target:**
```
1. Add  StepLR
2. Make model lighter
```

**Results:**
```
- Parameters: 6,768
- Best Train Accuracy: 98.79
- Best Test Accuracy: 99.45 (14th Epoch)
```
**Analysis:**
```
1. Consistent 99.4%+ accuracy
2. Decreasing the number of channels and increasing the layers seems to be good way to make model lighter
3. Having stepLR with step size 10 to adjust learning rate to 0.09 helps in learning.
```
