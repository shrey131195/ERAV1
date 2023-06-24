## Session_8.ipynb

The task was to create a train a CNN network that can classify MNIST data with the following constraints:
* 70% validation accuracy
* Less than 50k Parameters
* You can use anything from above you want. 
* Less than 20 Epochs

Parameters used : **39,552**

### Summary for model
![image](https://github.com/shrey131195/ERAV1/assets/26046930/a9c68c7c-5035-4d44-b0bd-6e38709c5402)

### Batch Normalisation
Best Accuracy : **76.04%**

#### Test Logs
![image](https://github.com/shrey131195/ERAV1/assets/26046930/65db84b9-a208-405c-a9f2-8337abb70c2b)

#### Plot
![image](https://github.com/shrey131195/ERAV1/assets/26046930/aa62146a-ec72-4890-8868-14630ad55f34)

#### Incorrect Pictures
![image](https://github.com/shrey131195/ERAV1/assets/26046930/633d09c7-2599-4214-bfce-07465d0837dc)

### Layer Normalisation
Best Accuracy : **74.16%**

#### Test Logs
![image](https://github.com/shrey131195/ERAV1/assets/26046930/27180bd3-5fa2-427c-aedd-3fe6a65b261d)

#### Plot
![image](https://github.com/shrey131195/ERAV1/assets/26046930/c3b9a267-1898-4428-97a2-ed19be72a1e5)

#### Incorrect Pictures
![image](https://github.com/shrey131195/ERAV1/assets/26046930/6db6e303-4f6f-4c26-b13f-b18c30f5437b)

### Group Normalisation
Best Accuracy : **72.4%**

#### Test Logs
![image](https://github.com/shrey131195/ERAV1/assets/26046930/ee90b073-8ee6-4ea9-a156-b70c1ca221f3)

#### Plot
![image](https://github.com/shrey131195/ERAV1/assets/26046930/e14aa70d-4153-45db-80b7-a21ef32542d9)

#### Incorrect Pictures
![image](https://github.com/shrey131195/ERAV1/assets/26046930/56e9a43c-900b-473e-8f53-2514d3ac0934)
