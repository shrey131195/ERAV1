# Session 6

## Part1: Neural Networks in Excel

### Example of a Neural Network

![image](https://github.com/shrey131195/ERAV1/assets/26046930/74caa60e-c3f9-48f5-ac35-65ec93b4c877)

### Basic Formula for back propagation
h1 = w1*i1 + w2*i2

h2 = w3*i1 + w4*i2

a_h1 = sigmoid(h1)		

a_h2 = sigmoid(h2)		

o1 = w5*a_h1 + w6*a_h2 		

o2 = w7*a_h1 + w8*a_h2		

a_o1 = sigmoid(o1)		

a_o2 = sigmoid(o2)		

E1 = 1/2*(t1-a_o1)^2		

E2 = 1/2*(t2-a_o2)^2		

E_total = E1 + E2		

sigmoid(x)  = 1/(1+exp(-x))

### Derive differentitaion for all weights

#### Differentiation for first layer weights back propagating

#### Pre requisite

ðE_t/ðw5 = ð(E1+E2)/ðw5 = ðE1/ðw5 = (ðE1/ða_o1)*(ða_o1/ðw5) = **(ðE1/ða_o1)** * **(ða_o1/ðo1)** * **(ðo1/ðw5)**										

**ðE1/ða_o1** = ð(1/2*(t1-a_o1)^2)/ða_o1  = -1*(t1-a_o1) = a_o1-t1										

**ða_o1/ðo1** = ð(sigmoid(o1))/ðo1 = sigmoid(o1)*(1-sigmoid(o1)) = a_o1 *(1-a_01)										

**ðo1/ðw5** = a_h1

#### Actual Differentiation
ðE_t/ðw5 = (a_o1-t1) * (a_o1 *(1-a_01)) * a_h1		

ðE_t/ðw6 = (a_o1-t1) * (a_o1 *(1-a_01)) * a_h2			

ðE_t/ðw7 = (a_o2-t2) * (a_o2 *(1-a_02)) * a_h1		

ðE_t/ðw7 = (a_o2-t2) * (a_o2 *(1-a_02)) * a_h2		

#### Differentiation for next layer weights back propagating

#### Pre requisite

ðE_t/ðw1 = (ðE_t/a_o1)*(ða_o1/ðo1)*(ðo1/ða_h1)*(ða_h1/ðh1)*(ðh1/ðw1) = (ðE_t/ða_h1)*(ða_h1/ðh1)*(ðh1/ðw1) = (ðE_t/ða_h1) * a_h1*(1-a_h1) * i1	

ðE_t/ðw2 = (ðE_t/ða_h1) * a_h1*(1-a_h1) * i2	

ðE_t/ðw3 = (ðE_t/ða_h2) * a_h2*(1-a_h2) * i1	

ðE_t/ðw4 = (ðE_t/ða_h2) * a_h2*(1-a_h2) * i2	

#### Substitute below in above accordingly

ðE_t/ða_h1 = ð(E1 + E2)/ða_h1		

ðE1/ða_h1 = (ðE1/ða_o1) * (ða_o1/ðo1)*ðo1/ða_h1 = (a_o1-t1)*(a_o1*(1-a_o1))*w5	

ðE2/ða_h1 =(ðE2/ða_o2) * (ða_o2/ðo2)*ðo2/ða_h1 = (a_o2-t2)*(a_o2*(1-a_o2))*w7		

ðE_t/ða_h1 = ((a_o1-t1)*(a_o1*(1-a_o1))*w5) + ((a_o2-t2)*(a_o2*(1-a_o2))*w7)		

#### Actual Differentiation
ðE_t/ðw1 = (((a_o1-t1)*(a_o1*(1-a_o1))*w5) + ((a_o2-t2)*(a_o2*(1-a_o2))*w7)) * a_h1*(1-a_h1) * i1			

ðE_t/ðw2 = (((a_o1-t1)*(a_o1*(1-a_o1))*w5) + ((a_o2-t2)*(a_o2*(1-a_o2))*w7)) * a_h1*(1-a_h1) * i2			

ðE_t/ðw3 = (((a_o1-t1)*(a_o1*(1-a_o1))*w6) + ((a_o2-t2)*(a_o2*(1-a_o2))*w8)) * a_h2*(1-a_h2) * i1			

ðE_t/ðw4 = (((a_o1-t1)*(a_o1*(1-a_o1))*w6) + ((a_o2-t2)*(a_o2*(1-a_o2))*w8)) * a_h2*(1-a_h2) * i2			

### Screenshots

#### Calculations
![image](https://github.com/shrey131195/ERAV1/assets/26046930/352e9370-385e-4cb1-9284-73d59fe9405f)

#### Gradients with different learning rates(0.1, 0.2, 0.5, 0.8, 1, 2)

![image](https://github.com/shrey131195/ERAV1/assets/26046930/b6cf0f15-0cd8-495b-908e-9fe541253e29)

## Part2

The task was to create a train a CNN network that can classify MNIST data with the following constraints:
* 99.4% validation accuracy
* Less than 20k Parameters
* You can use anything from above you want. 
* Less than 20 Epochs

Parameters used : **16,794**
Best Accuracy : **99.46%**

### Model
![image](https://github.com/shrey131195/ERAV1/assets/26046930/c35eab32-1d02-4a4d-8a3e-8515526baaae)
 
### Summary for model
![image](https://github.com/shrey131195/ERAV1/assets/26046930/09076f14-85df-467e-8fa9-5e873f04f742)

### Test Logs
![image](https://github.com/shrey131195/ERAV1/assets/26046930/1dbe2bc8-ec85-4211-8611-f2811f712ccc)
