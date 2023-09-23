# ERA-SESSION18 UNET & VAE Implementation in PyTorch Lightning

### Tasks:
1. Train your own UNet from scratch, you can use the dataset and strategy provided in this linkLinks to an external site.. However, you need to train it 4 times:
  - MP+Tr+BCE
  - MP+Tr+Dice Loss
  - StrConv+Tr+BCE
  - StrConv+Ups+Dice Loss
2. Design a variation of a VAE that:
  - takes in two inputs: an MNIST image, and its label (one hot encoded vector sent through an embedding layer)
  - Training as you would train a VAE
  - Now randomly send an MNIST image, but with a wrong label. Do this 25 times, and share what the VAE makes (25 images stacked in 1 image)!
  - Now do this for CIFAR10 and share 25 images (1 stacked image)!
  - Questions asked in the assignment are:
    - Share the MNIST notebook link ON GITHUB [100]
    - Share the CIFAR notebook link ON GITHUB [200]
    - Upload the 25 MNIST outputs PROPERLY labeled [250]
    - Upload the 25 CIFAR outputs PROPERLY labeled. [450]
   
##### UNET - OXford Pet Dataset Samples
![image](https://github.com/shrey131195/ERAV1/assets/26046930/fa3c7dc9-5c67-420b-8050-8a8351c20449)

##### MaxPool + ConvTranspose + Dice Loss
**Training log**
EPOCH: 1 (LR: 0.001)
100%|██████████| 115/115 [00:39<00:00,  2.93it/s]Training Loss: -14988.183938145638

Test set: Average loss=-74.22976052761078

 EPOCH: 2 (LR: 0.001)
100%|██████████| 115/115 [00:39<00:00,  2.91it/s]Training Loss: -3585.0284881591797

Test set: Average loss=6329.501517295837

 EPOCH: 3 (LR: 0.001)
100%|██████████| 115/115 [00:39<00:00,  2.92it/s]Training Loss: -3984.460178375244

Test set: Average loss=-5440.595603942871

 EPOCH: 4 (LR: 0.001)
100%|██████████| 115/115 [00:39<00:00,  2.94it/s]Training Loss: -4247.924533843994

Test set: Average loss=-2363.508373260498

 EPOCH: 5 (LR: 0.001)
100%|██████████| 115/115 [00:39<00:00,  2.94it/s]Training Loss: -4439.253379821777

Test set: Average loss=210.07060050964355

 EPOCH: 6 (LR: 0.001)
100%|██████████| 115/115 [00:39<00:00,  2.94it/s]Training Loss: -4628.617164611816

Test set: Average loss=249.01873016357422

 EPOCH: 7 (LR: 0.001)
100%|██████████| 115/115 [00:39<00:00,  2.94it/s]Training Loss: -4791.261848449707

Test set: Average loss=-14728.533610343933

 EPOCH: 8 (LR: 0.001)
100%|██████████| 115/115 [00:39<00:00,  2.90it/s]Training Loss: -4972.690010070801

Test set: Average loss=-3038.2406272888184

 EPOCH: 9 (LR: 0.001)
100%|██████████| 115/115 [00:39<00:00,  2.93it/s]Training Loss: -5108.707195281982

Test set: Average loss=-4077.7550802230835

 EPOCH: 10 (LR: 0.001)
100%|██████████| 115/115 [00:39<00:00,  2.93it/s]Training Loss: -5274.879146575928

Test set: Average loss=-16411.531860351562

 EPOCH: 11 (LR: 0.001)
100%|██████████| 115/115 [00:39<00:00,  2.91it/s]Training Loss: -5436.027931213379

Test set: Average loss=-2194.11537361145

 EPOCH: 12 (LR: 0.001)
100%|██████████| 115/115 [00:39<00:00,  2.93it/s]Training Loss: -5574.092422485352

Test set: Average loss=-1576.4799137115479

 EPOCH: 13 (LR: 0.001)
100%|██████████| 115/115 [00:39<00:00,  2.94it/s]Training Loss: -5766.529968261719

Test set: Average loss=-6387.144726753235

 EPOCH: 14 (LR: 0.001)
100%|██████████| 115/115 [00:39<00:00,  2.94it/s]Training Loss: -5912.000782012939

Test set: Average loss=-1709.083327293396

 EPOCH: 15 (LR: 0.001)
100%|██████████| 115/115 [00:39<00:00,  2.94it/s]Training Loss: -6080.21448135376

Test set: Average loss=2172.098279953003
**RESULTS**
![image](https://github.com/shrey131195/ERAV1/assets/26046930/c608f5fc-10fb-499e-8850-54b56e23f7ed)
**LOSS**
![image](https://github.com/shrey131195/ERAV1/assets/26046930/6c803490-0dfb-491e-a95b-25f53aff61d4)

##### MaxPool + ConvTranspose + BCE Loss
**Training log**
 EPOCH: 1 (LR: 0.001)
100%|██████████| 115/115 [00:37<00:00,  3.08it/s]Training Loss: 57.898028165102005

Test set: Average loss=59.21360296010971

 EPOCH: 2 (LR: 0.001)
100%|██████████| 115/115 [00:38<00:00,  3.00it/s]Training Loss: 39.275745540857315

Test set: Average loss=42.830748468637466

 EPOCH: 3 (LR: 0.001)
100%|██████████| 115/115 [00:39<00:00,  2.91it/s]Training Loss: 31.837250724434853

Test set: Average loss=29.658504828810692

 EPOCH: 4 (LR: 0.001)
100%|██████████| 115/115 [00:39<00:00,  2.93it/s]Training Loss: 28.874064594507217

Test set: Average loss=28.06464682519436

 EPOCH: 5 (LR: 0.001)
100%|██████████| 115/115 [00:39<00:00,  2.92it/s]Training Loss: 26.90036779642105

Test set: Average loss=26.067045778036118

 EPOCH: 6 (LR: 0.001)
100%|██████████| 115/115 [00:39<00:00,  2.94it/s]Training Loss: 25.092479169368744

Test set: Average loss=24.79764223098755

 EPOCH: 7 (LR: 0.001)
100%|██████████| 115/115 [00:39<00:00,  2.93it/s]Training Loss: 23.735425636172295

Test set: Average loss=26.813547417521477

 EPOCH: 8 (LR: 0.001)
100%|██████████| 115/115 [00:39<00:00,  2.93it/s]Training Loss: 22.93260495364666

Test set: Average loss=23.53752888739109

 EPOCH: 9 (LR: 0.001)
100%|██████████| 115/115 [00:39<00:00,  2.93it/s]Training Loss: 22.16013652086258

Test set: Average loss=23.511479809880257

 EPOCH: 10 (LR: 0.001)
100%|██████████| 115/115 [00:39<00:00,  2.92it/s]Training Loss: 20.991869926452637

Test set: Average loss=22.17194725573063

 EPOCH: 11 (LR: 0.001)
100%|██████████| 115/115 [00:39<00:00,  2.92it/s]Training Loss: 20.313602030277252

Test set: Average loss=22.723210334777832

 EPOCH: 12 (LR: 0.001)
100%|██████████| 115/115 [00:39<00:00,  2.93it/s]Training Loss: 19.345448315143585

Test set: Average loss=22.06530152261257

 EPOCH: 13 (LR: 0.001)
100%|██████████| 115/115 [00:39<00:00,  2.93it/s]Training Loss: 18.62416408956051

Test set: Average loss=23.47374503314495

 EPOCH: 14 (LR: 0.001)
100%|██████████| 115/115 [00:39<00:00,  2.93it/s]Training Loss: 18.154343709349632

Test set: Average loss=22.447478398680687

 EPOCH: 15 (LR: 0.001)
100%|██████████| 115/115 [00:39<00:00,  2.93it/s]Training Loss: 17.038126692175865

Test set: Average loss=20.75004430860281
**RESULTS**
![image](https://github.com/shrey131195/ERAV1/assets/26046930/6589085b-9643-4956-886d-88118c7d4102)
**LOSS**
![image](https://github.com/shrey131195/ERAV1/assets/26046930/caad7e01-5388-4233-a2b6-5b2d161d4208)

##### StrConv + ConvTranspose + BCE Loss
**Training log**
EPOCH: 1 (LR: 0.001)
100%|██████████| 115/115 [00:40<00:00,  2.83it/s]Training Loss: 58.572375148534775

Test set: Average loss=94.17986130714417

 EPOCH: 2 (LR: 0.001)
100%|██████████| 115/115 [00:40<00:00,  2.83it/s]Training Loss: 42.78858470916748

Test set: Average loss=41.9924800992012

 EPOCH: 3 (LR: 0.001)
100%|██████████| 115/115 [00:40<00:00,  2.85it/s]Training Loss: 35.750930815935135

Test set: Average loss=37.645336642861366

 EPOCH: 4 (LR: 0.001)
100%|██████████| 115/115 [00:40<00:00,  2.84it/s]Training Loss: 32.191057816147804

Test set: Average loss=29.499892607331276

 EPOCH: 5 (LR: 0.001)
100%|██████████| 115/115 [00:40<00:00,  2.84it/s]Training Loss: 29.380158185958862

Test set: Average loss=35.46665419638157

 EPOCH: 6 (LR: 0.001)
100%|██████████| 115/115 [00:40<00:00,  2.84it/s]Training Loss: 27.588099986314774

Test set: Average loss=28.469015330076218

 EPOCH: 7 (LR: 0.001)
100%|██████████| 115/115 [00:40<00:00,  2.83it/s]Training Loss: 26.66010370850563

Test set: Average loss=27.63331399857998

 EPOCH: 8 (LR: 0.001)
100%|██████████| 115/115 [00:40<00:00,  2.84it/s]Training Loss: 25.8313190639019

Test set: Average loss=25.681565687060356

 EPOCH: 9 (LR: 0.001)
100%|██████████| 115/115 [00:40<00:00,  2.84it/s]Training Loss: 24.451588571071625

Test set: Average loss=25.681929126381874

 EPOCH: 10 (LR: 0.001)
100%|██████████| 115/115 [00:40<00:00,  2.83it/s]Training Loss: 23.784069806337357

Test set: Average loss=24.173929557204247

 EPOCH: 11 (LR: 0.001)
100%|██████████| 115/115 [00:40<00:00,  2.84it/s]Training Loss: 22.916942417621613

Test set: Average loss=24.039068043231964

 EPOCH: 12 (LR: 0.001)
100%|██████████| 115/115 [00:40<00:00,  2.84it/s]Training Loss: 22.199568018317223

Test set: Average loss=23.420975409448147

 EPOCH: 13 (LR: 0.001)
100%|██████████| 115/115 [00:40<00:00,  2.82it/s]Training Loss: 22.00563046336174

Test set: Average loss=24.956779181957245

 EPOCH: 14 (LR: 0.001)
100%|██████████| 115/115 [00:40<00:00,  2.83it/s]Training Loss: 21.534393817186356

Test set: Average loss=26.75588171184063

 EPOCH: 15 (LR: 0.001)
100%|██████████| 115/115 [00:40<00:00,  2.84it/s]Training Loss: 20.785470128059387

Test set: Average loss=22.680479526519775
**RESULTS**
![image](https://github.com/shrey131195/ERAV1/assets/26046930/f5a4dbe5-28ec-4547-a0c7-e6c3e101575f)
**LOSS**
![image](https://github.com/shrey131195/ERAV1/assets/26046930/735a5953-fd52-4ef2-befa-d0559f0088d6)

##### StrConv + Upsample + Dice Loss
**Training log**
 EPOCH: 1 (LR: 0.001)
100%|██████████| 115/115 [00:40<00:00,  2.84it/s]Training Loss: -2260.9181845784187

Test set: Average loss=10827.439171791077

 EPOCH: 2 (LR: 0.001)
100%|██████████| 115/115 [00:40<00:00,  2.85it/s]Training Loss: -3416.9540100097656

Test set: Average loss=2041.7259950637817

 EPOCH: 3 (LR: 0.001)
100%|██████████| 115/115 [00:40<00:00,  2.83it/s]Training Loss: -5187.545337677002

Test set: Average loss=29367.897102355957

 EPOCH: 4 (LR: 0.001)
100%|██████████| 115/115 [00:40<00:00,  2.84it/s]Training Loss: -4349.939970493317

Test set: Average loss=-1589.1722502708435

 EPOCH: 5 (LR: 0.001)
100%|██████████| 115/115 [00:40<00:00,  2.84it/s]Training Loss: -2058.6840314865112

Test set: Average loss=-82.11089992523193

 EPOCH: 6 (LR: 0.001)
100%|██████████| 115/115 [00:40<00:00,  2.84it/s]Training Loss: -2350.4616355895996

Test set: Average loss=3493.6149005889893

 EPOCH: 7 (LR: 0.001)
100%|██████████| 115/115 [00:40<00:00,  2.84it/s]Training Loss: -2578.0293922424316

Test set: Average loss=-15874.881034851074

 EPOCH: 8 (LR: 0.001)
100%|██████████| 115/115 [00:40<00:00,  2.82it/s]Training Loss: -2778.583818435669

Test set: Average loss=-2631.0340089797974

 EPOCH: 9 (LR: 0.001)
100%|██████████| 115/115 [00:40<00:00,  2.84it/s]Training Loss: -2992.2215309143066

Test set: Average loss=9281.859058380127

 EPOCH: 10 (LR: 0.001)
100%|██████████| 115/115 [00:40<00:00,  2.84it/s]Training Loss: -3252.032434463501

Test set: Average loss=39236.94672584534

 EPOCH: 11 (LR: 0.001)
100%|██████████| 115/115 [00:40<00:00,  2.82it/s]Training Loss: -3537.6160373687744

Test set: Average loss=-3782.1420402526855

 EPOCH: 12 (LR: 0.001)
100%|██████████| 115/115 [00:40<00:00,  2.84it/s]Training Loss: -3852.7009601593018

Test set: Average loss=-8070.650800704956

 EPOCH: 13 (LR: 0.001)
100%|██████████| 115/115 [00:40<00:00,  2.84it/s]Training Loss: -4294.752021789551

Test set: Average loss=-12224.882089614868

 EPOCH: 14 (LR: 0.001)
100%|██████████| 115/115 [00:40<00:00,  2.84it/s]Training Loss: -4895.907039642334

Test set: Average loss=-1885.42604637146

 EPOCH: 15 (LR: 0.001)
100%|██████████| 115/115 [00:40<00:00,  2.83it/s]Training Loss: -5748.875831604004

Test set: Average loss=-1145.6378135681152
**RESULTS**
![image](https://github.com/shrey131195/ERAV1/assets/26046930/10584288-fc6f-4fde-b61c-3752cce75b70)

**LOSS**
![image](https://github.com/shrey131195/ERAV1/assets/26046930/a8ad4b08-37d8-46a6-bcd7-683ce82e47ff)

#### VAE

##### VAE MNIST Data samples
![image](https://github.com/shrey131195/ERAV1/assets/26046930/40d861be-e007-4820-b38b-6f4369905d8a)

##### VAE MNIST Training log
Training: 0it [00:00, ?it/s]
Validation: 0it [00:00, ?it/s]
Epoch  0
Train Loss:  4338.2333984375
Val Loss:  3695.6982421875
Validation: 0it [00:00, ?it/s]
Epoch  1
Train Loss:  3584.0322265625
Val Loss:  3493.22314453125
Validation: 0it [00:00, ?it/s]
Epoch  2
Train Loss:  3420.442626953125
Val Loss:  3353.46728515625
Validation: 0it [00:00, ?it/s]
Epoch  3
Train Loss:  3311.287841796875
Val Loss:  3266.262451171875
Validation: 0it [00:00, ?it/s]
Epoch  4
Train Loss:  3238.250732421875
Val Loss:  3200.45263671875
Validation: 0it [00:00, ?it/s]
Epoch  5
Train Loss:  3184.343017578125
Val Loss:  3159.07080078125
Validation: 0it [00:00, ?it/s]
Epoch  6
Train Loss:  3144.289794921875
Val Loss:  3119.613525390625
Validation: 0it [00:00, ?it/s]
Epoch  7
Train Loss:  3112.061767578125
Val Loss:  3090.404541015625
Validation: 0it [00:00, ?it/s]
Epoch  8
Train Loss:  3085.99609375
Val Loss:  3068.654541015625
Validation: 0it [00:00, ?it/s]
Epoch  9
Train Loss:  3063.714111328125
Val Loss:  3050.071533203125
Validation: 0it [00:00, ?it/s]
Epoch  10
Train Loss:  3043.17578125
Val Loss:  3028.111572265625
Validation: 0it [00:00, ?it/s]
Epoch  11
Train Loss:  3026.00048828125
Val Loss:  3015.209716796875
Validation: 0it [00:00, ?it/s]
Epoch  12
Train Loss:  3009.426025390625
Val Loss:  3000.0087890625
Validation: 0it [00:00, ?it/s]
Epoch  13
Train Loss:  2994.812744140625
Val Loss:  2987.32177734375
Validation: 0it [00:00, ?it/s]
Epoch  14
Train Loss:  2980.221923828125
Val Loss:  2972.18310546875
Validation: 0it [00:00, ?it/s]
Epoch  15
Train Loss:  2966.56982421875
Val Loss:  2956.776123046875
Validation: 0it [00:00, ?it/s]
Epoch  16
Train Loss:  2953.303466796875
Val Loss:  2944.7890625
Validation: 0it [00:00, ?it/s]
Epoch  17
Train Loss:  2940.31689453125
Val Loss:  2930.719482421875
Validation: 0it [00:00, ?it/s]
Epoch  18
Train Loss:  2926.958251953125
Val Loss:  2917.6630859375
Validation: 0it [00:00, ?it/s]
Epoch  19
Train Loss:  2914.64697265625
Val Loss:  2907.48046875

##### VAE MNIST Results
![image](https://github.com/shrey131195/ERAV1/assets/26046930/04e9b5e9-dcb0-401e-b083-a5178ece154c)

##### VAE CIFAR Data samples
![image](https://github.com/shrey131195/ERAV1/assets/26046930/7834f5fc-c143-4676-a642-7e41aa40cf25)

##### VAE CIFAR Training log
Training: 0it [00:00, ?it/s]
Validation: 0it [00:00, ?it/s]
Epoch  0
Train Loss:  13823.705078125
Val Loss:  4443.755859375
Validation: 0it [00:00, ?it/s]
Epoch  1
Train Loss:  4340.798828125
Val Loss:  4286.13818359375
Validation: 0it [00:00, ?it/s]
Epoch  2
Train Loss:  4230.43115234375
Val Loss:  4146.11474609375
Validation: 0it [00:00, ?it/s]
Epoch  3
Train Loss:  4089.507568359375
Val Loss:  4033.19775390625
Validation: 0it [00:00, ?it/s]
Epoch  4
Train Loss:  4003.513427734375
Val Loss:  3974.619873046875
Validation: 0it [00:00, ?it/s]
Epoch  5
Train Loss:  3958.8974609375
Val Loss:  3932.842529296875
Validation: 0it [00:00, ?it/s]
Epoch  6
Train Loss:  3919.6953125
Val Loss:  3889.658935546875
Validation: 0it [00:00, ?it/s]
Epoch  7
Train Loss:  3860.661865234375
Val Loss:  3826.85400390625
Validation: 0it [00:00, ?it/s]
Epoch  8
Train Loss:  3809.402587890625
Val Loss:  3796.16259765625
Validation: 0it [00:00, ?it/s]
Epoch  9
Train Loss:  3774.78759765625
Val Loss:  3754.677001953125
Validation: 0it [00:00, ?it/s]
Epoch  10
Train Loss:  3740.806640625
Val Loss:  3716.1484375
Validation: 0it [00:00, ?it/s]
Epoch  11
Train Loss:  3702.548095703125
Val Loss:  3680.977783203125
Validation: 0it [00:00, ?it/s]
Epoch  12
Train Loss:  3669.667724609375
Val Loss:  3657.096923828125
Validation: 0it [00:00, ?it/s]
Epoch  13
Train Loss:  3644.147216796875
Val Loss:  3621.921142578125
Validation: 0it [00:00, ?it/s]
Epoch  14
Train Loss:  3621.0791015625
Val Loss:  3607.129638671875
Validation: 0it [00:00, ?it/s]
Epoch  15
Train Loss:  3602.8232421875
Val Loss:  3588.550537109375
Validation: 0it [00:00, ?it/s]
Epoch  16
Train Loss:  3588.69580078125
Val Loss:  3575.15625
Validation: 0it [00:00, ?it/s]
Epoch  17
Train Loss:  3575.5244140625
Val Loss:  3569.548095703125
Validation: 0it [00:00, ?it/s]
Epoch  18
Train Loss:  3562.7265625
Val Loss:  3550.062255859375
Validation: 0it [00:00, ?it/s]
Epoch  19
Train Loss:  3547.565673828125
Val Loss:  3535.135986328125

##### VAE CIFAR Results
![image](https://github.com/shrey131195/ERAV1/assets/26046930/efca6cd6-b680-482b-bffb-8646015e406a)

