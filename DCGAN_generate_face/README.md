# Deep Convolution Generative Adversarial Network generate face

+ Implemented a `Deep Convolution GAN` model to generate face picture based on ` CelebA` dataset
+ In `generator` network, used `transposed convolution` to `upsample` from the input layers; applied `leaky ReLu` as activation function; applied `batch normalization` to help train and avoid poor initalization
+ In `discriminator` network, built `convolutional` classifer, applied `leaky ReLu` and `batch normalization`; used one `fully connected` layer and got `sigmoid` output
+ Calculated losses for discriminator and generator by using `sigmoid cross entropys`; applied label `smoothing` to help discriminator generalize; created separately `AdamOptimizer` for discriminator and generator to update network variables 

![DCGAN](./demo.png)