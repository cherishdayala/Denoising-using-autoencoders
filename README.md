# Denoising Using Autoencoders


  An autoencoder is an unsupervised neural network that accepts an input data set (i.e, input), internally compresses the input data into a latent spatial representation, and attempts to reconstruct the input data from that latent representation (i.e, output data).
  
  Image denoising is the process of removing noise from a noisy image in order to restore the true image. During the denoising process, it is hard to differentiate between high-frequency components like noise, texture and edge and there might be a loss of few details in denoised images.

As our inputs are images, convolutional neural networks (convnets) are used as encoders and decoders. The Convolutional Neural Network is the most well-known neural network for modeling image data (CNN, or ConvNet). It can better retain the connected information between the pixels of an image.  The layers in a CNN are designed in such a way that it is a better choice for processing image data. 
