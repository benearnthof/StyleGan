# StyleGan
A pytorch implementation of Nvidias StyleGan for the class of Deep Unsupervised Learning at LMU Munich 2021

To run training execute the file dataset.py first. For example: 
!python dataset.py --dataset_path --out_path --max_size --min_size --magic_number
dataset_path corresponds to the file path where the dataset is located. 
Has to be of the structure of a torchvision imagefolder: 
https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.ImageFolder
out_path is the file path of the lmdb database that will be generatedd
max_size is the maximum image size (default 128x128) 
min_size is the minimum image size (default 8x8)
magic_number is a multiplying constant to initialize the data base in the correct size on windows machines. 

After preprocessing the images StyleGAN can be trained with: 
!python train.py path --init_size --max_size --mixing --loss 
Where path is the path to the lmdb database created above
init_size is the size at which to start progressively growing (default 8x8) 
max_size is the maximum size of the training images (default 128x128) 
mixing specifies that mixing regularization will be used 
loss specifies the gan loss used (r1 or wgan-gp) 

to train stylegan2-ada the official pytorch implementation was used, 
to evaluate the checkpoints for interpolation only the model2 files are necessary to be loaded like they are
in the respective notebooks. 
