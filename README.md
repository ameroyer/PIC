![license](https://img.shields.io/github/license/ameroyer/PIC.svg)
![GitHub repo size in bytes](https://img.shields.io/github/repo-size/ameroyer/PIC.svg)
![GitHub top language](https://img.shields.io/github/languages/top/ameroyer/PIC.svg)
![Maintenance](https://img.shields.io/maintenance/yes/2018.svg)
![GitHub last commit](https://img.shields.io/github/last-commit/ameroyer/PIC.svg)


# [PIC] Probabilistic Image Colorization
Tensorflow implementation for [Probabilistic Image Colorization](https://arxiv.org/abs/1705.04258) - generating diverse and vibrant colorization using auto-regressive generative networks - on the CIFAR and ImageNet datasets.


![model](examples/model.png)

We develop a probabilistic technique for colorizing grayscale natural images. In light of the intrinsic uncertainty of this task, the proposed probabilistic framework has numerous desirable properties. In particular, our model is able to produce multiple plausible and vivid colorizations for a given grayscale image and is one of the first colorization models to provide a proper stochastic sampling scheme. 

Moreover, our training procedure is supported by a rigorous theoretical framework that does not require any ad hoc heuristics and allows for efficient modeling and learning of the joint pixel color distribution. We demonstrate strong quantitative and qualitative experimental results on the CIFAR-10 dataset and the challenging ILSVRC 2012 dataset.


![sample1](examples/1.jpg)
![sample1](examples/11.jpg)
![sample1](examples/12.jpg)
![sample1](examples/3.jpg)
![sample1](examples/5.jpg)
**Figure:** Input grayscale (left), original colored image (rigt) and samples from our model (middle columns)

If you find this work useful, please cite
```
"Probabilistic Image Colorization"
 Amélie Royer, Alexander Kolesnikov, Christoph H. Lampert
 British Machine Vision Conference (BMVC), 2017
```

## Instructions

### Dependencies
  * Python 2.6+ or 3+
  * Tensorflow 1.0+
  * Numpy
  * h5py (for loading the imagenet dataset)
  * skimage
  
  
The main training and evaluation code is in `main.py`. Use `python main.py -- help` to display the various options available.
  
### Train the model

Train on **ImageNet** on 4 GPUS with color channels lying in the LAB colorspace and being generated at 1/4 the resolution of the original image.
```bash
python main.py --nr_gpus 4 --batch_size 16 --test_batch_size 25 --init_batch_size 100  \
                -lr 0.00016 -p 0.999 -ld 0.99999 -c 160 -l 4 --downsample 4            \
                --color lab --dataset imagenet --gen_epochs 1 --data_dir [data_dir]
```

Same training on **CIFAR**.
```bash
python main.py --nr_gpus 4 --batch_size 16 --test_batch_size 16 --init_batch_size 100  \
                -lr 0.001 -p 0.999 -ld 0.99995 -c 160 -l 4 --downsample 2              \
                --color lab --dataset cifar --gen_epochs 1 --data_dir [data_dir]
```


### Evaluation of a pre-trained model on the test set

Download the public pre-trained models (ImageNet and CIFAR).
```bash
wget http://pub.ist.ac.at/~aroyer/Models/PIC/cifar_model.tar.gz
tar -xzvf cifar_model.tar.gz
```


```bash
wget http://pub.ist.ac.at/~aroyer/Models/PIC/imagenet_model.tar.gz
tar -xzvf imagenet_model.tar.gz
```

Evaluate the model on the dataset validation split. For instance for ImageNet:
```bash
python main.py --nr_gpus 4 --batch_size 16 --test_batch_size 25 --init_batch_size 100  \
               -c 160 -l 4 --downsample 4 --color lab --dataset imagenet --mode "eval" \
               --data_dir [data_dir] --model [path_to_checkpoint .ckpt]
```


### Apply a pre-trained model on selected samples

Apply the model on given colored images to generate (i) reconstruction and (ii) random samples from grayscale version of the input images. The generated images are saved as `demo_reconstructions.jpg` and `demo_generations.jpg` respectively. 

on **CIFAR**
```bash
python main.py --nr_gpus 1 -c 160 -l 4 --downsample 2 --color lab --dataset cifar --test \
               --mode "demo" --model [path_to_checkpoint .ckpt] --input [path to image(s)]
```
               
on **ImageNet**
```bash
python main.py --nr_gpus 1 -c 160 -l 4 --downsample 4 --color lab --dataset imagenet \
               --mode "demo" --model [path_to_checkpoint .ckpt] --input [path to image(s)]
```


## Demo example

For instance, to generate reconstructions and samples on the images in `samples_val`, which are samples from the validation set of the ImageNet dataset, with the pre-trained ImageNet model (*Note:* set `--nr_gpus 0` to run in CPU mode):

```bash
wget http://pub.ist.ac.at/~aroyer/Models/PIC/imagenet_model.tar.gz
tar -xzvf imagenet_model.tar.gz

python main.py --nr_gpus 1 -c 160 -l 4 --downsample 4 --color lab --dataset imagenet \
               --test_batch_size 16 --mode "demo" --model imagenet/model.ckpt         \
               --input "val_samples/*.JPEG"
```