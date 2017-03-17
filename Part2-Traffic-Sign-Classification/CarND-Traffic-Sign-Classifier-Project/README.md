## Building a Traffic Sign Recognition Program

## NOTE: This is an implementation for the Self-Driving Car Udacity Nanodegree, here is a condensed version which 
#### TODO:
- Add another [spatial transformer](https://arxiv.org/pdf/1506.02025.pdf) layer in the middle of the conv layers, instead of just at the beginning
- Use [center loss](http://ydwen.github.io/papers/WenECCV16.pdf) as the loss instead of just the cross-entropy loss
- Have a look at the model architecture for a single column in [MCDNN](http://people.idsia.ch/~juergen/nn2012traffic.pdf) for Traffic Sign Classification

### Overview
In this project, I will use deep neural networks and convolutional neural networks to classify traffic signs, exploring many different architectures and various loss functions. I will train a model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). 

This is an ongoing project that I have, the purpose of which is to play around and  implement various network architectures from scratch and using various loss functions so that I can have a better understanding of Tensorflow. So far I have implemented:
- LeNet, and LeNet with Spatial Transformers
- Inception (although only one layer deep due to computing limitations)


### Dependencies

This project requires **Python 3.5** and the following Python libraries installed:

- [Jupyter](http://jupyter.org/)
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [scikit-learn](http://scikit-learn.org/)
- [TensorFlow](http://tensorflow.org)
- [Matplotlib](http://matplotlib.org/)
- [Pandas](http://pandas.pydata.org/) (Optional)

Run this command at the terminal prompt to install [OpenCV](http://opencv.org/). Useful for image processing:

- `conda install -c https://conda.anaconda.org/menpo opencv3`

### Dataset

1. [Download the dataset](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip). This is a pickled dataset in which we've already resized the images to 32x32.


