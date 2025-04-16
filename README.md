# Deep Learning Scientist Challenge

This page challenges many issues and pitfalls one can face when training/predicting vehicles and aircrafts
in satellite aerial images.

In a first part, we propose tricks and improvments for an aircrafts detection model on high resolution image satellite, according to some last training feedbacks and inference latency time constraint.

in a second part, we have to analyse a typical overfitting case for a Unet segmentation model, and propose solutions to make it converge.

In a last part, we are called to propose a python code to post-process a segmentation mask failing at separating some vehicles and fully locate some single vehicles. The objective behind is to have the right count of vehicles present into the image.

Those three problems are assessed in the following sections

## I/ Improve the model, please, it’s not good enough

The product manager asks you to improve an aircraft detector in order to reach very good
performances for the product.
Here are some feedbacks regarding the last successful training.


| observation | value |
---------------|-------
| architecture | RetinaNet |
| backbone | Resnet50 |
| test precision    | 80%  |  
| test recall       | 80%  |
| batch size   | 8    |
| rotate augmentation | -90, 90, 180|
| flipping augmentation | H, V |
| epochs | 100 |
| training size | 50000 |
| image size | 512*512 |
| max inference time | 1min |

We do not tolerate inference time above **5 mins**, and we would like to reach following test metrics

| expectation | value |
---------------|-------
| test precision    | 85%  |  
| test recall       | 90%  |



In the object detection field, aerial Images are hard use cases where objects of interest are very small and subject to clutter obstructions (clouds, shadows), if you consider that objects can be very rare, you also end up with an extremely imbalanced problem.
In that context, [RetinaNet (FAIR - 2017)](https://arxiv.org/pdf/1708.02002) is a good solution to tackle thoses challenges:

- *small objects detection*

  Instead of relying only on backbone last convolution map (where small objects are likely to disappear due to successive max pooling), RetinaNet also takes into account intermediate low level feature maps (FPN) to get chance of classifying and localing small objects

- *class imbalancing*

  RetinaNet extends classical cross entropy loss by multiplying the class probability log with a specific term called "focal" term. This term is supposed to lower the loss contribution when probability is close to 1 (which likely correspond to "easy" background majority class examples). This factor comes with a positive integer exponent to control how much you need to counter balance background "easy" examples. The following figure illustrates well factor impact:

  ![img](.thumbnails/focal_loss.png "How focal loss penalize easy examples")

In the next subsections, we will depict how we can enhance a RetinaNet in the context of aircrafts detection on aerial images, taking into account all quantitatives assumptions we have, particularly the time budget of 5 mins. There are 5 subsections described by decreasing priority order (get simple things done first)

- *Inference-only tricks*

  There are quick-wins because are fast to experiment so you can iterate a lot.

- *Training hyperparameters tuning*

  There are straightforward and automatic to experiment/select thanks to cross-validation

- *Object-detection hyperparameters tuning*

  Many framework allows to change those high level parameters very simply using config file (like [mmdetection - MMLab](https://github.com/open-mmlab/mmdetection) ) but input statistics and results feedback are needed to find the optimal values

- *Input data modifications*

  it needs deeper data analysis and sometimes field knowledge to apply data strategies that matter

- *Architecture/weight modifications*

  There are very challenging because they can change metrics a lot (overfitting for example) and one could need many iterations to figure out a decent set of hyperparameters


### I.1/ Inference-only tricks

Here is a list of tricks that do not need any re-training:

* One can avoid false detections and increase precision by simply increasing threshold on class probability. Note that a different threshold can be applied to each class giving more degrees of freedom.

* All object detections post process detections by removing those that overlap (above a certain IOU threshold) with a more confident detection. By lowering this threshold, you can remove more false alarms and increase precision

* RetinaNet (like most object detections) rely on anchors on original image around which we generate size-based and ratio-based rectangles (proposals). The model is then supposed to correctly label and refine them. There is a room for improvment if you can adapt the sizes/ratios to better match the expected target geometrics.

* When one target occluded another, NMS can remove a good detection associated to the occluded target because of overlapping. So instead of removing it, [soft-NMS - 2017](https://arxiv.org/abs/1704.04503) propose to *decays the detection score as a continuous function of the overlap*. As a result, recall can be improved as good occluded detections are now kept and get chances to pass threshold. One should first carefully assess results to be sure that poor recall are due to missing occluded targets, otherwise precision is likely to decrease due to more false alarm.

* Data augmentation has become mandatory in object detection. It allows to augment data in the context of low labelled data regime but it also benefits to inference as stated by recent paper [TTA - 2023](https://arxiv.org/pdf/2401.01018) that 
promote Test-Time Augmentation. The idea is to transform image multiple times, performs detections process in each and apply aggregations strategies to ensemble detections. The number of transformations should be limited to not
reach the 5 minutes time budget.


### I.2/ Training hyperparameters tuning


* Different learning rate strategy can be experimented like SGD used in original RetinaPaper

* Batch size can be increased for more stable convergence, but always constrained by GPU memory capacity limit

* the focal loss gamma parameter can be increased to more focus on foreground loss, then improving both precision and recall

* after a per-class analysis results, one can adapt per class weighting loss to better handle per class poor results.

* There is no information about training/val strategy, so we could go with a common pattern that randomly divide training into 90% train and 10% validation

* There is no information about metrics, but we can use Average Precision as validation metric to early stop training and avoid overfitting. 
Additionally, we can monitor APs, APm and APl metrics to assess how model performs on small, medium and large targets respectively, provided that it makes sense regarding our aerial images.


### I.3/ Object-detection hyperparameters tuning

- as stated in the subsection dedicated to inference-only tricks, we can also adapt sizes/ratios proposals early at training stage to fit with ground truth boxes distribution and make training more efficient

### I.4/ Input data modifications

Data augmentation allows model to see same given input in many configurations and provide better generalisation capacity. While rotations and flipping are very usual some other are worth to experiment, python packages like [albumentations](https://www.albumentations.ai/docs/3-basic-usage/bounding-boxes-augmentations/) and [imgaug](https://imgaug.readthedocs.io/en/latest/) are off-the shelf toolkits that carry out more advanced image/mask transformations:

* *random cropping* : interesting to let model learns foreground-only features

* *random cut*: interesting to mimic occluded/cluttered objects

* *decreasing contrast*: can mimic shadowed planes in aerial images

* *warping*: can mimic different aerial views

* *downscaling/upscaling*: mimic different altitudes for aerial images

* *fog/haze/blurring*: help to simulate weather bad conditions
	

### I.5/ Architecture/weight modifications

Following changes are more groundbreaking and need much experimentation time to overcome possible unexpected behaviors like overfitting

* Training can be performed from a RetineNet checkpoint trained on larger dataset like COCO

* Backbone Resnet50 can be progressively replaced by Resnet100 and then resnet 152 while carrefully assessed that you are not overfitting (hopefully, our data size is in the order of magnitude of CIFAR used by Resnet authors). TFLOPS operations increasing is linear so we should still be under the 5 mins budget

![img](.thumbnails/resnet_tflops.png "Resnet architectures and TFLOPS")

* If RetinaNet FPN is not using all layer maps, try adding more (specifically add the first conv maps if small objects are missing)



