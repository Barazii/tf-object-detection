# Overview
Model tensorflow SSD Resnet50 is used to try to detect species of birds.\
The first part is deploy pretrained version of the model to test its initial capability of detecting any bird (no specific species).

![Alt Text](images/tf-od.drawio.png)

The second part is finetune the model on custom dataset of few bird classes/species to test feasibility of finetuning pretrained models to predict new unseen classes.