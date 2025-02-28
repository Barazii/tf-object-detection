Bad approach to deploy pretrained models from a pipeline-like standalone script, notebooks better. For finetuning, pipelining good. \
Increasing classes number in training/finetuning only consumes resources, doesn't improve model accuracy. But increasing samples number per class does improve. Economical method: use few classes for prototyping/development, then maybe include all classes.\
