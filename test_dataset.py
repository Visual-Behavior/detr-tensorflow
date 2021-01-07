from detr_tf.data import load_voc_dataset
from detr_tf.training_config import TrainingConfig


class MyConfig(TrainingConfig):
    def __init__(self):
        super().__init__()
        self.datadir = "/home/thibault/data/VOCdevkit/VOC2012/"

config = MyConfig()

# Load and exlude the person class
iterator, class_names = load_voc_dataset("train", VOC_CLASS_NAME, config.batch_size, config, augmentation=True)
print('class_names', class_names)