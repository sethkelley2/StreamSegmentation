import albumentations as A

from rastervision.pytorch_learner import (
    SemanticSegmentationRandomWindowGeoDataset,
    SemanticSegmentationSlidingWindowGeoDataset,
    SemanticSegmentationVisualizer)
from rastervision.core.data import ClassConfig
from tqdm.autonotebook import tqdm
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision.models.segmentation import deeplabv3_resnet50
from rastervision.pipeline.file_system import make_dir
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from os.path import join
from utils import SemanticSegmentation
import pytorch_lightning as pl
from torch.nn import functional as F
import json
from pytorch_lightning.callbacks.early_stopping import EarlyStopping



#REPLACE THIS WITH OUR DATA... Can use a list of uris
scene_id = 5631
train_image_uri = f's3://spacenet-dataset/spacenet/SN2_buildings/train/AOI_2_Vegas/PS-RGB/SN2_buildings_train_AOI_2_Vegas_PS-RGB_img{scene_id}.tif'
train_label_uri = f's3://spacenet-dataset/spacenet/SN2_buildings/train/AOI_2_Vegas/geojson_buildings/SN2_buildings_train_AOI_2_Vegas_geojson_buildings_img{scene_id}.geojson'


#configuration
class_config = ClassConfig(
    names=['stream', 'background'],
    colors=['blue', 'green'],
    null_class='background')

data_augmentation_transform = A.Compose([
    A.Flip(),
    A.ShiftScaleRotate(),
    A.RGBShift()
])

#training dataset
train_ds = SemanticSegmentationRandomWindowGeoDataset.from_uris(
    class_config=class_config,
    image_uri=train_image_uri,
    label_vector_uri=train_label_uri,
    label_vector_default_class_id=class_config.get_class_id('stream'),
    size_lims=(300, 350),
    out_size=325,
    max_windows=10,
    transform=data_augmentation_transform)

viz = SemanticSegmentationVisualizer(
    class_names=class_config.names, class_colors=class_config.colors)
x, y = viz.get_batch(train_ds, 4)
viz.plot_batch(x, y, show=True)



#REPLACE THIS WITH OUR DATA... Can use a list of uris
scene_id = 5632
val_image_uri = f's3://spacenet-dataset/spacenet/SN2_buildings/train/AOI_2_Vegas/PS-RGB/SN2_buildings_train_AOI_2_Vegas_PS-RGB_img{scene_id}.tif'
val_label_uri = f's3://spacenet-dataset/spacenet/SN2_buildings/train/AOI_2_Vegas/geojson_buildings/SN2_buildings_train_AOI_2_Vegas_geojson_buildings_img{scene_id}.geojson'


#validation dataset
val_ds = SemanticSegmentationSlidingWindowGeoDataset.from_uris(
    class_config=class_config,
    image_uri=val_image_uri,
    label_vector_uri=val_label_uri,
    label_vector_default_class_id=class_config.get_class_id('stream'),
    size=325,
    stride=325)

#trainer config
batch_size = 8
lr = 1e-4
epochs = 10
output_dir = './lightning-trainer/'
make_dir(output_dir)
fast_dev_run = False

#create dataloaders
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=4)
    

#create model
deeplab = deeplabv3_resnet50(num_classes=len(class_config) + 1)
model = SemanticSegmentation(deeplab, lr=lr)
tb_logger = TensorBoardLogger(save_dir=output_dir, flush_secs=10)

early_stop_callback = EarlyStopping(
   monitor='validation_loss',
   min_delta=0.00,
   patience=3,
   verbose=False,
   mode='max'
)

trainer = pl.Trainer(
    accelerator='auto',
    min_epochs=1,
    max_epochs=epochs+1,
    default_root_dir=output_dir,
    logger=[tb_logger],
    fast_dev_run=fast_dev_run,
    log_every_n_steps=1,
    callbacks=[early_stop_callback]
)

#fit model
trainer.fit(model, train_dl, val_dl)