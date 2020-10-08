from enum import Enum


class ModelStructure(Enum):
    def __str__(self):
        return '%s' % self.value

    VANILLA = 'vanilla'
    SENET = 'senet'
    SPNORM = 'spectral_norm'
    DEFORM = 'deformable'


class Config:
    # Basics
    n_gpus = 1
    class_wise_acc = True
    epochs_to_save_gradCAM = 1
    epochs_to_save_weights = 10

    # Openset
    threshold = 0.9

    # Data
    n_workers = 8
    input_shape = (224, 224, 3)               # Resize
    # TFDS
    use_tfds = False
    tfds_dir = 'data'
    dataset_name = 'cifar10'                  # TFDS dataset name
    # ImageNet
    root_dir = 'data/imagenet'
    train_txt_path = 'data/imagenet_LT/imagenet_oltr_train.txt'
    val_txt_path = 'data/imagenet_LT/imagenet_oltr_val.txt'
    test_txt_path = 'data/imagenet_LT/imagenet_oltr_test.txt'
    openset_txt_path = 'data/imagenet_LT/imagenet_oltr_open.txt'
    num_classes = 1000
    imagenet_resize = [256, 256]
    imagenet_crop = [224, 224, 3]

    # Model
    structure = ModelStructure.VANILLA
    kernel_size = 3
    strides = 2
    activation = 'lrelu'
    shared_conv_channels = [32, 64, 128]
    dense_branch_units = [256, 128]

    # GradCAM
    cam_layer = 'shared_conv_2'

    # Directories
    results_dir = 'exps/tmp_imgnet'
    classifier_weight_path = None #'exps/test/chkpt/vanilla-classifier-30.h5'

    # Trainer
    num_epochs = 30
    batch_size = 64
    lr = 1e-3
    beta1 = 0.5
    beta2 = 0.999

