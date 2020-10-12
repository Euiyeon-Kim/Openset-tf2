from enum import Enum


class ModelStructure(Enum):
    def __str__(self):
        return '%s' % self.value

    VANILLA = 'vanilla'
    SENET = 'senet'
    SPNORM = 'spectral_norm'
    DEFORM = 'deformable'
    CUSTOM = 'custom'


class Config:
    # Model
    structure = ModelStructure.CUSTOM
    kernel_size = 3
    strides = 2
    activation = 'lrelu'
    # Classifier
    shared_conv_channels = [32, 64, 128, 256, 512]
    dense_branch_units = [512, 256]
    # Deform
    deform_conv_channels = [32, 64, 128, 256, 512]
    deform_conv_offset_channels = [128, 256]

    # GradCAM
    cam_layer = 'sn_9'

    # Basics
    n_gpus = 1
    epochs_to_validate = 1
    epochs_to_save_gradCAM = 1
    epochs_to_save_weights = 10

    # Openset
    test_with_openset = True
    openset_rate = 0.4
    threshold = 0.5

    # Data
    n_workers = 8
    batch_size = 128
    split_weight = (9, 1)
    input_shape = (224, 224, 3)               # Resize
    # TFDS
    use_tfds = False
    tfds_dir = 'data'
    dataset_name = 'cifar100'                  # TFDS dataset name
    # ImageNet
    root_dir = 'data/imagenet'
    openset_dir = 'data/imagenet_LT/ImageNet_LT_open'
    train_txt_path = 'data/imagenet_LT/imagenet_oltr_train.txt'
    val_txt_path = 'data/imagenet_LT/imagenet_oltr_val.txt'
    test_txt_path = 'data/imagenet_LT/imagenet_oltr_test.txt'
    openset_txt_path = 'data/imagenet_LT/imagenet_oltr_open.txt'
    num_classes = 1000
    imagenet_resize = [256, 256]
    imagenet_crop = [224, 224, 3]

    # Directories
    results_dir = 'exps/spnorm_imagenet'
    classifier_weight_path = None

    # Trainer
    num_epochs = 1000
    lr = 1e-4
    beta1 = 0.5
    beta2 = 0.999

