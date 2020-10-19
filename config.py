from enum import Enum


class ModelStructure(Enum):
    def __str__(self):
        return '%s' % self.value

    RESNET50 = 'resnet50'
    SENET = 'senet'
    SPNORM = 'spectral_norm'
    DEFORM = 'deformable'
    CUSTOM = 'custom'


class Config:
    exp_name = 'imagenet_resnet50_in'

    # Model
    structure = ModelStructure.RESNET50
    kernel_size = 3
    strides = 2
    activation = 'lrelu'
    # Classifier
    shared_conv_channels = [128, 256, 256, 512, 512]
    dense_branch_units = [512, 256]
    # Deform
    deform_conv_channels = [32, 64, 128, 256, 512]
    deform_conv_offset_channels = [128, 256]

    # GradCAM
    cam_layer = 'conv2d_52'

    # Basics
    n_gpus = 1
    epochs_to_validate = 1
    epochs_to_save_gradCAM = 1
    epochs_to_save_weights = 2

    # Openset
    test_with_openset = True
    openset_rate = 0
    threshold = 0

    # Data
    n_workers = 8
    batch_size = 32
    split_weight = (9, 1)
    input_shape = (224, 224, 3)               # Resize
    # TFDS
    use_tfds = False
    tfds_dir = 'data'
    dataset_name = 'cifar100'                  # TFDS dataset name
    # ImageNet
    root_dir = 'data/imagenet'
    train_txt_path = 'data/imagenet/imagenet2012_train_image_infos.txt'
    val_txt_path = 'data/imagenet/imagenet2012_val_image_infos.txt'
    # test_txt_path = 'data/hr-ting/hr-ting_test_infos.txt'
    openset_dir = 'data/imagenet_LT/ImageNet_LT_open'
    openset_txt_path = 'data/imagenet_LT/imagenet_oltr_open.txt'
    num_classes = 1000
    imagenet_resize = [224, 224]
    imagenet_crop = [224, 224, 3]

    # Directories
    results_dir = f'exps/{exp_name}'
    classifier_weight_path = None

    # Trainer
    num_epochs = 1000
    lr = 1e-4
    beta1 = 0.5
    beta2 = 0.999

