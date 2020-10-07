from enum import Enum


class ModelStructure(Enum):
    def __str__(self):
        return '%s' % self.value

    VANILLA = 'vanilla'
    SENET = 'senet'
    SPNORM = 'spectral_norm'
    DEFORM = 'deformable'


class Config:
    n_gpus = 1
    epochs_to_save_gradCAM = 1
    epochs_to_save_weights = 10

    # Data
    n_workers = 4

    # Model
    structure = ModelStructure.VANILLA
    kernel_size = 3
    strides = 2
    activation = 'lrelu'
    shared_conv_channels = [32, 64, 128, 256]
    dense_branch_units = [512, 256]

    # Directories
    root_dir = 'data'
    results_dir = f'exps/svhn'

    # Trainer
    num_epochs = 1000
    batch_size = 64
    lr = 1e-3
    beta1 = 0.5
    beta2 = 0.999

