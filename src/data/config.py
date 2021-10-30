from common.config import ex, data_dir, cache_dir, data_download

data_dir = data_dir
cache_dir = cache_dir
download = data_download

num_trains = {
    'celeba': 81385,  # total 162770
    'cifar': 50000,
    'cifar100': 50000,
    'cococap': 59144,  # total 118287
    'cocodet': 59144,  # total 118287
    'fmnist': 60000,
    'imagenet': 1281167,
    'miniimagenet': 50000,
    'mnist': 60000,
    'svhn': 73257,
    'voc': 1464,
}

num_tests = {
    'celeba': 9934,  # total 19867
    'cifar': 10000,
    'cifar100': 10000,
    'cococap': 5000,  # total 5000
    'cocodet': 5000,  # total 5000
    'fmnist': 10000,
    'imagenet': 50000,
    'miniimagenet': 10000,
    'mnist': 10000,
    'svhn': 26032,
    'voc': 1449,
}

input_sizes = {
    'celeba': 3*32*32,
    'cifar': 3*32*32,
    'cifar100': 3*32*32,
    'cococap': 3*32*32,
    'cocodet': 3*32*32,
    'fmnist': 28*28,
    'imagenet': 3*32*32,
    'miniimagenet': 3*32*32,
    'mnist': 28*28,
    'svhn': 3*32*32,
    'voc': 3*32*32,
}

output_sizes = {
    'celeba': 10,
    'cifar': 10,
    'cifar100': 100,
    'cococap': 768,
    'cocodet': 2,
    'fmnist': 10,
    'imagenet': 1000,
    'miniimagenet': 100,
    'mnist': 10,
    'svhn': 10,
    'voc': 22,
}

num_channels = {
    'celeba': 3,
    'cifar': 3,
    'cifar100': 3,
    'cococap': 3,
    'cocodet': 3,
    'fmnist': 1,
    'imagenet': 3,
    'miniimagenet': 3,
    'mnist': 1,
    'svhn': 3,
    'voc': 3,
}

cnn_view = {
    'celeba': 5*5*50,
    'cifar': 5*5*50,
    'cifar100': 5*5*50,
    'cococap': 5*5*50,
    'cocodet': 5*5*50,
    'fmnist': 4*4*50,
    'imagenet': 5*5*50,
    'miniimagenet': 5*5*50,
    'mnist': 4*4*50,
    'svhn': 5*5*50,
    'voc': 5*5*50,
}

im_size = {
    'celeba': 32,
    'cococap': 32,
    'cocodet': 32,
    'imagenet': 32,
    'miniimagenet': 32,
    'voc': 64,
}

model_im_size = {
    'cnn': 0,
    'fcn': 0,
    'resnet18': 0,
    'resnet34': 0,
    'resnet101': 0,
    'resnet152': 0,
    'svm': 0,
    'unet': 0,
    'vgg19': 32,
}
