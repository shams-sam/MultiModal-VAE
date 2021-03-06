import torch

from common.config import ex
from data import loader


@ex.automain
def main(_config):
    if _config["unit_test"] == "gpu":
        print(f"\ncuda ready? {torch.cuda.is_available()}\n")

    elif _config["unit_test"] == "loader":
        for dataset in loader.__all__:
            for split in [True, False]:
                print('-' * 80)
                print('dataset: {}, train:{}'.format(dataset, split))
                data_loader = loader.get_loader(
                    dataset, batch_size=16, train=split, shuffle=True)
                print('\tdata_size:', len(data_loader.dataset))
                for data, label in data_loader:
                    print('\tbatch_size:', data.shape, label.shape)
                    break

                if dataset not in loader.__subset__:
                    continue
                data_loader = loader.get_loader(
                    dataset, batch_size=16, train=split, shuffle=True, subset=0.1)
                print('\tdata_size:', len(data_loader.dataset))
                for data, label in data_loader:
                    print('\tbatch_size:', data.shape, label.shape)
                    break
