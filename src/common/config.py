from sacred import Experiment


data_dir = "/home/azam1/workspace/data"
cache_dir = f"{data_dir}/feather"
data_download = False
ex = Experiment("multimodal")


@ex.config
def base():
    base_dir = "/home/azam1/workspace"
    data_dir = data_dir
    unit_test = "gpu"


@ex.named_config
def test_loader():
    unit_test = "loader"
