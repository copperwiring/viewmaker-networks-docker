```
docker pull nvcr.io/nvidia/pytorch:20.11-py3

docker run --gpus=all --rm -it -d -p 8888:8888 -p 6006:6006 -v /home/das/srishtiy/viewmaker:/viewmaker --ipc=host nvcr.io/nvidia/pytorch:20.11-py3 /bin/bash

docker ps
docker exec -it <change as required, e.g. beautiful_merkel> /bin/bash

jupyter-lab --allow-root --ip='0.0.0.0' --NotebookApp.token='' & 
```

You can then open your code on `http://localhost:8888/`

## 1) Install Dependencies

```
Install other dependencies:
```console
pip install -r requirements.txt
```

## 2) Running experiments

Start by running
```console
source init_env.sh
```

Now, you can run experiments for the different modalities as follows:

```console
scripts/run_sensor.py config/sensor/pretrain_viewmaker_pamap2_simclr.json --gpu-device 0
```


The `scripts` directory holds:
- `run_image.py`: for pretraining and running linear evaluation on CIFAR-10
- `run_meta_transfer.py`: for running linear evaluation on a range of transfer datasets, including many from MetaDataset
- `run_audio.py`: for pretraining on LibriSpeech and running linear evaluation on a range of transfer datasets
- `run_sensor.py`: for pretraining on Pamap2 and running transfer, supervised, and semi-supervised learning on different splits of Pamap2
- `eval_cifar10_c.py`: for evaluating a linear evaluation model on the CIFAR-10-C dataset for assessing robustness to common corruptions

The `config` directory holds configuration files for the different experiments,  specifying the hyperparameters from each experiment. The first field in every config file is `exp_base` which specifies the base directory to save experiment outputs, which you should change for your own setup.

You are responsible for downloading the datasets. Update the paths in `src/datasets/root_paths.py`.

Training curves and other metrics are logged using [wandb.ai](wandb.ai)
