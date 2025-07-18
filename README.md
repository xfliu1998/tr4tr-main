# TR4TR 

This is an official pytorch implementation of our paper [Spatial-Temporal Transformer for Single RGB-D Camera Synchronous Tracking and Reconstruction of Non-rigid Dynamic Objects](https://link.springer.com/article/10.1007/s11263-025-02469-5).
In this repository, we provide PyTorch code for training and testing our proposed TR4TR model. 

## Setup
First, clone the repository locally:
```shell script
git clone https://github.com/xfliu1998/tr4tr-main.git
```
After cloning this repo, `cd` into it and create a conda environment for the project:
```shell script
cd resources
conda env create --file env.yaml
cd ..
```
Then, activate the environment:
```shell script
conda activate tr4tr
```

## Usage
### Dataset Preparation
We train and evaluate our network using the DeepDeform dataset, 
the original data can be obtained at the [DeepDeform](https://github.com/AljazBozic/DeepDeform) repository.
After downloading the data, you need to change the dataset path in file `config.yaml` and `utils/data_utils.py` to the path where you downloaded the data.
Then generate the json file to train and evaluate with the following command:
```shell script
cd utils
sh data_utils.sh
cd ..
```

### Train and Evaluate
You can set your customized model parameters by modifying the file `config.yaml`, 
including modifying the input form of data, network architecture parameters, and training hyperparameters.
Then you need to modify the GPU parameters in file `main.sh`.
You can train and evaluate the model with the following command:
```shell script
sh main.sh
```

## Visualize
If you want to visualize the results, you need to specify the path of the pre-trained model in file `config.yaml` and your own file path, 
and write the following json file `val_.json` to put under the same level file as the data you want to visualize.
```json
[
	{
	"source_color": "val/color/shirt_000000.jpg",
	"source_depth": "val/depth/shirt_000000.png",
	"target_color": "val/color/shirt_000100.jpg",
	"target_depth": "val/depth/shirt_000100.png",
	"object_id": "shirt",
	"source_id": "000000",
        "target_id": "000100",
	"optical_flow": "val/optical_flow/shirt_000000_000100.oflow",
        "scene_flow": "val/scene_flow/shirt_000000_000100.sflow"
	}
]
```

Then you need to modify the parameter `experiment_mode='predict'` in the file `main.sh` and run the command:
```shell script
sh main.sh
```
You can refer to file `utils/visual_utils.py` for related view instructions.

## Related File
- [model checkpoint](https://drive.google.com/open?id=1TgIsS4jdYpxP_qDazbF6v42ySwSss5BA&usp=drive_fs)
- [pcd result](https://drive.google.com/open?id=1sLQ_vsyQd9D7mJnw_8LOlcJ9u7EUsHg8&usp=drive_fs)

# License
TR4TR is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.
