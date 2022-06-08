### This is the repository for the Bachelor Thesis "Mesh Repairing using Deep Networks", written by Benjamin Sühling

The directory `Blender` contains the dataset generator.
The scripts `knife.sh` and `pillows.sh` serve the purpose of creating the knife and pillow datasets used for evaluation in this thesis.
For the pillows dataset, the ground truth meshes have to be exchanged manually.

The directory `MeshRepairing` contains the two implemented approaches to repair meshes under utilization of deep neural networks.
For these two algorithms, the basic project structure was adopted from [MeshCNN](https://ranahanocka.github.io/MeshCNN/).
This especially includes the Mesh data structure and the convolution and pooling layer, but also some other methods.
The `global.sh` files are scripts to start the learning process of the global approach, and the `hole.sh` files are scripts to start the learing process of the hole filling approach.
The `global_eval.sh` and `hole_eval.sh` files serve to start the evaluation processes. The paths to the PyTorch models have to be specified in the respective Python files.
The purpose of `evaluation.sh` is to obtain the quantitative results of the evaluation meshes.
