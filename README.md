# PointGrow: Autoregressively Learned Point Cloud Generation with Self-Attention
This work presents a novel autoregressive model, PointGrow, which generates realistic point cloud samples from scratch or conditioned on given semantic contexts. This model operates recurrently, with each point sampled according to a conditional distribution given its previously-generated points. It is further augmented with dedicated self-attention modules to capture long-range interpoint dependencies during the generation process.

## Data
We provided processed point clouds from 7 categories in [ShapeNet](https://www.shapenet.org), including airplane, car, table, chair, bench, cabinet and lamp. The coordinates of those point clouds, arranged as (z, y, x), range from 0 to 1. They are sorted in the order of z, y and x, and can be downloaded from [here](https://www.dropbox.com/s/nlcswrxul1ymypw/ShapeNet7.zip).


## Unconditional PointGrow
The model is trained per category, change the ShapeNet category id when working on different categories.

|    Category   |      Id       | 
| ------------- | ------------- |
| Airplane      | 02691156      |
| Car           | 02958343      |
| Table         | 04379243      |
| Chair         | 03001627      |
| Bench         | 02828884      |
| Cabinet       | 02933112      |
| Lamp          | 03636649      |

* Run unconditional PointGrow training script for airplane category with SACA-A module:
``` bash
python train_unconditional.py --cat 02691156 --model unconditional_model_saca_a
```
Model parameters will be stored under "_log/unconditional_model_saca_a/02691156_".

* To generate 300 point clouds for airplane category using the pre-trained model:
``` bash
python generate_unconditional.py --cat 02691156 --model unconditional_model_saca_a --tot_pc 300
```
The generated point clouds will be stored in the format of numpy array under "_res/unconditional_model_saca_a/res_02691156.npy_".

## Conditional PointGrow
### One-hot categorical vectors
Generate point clouds conditioned on additional one-hot vectors, with their non-empty elements indicating ShapeNet categories. For example, following the order in the above table, the one-hot vector for airplane can be expressed as [1, 0, 0, 0, 0, 0, 0]. 

* Train conditional PointGrow for one-hot vectors:
``` bash
python train_conditional_one_hot.py
```
Model parameters will be saved under "_log/conditional_model_one_hot_".

* To generate 50 point clouds for airplane (cat_idx = 0) using the pre-trained model:
``` bash
python generate_conditional_one_hot.py --cat_idx 0 --tot_pc 50
```
The generated point clouds will be stored in the format of numpy array under "_res/conditional_model_one_hot/res_02691156.npy_".

### Image embeddings 
Generate point clouds conditioned on the embedding vectors of given 2D images. We still use ShapeNet point clouds, and obtain their 2D renderings from 3D-R2N2(https://github.com/chrischoy/3D-R2N2). A collection of 2D images of the shape categories used in this project, and the shape ids that match the point clouds provided in this project can be found here(https://www.dropbox.com/s/cf1fak6ssvauqmr/ShapeNetRenderings.zip). 


