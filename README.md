# NoCGNN
SEEKING SIMILARITIES WHILE REMOVING DIFFERENCES: GRAPH NEURAL NETWORKS BASED ON NODE CORRELATION

## Dependencies

The script has been tested running under Python 3.7.4, with the following packages installed (along with their dependencies):

- `dgl-cpu==0.4.3.post2`
- `dgl-gpu==0.4.3.post2`
- `ogb==1.3.1`
- `numpy==1.19.2`
- `scipy==1.4.1`
- `networkx==2.5`
- `torch==1.5.0`
- `torch-cluster==1.5.7`
- `torch-geometric==1.6.3`
- `torch-scatter==2.0.5`
- `torch-sparse==0.6.6`
- `torch-spline-conv==1.2.0`


```
pip install -r requirements.txt
```

## Running Experiments 

### Training or hyperparameter searching

```
# training with default hyperparameters (e.g. ACM-GCN+ on Texas)
python train.py --model acmgcnp --dataset_name texas

# training with user defined hyperparameters
python train.py --model acmgcnp --dataset_name texas --lr 0.06 --weight_decay 0.0006 --dropout 0.6

# hyperparameter searching (learning rate, weight_decay & dropout)
python hyperparameter_searching.py --model acmgcnp --dataset_name texas
python hyperparameter_searching.py  --dataset_name citeseer

```
The training/hyperparameter seacrhing logs are saved into the `logs/` folder located at `<your-working-directory>/Adaptive-Channel-Mixing-GNN/ACM-PyTorch/logs`.


## Reference
If you make advantage of the ACM framework in your research, please cite the following in your manuscript:

```
@article{luan2022revisiting,
  title={Revisiting Heterophily For Graph Neural Networks},
  author={Luan, Sitao and Hua, Chenqing and Lu, Qincheng and Zhu, Jiaqi and Zhao, Mingde and Zhang, Shuyuan and Chang, Xiao-Wen and Precup, Doina},
  journal={Conference on Neural Information Processing Systems},
  year={2022}
}
```

