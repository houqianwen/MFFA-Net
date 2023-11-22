# 
Title `Semi-Supervised Single Image Dehazing with Dual Teachers-Student Based on Knowledge Transfer` 
## Dependencies

- Ubuntu==18.04
- Pytorch==1.8.1
- CUDA==11.1

## Data Preparation
the structure of  `data`  are aligned as follows:

```
data
├── labeled
│   ├── clear
│   └── haze
│   └── result
│          └── smodel
├── unlabeled
│   ├── haze
│   └── result 
│         ├── smodel
│         └── tmodel
│── pre_teacher
│   └── val_best_result 
└── val
    ├── clear
    └── haze
    └── result
```


```latex
@inproceedings{Qianwen Hou2023,
  title={Semi-Supervised Image Dehazing with Dual Teachers-Student Based on Knowledge Transfer},
  author={Jianlei Liu,Qianwen Hou,Shilong Wang},
  year={2023}
}
```



