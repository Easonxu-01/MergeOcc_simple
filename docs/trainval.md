# Training
1. **Train a unified occupancy prediction model with 8 GPUs**

   ```
   bash run.sh ./projects/configs/baselines/LiDAR_RPR_bbnorm_unilidar.py 8
   ```

2. **Train a coarse-to-fine unified occupancy prediction model with 8 GPUs**

   ```
   bash run.sh ./projects/configs/coarse-to-fine/LiDAR_cascade_RPR_bbnorm_unilidar.py 8
   ```

# Evaluation

**Evaluation example.**

```
bash run_eval.sh $PATH_TO_CFG $PATH_TO_CKPT $GPU_NUM
```

# Visualization

1. **generate the occupancy prediction results  to $show-dir**

```
bash run_eval.sh $PATH_TO_CFG $PATH_TO_CKPT $GPU_NUM --show --show-dir $PATH
```

2. **replace the result path in visualize.py and run the code.**
