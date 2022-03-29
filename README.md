# Turtle Recall: Conservation Challenge

To get best model run two commands sequentially:

```{bash}
python train.py \
    --mode train \
    --use_extra_data \
    --new_turtles_fraq 4.5
```

```{bash}
python train.py \
    --mode train \
    --use_extra_data \
    --new_turtles_fraq 0.05 \
    --checkpoint checkpoints/model.pt
```

To generate submission:
```{bash}
python3 submission.py --checkpoint checkpoints/model.pt
```
