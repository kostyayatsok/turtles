import torch
import torchvision
from src.model import get_model
from src.data import load_csv, load_images, get_dataloader
from src.train import predict
from config import *
import argparse
import pandas as pd

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=argparse.SUPPRESS)
    parser.add_argument("--views_model", default=argparse.SUPPRESS)
    args = parser.parse_args()

    train, _, test = load_csv(0.)
    load_images()

    known_ids = train[train["is_known"]].unique()
    idx2id = train["turtle_id"].unique()
    id2idx = {v : i for i, v in enumerate(idx2id)}
    
    views_model = None
    model = get_model(num_classes, device, model_type)
    if "checkpoint" in args:
        model.load_state_dict(torch.load(args.checkpoint))
    if "views_model" in args:
        views_model = get_model(3, device, "simple")
        views_model.load_state_dict(torch.load(args.views_model))

    test["turtle_id"] = "new_turtle" 
    loader = get_dataloader(test, val_transforms, id2idx, batch_size, shuffle=False)

    for i in range(5):
        test[f'prediction{i+1}'] = "new_turtle"

    prediction = []
    for inputs, labels, views in loader:
        outputs, pred = predict(
            model, inputs, labels, views, device,
            phase='val', views_model=None, mode='simple'
        )
        outputs = torch.softmax(outputs, axis=-1).cpu()

        for out in outputs:
            idxs = np.argsort(out)
            new_turtle_flag = True
            pred = []
            for idx in idxs:
                if idx2id[idx] in known_ids:
                    pred.append(idx2id[idx])
                elif new_turtle_flag:
                    pred.append("new_turtle")
                    new_turtle_flag = False

                if len(pred) == 5:
                    break
            prediction.append(pred)
    prediction = pd.DataFrame(prediction, columns=[f"prediction{i+1}" for i in range(5)])
    prediction["image_id"] = test["image_id"]
    prediction.to_csv("submission_turtles.csv")