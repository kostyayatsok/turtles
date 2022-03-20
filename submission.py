import torch
import torchvision
from src.model import get_model
from src.data import load_csv, load_images, get_dataloader, train_val_split
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

    load_images()

    train, extra, test = load_csv()
    idx2id = torch.load("idx2id.pt")
    id2idx = {v : i for i, v in enumerate(idx2id)}
    known_ids = train.loc[train["is_known_id"], "turtle_id"].unique()
     
    views_model = None
    model = get_model(num_classes, device, model_type)
    if "checkpoint" in args:
        model.load_state_dict(torch.load(args.checkpoint))
    if "views_model" in args:
        views_model = get_model(3, device, "simple")
        views_model.load_state_dict(torch.load(args.views_model))

    test["turtle_id"] = idx2id[0]
    loader = get_dataloader(test, val_transforms, id2idx, batch_size, shuffle=False)

    for i in range(5):
        test[f'prediction{i+1}'] = "new_turtle"

    prediction = []
    for inputs, labels, views in loader:
        outputs, pred = predict(
            model, inputs, labels, views, device,
            phase='test', views_model=views_model, mode=model_type
        )
        outputs = torch.softmax(outputs, axis=-1).cpu()

        for out in outputs:
            idxs = np.argsort(-out)
            new_turtle_flag = True
            pred = []
            for idx in idxs:
                if idx < idx2id.shape[0] and idx2id[idx] in known_ids:
                    pred.append(idx2id[idx])
                elif new_turtle_flag:
                    pred.append("new_turtle")
                    new_turtle_flag = False

                if len(pred) == 5:
                    break
            prediction.append(pred)
    prediction = pd.DataFrame(prediction, columns=[f"prediction{i+1}" for i in range(5)])
    prediction["image_id"] = test["image_id"]
    prediction[["image_id", "prediction1", "prediction2", "prediction3", "prediction4", "prediction5"]].to_csv("submission_turtles.csv", index=False)
