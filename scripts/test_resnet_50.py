import os
# Set cache directory BEFORE importing datasets library
os.environ['HF_DATASETS_CACHE'] = '/simdata/pahrendt/datasets'

from distqat.models import ResNet50Full
from datasets import load_dataset
from torch.utils.data import DataLoader
from distqat.data import CVDataset
from torchvision import transforms
import torch
import time
from tqdm import tqdm

model = ResNet50Full(num_classes=1000, num_channels=3)

# print(model)


ds = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)
# ds = SyntheticImageNetDataset(img_size=224, num_classes=1000, seed=42)
# torch.backends.cudnn.benchmark = True
# torch.set_float32_matmul_precision("high")
transform = transforms.Compose([
                    transforms.Lambda(lambda img: img.convert("RGB")),
                    transforms.Resize(232),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # ImageNet mean and std
                ])
ds = CVDataset(ds, content_key="image", transform=transform)

def cv_collate_fn(batch):
    import torch
    uids, batch = zip(*batch)
    images = [
        (b["image"].to(dtype=torch.float32) if torch.is_tensor(b["image"]) else torch.as_tensor(b["image"], dtype=torch.float32))
        for b in batch
    ]
    labels = [torch.as_tensor(b["label"], dtype=torch.long) for b in batch]
    return uids, {
        "inputs": torch.stack(images, dim=0),
        "labels": torch.stack(labels, dim=0),
    }

    
loader = DataLoader(
    ds,                                 # yields (uid, sample) or sample
    batch_size=1024,
    num_workers=16,
    pin_memory=True,
    drop_last=True,
    shuffle=False,                            # shuffle ignored for IterableDataset
    collate_fn=cv_collate_fn,
    persistent_workers=True,
    prefetch_factor=2,
)

model = model.to("cuda")#.half()
optimizer = torch.optim.Adam(model.parameters(), lr=4e-4, weight_decay=0.1, betas=(0.9, 0.95))
criterion = torch.nn.CrossEntropyLoss()


start_time = time.time()
# scaler = torch.amp.GradScaler('cuda')
for epoch in range(10):
    # You can try to estimate length by dividing a fixed subset size by batch size if known, for IterableDataset set total to None
    with tqdm(loader, desc=f"Epoch {epoch}", unit="batch") as pbar:
        for i, batch in enumerate(pbar):
            uids, batch = batch
            inputs, labels = batch["inputs"], batch["labels"]
            inputs = inputs.to("cuda")
            labels = labels.to("cuda")
            # with torch.amp.autocast("cuda", dtype=torch.bfloat16):  # or fp16
            outputs = model(inputs)
            # compute loss in fp32 for a bit more numerical stability
            loss = criterion(outputs.float(), labels)
            loss.backward()
            optimizer.step()
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            optimizer.zero_grad(set_to_none=True)
            # pbar.set_postfix({"loss": loss.item()})
            if i % 10 == 0:
                pbar.set_postfix({"loss": float(loss.detach().cpu())})
            
            if i == 100:
                break
    end_time = time.time()
    print(f"Epoch {epoch} time: {end_time - start_time}")
    start_time = end_time
    break
