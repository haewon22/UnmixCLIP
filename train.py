import os
import json
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import clip
import glob
import logging
import wandb

from model import UnmixCLIP, MLPProjector
from losses import MFILoss, AsymmetricLoss
from Dataset import Coco14Dataset

from Cutout import Cutout

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)

@torch.no_grad()
def validate(model, dataloader, text_proj, asl_loss_fn, device):
    model.eval()
    total_loss = 0.0
    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        img_proj = model(images) 
        logits = img_proj @ text_proj.t()
        loss = asl_loss_fn(logits, targets)
        total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    device = "mps"
    logging.info(f"🖥️  Device set to: {device}")

    clip_model, _ = clip.load("RN101", device=device)

    wandb.init(
        project="unmix-clip",
        name="",
        config={
            "lr": 0.002,
            "momentum": 0.9,
            "batch_size": 32,
            "epochs": 50,
            "text_proj": [512, 384, 256],
            "image_proj": [clip_model.visual.output_dim, 256],
            "lambda_mfi": 0.2
        }
    )

    image_projector = MLPProjector(clip_model.visual.output_dim, [], 256)
    text_projector = MLPProjector(512, [384], 256)
    model = UnmixCLIP(clip_model, image_projector, text_projector).to(device)

    mfi_loss_fn = MFILoss(lambda_=0.2)
    asl_loss_fn = AsymmetricLoss()


    optimizer = optim.SGD(
        list(model.image_projector.parameters()) + list(model.text_projector.parameters()),
        lr=0.002,
        momentum=0.9
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        Cutout(1, 16),
        transforms.RandAugment(),
        transforms.ToTensor(),
    ])

    train_dataset = Coco14Dataset("./data/train2014", "./data/annotations/instances_train2014.json", transform)
    val_dataset = Coco14Dataset("./data/val2014", "./data/annotations/instances_val2014.json", transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    with open("prompts.json", "r") as f:
        prompt_config = json.load(f)

    class_names = train_dataset.classes
    pos_prompts, neg_prompts = [], []
    for c in class_names:
        if c in prompt_config:
            pos_prompts.append(prompt_config[c]["positive"])
            neg_prompts.append(prompt_config[c]["negative"])
        else:
            pos_prompts.append(f"A photo of a {c}.")
            neg_prompts.append(f"A photo without a {c}.")

    pos_tokens = clip.tokenize(pos_prompts).to(device)
    neg_tokens = clip.tokenize(neg_prompts).to(device)

    num_epochs = 50
    log_interval = 10
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        logging.info(f"▶️ Start Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            pos_text_proj = model.forward_text(pos_tokens)
            neg_text_proj = model.forward_text(neg_tokens)
            final_text_proj = F.normalize(pos_text_proj - neg_text_proj, dim=-1)

            img_proj = model(images)
            loss_mfi = mfi_loss_fn(final_text_proj.unsqueeze(0))
            logits = img_proj @ final_text_proj.t()
            loss_asl = asl_loss_fn(logits, targets)
            loss = loss_asl + 7e-5 * loss_mfi

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            if batch_idx % log_interval == 0:
                current_lr = scheduler.get_last_lr()[0]
                logging.info(
                    f"Epoch {epoch+1}/{num_epochs} | Step {batch_idx+1}/{len(train_loader)} | "
                    f"Global Step {global_step} | Loss: {loss.item():.4f} | LR: {current_lr:.6f}"
                )
                wandb.log({
                    "step_loss": loss.item(),
                    "lr": current_lr
                }, step=global_step)

        avg_loss = running_loss / len(train_loader)
        scheduler.step()
        logging.info(f"✅ End of Epoch {epoch+1}/{num_epochs} | Avg Train Loss: {avg_loss:.4f}")
        wandb.log({
            "epoch": epoch + 1,
            "loss": avg_loss
        }, step=global_step)

        pos_text_proj = model.forward_text(pos_tokens)
        neg_text_proj = model.forward_text(neg_tokens)
        final_text_proj = F.normalize(pos_text_proj - neg_text_proj, dim=-1)

        val_loss = validate(model, val_loader, final_text_proj, asl_loss_fn, device)
        logging.info(f"📊 Validation Loss: {val_loss:.4f}")
        wandb.log({"val_loss": val_loss}, step=global_step)

        ckpt_dir = "ckpt"
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"unmix_clip_projectors_step{global_step}.pth")
        torch.save({
            "epoch": epoch + 1,
            "global_step": global_step,
            "image_projector": model.image_projector.state_dict(),
            "text_projector": model.text_projector.state_dict()
        }, ckpt_path)
        logging.info(f"💾 Checkpoint saved to {ckpt_path}")
        wandb.save(ckpt_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(ckpt_dir, "best_checkpoint.pth")
            torch.save({
                "epoch": epoch + 1,
                "global_step": global_step,
                "val_loss": val_loss,
                "image_projector": model.image_projector.state_dict(),
                "text_projector": model.text_projector.state_dict()
            }, best_path)
            logging.info(f"🏅 Best checkpoint updated at epoch {epoch+1} (val_loss={val_loss:.4f})")
            wandb.save(best_path)

        ckpt_files = sorted(
            glob.glob(os.path.join(ckpt_dir, "unmix_clip_projectors_step*.pth")),
            key=os.path.getmtime
        )
        if len(ckpt_files) > 10:
            for old_ckpt in ckpt_files[:-10]:
                os.remove(old_ckpt)
                logging.info(f"🧹 Deleted old checkpoint: {old_ckpt}")

    wandb.finish()

if __name__ == "__main__":
    main()
