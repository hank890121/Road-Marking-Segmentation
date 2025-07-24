import torch
from ema_pytorch import EMA
from rich import print
from rich.progress import Progress
from torch.nn import functional as F
from simple_parsing import Serializable
from dataclasses import dataclass
import os

from engine import builder, transform
from engine.category import Category
from engine.dataloader import (
    ImgAnnDataset,
    RCSConfig,
    RCSImgAnnDataset,
    RareCategoryManager,
)
from engine.logger import Logger
from engine.metric import Metrics
from engine.misc import set_seed
from engine.optimizer import AdamW
from engine.visualizer import IdMapVisualizer, ImgSaver
from engine.models.segformer import SegformerDiscriminator


from engine.validator import Validator


def compute_domain_discrimination_loss(
        model: torch.nn.Module,
        discriminator: torch.nn.Module,
        img: torch.Tensor,
        ann: torch.Tensor,
        domain_label: int,
        domain_class_weight: torch.Tensor
    ) -> torch.Tensor:
    latent = model.segformer(img, output_hidden_states=True).hidden_states
    dis_pred = discriminator(latent)
    dis_pred = F.interpolate(dis_pred, cfg.crop_size, mode="bilinear")
    dis_label = torch.zeros(dis_pred.shape).cuda()
    dis_label[:] = domain_label
    return F.binary_cross_entropy_with_logits(
        dis_pred,
        dis_label,
        domain_class_weight[ann][:, None, :],
    ).mean()


@dataclass
class TrainingConfig(Serializable):
    category_csv: str
    rcs_path: str | None
    source_train_root: str
    source_val_root: str
    target_train_root: str
    target_val_root: str

    logdir: str
    log_interval: int
    checkpoint_interval: int

    max_iters: int
    train_batch_size: int
    val_batch_size: int
    val_interval: int
    image_scale: tuple[int, int]
    crop_size: tuple[int, int]
    stride: tuple[int, int] | None
    random_resize_ratio: tuple[float, float]

    seed: int
    num_workers: int
    pin_memory: bool

    with_uda: bool
    ema_update_interval: list[int]
    rcs_temperature: float
    rcs_ignore_ids: tuple[int]
    domain_loss_weight: float
    num_masks: int


def main(cfg: TrainingConfig, exp_name: str, checkpoint: str):

    set_seed(cfg.seed)

    if cfg.with_uda:
        current_ema_update_interval_id = 0
        ema_update_interval_update_interval = cfg.max_iters // len(cfg.ema_update_interval)

    categories = Category.load(cfg.category_csv)
    num_categories = len(categories)

    source_domain_class_weight = torch.zeros((num_categories)).cuda()
    target_domain_class_weight = torch.zeros((num_categories)).cuda()

    model = builder.build_model(
        {
            "name": "segformer",
            "pretrained": "nvidia/mit-b0",
            "num_classes": num_categories,
        }
    ).cuda()

    if cfg.with_uda:
        ema = EMA(model, beta=0.999, update_after_step=-1, update_every=cfg.ema_update_interval).cuda()
        soft_loss_computer = builder.build_soft_loss_computer({"name": "PixelThreshold", "threshold": 0.968})
        discriminator: torch.nn.Module = SegformerDiscriminator.from_pretrained("nvidia/mit-b0", num_labels=1).cuda()

    criterion = builder.build_criterion(
        {
            "name": "cross_entropy_loss",
            "ignore_index": 255,
            "reduction": "none",  # Should be none
            "label_smoothing": 0,
        }
    ).cuda()

    if cfg.with_uda:
        optimizer = AdamW(
            [
                {"name": "backbone", "params": model.segformer.parameters(), "lr": 6e-5},
                {"name": "head", "params": model.decode_head.parameters(), "lr": 6e-4},
                {"name": "dicriminator", "params": discriminator.parameters(), "lr": 6e-4},
            ],
        )
    else:
        optimizer = AdamW(
            [
                {"name": "backbone", "params": model.segformer.parameters(), "lr": 6e-5,},
                {"name": "head", "params": model.decode_head.parameters(), "lr": 6e-4},
            ],
        )

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer.torch(), 1e-4, 1, 1500)
    poly_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer.torch(), cfg.max_iters, 1)

    source_train_transforms = [
        transform.LoadImg(),
        transform.LoadAnn(),
        transform.RandomResizeCrop(cfg.image_scale, cfg.random_resize_ratio, cfg.crop_size),
        transform.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transform.RandomGaussian(kernel_size=5),
        transform.Normalize(),
    ]
    if cfg.with_uda:
        target_train_transforms = [
            transform.LoadImg(),
            transform.RandomResizeCrop(cfg.image_scale, cfg.random_resize_ratio, cfg.crop_size),
            *[transform.RandomErase(scale=(0.02, 0.04)) for _ in range(cfg.num_masks)],
            transform.Normalize(),
        ]
    val_transforms = [
        transform.LoadImg(),
        transform.LoadAnn(),
        transform.Resize(cfg.image_scale),
        transform.Normalize(),
    ]

    if cfg.rcs_path:
        source_train_dataloader = RCSImgAnnDataset(
            root=cfg.source_train_root,
            transforms=source_train_transforms,
            img_prefix="images",
            ann_prefix="labels",
            img_suffix=".jpg",
            ann_suffix=".png",
            categories=categories,
            rcm=RareCategoryManager(categories, RCSConfig(cfg.rcs_path, cfg.rcs_ignore_ids, cfg.rcs_temperature)),
        ).get_loader(
            batch_size=cfg.train_batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            drop_last=True,
            pin_memory=cfg.pin_memory,
            infinite=True,
        )
    else:
        source_train_dataloader = ImgAnnDataset(
            root=cfg.source_train_root,
            transforms=source_train_transforms,
            img_prefix="images",
            ann_prefix="labels",
            img_suffix=".jpg",
            ann_suffix=".png",
        ).get_loader(
            batch_size=cfg.train_batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            drop_last=True,
            pin_memory=cfg.pin_memory,
            infinite=True,
        )

    if cfg.with_uda:
        target_train_dataloader = ImgAnnDataset(
            root=cfg.target_train_root,
            transforms=target_train_transforms,
            img_prefix="images",
            ann_prefix="labels",
            img_suffix=".jpg",
            ann_suffix=".png",
            check_exist=False,
        ).get_loader(
            batch_size=cfg.train_batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            drop_last=True,
            pin_memory=cfg.pin_memory,
            infinite=True,
        )

    source_val_dataloader = ImgAnnDataset(
        root=cfg.source_val_root,
        transforms=val_transforms,
        img_prefix="images",
        ann_prefix="labels",
        img_suffix=".jpg",
        ann_suffix=".png",
    ).get_loader(
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
        pin_memory=cfg.pin_memory,
    )

    target_val_dataloader = ImgAnnDataset(
        root=cfg.target_val_root,
        transforms=val_transforms,
        img_prefix="images",
        ann_prefix="labels",
        img_suffix=".jpg",
        ann_suffix=".png",
    ).get_loader(
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
        pin_memory=cfg.pin_memory,
    )

    logdir = os.path.join(cfg.logdir, exp_name)
    logger = Logger(logdir)
    img_saver = ImgSaver(os.path.join(logdir, "saved_images"), IdMapVisualizer(categories))
    metrics = Metrics(num_categories=num_categories, nan_to_num=0)

    if cfg.stride is not None:
        inferencer = builder.build_inferencer(
            {
                "mode": "slide",
                "crop_size": cfg.crop_size,
                "stride": cfg.stride,
                "num_categories": num_categories,
            }
        )
    else:
        inferencer = builder.build_inferencer({"mode": "basic"})

    validator = Validator(
        [("Source", source_val_dataloader), ("Target", target_val_dataloader)],
        model,
        criterion,
        inferencer,
        metrics,
        categories,
    )

    torch.compile(model)
    torch.compile(criterion)
    if cfg.with_uda:
        torch.compile(ema.ema_model)
        torch.compile(discriminator)

    scaler = torch.cuda.amp.GradScaler()

    start_it = 1
    if checkpoint is not None:
        ckpt = torch.load(os.path.join(cfg.logdir, exp_name, checkpoint))
        start_it = ckpt["iterations"]
        model.load_state_dict(ckpt["model"])
        optimizer.optimizer.load_state_dict(ckpt["optimizer"])
        warmup_scheduler.load_state_dict(ckpt["warmup_scheduler"])
        poly_scheduler.load_state_dict(ckpt["poly_scheduler"])
        if cfg.with_uda:
            ema.load_state_dict(ckpt["ema"])
            discriminator.load_state_dict(ckpt["discriminator"])
        print(f"Checkpoint loaded. Starting from iterations {start_it}")

    with Progress() as progress:
        train_task = progress.add_task("Training", total=cfg.max_iters)
        for it in range(start_it, cfg.max_iters + 1):

            if cfg.with_uda:
                ema.update_every = cfg.ema_update_interval[current_ema_update_interval_id]
                if it % ema_update_interval_update_interval == 0:
                    current_ema_update_interval_id += 1

                ema.update()
                discriminator.train()

            model.train()
            optimizer.zero_grad()

            # CBSS
            data = next(source_train_dataloader)
            source_img = data["img"].cuda()
            source_ann = data["ann"].cuda()

            with torch.cuda.amp.autocast():
                source_pred = model(source_img)
                source_loss = criterion(source_pred, source_ann).mean()
            scaler.scale(source_loss).backward()

            if cfg.with_uda and (it % ema.update_every == 0):
                data = next(target_train_dataloader)
                target_img = data["img"].cuda()
                if cfg.num_masks > 0:
                    erased_target_img = data["erased img"].cuda()

                # Pseudo labeling for SEMA and compute class-conditional pixel weights for CCDD
                with torch.no_grad(), torch.cuda.amp.autocast():
                    pseudo_source_ann = ema(source_img).softmax(1)
                    target_ann = ema(target_img).softmax(1)
                    max_pseudo_source_ann = torch.max(pseudo_source_ann, 1)
                    max_pseudo_target_ann = torch.max(target_ann, 1)
                    for cat in categories:
                        source_confidences = max_pseudo_source_ann.values[max_pseudo_source_ann.indices == cat.id].flatten()
                        source_confidences.sort()  # min to max
                        if len(source_confidences) > 0:
                            source_domain_class_weight[cat.id] = -source_confidences[int(len(source_confidences) * 0.8)].log()

                        target_confidences = max_pseudo_target_ann.values[max_pseudo_target_ann.indices == cat.id].flatten()
                        target_confidences.sort()  # min to max
                        if len(target_confidences) > 0:
                            target_domain_class_weight[cat.id] = -target_confidences[int(len(target_confidences) * 0.8)].log()

                with torch.cuda.amp.autocast():
                    target_pred = model(target_img)
                    target_loss = soft_loss_computer.compute(target_pred, target_ann, criterion)
                scaler.scale(target_loss).backward()

                if cfg.num_masks > 0:
                    with torch.cuda.amp.autocast():
                        erased_target_pred = model(erased_target_img)
                        erased_target_loss = soft_loss_computer.compute(erased_target_pred, target_ann, criterion)
                    scaler.scale(erased_target_loss).backward()

                # Latent Domain Discriminator
                with torch.cuda.amp.autocast():
                    source_dis_loss = compute_domain_discrimination_loss(model, discriminator, source_img, source_ann, 0, source_domain_class_weight)
                scaler.scale(source_dis_loss * cfg.domain_loss_weight).backward()

                with torch.cuda.amp.autocast():
                    target_dis_loss = compute_domain_discrimination_loss(model, discriminator, target_img, target_ann.argmax(1), 1, target_domain_class_weight)
                scaler.scale(target_dis_loss * cfg.domain_loss_weight).backward()

                source_domain_class_weight.fill_(0)
                target_domain_class_weight.fill_(0)

            optimizer.step(scaler)

            # Logging
            if it % cfg.log_interval == 0:
                if cfg.with_uda:
                    if it % ema.update_every == 0:
                        if cfg.num_masks > 0:
                            logger.info(
                                "Train",
                                f"Iteration: {it}, Source Loss: {source_loss.item(): .5f}, Target Loss: {target_loss.item(): .5f}, Erased Target Loss: {erased_target_loss.item(): .5f}, Discriminator Source Loss: {source_dis_loss: .5f}, Discriminator Target Loss: {target_dis_loss: .5f}",
                            )
                        else:
                            logger.info(
                                "Train",
                                f"Iteration: {it}, Source Loss: {source_loss.item(): .5f}, Target Loss: {target_loss.item(): .5f}, Discriminator Source Loss: {source_dis_loss: .5f}, Discriminator Target Loss: {target_dis_loss: .5f}",
                            )
                    else:
                        logger.info("Train", f"Iteration: {it}, Source Loss: {source_loss.item(): .5f}")
                else:
                    logger.info("Train", f"Iteration: {it}, Source Loss: {source_loss.item(): .5f}",)

                logger.tb_log("Train/Source Loss", source_loss.item(), it)
                if cfg.with_uda and (it % ema.update_every == 0):
                    logger.tb_log("Train/Target Loss", target_loss.item(), it)
                    if cfg.num_masks > 0:
                        logger.tb_log("Train/Target Erased Loss", erased_target_loss.item(), it)
                    logger.tb_log("Train/Source Discriminator Loss", source_dis_loss.item(), it)
                    logger.tb_log("Train/Target Discriminator Loss", target_dis_loss.item(), it)

                img_saver.save_img(source_img, f"train_source_{it}_img.jpg")
                img_saver.save_ann(source_ann, f"train_source_{it}_ann.jpg")
                img_saver.save_pred(source_pred, f"train_source_{it}_pred.jpg")

                if cfg.with_uda and (it % ema.update_every == 0):
                    img_saver.save_img(target_img, f"train_target_{it}_img.jpg")
                    img_saver.save_pred(target_ann, f"train_target_{it}_ann.jpg")
                    img_saver.save_pred(target_pred, f"train_target_{it}_pred.jpg")
                    if cfg.num_masks > 0:
                        img_saver.save_img(erased_target_img, f"train_target_{it}_erased_img.jpg")
                        img_saver.save_pred(erased_target_pred, f"train_target_{it}_erased_pred.jpg")

            warmup_scheduler.step()
            poly_scheduler.step()

            if it % cfg.val_interval == 0:
                val_outputs = validator.validate(progress)
                for name, loss, iou in val_outputs:
                    logger.info("Validation", f"Iteration: {it}, {name} Loss: {loss: .5f}, mIoU: {iou.mean()}")
                    logger.tb_log(f"Validation/{name} Loss", loss, it)
                    logger.tb_log(f"mIoU/{name}", iou.mean(), it)
                    for cat, _iou in zip(categories, iou):
                        logger.tb_log(f"IoU/{name}-{cat.name}", _iou, it)

            if cfg.with_uda:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "ema": ema.state_dict(),
                        "discriminator": discriminator.state_dict(),
                        "optimizer": optimizer.torch().state_dict(),
                        "iterations": it,
                        "warmup_scheduler": warmup_scheduler.state_dict(),
                        "poly_scheduler": poly_scheduler.state_dict(),
                    },
                    f"{logdir}/checkpoint_latest.pth",
                )
            else:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.torch().state_dict(),
                        "iterations": it,
                        "warmup_scheduler": warmup_scheduler.state_dict(),
                        "poly_scheduler": poly_scheduler.state_dict(),
                    },
                    f"{logdir}/checkpoint_latest.pth",
                )
            if it % cfg.checkpoint_interval == 0:
                if cfg.with_uda:
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "ema": ema.state_dict(),
                            "discriminator": discriminator.state_dict(),
                            "optimizer": optimizer.torch().state_dict(),
                            "iterations": it,
                            "warmup_scheduler": warmup_scheduler.state_dict(),
                            "poly_scheduler": poly_scheduler.state_dict(),
                        },
                        f"{logdir}/checkpoint_iter_{it}.pth",
                    )
                else:
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.torch().state_dict(),
                            "iterations": it,
                            "warmup_scheduler": warmup_scheduler.state_dict(),
                            "poly_scheduler": poly_scheduler.state_dict(),
                        },
                        f"{logdir}/checkpoint_iter_{it}.pth",
                    )

            progress.update(train_task, completed=it)
        progress.remove_task(train_task)


if __name__ == "__main__":
    import sys

    assert len(sys.argv) == 3 or len(sys.argv) == 4
    cfg = TrainingConfig.load(sys.argv[1])
    exp_name = sys.argv[2]
    checkpoint = sys.argv[3] if len(sys.argv) == 4 else None
    main(cfg, exp_name, checkpoint)
