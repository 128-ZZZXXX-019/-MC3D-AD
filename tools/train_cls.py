# -*- coding: utf-8 -*-
import argparse
import logging
import os
import pprint
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from easydict import EasyDict
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets.data_builder import build_dataloader
from models.model_helper import ModelHelper
from utils.dist_helper import setup_distributed
from utils.lr_helper import get_scheduler
from utils.misc_helper import (
    AverageMeter,
    create_logger,
    get_current_time,
    load_state,
    save_checkpoint,
    set_random_seed,
)
from utils.optimizer_helper import get_optimizer


parser = argparse.ArgumentParser(description="PointMAE classification training") #描述
parser.add_argument("--config", default="./config.yaml")
parser.add_argument("-e", "--evaluate", action="store_true") #布尔开关，出现-e或者--evaluate表示true，开启评估模式
parser.add_argument("--local_rank", default=None) #分布式训练进程在本机上的序号，默认none

#分布式是否可用
def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

#把不同类型的类别名统一转换为字符串
def normalize_cls_name(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")

    if torch.is_tensor(x):
        if x.numel() == 1:
            return str(x.item())
        return str(x.detach().cpu().tolist())

    return str(x)


def build_class_names(dataset_cfg):
    class_names = dataset_cfg.get("class_names", None)

    if class_names is not None: #类别名文件的路径
        if isinstance(class_names, str): #如果是字符串
            with open(class_names, "r") as f: #打开保存类别名的文件
                return [line.strip() for line in f if line.strip()] #逐行读取，去掉首尾空白，忽略空行
        return [str(x) for x in class_names] #返回类别名列表["airplane","chair","table"]

    #配置里没有 class_names，认为cls_name来自data文件夹下的子文件夹名
    data_dir = dataset_cfg.data_dir
    class_names = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]
    class_names = sorted(class_names)

    if len(class_names) == 0:
        raise RuntimeError(
            f"No class folders found under dataset.data_dir={data_dir}. "
            f"Please set dataset.class_names manually."
        )

    return class_names

#从config中获取分类器类别数量
def get_classifier_cls_num(config):
    for module_cfg in config.net:
        if module_cfg["name"] == "classifier":
            return int(module_cfg["kwargs"]["cls_num"])

    raise RuntimeError("Cannot find module named 'classifier' in config.net")

#提取一个batch中的每个样本的真实标签，并同义转换成指定设备上的一维张量
def get_cls_targets(batch, class_to_idx, device):
    # 不要使用 batch['label']，因为在异常检测数据集中，label 通常表示“正常/异常”，而不是具体的类别标签
    for label_key in ["cls_label", "class_label", "category_id"]:
        if label_key in batch: #检查以上key是否在batch字典中
            value = batch[label_key] #取出值 例如[1,0,3,4]

            #转一维张量
            if torch.is_tensor(value):
                return value.view(-1).to(device=device, dtype=torch.long)

            if isinstance(value, (list, tuple)):
                return torch.tensor(value, device=device, dtype=torch.long)

            return torch.tensor([value], device=device, dtype=torch.long)

    #备选方案
    name_key = None
    for candidate in ["clsname", "class_name", "category", "class"]:
        if candidate in batch: #看是否有这些key名
            name_key = candidate
            break

    if name_key is None:
        raise KeyError(
            "Cannot build classification target. "
            "Expected one of ['cls_label', 'class_label', 'category_id', 'clsname'] in batch."
        )

    #取出类别名列表
    cls_names = batch[name_key]

    if isinstance(cls_names, (str, bytes)):
        cls_names = [cls_names]

    #转换为数字索引
    labels = []
    for cls_name in cls_names:
        name = normalize_cls_name(cls_name)

        if name not in class_to_idx:
            raise KeyError(
                f"Class name '{name}' not found in class_to_idx. "
                f"Available classes={list(class_to_idx.keys())}"
            )

        labels.append(class_to_idx[name])

    return torch.tensor(labels, device=device, dtype=torch.long)

#接下来两个函数：
# 分布式训练/推理中跨进程收集张量，并解决各进程张量大小不一致的问题
def all_gather_1d_tensor(tensor):
    if not is_dist_avail_and_initialized():
        return tensor

    tensor = tensor.contiguous()
    world_size = dist.get_world_size()

    local_size = torch.tensor(
        [tensor.numel()],
        dtype=torch.long,
        device=tensor.device,
    )

    size_list = [
        torch.zeros_like(local_size)
        for _ in range(world_size)
    ]

    dist.all_gather(size_list, local_size)

    max_size = int(torch.stack(size_list).max().item())

    if tensor.numel() < max_size:
        pad = torch.empty(
            max_size - tensor.numel(),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        tensor = torch.cat([tensor, pad], dim=0)

    gathered = [
        torch.empty(max_size, dtype=tensor.dtype, device=tensor.device)
        for _ in range(world_size)
    ]

    dist.all_gather(gathered, tensor)

    output = []
    for item, size in zip(gathered, size_list):
        output.append(item[: int(size.item())])

    return torch.cat(output, dim=0)


def all_gather_first_dim_tensor(tensor):
    if not is_dist_avail_and_initialized():
        return tensor

    tensor = tensor.contiguous()
    world_size = dist.get_world_size()

    local_size = torch.tensor(
        [tensor.shape[0]],
        dtype=torch.long,
        device=tensor.device,
    )

    size_list = [
        torch.zeros_like(local_size)
        for _ in range(world_size)
    ]

    dist.all_gather(size_list, local_size)

    max_size = int(torch.stack(size_list).max().item())

    if tensor.shape[0] < max_size:
        pad_shape = list(tensor.shape)
        pad_shape[0] = max_size - tensor.shape[0]
        pad = torch.zeros(
            pad_shape,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        tensor = torch.cat([tensor, pad], dim=0)

    gathered = [
        torch.zeros_like(tensor)
        for _ in range(world_size)
    ]

    dist.all_gather(gathered, tensor)

    output = []
    for item, size in zip(gathered, size_list):
        output.append(item[: int(size.item())])

    return torch.cat(output, dim=0)

#获取分布式训练的底层模型对象
def _get_base_model(model):
        #如果model中有module参数，则返回它
    return model.module if hasattr(model, "module") else model


def _get_classifier(model):
    base_model = _get_base_model(model)
    #如果底层模型有classifier参数，则返回他
    if hasattr(base_model, "classifier"):
        return base_model.classifier
    return None

#在每个训练 step 中，跨所有进程收集特征，更新共享的 EMA 原型，然后用更新后的原型重新计算 logits，从而使分类器在分布式训练下保持一致性
def maybe_update_ema_prototypes_and_recompute_logits(model, outputs, target):
    # For EMA prototype classifier:
    # 1. gather raw/reg proto features from all ranks;
    # 2. update shared prototypes;
    # 3. recompute raw/reg/fused logits with updated prototypes.
    if "cls_logits" not in outputs:
        raise KeyError(f"Model output must contain 'cls_logits', got keys={list(outputs.keys())}")

    classifier = _get_classifier(model)
    if classifier is None:
        return outputs["cls_logits"]

    prototype_mode = getattr(classifier, "prototype_mode", None)
    if prototype_mode != "ema":
        return outputs["cls_logits"]

    branch_items = []
    if "raw_proto_feature" in outputs:
        branch_items.append(("raw", outputs["raw_proto_feature"]))

    if "reg_proto_feature" in outputs:
        branch_items.append(("reg", outputs["reg_proto_feature"]))
    elif "registered_proto_feature" in outputs:
        branch_items.append(("reg", outputs["registered_proto_feature"]))

    if len(branch_items) == 0:
        if "proto_feature" not in outputs:
            raise KeyError(
                "EMA prototype classifier needs proto_feature/raw_proto_feature/reg_proto_feature. "
                f"Got keys={list(outputs.keys())}"
            )
        branch_items.append(("main", outputs["proto_feature"]))

    z_cat = torch.cat([item[1].detach() for item in branch_items], dim=0)
    target_cat = torch.cat(
        [target.detach().view(-1).long() for _ in branch_items],
        dim=0,
    )

    z_all = all_gather_first_dim_tensor(z_cat)
    target_all = all_gather_first_dim_tensor(target_cat)

    with torch.no_grad():
        classifier.update_prototypes(z_all, target_all)

    proto = F.normalize(classifier.prototypes, dim=1)
    temperature = float(classifier.temperature)

    for branch_name, z in branch_items:
        logits = torch.matmul(z, proto.t()) / temperature

        if branch_name == "raw":
            outputs["raw_cls_logits"] = logits
        elif branch_name == "reg":
            outputs["reg_cls_logits"] = logits
            outputs["registered_cls_logits"] = logits
        else:
            outputs["cls_logits"] = logits

    if hasattr(classifier, "fuse_logits"):
        outputs["cls_logits"] = classifier.fuse_logits(outputs)
    elif "raw_cls_logits" in outputs and "reg_cls_logits" in outputs:
        outputs["cls_logits"] = 0.5 * (outputs["raw_cls_logits"] + outputs["reg_cls_logits"])

    return outputs["cls_logits"]

#计算分类损失
#logits：指的神经网络输出层的每个类别的概率分布
def compute_classification_loss(outputs, target, cfg):
    # Default: when both branches exist, optimize raw and registered branch CE equally.
    # Optional config:
    # trainer:
    #   cls_loss:
    #     raw_weight: 1.0
    #     reg_weight: 1.0
    #     fused_weight: 0.0
    loss_cfg = cfg.trainer.get("cls_loss", {})
    raw_weight = float(loss_cfg.get("raw_weight", 1.0))
    reg_weight = float(loss_cfg.get("reg_weight", 1.0))
    fused_weight = float(loss_cfg.get("fused_weight", 0.0))

    terms = []
    log_info = {}

    if "raw_cls_logits" in outputs and raw_weight > 0.0:
        raw_loss = F.cross_entropy(outputs["raw_cls_logits"], target)
        terms.append((raw_weight, raw_loss))
        log_info["loss_raw"] = raw_loss.detach()

    reg_logits = outputs.get("reg_cls_logits", outputs.get("registered_cls_logits", None))
    if reg_logits is not None and reg_weight > 0.0:
        reg_loss = F.cross_entropy(reg_logits, target)
        terms.append((reg_weight, reg_loss))
        log_info["loss_reg"] = reg_loss.detach()

    if fused_weight > 0.0:
        fused_loss = F.cross_entropy(outputs["cls_logits"], target)
        terms.append((fused_weight, fused_loss))
        log_info["loss_fused"] = fused_loss.detach()

    if len(terms) == 0:
        main_loss = F.cross_entropy(outputs["cls_logits"], target)
        log_info["loss_main"] = main_loss.detach()
        return main_loss, log_info

    weight_sum = sum(w for w, _ in terms)
    loss = sum(w * item for w, item in terms) / max(weight_sum, 1e-12)
    return loss, log_info

#计算分类任务的评估指标，包括整体准确率、平均类别准确率和宏平均 F1
def compute_cls_metrics(pred, target, num_classes):
    pred = pred.cpu().long()
    target = target.cpu().long()

    total = int(target.numel())
    correct = int((pred == target).sum().item())

    acc1 = correct / max(total, 1)

    conf_mat = torch.zeros(
        num_classes,
        num_classes,
        dtype=torch.long,
    )

    for t, p in zip(target.tolist(), pred.tolist()):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            conf_mat[t, p] += 1

    tp = conf_mat.diag().float()
    support = conf_mat.sum(dim=1).float()
    pred_count = conf_mat.sum(dim=0).float()

    valid_class = support > 0

    recall = torch.zeros(num_classes)
    precision = torch.zeros(num_classes)

    recall[valid_class] = tp[valid_class] / support[valid_class].clamp_min(1.0)

    valid_pred = pred_count > 0
    precision[valid_pred] = tp[valid_pred] / pred_count[valid_pred].clamp_min(1.0)

    f1 = torch.zeros(num_classes)
    denom = precision + recall
    valid_f1 = denom > 0
    f1[valid_f1] = 2.0 * precision[valid_f1] * recall[valid_f1] / denom[valid_f1]

    if valid_class.any():
        mean_class_acc = recall[valid_class].mean().item()
        macro_f1 = f1[valid_class].mean().item()
    else:
        mean_class_acc = 0.0
        macro_f1 = 0.0

    return {
        "acc1": acc1,
        "mean_class_acc": mean_class_acc,
        "macro_f1": macro_f1,
    }

#冻结模型指定的层
def freeze_selected_layers(model, frozen_layers):
    for layer in frozen_layers:
        module = getattr(model.module, layer)
        module.eval()

        for param in module.parameters():
            param.requires_grad = False

#把模型预测的类别索引（整数）转换为类别名称字符串列表
def pred_to_cls_names(pred, class_names):
    pred_cpu = pred.detach().cpu().long().tolist()
    names = []
    for item in pred_cpu:
        idx = int(item)
        if idx < 0 or idx >= len(class_names):
            raise RuntimeError(f"Predicted class index {idx} out of range [0,{len(class_names)-1}]")
        names.append(class_names[idx])
    return names

#stage1.先不做配准，得到一个初始类别预测
#stage2；再根据这个预测的类别，进行有针对性的配准，提取更对齐的特征
#last：最后可选择用第一阶段或第二阶段的分类结果作为最终指标。
@torch.no_grad()
def forward_two_stage_for_inference(model, batch, class_names, cfg):
    # Stage 1: no registration. Model1 will expose raw features as xyz_features.
    batch_stage1 = dict(batch) #复制batch
    batch_stage1["is_train"] = False #推理模式
    batch_stage1["force_no_registration"] = True #强制不配准，使用原始数据
    outputs_stage1 = model(batch_stage1)    #推理得到输出的分类结果

    if "cls_logits" not in outputs_stage1:  #如果没有分类logits，返回错误
        raise KeyError(
            f"Model output must contain 'cls_logits', got keys={list(outputs_stage1.keys())}"
        )

    logits_stage1 = outputs_stage1["cls_logits"]

    #获取评估参数
    evaluator_cfg = cfg.get("evaluator", {})
    use_two_stage = bool(evaluator_cfg.get("two_stage_registration", True))#默认开启两阶段的推理

    #非两阶段，直接返回stage1的分类结果
    if not use_two_stage:
        outputs_stage1["metric_cls_logits"] = logits_stage1
        return outputs_stage1, logits_stage1

    #获取第一阶段的预测类别名称索引
    pred_stage1 = logits_stage1.argmax(dim=1)
    #索引转类别名
    pred_names = pred_to_cls_names(pred_stage1, class_names)

    # Stage 2: register by predicted object class, then re-extract registered tokens.
    batch_stage2 = dict(batch)
    batch_stage2["pred_clsname"] = pred_names
    batch_stage2["registration_clsname"] = pred_names
    batch_stage2["force_no_registration"] = False
    batch_stage2["is_train"] = False

    #stage2 推理
    outputs_stage2 = model(batch_stage2)

    #保存第一阶段的信息
    outputs_stage2["stage1_cls_logits"] = logits_stage1
    outputs_stage2["stage1_pred"] = pred_stage1
    outputs_stage2["stage1_pred_clsname"] = pred_names

    #选择最终用于评估的 logits，默认第一阶段
    metric_logits_mode = str(evaluator_cfg.get("metric_logits", "stage1")).lower()
    if metric_logits_mode in ["stage2", "registered", "reg"]:
        metric_logits = outputs_stage2["cls_logits"]
    else:
        # Default: the category decision is the first raw prediction.
        metric_logits = logits_stage1

    outputs_stage2["metric_cls_logits"] = metric_logits
    return outputs_stage2, metric_logits


def train_one_epoch(
    train_loader, #数据加载器
    model, #模型
    optimizer,  #优化器
    lr_scheduler, #学习率设置
    epoch,  #当前epoch
    start_iter, #全局的起始步数
    tb_logger,  #Tensor Board
    logger,
    class_to_idx, #类别到索引的映射
    frozen_layers, #需要冻结的层
    task_id=None,   #多任务中的任务id
):
    #记录最近若干步的平均 batch 时间、数据加载时间、损失、准确率。
    batch_time = AverageMeter(config.trainer.print_freq_step)
    data_time = AverageMeter(config.trainer.print_freq_step)
    losses = AverageMeter(config.trainer.print_freq_step)
    accs = AverageMeter(config.trainer.print_freq_step)

    model.train() #训练模式（启用 BatchNorm 更新、Dropout 等
    freeze_selected_layers(model, frozen_layers)

    #获取当前进程的 rank 和总进程数
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    end = time.time()

    #遍历一个epoch的所有batch
    for i, batch in enumerate(train_loader):
        batch["task_id"] = task_id #多任务/增量学习场景下告知模型当前任务
        batch["is_train"] = True
        batch["use_gt_registration"] = True #训练时使用真实类别做配准

        curr_step = start_iter + i
        data_time.update(time.time() - end)

        outputs = model(batch)

        if "cls_logits" not in outputs:
            raise KeyError(
                f"Model output must contain 'cls_logits', got keys={list(outputs.keys())}"
            )

        logits = outputs["cls_logits"]
        target = get_cls_targets(batch, class_to_idx, logits.device)

        logits = maybe_update_ema_prototypes_and_recompute_logits(
            model=model,
            outputs=outputs,
            target=target,
        )

        loss, loss_info = compute_classification_loss(outputs, target, config)

        #计算当前 batch 的平均准确率
        pred = logits.argmax(dim=1)
        acc = (pred == target).float().mean()

        #分布式聚合损失和准确率

        #这里 acc 是每个进程 batch 内的平均，各进程 batch 大小可能不同（如最后一个 batch 不满），严格来说应该用加权平均，但通常各进程样本数接近，差别不大
        reduced_loss = loss.detach().clone()
        reduced_acc = acc.detach().clone()

        dist.all_reduce(reduced_loss)
        dist.all_reduce(reduced_acc)

        reduced_loss = reduced_loss / world_size
        reduced_acc = reduced_acc / world_size


        # losses.update(reduced_loss.item(), target.size(0))
        # accs.update(reduced_acc.item(), target.size(0))
        #不带权重，简单平均最近 print_freq_step 步的值
        losses.update(reduced_loss.item())
        accs.update(reduced_acc.item())

        #反向传播
        #清空梯度 → 反向传播 → 可选梯度裁剪 → 更新参数
        optimizer.zero_grad()
        loss.backward()

        if config.trainer.get("clip_max_norm", None):
            max_norm = config.trainer.clip_max_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        batch_time.update(time.time() - end)

        #日志打印与 TensorBoard

        if (curr_step + 1) % config.trainer.print_freq_step == 0 and rank == 0:
            current_lr = optimizer.param_groups[0]["lr"]

            tb_logger.add_scalar("loss_train", losses.avg, curr_step + 1)
            tb_logger.add_scalar("acc1_train", accs.avg, curr_step + 1)
            tb_logger.add_scalar("lr", current_lr, curr_step + 1)

            for key, value in loss_info.items():
                tb_logger.add_scalar(key, float(value.item()), curr_step + 1)

            tb_logger.flush()

            logger.info(
                "Epoch: [{0}/{1}]\t"
                "Iter: [{2}/{3}]\t"
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "Data {data_time.val:.2f} ({data_time.avg:.2f})\t"
                "Loss {loss.val:.5f} ({loss.avg:.5f})\t"
                "Acc@1 {acc.val:.4f} ({acc.avg:.4f})\t"
                "LR {lr:.6f}".format(
                    epoch + 1,
                    config.trainer.max_epoch,
                    curr_step + 1,
                    len(train_loader) * config.trainer.max_epoch,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    acc=accs,
                    lr=current_lr,
                )
            )

        end = time.time()


@torch.no_grad()
def validate(
    val_loader,
    model,
    logger,
    class_to_idx,
    num_classes,
    class_names,
    task_id=None,
):
    model.eval() #评估模式

    #进程和设备
    rank = dist.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())

    #累计本进程的损失总和与样本数
    local_loss_sum = 0.0
    local_num = 0

    #收集本进程所有 batch 的预测和标签，最后拼接
    pred_list = []
    target_list = []

    end = time.time()
    batch_time = AverageMeter(0)

    for i, batch in enumerate(val_loader):
        batch["task_id"] = task_id

        #两步推理
        outputs, logits = forward_two_stage_for_inference(
            model=model,
            batch=batch,
            class_names=class_names,
            cfg=config,
        )

        if "cls_logits" not in outputs:
            raise KeyError(
                f"Model output must contain 'cls_logits', got keys={list(outputs.keys())}"
            )

        #GT
        target = get_cls_targets(batch, class_to_idx, logits.device)

        #计算损失
        loss = F.cross_entropy(logits, target, reduction="mean")
        pred = logits.argmax(dim=1)

        #累计损失和样本数
        local_loss_sum += float(loss.item()) * target.numel()
        local_num += int(target.numel())

        pred_list.append(pred.detach())
        target_list.append(target.detach())

        #统计时间和日志
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % config.trainer.print_freq_step == 0 and rank == 0:
            logger.info(
                "Test: [{0}/{1}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})".format(
                    i + 1,
                    len(val_loader),
                    batch_time=batch_time,
                )
            )

    #拼接本进程的预测和标签
    if len(pred_list) > 0:
        local_pred = torch.cat(pred_list, dim=0).to(device)
        local_target = torch.cat(target_list, dim=0).to(device)
    else:
        local_pred = torch.empty(0, dtype=torch.long, device=device)
        local_target = torch.empty(0, dtype=torch.long, device=device)

    #跨进程收集预测和标签
    all_pred = all_gather_1d_tensor(local_pred)
    all_target = all_gather_1d_tensor(local_target)

    #跨进程汇总损失
    loss_info = torch.tensor(
        [local_loss_sum, local_num],
        dtype=torch.float64,
        device=device,
    )


    dist.all_reduce(loss_info)

    final_loss = float(loss_info[0].item() / max(loss_info[1].item(), 1.0))

    #计算指标
    ret_metrics = {}

    if rank == 0:
        ret_metrics = compute_cls_metrics(
            pred=all_pred,
            target=all_target,
            num_classes=num_classes,
        )
        ret_metrics["loss"] = final_loss

        logger.info(
            " * Loss {loss:.5f}\t"
            "Acc@1 {acc1:.4f}\t"
            "MeanClassAcc {mean_class_acc:.4f}\t"
            "MacroF1 {macro_f1:.4f}".format(**ret_metrics)
        )

    #恢复为训练模式
    model.train()

    return ret_metrics


def main():
    global args, config, key_metric, best_metric

    args = parser.parse_args()

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    config.port = config.get("port", None)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)


    rank, world_size = setup_distributed(port=config.port)

    config.exp_path = os.path.dirname(args.config)#配置文件所在目录，作为实验根目录
    config.save_path = os.path.join(config.exp_path, config.saver.save_dir)#模型检查点保存目录
    config.log_path = os.path.join(config.exp_path, config.saver.log_dir)#日志和 TensorBoard 目录

    #只有第一个进程创建目录和日志文件，其他进程不创建
    if rank == 0:
        os.makedirs(config.save_path, exist_ok=True)
        os.makedirs(config.log_path, exist_ok=True)

        current_time = get_current_time()
        tb_logger = SummaryWriter(config.log_path + "/events_cls/" + current_time)
        logger = create_logger(
            "global_logger",
            config.log_path + "/cls_{}.log".format(current_time),
        )

        logger.info("args: {}".format(pprint.pformat(args)))
        logger.info("config: {}".format(pprint.pformat(config)))
    else:
        tb_logger = None
        logger = None

    #设置随机种子
    random_seed = config.get("random_seed", None)
    reproduce = config.get("reproduce", None)

    if random_seed:
        set_random_seed(random_seed, reproduce)

    class_names = build_class_names(config.dataset)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    num_classes = get_classifier_cls_num(config)

    if len(class_names) != num_classes:
        raise RuntimeError(
            f"len(class_names)={len(class_names)} but classifier.cls_num={num_classes}. "
            f"Please check dataset.class_names and net.classifier.kwargs.cls_num."
        )

    if rank == 0:
        logger.info("class_names: {}".format(class_names))
        logger.info("class_to_idx: {}".format(class_to_idx))

    model = ModelHelper(config.net)
    model.cuda()

    #模型构建与DDP封装
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    layers = []
    for module in config.net:
        layers.append(module["name"])

    frozen_layers = config.get("frozen_layers", [])
    active_layers = list(set(layers) ^ set(frozen_layers))

    if rank == 0:
        logger.info("layers: {}".format(layers))
        logger.info("frozen layers: {}".format(frozen_layers))
        logger.info("active layers: {}".format(active_layers))

    #仅把 active_layers 的参数传给优化器，冻结层的参数不参与优化
    parameters = [
        {"params": getattr(model.module, layer).parameters()}
        for layer in active_layers
    ]

    #构建优化器和学习率调度器
    optimizer = get_optimizer(parameters, config.trainer.optimizer)
    lr_scheduler = get_scheduler(optimizer, config.trainer.lr_scheduler)

    key_metric = config.evaluator.get("key_metric", "acc1")#选择关键指标，默认acc1
    best_metric = 0.0
    last_epoch = 0 #恢复训练的epoch数，默认0

    #是否自动恢复训练
    auto_resume = config.saver.get("auto_resume", True)
    resume_model = config.saver.get("resume_model", None)
    load_path = config.saver.get("load_path", None)

    #如果 resume_model 是相对路径，则将其转换为绝对路径
    if resume_model and not resume_model.startswith("/"):
        resume_model = os.path.join(config.exp_path, resume_model)

    latest_model = os.path.join(config.save_path, "ckpt.pth.tar")

    #如果 auto_resume 为 True 且最新模型文件存在，则将 resume_model 设置为最新模型路径
    if auto_resume and os.path.exists(latest_model):
        resume_model = latest_model

    #恢复训练或加载模型
    if resume_model:
        best_metric, last_epoch = load_state(
            resume_model,
            model,
            optimizer=optimizer,
        )
    elif load_path:
        if not load_path.startswith("/"):
            load_path = os.path.join(config.exp_path, load_path)

        if os.path.exists(load_path):
            load_state(load_path, model)
        elif rank == 0:
            logger.info(f"Skip load_path because file does not exist: {load_path}")

    #构建训练和验证数据加载器
    train_loaders, val_loaders = build_dataloader(
        config.dataset,
        distributed=False,
    )

    #如果只进行评估，则不进行训练，直接在验证集上评估模型性能
    if args.evaluate:
        for task_id, val_loader_task in enumerate(val_loaders):
            validate(
                val_loader=val_loader_task,
                model=model,
                logger=logger,
                class_to_idx=class_to_idx,
                num_classes=num_classes,
                class_names=class_names,
                task_id=task_id,
            )
        return

    #训练循环，针对每个任务分别训练和验证
    for task_id, (train_loader_task, val_loader_task) in enumerate(
        zip(train_loaders, val_loaders)
    ):
        if rank == 0:
            logger.info(f"Training classification task {task_id}")

        best_metric = 0.0

        for epoch in range(last_epoch, config.trainer.max_epoch):
            train_one_epoch(
                train_loader=train_loader_task,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                epoch=epoch,
                start_iter=epoch * len(train_loader_task),
                tb_logger=tb_logger,
                logger=logger,
                class_to_idx=class_to_idx,
                frozen_layers=frozen_layers,
                task_id=task_id,
            )

            lr_scheduler.step(epoch)

            #在每个验证周期结束后进行验证，并保存最佳模型
            if (epoch + 1) % config.trainer.val_freq_epoch == 0:
                ret_metrics = validate(
                    val_loader=val_loader_task,
                    model=model,
                    logger=logger,
                    class_to_idx=class_to_idx,
                    num_classes=num_classes,
                    class_names=class_names,
                    task_id=task_id,
                )

                if rank == 0:
                    ret_key_metric = ret_metrics[key_metric]
                    is_best = ret_key_metric >= best_metric
                    best_metric = max(ret_key_metric, best_metric)

                    save_checkpoint(
                        {
                            "epoch": epoch + 1,
                            "task_id": task_id,
                            "arch": config.net,
                            "state_dict": model.state_dict(),
                            "best_metric": best_metric,
                            "optimizer": optimizer.state_dict(),
                            "class_names": class_names,
                            "class_to_idx": class_to_idx,
                        },
                        is_best,
                        config,
                    )


if __name__ == "__main__":
    main()
