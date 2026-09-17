import glob
import logging
import os
from sklearn.neighbors import NearestNeighbors
import numpy as np
import tabulate
import torch
import torch.nn.functional as F
from datasets.metrics import AUPRO
from sklearn import metrics


def dump(save_dir, outputs):
    filenames = outputs["filename"]
    batch_size = len(filenames)
    preds = outputs["pred"].squeeze(1).cpu().numpy()  # B x 1 x H x W
    masks = outputs["mask"].cpu().numpy()  # B x 1 x H x W
    point_cloud = outputs["pointcloud"].cpu().numpy()  # B x 1 x H x W
    center_idx = outputs["center_idx"].cpu().numpy()
    clsnames = outputs["clsname"]
    labels = outputs["label"].cpu().numpy()
    cls_labels = outputs["cls_label"].cpu().numpy()
    cls_preds = torch.argmax(outputs["cls_pred"],dim=1).cpu().numpy()
    for i in range(batch_size):
        file_dir, filename = os.path.split(filenames[i])
        _, subname = os.path.split(file_dir)
        filename = "{}_{}_{}".format(clsnames[i], subname, filename)
        filename, _ = os.path.splitext(filename)
        save_file = os.path.join(save_dir, filename + ".npz")
        np.savez(
            save_file,
            filename=filenames[i],
            pred=preds[i],
            mask=masks[i],
            point_cloud=point_cloud[i],
            label=labels[i],
            center_idx=center_idx[i],
            clsname=clsnames[i],
            cls_label=cls_labels[i],
            cls_pred=cls_preds[i]
        )


def fill_missing_values(x_data,x_label,y_data, k=1):
    # 创建最近邻居模型
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(x_data)

    # 找到每个点的最近邻居
    distances, indices = nn.kneighbors(y_data)
    # print(distances.shape)
    # print(indices.shape)
    avg_values = np.mean(x_label[indices], axis=1)
    # print("avg_values.shape",avg_values.shape)
    return avg_values

def merge_together(save_dir):
    # 关键：排序，保证每次读取 npz 的顺序一致（可复现、也方便 debug）
    npz_file_list = sorted(glob.glob(os.path.join(save_dir, "*.npz")))

    fileinfos = []
    preds = []
    masks = []
    labels = []
    image_max_score = []
    cls_label = []
    cls_pred = []
    points = []

    for npz_file in npz_file_list:
        npz = np.load(npz_file)
        fileinfos.append(
            {
                "filename": str(npz["filename"]),
                "clsname": str(npz["clsname"]),
            }
        )
        point_cloud = npz["point_cloud"]
        points.append(point_cloud)
        sample_idx = npz["center_idx"]
        mask_idx = sample_idx.squeeze().astype(np.int64)

        xyz_sampled = point_cloud[mask_idx, :]
        pred = npz["pred"]

        preds_all = fill_missing_values(xyz_sampled, pred, point_cloud)
        preds_all = (
            F.avg_pool1d(
                torch.from_numpy(preds_all).unsqueeze(0),
                kernel_size=511,
                padding=511 // 2,
                stride=1,
            )
            .squeeze(0)
            .numpy()
        )

        preds.append(preds_all)
        masks.append(npz["mask"])
        labels.append(npz["label"])
        image_max_score.append(preds_all.max())
        cls_label.append(npz["cls_label"])
        cls_pred.append(npz["cls_pred"])

    return fileinfos, labels, image_max_score, masks, preds, cls_label, points, cls_pred




class Report:
    def __init__(self, heads=None):
        if heads:
            self.heads = list(map(str, heads))
        else:
            self.heads = ()
        self.records = []

    def add_one_record(self, record):
        if self.heads:
            if len(record) != len(self.heads):
                raise ValueError(
                    f"Record's length ({len(record)}) should be equal to head's length ({len(self.heads)})."
                )
        self.records.append(record)

    def __str__(self):
        return tabulate.tabulate(
            self.records,
            self.heads,
            tablefmt="pipe",
            numalign="center",
            stralign="center",
        )


class EvalDataMeta:
    def __init__(self, preds, masks):
        self.preds = preds  # N x H x W
        self.masks = masks  # N x H x W


class EvalImage:
    def __init__(self, data_meta, **kwargs):
        self.preds = self.encode_pred(data_meta.preds, **kwargs)
        self.masks = self.encode_mask(data_meta.masks)
        self.preds_good = sorted(self.preds[self.masks == 0], reverse=True)
        self.preds_defe = sorted(self.preds[self.masks == 1], reverse=True)
        self.num_good = len(self.preds_good)
        self.num_defe = len(self.preds_defe)

    @staticmethod
    def encode_pred(preds):
        raise NotImplementedError

    def encode_mask(self, masks):
        N, _, _ = masks.shape
        masks = (masks.reshape(N, -1).sum(axis=1) != 0).astype(np.int)  # (N, )
        return masks

    def eval_auc(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.masks, self.preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        if auc < 0.5:
            auc = 1 - auc
        return auc


class EvalImageMean(EvalImage):
    @staticmethod
    def encode_pred(preds):
        N, _, _ = preds.shape
        return preds.reshape(N, -1).mean(axis=1)  # (N, )


class EvalImageStd(EvalImage):
    @staticmethod
    def encode_pred(preds):
        N, _, _ = preds.shape
        return preds.reshape(N, -1).std(axis=1)  # (N, )


class EvalImageMax(EvalImage):
    @staticmethod
    def encode_pred(preds, avgpool_size):
        N, _, _ = preds.shape
        preds = torch.tensor(preds[:, None, ...]).cuda()  # N x 1 x H x W
        preds = (
            F.avg_pool2d(preds, avgpool_size, stride=1).cpu().numpy()
        )  # N x 1 x H x W
        return preds.reshape(N, -1).max(axis=1)  # (N, )


class EvalPerPixelAUC:
    def __init__(self, data_meta):
        self.preds = np.concatenate(
            [pred.flatten() for pred in data_meta.preds], axis=0
        )
        self.masks = np.concatenate(
            [mask.flatten() for mask in data_meta.masks], axis=0
        )
        self.masks[self.masks > 0] = 1

    def eval_auc(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.masks, self.preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        if auc < 0.5:
            auc = 1 - auc
        return auc
def min_max_normalize(data):
    """
    对给定的 NumPy 数组进行 Min-Max 归一化。

    参数:
        data (np.ndarray): 输入的 NumPy 数组。

    返回:
        np.ndarray: 归一化后的数组。
    """
    # 计算最小值和最大值
    min_val = np.min(data)
    max_val = np.max(data)
    
    # 进行归一化
    normalized_data = (data - min_val) / (max_val - min_val)
    
    return normalized_data

eval_lookup_table = {
    "mean": EvalImageMean,
    "std": EvalImageStd,
    "max": EvalImageMax,
    "pixel": EvalPerPixelAUC,
}

def get_auc(label,pred):
    auc = metrics.roc_auc_score(np.asarray(label),min_max_normalize(np.asarray(pred)))
    # if auc < 0.5:
    #     auc = 1-auc
    return auc

def get_aupr(label, pred, use_ap: bool = True):
    """
    PR-AUC (AUPR).
    - use_ap=True: average_precision_score（异常检测里最常用的 AUPR/AP）
    - use_ap=False: precision_recall_curve + auc(recall, precision) 的梯形积分
    """
    y_true = (np.asarray(label) > 0).astype(np.uint8)
    y_score = min_max_normalize(np.asarray(pred, dtype=np.float32))
    if y_true.max() == y_true.min():
        return float("nan")
    if use_ap:
        return float(metrics.average_precision_score(y_true, y_score))
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
    return float(metrics.auc(recall, precision))

def get_pro(label,pred):
    point_AUPRO = AUPRO().cuda()
    point_AUPRO.update(torch.from_numpy(min_max_normalize(np.asarray(pred))).cuda,torch.from_numpy(np.asarray(label)).cuda())

def performances(fileinfos, labels, image_max_score, masks, preds, cls_labels, cls_preds):
    ret_metrics = {}

    # 关键：sorted + set -> 稳定顺序（默认按字母序）
    clsnames = sorted({fileinfo["clsname"] for fileinfo in fileinfos})

    for clsname in clsnames:
        labels_l = []
        image_max_cls = []
        mask_l = []
        pred_l = []
        cls_label_l = []
        cls_pred_l = []

        for fileinfo, label, max_score, mask, pred, cls_label, cls_pred in zip(
            fileinfos, labels, image_max_score, masks, preds, cls_labels, cls_preds
        ):
            if fileinfo["clsname"] == clsname:
                labels_l.append(float(label))
                image_max_cls.append(max_score)
                mask_l.append(mask)
                pred_l.append(pred)
                cls_label_l.append(cls_label)
                cls_pred_l.append(cls_pred)

        preds_l = np.concatenate(pred_l)   # B x N
        masks_l = np.concatenate(mask_l)   # B x N
        cls_label_l = np.array(cls_label_l)
        cls_pred_l = np.array(cls_pred_l)

        # ret_metrics[f"{clsname}_pixel-AUROC_auc"] = get_auc(masks_l, preds_l)
        # ret_metrics[f"{clsname}_obj-AUROC_auc"] = get_auc(labels_l, image_max_cls)
        # ret_metrics[f"{clsname}_cls-ACC_auc"] = np.mean(cls_label_l == cls_pred_l)
        # pixel-level
        ret_metrics[f"{clsname}_pixel-AUROC_auc"] = get_auc(masks_l, preds_l)
        ret_metrics[f"{clsname}_pixel-AUPR_auc"]  = get_aupr(masks_l, preds_l)

        # object-level
        ret_metrics[f"{clsname}_obj-AUROC_auc"]   = get_auc(labels_l, image_max_cls)
        ret_metrics[f"{clsname}_obj-AUPR_auc"]    = get_aupr(labels_l, image_max_cls)

        ret_metrics[f"{clsname}_cls-ACC_auc"] = np.mean(cls_label_l == cls_pred_l)

    # mean 指标（顺序固定）
    # for metric in ["obj-AUROC", "pixel-AUROC", "cls-ACC"]:
    for metric in ["obj-AUROC", "obj-AUPR", "pixel-AUROC", "pixel-AUPR", "cls-ACC"]:
        evalvalues = [ret_metrics[f"{clsname}_{metric}_auc"] for clsname in clsnames]
        ret_metrics[f"mean_{metric}_auc"] = float(np.mean(np.array(evalvalues)))

    return ret_metrics



def log_metrics(ret_metrics, config):
    """
    打印表格时保证行/列顺序稳定。

    支持（可选）在 YAML 里配置：
      evaluator:
        metrics:
          auc: true
          metric_order: ["obj-AUROC", "cls-ACC", "pixel-AUROC"]
          class_order: ["cube", "light", "spring_pad", ...]   # 想按持续学习任务顺序就填这个
    """
    logger = logging.getLogger("global_logger")

    if not config.get("auc", None):
        return

    # 只取形如: "<clsname>_<metric>_auc" 的 key
    auc_keys = [k for k in ret_metrics.keys() if k.endswith("_auc")]
    if len(auc_keys) == 0:
        logger.info("No *_auc metrics found to log.")
        return

    # ----------------------------
    # 列顺序（metrics）
    # ----------------------------
    all_evalnames = sorted({k.rsplit("_", 2)[1] for k in auc_keys})

    # default_metric_order = ["obj-AUROC", "cls-ACC", "pixel-AUROC"]
    default_metric_order = ["obj-AUROC", "obj-AUPR", "cls-ACC", "pixel-AUROC", "pixel-AUPR"]
    metric_order = config.get("metric_order", default_metric_order)

    if isinstance(metric_order, (list, tuple)) and len(metric_order) > 0:
        evalnames = [m for m in metric_order if m in all_evalnames]
        evalnames += sorted([m for m in all_evalnames if m not in metric_order])
    else:
        evalnames = all_evalnames

    # ----------------------------
    # 行顺序（classes）
    # ----------------------------
    all_clsnames = {k.rsplit("_", 2)[0] for k in auc_keys}
    mean_present = "mean" in all_clsnames
    all_clsnames.discard("mean")

    class_order = config.get("class_order", None)
    if isinstance(class_order, (list, tuple)) and len(class_order) > 0:
        # 先按你给的持续学习顺序排
        ordered = [c for c in class_order if c in all_clsnames]
        # 剩下的再按字母序补齐（防止漏配）
        remaining = sorted([c for c in all_clsnames if c not in class_order])
        clsnames = ordered + remaining
    else:
        # 默认稳定字母序
        clsnames = sorted(all_clsnames)

    if mean_present:
        clsnames.append("mean")

    # ----------------------------
    # 打印表格
    # ----------------------------
    record = Report(["clsname"] + evalnames)
    for clsname in clsnames:
        clsvalues = [
            ret_metrics.get(f"{clsname}_{evalname}_auc", float("nan"))
            for evalname in evalnames
        ]
        record.add_one_record([clsname] + clsvalues)

    logger.info(f"\n{record}")
