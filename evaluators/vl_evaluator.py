import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix
from dassl.evaluation.build import EVALUATOR_REGISTRY
from dassl.evaluation.evaluator import Classification

from tools.metrics import ECE, MCE, AdaptiveECE, PIECE
from tools.plot import plot_reliability_diagram

@EVALUATOR_REGISTRY.register()
class VLClassification(Classification):
    """Evaluator for Vision-Language models."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_score = []
        self._y_true = []
        self._y_pred = []
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_score = []
        self._y_true = []
        self._y_pred = []
        self._text_features = []
        self._image_features = []
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def process(self, mo, gt, image_features, text_features):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        # pred = mo.max(1)[1]
        # matches = pred.eq(gt).float()
        # self._correct += int(matches.sum().item())
        # self._total += gt.shape[0]
        self._y_score.extend(mo.data.cpu().numpy().tolist())
        self._y_true.extend(gt.data.cpu().numpy().tolist())
        # self._y_pred.extend(pred.data.cpu().numpy().tolist())
        self._text_features.extend(text_features.data.cpu().numpy().tolist()) # record text feature and image features in CLIP
        self._image_features.extend(image_features.data.cpu().numpy().tolist())

        # if self._per_class_res is not None:
        #     for i, label in enumerate(gt):
        #         label = label.item()
        #         matches_i = int(matches[i].item())
        #         self._per_class_res[label].append(matches_i)

    def evaluate(self, probs, labels, text_proximity):

        results = OrderedDict()
        ece_bin = self.cfg.CALIBRATION.METRICS.ECE_BINS
        piece_bin = self.cfg.CALIBRATION.METRICS.PIECE_BINS

        total = len(labels)

        # make the prediction
        preds = np.argmax(probs, axis=1)

        correct = np.sum(preds == labels)

        accuracy = 100.0 * correct / total

        error = 100.0 - accuracy

        macro_f1 = 100.0 * f1_score(
            labels,
            preds,
            average="macro",
            labels=np.unique(labels)
        )

        confs = probs[range(probs.shape[0]), preds]
        avg_conf = np.mean(confs)

        ece = 100.0 * ECE(confs, preds, labels, ece_bin)

        mce = 100.0 * MCE(confs, preds, labels, ece_bin)

        ace = 100.0 * AdaptiveECE(confs, preds, labels, ece_bin)

        piece = 100.0 * PIECE(confs, text_proximity, preds, labels, piece_bin, ece_bin)

        # The first value will be returned by trainer.test()
        results["accuracy"] = accuracy
        results["error_rate"] = error
        results["macro_f1"] = macro_f1
        results["confidence"] = avg_conf
        results["ece"] = ece
        results["mce"] = mce
        results["ace"] = ace
        results["piece"] = piece

        print(
            "=> result\n"
            f"* total: {total:,}\n"
            f"* correct: {correct:,}\n"
            f"* accuracy: {accuracy:.2f}%\n"
            f"* error: {error:.2f}%\n"
            f"* macro_f1: {macro_f1:.2f}%\n"
            f"* confidence: {avg_conf:.2f}%\n"
            f"* ece: {ece:.2f}%\n"
            f"* mce: {mce:.2f}%\n"
            f"* ace: {ace:.2f}%\n"
            f"* piece: {piece:.2f}%"
        )
        
        # plot ece
        base_dir = self.cfg.OUTPUT_DIR
        base_name = self.cfg.DATASET.NAME + '_' + self.cfg.TRAINER.NAME

        if self.cfg.CALIBRATION.SCALING.IF_SCALING:
            base_name = base_name + '_' + str(self.cfg.CALIBRATION.SCALING.MODE)

        if self.cfg.CALIBRATION.BIN.BIN_CALIBRATOR_NAME:
            base_name = base_name + '_' + str(self.cfg.CALIBRATION.BIN.BIN_CALIBRATOR_NAME)

        if self.cfg.CALIBRATION.DAC.IF_DAC:
            base_name = base_name + '_dac'

        if self.cfg.CALIBRATION.PROCAL.IF_PROCAL:
            base_name = base_name + '_procal'

        base_name  = base_name + '_ece.png'
        plot_dir = osp.join(base_dir, base_name)

        plot_reliability_diagram(preds, confs, labels, ece_bin, None, plot_dir)

        # if self._per_class_res is not None:
        #     labels = list(self._per_class_res.keys())
        #     labels.sort()

        #     print("=> per-class result")
        #     accs = []

        #     for label in labels:
        #         classname = self._lab2cname[label]
        #         res = self._per_class_res[label]
        #         correct = sum(res)
        #         total = len(res)
        #         acc = 100.0 * correct / total
        #         accs.append(acc)
        #         print(
        #             f"* class: {label} ({classname})\t"
        #             f"total: {total:,}\t"
        #             f"correct: {correct:,}\t"
        #             f"acc: {acc:.1f}%"
        #         )
        #     mean_acc = np.mean(accs)
        #     print(f"* average: {mean_acc:.1f}%")

        #     results["perclass_accuracy"] = mean_acc

        # if self.cfg.TEST.COMPUTE_CMAT:
        #     cmat = confusion_matrix(
        #         self._y_true, self._y_pred, normalize="true"
        #     )
        #     save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
        #     torch.save(cmat, save_path)
        #     print(f"Confusion matrix is saved to {save_path}")

        return results
