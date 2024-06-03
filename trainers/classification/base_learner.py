import time
import numpy as np
import os.path as osp
import datetime
from collections import OrderedDict
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from scipy.special import softmax


from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)

from copy import deepcopy
from clip import clip

from tools.zsclip_encoder import build_zsclip, build_clip_templates
from tools.plot import plot_reliability_diagram
from trainers.calibration.proximity import mkdir_if_missing, get_knn_dists, get_val_image_knn_dists
from trainers.calibration.vl_calibrator import VLCalibration


@TRAINER_REGISTRY.register()
class VLBaseLearner(TrainerX):
    """A base trainer for vision language tuning and calibration"""


    def after_train(self):
        print("Finish training")

        print("Testing")
        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")
            self.test()

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()    



    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        
        if not self.cfg.CALIBRATION.SCALING.IF_SCALING:  # few shot, not calibration
            if self.cfg.TRAINER.NAME == 'ProDA':
                self.model.set_classifier()

        # prepare the dataset
        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader 

        print(f"Evaluate on the *{split}* set")

        # calcualte the output
        image_features_test = [] 
        text_features_test = []
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output, image_features_test_i, text_features_test_i = self.model_inference(input)
            self.evaluator.process(output, label, image_features_test_i, text_features_test_i)
            image_features_test.append(image_features_test_i.data.cpu())

        image_features_test = np.array(torch.cat(image_features_test))  
        text_features_test.append(text_features_test_i.data.cpu()) # only record once
        text_features_test = torch.cat(text_features_test, dim=0).numpy() 

        logits = np.array(self.evaluator._y_score) # logits
        # preds = np.array(self.evaluator._y_pred)
        labels = np.array(self.evaluator._y_true)
        # image_features_test = np.array(self.evaluator._image_features)
        # text_features_test = np.array(self.evaluator._text_features)


        # save info from val dataloader on base class using the tuned CLIP,  and use them to train the calibrator
        if self.cfg.DATASET.SUBSAMPLE_CLASSES == 'base':
            self.save_base_val_features()
            
        # get val features on base class using tuned model for further calculation
        val_feature_dir = osp.join('./temp/base_features', self.cfg.DATASET.NAME,  self.cfg.TRAINER.NAME, 'shots' + \
                                   str(self.cfg.DATASET.NUM_SHOTS), self.cfg.MODEL.BACKBONE.NAME, 'base', 'seed' + str(self.cfg.SEED), 'base_features.pt')
        val_dict = torch.load(val_feature_dir)
        

        #build the calibrator
        base_calibration_mode = self.cfg.CALIBRATION.BASE_CALIBRATION_MODE
        base_bin_calibrator_name = self.cfg.CALIBRATION.BIN.BIN_CALIBRATOR_NAME
        dac_flag = self.cfg.CALIBRATION.DAC.IF_DAC 
        procal_flag = self.cfg.CALIBRATION.PROCAL.IF_PROCAL
        val_dict = val_dict
        text_feature_dict = self.get_text_features()
        calibrator = VLCalibration(self.cfg, base_calibration_mode, base_bin_calibrator_name, dac_flag, procal_flag, val_dict, text_feature_dict) # build the calibrator use val dataset
        calibrator.fit() # calibrator initialization

        # get test set proximity
        base_val_image_features = val_dict['val_image_features']
        base_dists_dir =  osp.join('./temp/knndist', self.cfg.DATASET.NAME, self.cfg.TRAINER.NAME, 'shots' + str(self.cfg.DATASET.NUM_SHOTS), \
                             self.cfg.MODEL.BACKBONE.NAME, self.cfg.DATASET.SUBSAMPLE_CLASSES, 'seed' + str(self.cfg.SEED), 'nn' + str(self.cfg.CALIBRATION.PROCAL.IMAGE_K))   # text_knndists
        K = self.cfg.CALIBRATION.PROCAL.IMAGE_K

        dist_dir = osp.join(base_dists_dir, 'knndist.npy')   # save the test image distance for quick inference next time
        if osp.exists(dist_dir):
            print('load the knn distance from:', dist_dir)
            text_knndists = np.load(dist_dir)
        else:
            text_knndists = get_knn_dists(base_val_image_features, image_features_test, K)
            mkdir_if_missing(base_dists_dir)
            np.save(dist_dir, text_knndists)

        text_knndists = np.mean(text_knndists, axis=1) # use the average distance to K nn, TODO: need to be modified
        test_img_proximity = np.exp(-text_knndists) # knndist to proximity
        

        # confidence calibration
        probs = calibrator.predict(logits, test_img_proximity)

        # evaluate, log and plot the results
        results = self.evaluator.evaluate(probs, labels, test_img_proximity)


        for k, v in results.items():
            tag = f"{split}/{k}"
            # print(tag)
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]



    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]

        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        return input, label, domain

    def count_unique_labels(self, dataloader):
        unique_labels = set()

        for batch_idx, batch in enumerate(tqdm(dataloader)):
            input, label = self.parse_batch_test(batch)
            unique_labels.update(label.cpu().numpy().tolist())
        print(f"There are {len(unique_labels)} unique labels in the DataLoader.")


    @torch.no_grad()
    def save_base_val_features(self):

        # only save the feature when evaluating base class 
        base_dir = osp.join('./temp/base_features', self.cfg.DATASET.NAME,  self.cfg.TRAINER.NAME, \
                            'shots' + str(self.cfg.DATASET.NUM_SHOTS), self.cfg.MODEL.BACKBONE.NAME, self.cfg.DATASET.SUBSAMPLE_CLASSES, 'seed' + str(self.cfg.SEED))
        if not os.path.exists(base_dir): 
            os.makedirs(base_dir)
        save_dir = osp.join(base_dir, 'base_features.pt')

        # Check if the file already exists
        if os.path.exists(save_dir):
            print(f"File {save_dir} already exists. Skipping save operation.")
            return
        

        print("Saving base features from val dataset")
        self.set_model_mode("eval")
        

        if not self.cfg.CALIBRATION.SCALING.IF_SCALING:  # few shot, not calibration
            if self.cfg.TRAINER.NAME == 'ProDA':
                self.model.set_classifier()

        data_loader = self.val_loader # use val loader of base class
        # data_loader = self.train_loader_x

        labels = []
        image_feautures_val = []
        text_features_val = []
        logits_val = []
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output, image_features_val_i, text_features_val_i = self.model_inference(input)
            labels.append(label.data.cpu())
            logits_val.append(output.data.cpu())
            image_feautures_val.append(image_features_val_i.data.cpu())
        
        text_features_val.append(text_features_val_i.data.cpu())

        logits_val = torch.cat(logits_val, dim=0).numpy()
        labels = torch.cat(labels, dim=0).numpy()
        image_feautures_val = torch.cat(image_feautures_val, dim=0).numpy()
        text_features_val = torch.cat(text_features_val, dim=0).numpy()

        predicted_classes = np.argmax(logits_val, axis=1)
        correct_predictions = np.sum(predicted_classes == labels)
        accuracy = correct_predictions / len(labels)
        # print(f"Val Accuracy: {accuracy * 100:.2f}%")

        # save the image proximity
        val_image_knn_dists = get_val_image_knn_dists(image_feautures_val, self.cfg.CALIBRATION.PROCAL.IMAGE_K)

        # Store the info in a dictionary
        feature_dict = {
            "val_logits": logits_val,
            "val_image_features": image_feautures_val,
            'val_text_features': text_features_val,
            "val_labels": labels,
            "val_image_knn_dists": val_image_knn_dists

        }
        
        torch.save(feature_dict, save_dir)


    @torch.no_grad()
    def get_text_features(self,):

        # get base val feature using tuned model,
        val_feature_dir = osp.join('./temp/base_features', self.cfg.DATASET.NAME,  self.cfg.TRAINER.NAME, 'shots' + \
                                   str(self.cfg.DATASET.NUM_SHOTS), self.cfg.MODEL.BACKBONE.NAME,'base', 'seed' + str(self.cfg.SEED), 'base_features.pt')
        val_dict = torch.load(val_feature_dir)
        val_text_features = val_dict['val_text_features']
        val_image_knn_dists = val_dict['val_image_knn_dists']

        # get base val feature using zero shot model
        zs_base_feature_dir = osp.join('./temp/base_features', self.cfg.DATASET.NAME,  'ZeroshotCLIP', \
                                       'shots' + str(self.cfg.DATASET.NUM_SHOTS), self.cfg.MODEL.BACKBONE.NAME, 'base', 'seed1', 'base_features.pt')
        zs_base_dict = torch.load(zs_base_feature_dir)
        
        # 1. get the base text features from zero-shot model
        base_text_features_zs = zs_base_dict['val_text_features']


        # 2. get the current text features from zero-shot model
        zs_clip  =  build_zsclip(self.cfg.MODEL.BACKBONE.NAME) # get the base model
        zs_clip.cuda()
        classnames = self.dm.dataset.classnames
        temp = build_clip_templates(self.cfg.DATASET.NAME)
        prompts_ = [temp.format(c.replace("_", " ")) for c in classnames]
        prompts_ = torch.cat([clip.tokenize(p) for p in prompts_])
        prompts_ = prompts_.cuda()
        with torch.no_grad():
            text_features = zs_clip.encode_text(prompts_)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            current_text_features_zs = text_features.data.cpu().numpy()


        # 3. get the base text features from tuned model
        base_text_features_tuned = val_text_features
        

        # 4. get the current text features from tuned model
        data_loader_temp = deepcopy(self.test_loader)
        batch_temp = next(iter(data_loader_temp))
        input, _ = self.parse_batch_test(batch_temp)
        _, _, current_text_features = self.model_inference(input)
        current_text_features_tuned = current_text_features.data.cpu().numpy()

        text_feature_dict = {
            "base_text_features_zs": base_text_features_zs,
            "current_text_features_zs": current_text_features_zs,
            'base_text_features_tuned': base_text_features_tuned,
            "current_text_features_tuned": current_text_features_tuned,

        }

        return text_feature_dict