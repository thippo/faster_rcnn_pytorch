import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.timer import Timer
from ..utils.py_cpu_nms import nms
from ..rpn_msr.proposal_layer import proposal_layer as proposal_layer_py
from ..rpn_msr.anchor_target_layer import anchor_target_layer as anchor_target_layer_py
from ..rpn_msr.proposal_target_layer import proposal_target_layer as proposal_target_layer_py
from ..utils.bbox_transform import bbox_transform_inv, clip_boxes

from .network import Conv2d, FC
from .resnet import resnet18


def nms_detections(pred_boxes, scores, nms_thresh, inds=None):
    dets = np.hstack((pred_boxes,
                      scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, nms_thresh)
    if inds is None:
        return pred_boxes[keep], scores[keep]
    return pred_boxes[keep], scores[keep], inds[keep]

class ROIpooling(nn.Module):

    def __init__(self, size=(7, 7), spatial_scale=1.0 / 16.0):
        super(ROIpooling, self).__init__()
        self.adapmax2d = nn.AdaptiveMaxPool2d(size)
        self.spatial_scale = spatial_scale

    def forward(self, features, rois_boxes):

        # rois_boxes : [x, y, x`, y`]
        rois_boxes = rois_boxes[:, 1:]

        if type(rois_boxes) == np.ndarray:
            rois_boxes = to_var(torch.from_numpy(rois_boxes))

        rois_boxes = rois_boxes.data.float().clone()
        rois_boxes.mul_(self.spatial_scale)
        rois_boxes = rois_boxes.long()

        output = []

        for i in range(rois_boxes.size(0)):
            roi = rois_boxes[i]

            try:

                roi_feature = features[:, :, roi[1]:(roi[3] + 1), roi[0]:(roi[2] + 1)]
            except Exception as e:
                print(e, roi)


            pool_feature = self.adapmax2d(roi_feature)
            output.append(pool_feature)

        return torch.cat(output, 0)

class RPN(nn.Module):
    _feat_stride = 16
    anchor_ratios = [0.5, 1, 2]
    anchor_scales = [8, 16, 32]

    def __init__(self):
        super(RPN, self).__init__()

        self.features = resnet18()
        self.conv1 = Conv2d(512, 512, 3, same_padding=True)
        self.score_conv = Conv2d(512, len(self.anchor_scales) * 3 * 2, 1, relu=False, same_padding=False)
        self.bbox_conv = Conv2d(512, len(self.anchor_scales) * 3 * 4, 1, relu=False, same_padding=False)

        # loss
        self.cross_entropy = None
        self.los_box = None

    @property
    def loss(self):
        return self.cross_entropy + self.loss_box * 10

    def forward(self, im_data, im_info, gt_boxes=None):
        features = self.features(im_data)
        height, width = features.size()[-2:]

        rpn_conv1 = self.conv1(features)

        # rpn score
        rpn_cls_score = self.score_conv(rpn_conv1)
        rpn_cls_score = rpn_cls_score.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 2)
        rpn_cls_prob = F.softmax(rpn_cls_score, dim=1)
        rpn_cls_prob = rpn_cls_prob.view(height, width, 18)
        rpn_cls_prob = rpn_cls_prob.permute(2, 0, 1).contiguous().unsqueeze(0)

        # rpn boxes
        rpn_bbox_pred = self.bbox_conv(rpn_conv1)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'
        rois = self.proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info,
                                   cfg_key, self._feat_stride, self.anchor_ratios, self.anchor_scales)

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None
            rpn_data = self.anchor_target_layer(rpn_cls_prob, gt_boxes,
                                                im_info, self._feat_stride, self.anchor_ratios, self.anchor_scales)
            self.cross_entropy, self.loss_box = self.build_loss(rpn_cls_prob, rpn_bbox_pred, rpn_data)

        return features, rois

    def build_loss(self, rpn_cls_score_reshape, rpn_bbox_pred, rpn_data):
        # classification loss
        rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        rpn_label = rpn_data[0].view(-1)

        rpn_keep = rpn_label.data.ne(-1).nonzero().squeeze().cuda()
        rpn_cls_score = torch.index_select(rpn_cls_score, 0, rpn_keep)
        rpn_label = torch.index_select(rpn_label, 0, rpn_keep)

        fg_cnt = torch.sum(rpn_label.data.ne(0)).float()

        rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label)

        # box loss
        rpn_bbox_targets, rpn_bbox_inside_weights = rpn_data[1:]
        rpn_bbox_targets = torch.mul(rpn_bbox_targets, rpn_bbox_inside_weights)
        rpn_bbox_pred = torch.mul(rpn_bbox_pred, rpn_bbox_inside_weights)

        rpn_loss_box = F.smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, size_average=False) / (fg_cnt + 1e-4)

        return rpn_cross_entropy, rpn_loss_box


    @staticmethod
    def proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchor_ratios, anchor_scales):
        rpn_cls_prob_reshape = rpn_cls_prob_reshape.data.cpu().numpy()
        rpn_bbox_pred = rpn_bbox_pred.data.cpu().numpy()
        x = proposal_layer_py(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchor_ratios, anchor_scales)
        x = torch.from_numpy(x).cuda()
        return x.view(-1, 5)

    @staticmethod
    def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, _feat_stride, anchor_ratios, anchor_scales):
        """
        rpn_cls_score: for pytorch (1, Ax2, H, W) bg/fg scores of previous conv layer
        gt_boxes: (G, 5) vstack of [x1, y1, x2, y2, class]
        gt_ishard: (G, 1), 1 or 0 indicates difficult or not
        dontcare_areas: (D, 4), some areas may contains small objs but no labelling. D may be 0
        im_info: a list of [image_height, image_width, scale_ratios]
        _feat_stride: the downsampling ratio of feature map to the original input image
        anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
        ----------
        Returns
        ----------
        rpn_labels : (1, 1, HxA, W), for each anchor, 0 denotes bg, 1 fg, -1 dontcare
        rpn_bbox_targets: (1, 4xA, H, W), distances of the anchors to the gt_boxes(may contains some transform)
                        that are the regression objectives
        rpn_bbox_inside_weights: (1, 4xA, H, W) weights of each boxes, mainly accepts hyper param in cfg
        beacuse the numbers of bgs and fgs mays significiantly different
        """
        rpn_cls_score = rpn_cls_score.data.cpu().numpy()
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights = \
            anchor_target_layer_py(rpn_cls_score, gt_boxes, im_info, _feat_stride, anchor_ratios, anchor_scales)

        rpn_labels = torch.from_numpy(rpn_labels).long().cuda()
        rpn_bbox_targets = torch.from_numpy(rpn_bbox_targets).cuda()
        rpn_bbox_inside_weights = torch.from_numpy(rpn_bbox_inside_weights).cuda()

        return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights

class FasterRCNN(nn.Module):

    def __init__(self, classes=None, debug=False):
        super(FasterRCNN, self).__init__()

        self.classes = classes
        self.n_classes = len(classes)

        self.rpn = RPN()
        self.roi_pool = ROIpooling((7, 7), 1.0/16)
        self.fc6 = FC(512 * 7 * 7, 4096)
        self.fc7 = FC(4096, 4096)
        self.score_fc = FC(4096, self.n_classes, relu=False)
        self.bbox_fc = FC(4096, self.n_classes * 4, relu=False)

        # loss
        self.cross_entropy = None
        self.loss_box = None

        # for log
        self.debug = debug

    @property
    def loss(self):
        return self.cross_entropy + self.loss_box * 10

    def forward(self, im_data, im_info, gt_boxes=None):
        features, rois = self.rpn(im_data, im_info, gt_boxes)

        if self.training:
            roi_data = self.proposal_target_layer(rois, gt_boxes, self.n_classes)
            rois = roi_data[0]

        # roi pool
        pooled_features = self.roi_pool(features, rois)
        x = pooled_features.view(pooled_features.size()[0], -1)
        x = self.fc6(x)
        x = F.dropout(x, training=self.training)
        x = self.fc7(x)
        x = F.dropout(x, training=self.training)

        cls_score = self.score_fc(x)
        cls_prob = F.softmax(cls_score, dim=1)
        bbox_pred = self.bbox_fc(x)

        if self.training:
            self.cross_entropy, self.loss_box = self.build_loss(cls_score, bbox_pred, roi_data)

        return cls_prob, bbox_pred, rois

    def build_loss(self, cls_score, bbox_pred, roi_data):
        # classification loss
        label = roi_data[1].squeeze()
        fg_cnt = torch.sum(label.data.ne(0))
        bg_cnt = label.data.numel() - fg_cnt

        # for log
        if self.debug:
            maxv, predict = cls_score.data.max(1)
            self.tp = torch.sum(predict[:fg_cnt].eq(label.data[:fg_cnt])) if fg_cnt > 0 else 0
            self.tf = torch.sum(predict[fg_cnt:].eq(label.data[fg_cnt:]))
            self.fg_cnt = fg_cnt
            self.bg_cnt = bg_cnt

        ce_weights = torch.ones(cls_score.size()[1])
        ce_weights[0] = float(fg_cnt.item()) / bg_cnt.item()
        ce_weights = ce_weights.cuda()
        cross_entropy = F.cross_entropy(cls_score, label, weight=ce_weights)

        # bounding box regression L1 loss
        bbox_targets, bbox_inside_weights = roi_data[2:]
        bbox_targets = torch.mul(bbox_targets, bbox_inside_weights)
        bbox_pred = torch.mul(bbox_pred, bbox_inside_weights)

        loss_box = F.smooth_l1_loss(bbox_pred, bbox_targets, size_average=False) / (fg_cnt.float() + 1e-4)

        return cross_entropy, loss_box

    @staticmethod
    def proposal_target_layer(rpn_rois, gt_boxes, num_classes):
        """
        ----------
        rpn_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2]
        gt_boxes: (G, 5) [x1 ,y1 ,x2, y2, class] int
        # gt_ishard: (G, 1) {0 | 1} 1 indicates hard
        dontcare_areas: (D, 4) [ x1, y1, x2, y2]
        num_classes
        ----------
        Returns
        ----------
        rois: (1 x H x W x A, 5) [0, x1, y1, x2, y2]
        labels: (1 x H x W x A, 1) {0,1,...,_num_classes-1}
        bbox_targets: (1 x H x W x A, K x4) [dx1, dy1, dx2, dy2]
        bbox_inside_weights: (1 x H x W x A, Kx4) 0, 1 masks for the computing loss
        bbox_outside_weights: (1 x H x W x A, Kx4) 0, 1 masks for the computing loss
        """
        rpn_rois = rpn_rois.data.cpu().numpy()
        rois, labels, bbox_targets, bbox_inside_weights = \
            proposal_target_layer_py(rpn_rois, gt_boxes, num_classes)
        # print labels.shape, bbox_targets.shape, bbox_inside_weights.shape
        rois = torch.from_numpy(rois).cuda()
        labels = torch.from_numpy(labels).long().cuda()
        bbox_targets = torch.from_numpy(bbox_targets).cuda()
        bbox_inside_weights = torch.from_numpy(bbox_inside_weights).cuda()

        return rois, labels, bbox_targets, bbox_inside_weights

    def interpret_faster_rcnn(self, cls_prob, bbox_pred, rois, im_info, im_shape, nms=True, clip=True, min_score=0.0):
        # find class
        scores, inds = cls_prob.data.max(1)
        scores, inds = scores.cpu().numpy(), inds.cpu().numpy()

        keep = np.where((inds > 0) & (scores >= min_score))
        scores, inds = scores[keep], inds[keep]

        # Apply bounding-box regression deltas
        keep = keep[0]
        box_deltas = bbox_pred.data.cpu().numpy()[keep]
        box_deltas = np.asarray([
            box_deltas[i, (inds[i] * 4): (inds[i] * 4 + 4)] for i in range(len(inds))
        ], dtype=np.float)
        boxes = rois.data.cpu().numpy()[keep, 1:5] / im_info[0][2]
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        if clip:
            pred_boxes = clip_boxes(pred_boxes, im_shape)

        # nms
        if nms and pred_boxes.shape[0] > 0:
            pred_boxes, scores, inds = nms_detections(pred_boxes, scores, 0.3, inds=inds)

        return pred_boxes, scores, self.classes[inds]

    def detect(self, image, thr=0.3):
        im_data, im_scales = self.get_image_blob(image)
        im_info = np.array(
            [[im_data.shape[1], im_data.shape[2], im_scales[0]]],
            dtype=np.float32)

        cls_prob, bbox_pred, rois = self(im_data, im_info)
        pred_boxes, scores, classes = \
            self.interpret_faster_rcnn(cls_prob, bbox_pred, rois, im_info, image.shape, min_score=thr)
        return pred_boxes, scores, classes

    def get_image_blob_noscale(self, im):
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= self.PIXEL_MEANS

        processed_ims = [im]
        im_scale_factors = [1.0]

        blob = im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)

    def get_image_blob(self, im):
        """Converts an image into a network input.
        Arguments:
            im (ndarray): a color image in BGR order
        Returns:
            blob (ndarray): a data blob holding an image pyramid
            im_scale_factors (list): list of image scales (relative to im) used
                in the image pyramid
        """
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= self.PIXEL_MEANS

        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        processed_ims = []
        im_scale_factors = []

        for target_size in self.SCALES:
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > self.MAX_SIZE:
                im_scale = float(self.MAX_SIZE) / float(im_size_max)
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                            interpolation=cv2.INTER_LINEAR)
            im_scale_factors.append(im_scale)
            processed_ims.append(im)

        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)
