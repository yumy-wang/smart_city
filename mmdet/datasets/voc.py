# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

from mmcv.utils import print_log

from mmdet.core import eval_map, eval_recalls
from .builder import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module()
class VOCDataset(XMLDataset):

    # CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    #            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    #            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
    #            'tvmonitor')
    CLASSES = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60')

    def __init__(self, **kwargs):
        super(VOCDataset, self).__init__(**kwargs)
        if 'VOC2007' in self.img_prefix:
            self.year = 2007
        elif 'VOC2012' in self.img_prefix:
            self.year = 2012
        else:
            raise ValueError('Cannot infer dataset year from img_prefix')

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=[0.3, 0.5, 0.7],
                 scale_ranges=None):
        """Evaluate in VOC protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'mAP', 'recall'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple], optional): Scale ranges for evaluating
                mAP. If not specified, all bounding boxes would be included in
                evaluation. Default: None.

        Returns:
            dict[str, float]: AP/recall metrics.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            if self.year == 2007:
                ds_name = 'voc07'
            else:
                ds_name = self.CLASSES
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=None,
                    iou_thr=iou_thr,
                    dataset=ds_name,
                    logger=logger)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou_thr in enumerate(iou_thrs):
                    eval_results[f'recall@{num}@{iou_thr}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results

    ##########################################
    # import tempfile
    # import os.path as osp
    # import mmcv
    # import numpy as np
    # def format_results(self, results, jsonfile_prefix=None, **kwargs):
    #         """Format the results to json (standard format for COCO evaluation).

    #         Args:
    #             results (list[tuple | numpy.ndarray]): Testing results of the
    #                 dataset.
    #             jsonfile_prefix (str | None): The prefix of json files. It includes
    #                 the file path and the prefix of filename, e.g., "a/b/prefix".
    #                 If not specified, a temp file will be created. Default: None.

    #         Returns:
    #             tuple: (result_files, tmp_dir), result_files is a dict containing \
    #                 the json filepaths, tmp_dir is the temporal directory created \
    #                 for saving json files when jsonfile_prefix is not specified.
    #         """
    #         assert isinstance(results, list), 'results must be a list'
    #         assert len(results) == len(self), (
    #             'The length of results is not equal to the dataset len: {} != {}'.
    #             format(len(results), len(self)))

    #         if jsonfile_prefix is None:
    #             tmp_dir = tempfile.TemporaryDirectory()
    #             jsonfile_prefix = osp.join(tmp_dir.name, 'results')
    #         else:
    #             tmp_dir = None
    #         result_files = self.results2json(results, jsonfile_prefix)
    #         return result_files, tmp_dir

    # def results2json(self, results, outfile_prefix):
    #     """Dump the detection results to a COCO style json file.

    #     There are 3 types of results: proposals, bbox predictions, mask
    #     predictions, and they have different data types. This method will
    #     automatically recognize the type, and dump them to json files.

    #     Args:
    #         results (list[list | tuple | ndarray]): Testing results of the
    #             dataset.
    #         outfile_prefix (str): The filename prefix of the json files. If the
    #             prefix is "somepath/xxx", the json files will be named
    #             "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
    #             "somepath/xxx.proposal.json".

    #     Returns:
    #         dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
    #             values are corresponding filenames.
    #     """
    #     result_files = dict()
    #     if isinstance(results[0], list):
    #         json_results = self._det2json(results)
    #         result_files['bbox'] = f'{outfile_prefix}.bbox.json'
    #         result_files['proposal'] = f'{outfile_prefix}.bbox.json'
    #         mmcv.dump(json_results, result_files['bbox'])
    #     else:
    #         raise TypeError('invalid type of results')
    #     return result_files

    # def _det2json(self, results):
    #     """Convert detection results to COCO json style."""
    #     json_results = []
    #     for idx in range(len(self)):
    #         img_id = self.img_ids[idx]
    #         result = results[idx]
    #         for label in range(len(result)):
    #             bboxes = result[label]
    #             for i in range(bboxes.shape[0]):
    #                 data = dict()
    #                 data['image_id'] = img_id
    #                 data['bbox'] = self.xyxy2xywh(bboxes[i])
    #                 data['score'] = float(bboxes[i][4])
    #                 data['category_id'] = self.cat_ids[label]
    #                 json_results.append(data)
    #     return json_results

    # def xyxy2xywh(self, bbox):
    #     """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
    #     evaluation.

    #     Args:
    #         bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
    #             ``xyxy`` order.

    #     Returns:
    #         list[float]: The converted bounding boxes, in ``xywh`` order.
    #     """

    #     _bbox = bbox.tolist()
    #     return [
    #         _bbox[0],
    #         _bbox[1],
    #         _bbox[2] - _bbox[0],
    #         _bbox[3] - _bbox[1],
    #     ]
