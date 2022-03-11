_base_ = '../cascade_rcnn/cascade_rcnn_r101_fpn_20e_coco.py'
model = dict(
    backbone=dict(
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))
