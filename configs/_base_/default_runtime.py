checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = "/mmdetection/mmdetection/checkpoints/cascade_rcnn_r50_fpn_1x_coco_classes_80.pth"
# load_from = None
resume_from = None
# workflow = [('train', 3), ('val', 1)]
workflow = [('train', 1)]
