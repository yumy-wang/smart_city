# dataset settings
dataset_type = 'CocoDataset'
data_root = '/mmdetection/mmdetection/data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(1280, 720), keep_ratio=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    # dict(
    #     type='Albu',
    #     transforms=[
    #         dict(
    #             type='OneOf',
    #             transforms=[
    #                 dict(
    #                     type='GaussianBlur',
    #                     blur_limit=7,
    #                     sigma_limit=0,
    #                     p=0.1),
    #                 dict(type='MotionBlur', blur_limit=7, p=0.1)
    #             ],
    #             p=0),
    #         dict(
    #             type='OneOf',
    #             transforms=[
    #                 dict(type='RandomGamma', gamma_limit=(60, 120), p=0.1),
    #                 dict(
    #                     type='RandomBrightnessContrast',
    #                     brightness_limit=0.1,
    #                     contrast_limit=0.1,
    #                     p=0.1),
    #                 dict(
    #                     type='CLAHE',
    #                     clip_limit=4.0,
    #                     tile_grid_size=(4, 4),
    #                     p=0.1)
    #             ],
    #             p=0),
    #         dict(type='Rotate', limit=90, border_mode=4, p=0.1),
    #         dict(type='Blur', blur_limit=7, p=0)
    #     ]),
    
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1280, 720),
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,      # 每张gpu训练多少张图片  batch_size = gpu_num(训练使用gpu数量) * imgs_per_gpu
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/test.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
# evaluation = dict(interval=1, metric='bbox', iou_thrs=[0.3, 0.5, 0.7])

