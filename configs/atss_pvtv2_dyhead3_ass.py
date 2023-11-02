_base_ = [
    # '_base_/datasets/coco_detection.py',
    '_base_/datasets/voc0712.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py'
]
model = dict(
    type='ATSS',
    # pretrained='pretrained/pvt_v2_b2.pth',
    pretrained='/media/wcj/A4D4C4CFD4C4A4C0/chr/PVT-2/detection/work_dirs/atss_pvtv2_dyhead_ass/epoch_55.pth',
    backbone=dict(
        type='pvt_v2_b2',
        style='pytorch'),
    neck=[dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
        dict(type='DyHead', in_channels=256, out_channels=256, num_blocks=3)
        ],
    bbox_head=dict(
        type='ATSSHead',
        num_classes=24,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
# augmentation strategy originates from DETR / Sparse RCNN
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    
    dict(type='Resize', img_scale=(224,224),keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
data = dict(train=dict(pipeline=train_pipeline))
evaluation = dict(interval=100)
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(step=[27, 33])


# use apex fp16
runner = dict(type='EpochBasedRunnerAmp', max_epochs=100)
# runner = dict(type='EpochBasedRunner', max_epochs=36)
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)

find_unused_parameters = True
