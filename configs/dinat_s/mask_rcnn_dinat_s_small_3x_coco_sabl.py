_base_ = [
    '../_base_/models/mask_rcnn_dinats.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_3x.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        drop_path_rate=0.35,
        patch_norm=True,
        kernel_size=7,
        dilations=[[1, 28], [1, 14], [1, 3, 1, 5, 1, 7, 1, 3, 1, 5, 1, 7, 1, 3, 1, 5, 1, 7], [1, 3]],
        pretrained='https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_s_small_in1k_224.pth',
    ),
    neck=dict(in_channels=[96, 192, 384, 768]),
    roi_head=dict(
        bbox_head=dict(
            _delete_=True,
            type='SABLHead',
            num_classes=6,
            cls_in_channels=256,
            reg_in_channels=256,
            roi_feat_size=7,
            reg_feat_up_ratio=2,
            reg_pre_kernel=3,
            reg_post_kernel=3,
            reg_pre_num=2,
            reg_post_num=1,
            cls_out_channels=1024,
            reg_offset_out_channels=256,
            reg_cls_out_channels=256,
            num_cls_fcs=1,
            num_reg_fcs=0,
            reg_class_agnostic=True,
            norm_cfg=None,
            bbox_coder=dict(
                type='BucketingBBoxCoder', num_buckets=14, scale_factor=1.7),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox_reg=dict(type='SmoothL1Loss', beta=0.1,
                               loss_weight=1.0)))
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    # dict(type='Sharpen'),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                                 (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                                 (736, 1333), (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True)
             ],
             [
                 dict(type='Resize',
                      img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(384, 600),
                      allow_negative_crop=True),
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                 (576, 1333), (608, 1333), (640, 1333),
                                 (672, 1333), (704, 1333), (736, 1333),
                                 (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True)
             ]
         ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
data = dict(train=dict(pipeline=train_pipeline))

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'rpb': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)

# Mixed precision?
fp16 = None
optimizer_config = dict(
    type="Fp16OptimizerHook",
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
)
