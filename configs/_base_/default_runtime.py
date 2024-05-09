# Copyright (c) OpenMMLab. All rights reserved.

checkpoint_config = dict(interval=1)
default_hooks = dict(
checkpoint = dict(type='CheckpointHook',
                  interval=1,  # 控制多少个迭代/批次保存一次
                  max_keep_ckpts=3,  # 保留的最大.pth文件数量
                  create_symlink=False)
)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
