"""Import-only stub for the compiled ``mmcv._ext`` extension module.

``mmcv-lite`` ships the pure-Python ``mmcv/ops/*.py`` wrappers but not the
compiled ``_ext`` extension they bind to.  Importing ``mmpose.apis`` pulls in
``mmpose.models.heads.transformer_heads.edpose_head``, which does an
unconditional ``from mmcv.ops import MultiScaleDeformableAttention``, so the
whole of ``mmcv.ops`` gets imported even when no op is ever used.

RTMO uses no mmcv op at all: its head only needs ``mmcv.cnn.ConvModule`` /
``Scale`` and mmpose's own pure-PyTorch ``nms_torch``.  This stub therefore
satisfies the import-time ``hasattr(ext, fn)`` probe in
``mmcv.utils.ext_loader.load_ext`` and defers failure to actual call time,
where it raises a NotImplementedError naming the missing op.

If you later need real ops (mmdet detectors, mmcv batched_nms, DCN,
MultiScaleDeformableAttention), delete this file and install a full mmcv
build instead.
"""


class _MissingOp:
    """Placeholder that is importable but raises when actually invoked."""

    __slots__ = ('_name',)

    def __init__(self, name):
        self._name = name

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            f"mmcv op '{self._name}' is unavailable: this environment uses "
            'mmcv-lite plus a stubbed mmcv._ext, which provides no compiled '
            'ops. Install a full mmcv build if you need this operator.')

    def __repr__(self):
        return f'<stubbed mmcv._ext op {self._name!r}>'


def __getattr__(name):
    # Dunders must keep failing so Python still treats this as a plain
    # module (e.g. a truthy __path__ would make it look like a package).
    if name.startswith('__') and name.endswith('__'):
        raise AttributeError(name)
    return _MissingOp(name)
