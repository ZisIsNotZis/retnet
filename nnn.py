from functools import reduce

import numpy as np
import torch
from torch.nn import *


def str(_, str=str):
    return "×".join(map(str, _.shape)) if isinstance(_, torch.Tensor) else getattr(_, "__name__", str(_))


class Dat:
    def __init__(s, a, l, *bs):
        s.a = np.memmap(a)
        try:
            s.map = np.memmap(a + ".map")
            s.unmap = (s.map < 255).nonzero()[0]
        except FileNotFoundError:
            s.unmap = np.zeros(127, "?")
            s.unmap[s.a] = True
            s.unmap = s.unmap.nonzero()[0]
            s.map = np.memmap(a + ".map", mode="w+", shape=127)
            s.map[:] = 255
            s.map[s.unmap] = range(len(s.unmap))
        s.a = np.lib.stride_tricks.sliding_window_view(s.a, l)
        s.bs = bs

    def __call__(s):
        return torch.from_numpy(s.map[s.a[np.random.randint(0, len(s.a), s.bs)]]).to("cuda", torch.long)


class F(Module):
    _ = ()

    def __init__(s, f, *_, unpak=False, **__):
        super().__init__()
        s.f = f
        s.unpak = unpak
        try:
            i = _.index(F)
            s._ = _[:i]
            s.__ = _[i + 1 :]
        except ValueError:
            s.__ = _
        s.___ = __
        for _ in enumerate(_), __.items():
            for __, _ in _:
                if isinstance(_, Parameter):
                    s.register_parameter(str(__), _)
                elif isinstance(_, Module):
                    s.register_module(str(__), _)

    def forward(s, x):
        return s.f(*s._, *x if s.unpak else [x], *s.__, **s.___)

    def _get_name(s):
        return str(s.f)

    def extra_repr(s):
        return ", ".join(("" if isinstance(__, int) else __ + "=") + str(_) for _ in [enumerate(s._), enumerate(s.__), s.___.items()] for __, _ in _ if not isinstance(_, Module))


def Tup(x, *_):
    if isinstance(x, torch.Tensor):
        return [_(x) for _ in _]
    return F(Tup, x, *_)


def Add(x, *_):
    if isinstance(x, torch.Tensor):
        return reduce(torch.add, [_(x) for _ in _])
    _ = [_ for _ in (x,) + _ if _]
    return _[0] if len(_) < 2 else F(Add, *_)


def Mul(x, *_):
    if isinstance(x, torch.Tensor):
        return reduce(torch.mul, [_(x) for _ in _])
    _ = [_ for _ in (x,) + _ if _]
    return _[0] if len(_) < 2 else F(Mul, *_)


def Dot(x, *_):
    if isinstance(x, torch.Tensor):
        return reduce(lambda a, b: a @ b.mT, [_(x) for _ in _])
    _ = [_ for _ in (x,) + _ if _]
    return _[0] if len(_) < 2 else F(Dot, *_)


def Seq(*_):
    o = []
    for _ in _:
        if not _ or isinstance(_, Identity):
            continue
        for _ in _ if isinstance(_, Sequential) else (_,):
            if _ == T and o and o[-1] == T:
                del o[-1]
                continue
            o.append(_)
    return Sequential(*o) if len(o) > 1 else o[0] if o else Identity()


def BatchNorm(*_, **__):
    return Seq(T, BatchNorm1d(*_, track_running_stats=False, **__), T)


def Const(x, _=None):
    if _ is None:
        return F(Const, x)
    return _


def ConvLeft(i, o, r, **_):
    return Seq(T, Conv1d(i, o, r, padding=r - 1, bias=False, **_), F(torch.Tensor.__getitem__, (..., slice(1 - r))), T) if r > 1 else Linear(i, o, False)


def SepConvLeft(i, h, o, r, **_):
    return MinModel(
        Seq(
            Linear(i, h, False),
            MinModel(
                Seq(
                    ConvLeft(h, h, r, groups=h, **_),
                    Linear(h, o, False),
                ),
                ConvLeft(h, o, r, **_),
            ),
        ),
        Seq(
            ConvLeft(i, h, r, **_),
            Linear(h, o, False),
        ),
        ConvLeft(i, o, r, **_),
    )


def MinModel(*_):
    return min(_, key=lambda _: sum(_.numel() for _ in _.parameters()) if _ else 9e9)



T = F(torch.transpose, -2, -1)
