import functools
import itertools
import numpy as np
import torch
import re
from torch.nn import *


class deprecate:
    from copy import deepcopy
    from re import findall

    def arg(**value):
        tree = {}
        for names, value in value.items():
            value = iter(value).__next__ if isinstance(value, list) else lambda _=value: _
            for names in names.split("_"):
                parents = [tree]
                names = findall("([A-Za-z][a-z_]*)([0-9_]*)", names)
                for nonlast, (attr, item) in enumerate(names, 1 - len(names)):
                    if item:
                        item = [int(i) for i in item.split("_")]
                        item += [item[-1] + 1, 1][(len(item) - 1) % 3 :]
                    childs = []
                    for parent in parents:
                        for attr in attr.lower().split("_"):
                            if not nonlast:
                                parent[attr] = value()
                            elif item:
                                parent.setdefault(attr, [{}])[:] += [deepcopy(parent[attr][-1]) for i in range(item[-2] - len(parent[attr]))]
                                for i in zip(item[::3], item[1::3], item[2::3]):
                                    for i in range(*i):
                                        childs.append(parent[attr][i])
                            else:
                                childs.append(parent.setdefault(attr, {}))
                    parents = childs
        return tree

    class param(dict):
        def __mul__(s, x):
            return param({_: (s * x).__ceil__() if isinstance(s, float) else s for _, s in s.items()})


def pk(a):
    return zip(range(2 - len(a), 1), a, a[1:])


def mul(a, b):
    return [mul(a, b) for a in a] if isinstance(a, (list, tuple)) else {i: mul(a, b) for i, a in a.items()} if isinstance(a, dict) else int(a(b)) if hasattr(a, "__call__") else int(a * b) if isinstance(a, float) else a


def bisect(f, g, x, t, step=1.5, tol=1.01, log=lambda *_: 0):
    a, b, Z = 1e-9, 9e9, 9e9
    while x / a > tol and b / x > tol:
        y = f(x)
        z = g(y)
        log(x, z)
        if abs(np.log(z / t)) < abs(np.log(Z / t)):
            X = x
            Y = y
            Z = z
        if z < t:
            a = x
            x = min(x * step, (x * b) ** 0.5)
        else:
            b = x
            x = max(x / step, (x * a) ** 0.5)
    return X, Y, Z


class Dat:
    def __init__(s, a, n, b, l, dev="cuda"):
        s.a = np.memmap(a, mode="r")
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
        s.n = n
        s.b = b
        s.dev = dev

    def __call__(s):
        return torch.from_numpy(s.map[s.a[np.random.randint(0, len(s.a), s.b)]]).to(s.dev, torch.long, True)

    def __len__(s):
        return int(len(s.a) * s.n / s.b)

    def __iter__(s):
        X = s()
        for _ in range(len(s)):
            x, X = X, s()
            yield x


class F(Module):
    def __init__(s, f, *_):
        super().__init__()
        s.f = f
        s._ = _

    def forward(s, x):
        return s.f(x, *s._)

    def extra_repr(s):
        return ", ".join(map(str, (s.f,) + s._))


def Seq(*_):
    _ = [_ for _ in _ for _ in (_ if type(_) == Sequential else [_]) if _ and not isinstance(_, Identity)]
    return Sequential(*_) if len(_) > 1 else _[0] if _ else Identity()


class Add(Sequential):
    def forward(s, x):
        if not isinstance(x, tuple):
            x = (x,)
        return functools.reduce(torch.add, ((i or s[-1])(x[-1] if j is None else j) for i, j in itertools.zip_longest(s, x)))


class Mul(Sequential):
    def forward(s, x):
        if not isinstance(x, tuple):
            x = (x,)
        return functools.reduce(torch.mul, ((i or s[-1])(x[-1] if j is None else j) for i, j in itertools.zip_longest(s, x)))


class Res(Sequential):
    def forward(s, x):
        return super().forward(x) + x


class Stack(Sequential):
    def forward(s, x):
        return torch.stack([s(x) for s in s])


class Dot(Sequential):
    def forward(s, x):
        return s[0](x) @ s[1](x).mT


class T(Sequential):
    def forward(s, x):
        return super().forward(x.mT).mT


def size(m):
    return sum(m.numel() for m in m.parameters())


def sized(f, _, x, y=100000):
    return bisect(lambda x: f(**mul(_, x)), size, x, _.get("sz", y), log=lambda *_: 0, step=1.2)[1]


def MinModel(*m, size=size):
    return min(filter(None, m), key=size)


class Decay(Module):
    def __init__(s, *k, d=-2, shift=0, min=4, max=16, logspace=False):
        super().__init__()
        for j, i in enumerate(k):
            setattr(s, str(j), Parameter(init.uniform_(torch.empty(i), np.log(0.1) / min / len(k), np.log(0.1) / max / len(k))))
        s.d = d
        s.shift = shift
        s.logspace = logspace

    def forward(s, x):
        if s.shift:
            x = x[(..., slice(-s.shift), *[slice(None)] * (-s.d - 1))]
        k = sum(s.parameters()) * torch.arange(x.size(s.d) / 2, -x.size(s.d) / 2, -1, dtype=x.dtype, device=x.device)[(..., *[None] * (-s.d - 1))]
        if s.logspace:
            x = x.sign() * (((x.abs() + 5e-41).log() + k).logcumsumexp(s.d) - k.logcumsumexp(s.d)).exp()
        else:
            k = k.exp()
            x = (x * k).cumsum(s.d) / k.cumsum(s.d)
        if s.shift:
            x = functional.pad(x, [*[0, 0] * (-s.d - 1), s.shift, 0])
        return x

    def extra_repr(s):
        return ", ".join(map(str, (s.d, s.shift)))


def Lin(i, o, b=False):
    return Linear(i, o, b)


def Sep(i, h, o, b=False, m=None):
    return MinModel(
        h
        and Seq(
            Lin(i, h),
            m,
            Lin(h, o, b),
        ),
        Seq(
            i <= o and m,
            Lin(i, o, b),
            o < i and m,
        ),
    )


def Left(m, r):
    return T(
        ZeroPad1d((r - 1, 0)),
        m,
    )


def CovLeft(i, o, r=1, g=1, b=False):
    return Left(Conv1d(i, o, r, groups=g, bias=b), r) if r > 1 else Linear(i, o, b)


def ApLeft(i, h, o, r=1, b=False):
    return Sep(i, h, o, b, r > 1 and Left(AvgPool1d(r), r))


def MpLeft(i, h, o, r=1, b=False):
    return Sep(i, h, o, b, r > 1 and Left(MaxPool1d(r), r))


def SeLeft(i, h, o, b=False):
    return Sep(i, h, o, b, F(lambda x: x.cumsum(-1) / torch.arange(1, 1 + x.size(-1), dtype=x.dtype, device=x.device)))


def Bn(i, b=True):
    return T(BatchNorm1d(i, track_running_stats=False, affine=b))


def Ret(i, qh=0, kh=0, k=0, vh=0, v=0, oh=0, o=0, h=0, r=1, b=False, **_):
    k = k or i
    v = v or k
    h = h or k
    o = o or i
    return Sequential(
        Dot(
            Sequential(Sep(i, h * qh, h * k, True), Unflatten(-1, [h, 1, k])),
            Sequential(
                Mul(
                    Sequential(Sep(i, h * kh, h * k, True), Unflatten(-1, [h, 1, k])),
                    Sequential(Sep(i, h * vh, h * v, True), Unflatten(-1, [h, v, 1])),
                ),
                Decay((h, 1, k), (h, v, 1), d=-4, shift=1, **_),
            ),
        ),
        Flatten(-3),
        Sep(h * v, h * oh, o),
    )


def SepCovLeft(i, ch=0, c=0, oh=0, o=0, g=1, r=1, b=False):
    o = o or i
    return MinModel(
        c
        and Seq(
            Sep(i, g * ch, g * c),
            MinModel(
                Seq(
                    CovLeft(g * c, g * c, r, g * c),
                    Sep(g * c, g * oh, o, b),
                ),
                oh
                and Seq(
                    CovLeft(g * c, g * oh, r, g),
                    Lin(g * oh, o, b),
                ),
                CovLeft(g * c, o, r),
            ),
        ),
        ch
        and Seq(
            Lin(i, g * ch),
            MinModel(
                c
                and Seq(
                    CovLeft(g * ch, g * c, r, g),
                    Sep(g * c, g * oh, o, b),
                ),
                oh
                and Seq(
                    CovLeft(g * ch, g * oh, r, g),
                    Linear(g * oh, o, b),
                ),
                CovLeft(g * ch, o, r),
            ),
        ),
        c
        and Seq(
            CovLeft(i, g * c, r),
            Sep(g * c, g * oh, o, b),
        ),
        oh
        and Seq(
            CovLeft(i, g * oh, r),
            Linear(g * oh, o, b),
        ),
        CovLeft(i, o, r),
    )


def Lu(a, b, n, bw=True):
    m = {Sequential: Sequential, Identity: Identity, T: T, Flatten: Flatten, Unflatten: Unflatten}
    if a:
        n /= np.log(b) - np.log(a)

    class LU(torch.autograd.function.InplaceFunction):
        def forward(s, x):
            return x.sign() * (torch.clamp(x.abs(), a, b) if a else x.abs() + 8e-46).log().mul(n).round().div(n).exp()

        def backward(s, x):
            return x.sign() * (torch.clamp(x.abs(), a, b) if a else x.abs() + 8e-46).log().mul(n).round().div(n).exp() if bw else x

    class Lu:
        def reset_parameters(s):
            super().reset_parameters()
            s._ = None

        def __getattr__(s, x):
            x = super().__getattr__(x)
            return lu(x) if hasattr(s, "_") and not isinstance(s, Embedding) else x

        def forward(s, x):
            y = super().forward(x)
            if y.untyped_storage() is x.untyped_storage():
                y._ = None
            else:
                y = lu(y)
            return y

    def lu(x):
        if isinstance(x, torch.Tensor):
            if not hasattr(x, "_"):
                x = LU.apply(x)
                x._ = None
        elif isinstance(x, type):
            if not issubclass(x, Lu):
                x = m.setdefault(x, type("Lu" + x.__name__, (Lu, x), {}))
        elif isinstance(x, Module) and not isinstance(x, Lu):
            for i in x.modules():
                i.__class__ = lu(type(i))
        return x

    return lu


class Stat:
    min = 9e9
    max = -9e9
    sum5 = 0
    sum = 0
    sum2 = 0
    numel = 0

    def __call__(s, x):
        if isinstance(x, Stat):
            s.min = min(s.mim, x.min)
            s.max = min(s.max, x.max)
            s.sum5 += x.sum5
            s.sum += x.sum
            s.sum2 += x.sum2
            s.numel += x.numel
        else:
            x = x.data
            s.min = min(s.min, x.min())
            s.max = max(s.max, x.max())
            s.sum5 += x.sqrt().sum()
            s.sum += x.sum()
            s.sum2 += x.square().sum()
            s.numel += x.numel()
        return x

    @property
    def mean5(s):
        return s.sum5 / s.numel

    @property
    def mean(s):
        return s.sum / s.numel

    @property
    def mean2(s):
        return s.sum2 / s.numel

    @property
    def var(s):
        return s.mean2 - s.mean**2

    @property
    def std(s):
        return s.var**0.5


def Log(N=1):
    X = n = 0
    a = []
    A = torch.empty(0)

    def Log(x=None):
        nonlocal X, n, A
        if x is None:
            if a:
                A = torch.cat((A, torch.stack(a).mul_(1 / N).exp_().cpu()))
                a.clear()
            return A
        X += x.data.float().log_()
        n += 1
        if n == N:
            a.append(X)
            X = n = 0

    return Log
