"""utils functions copied from https://raw.githubusercontent.com/fastai/fastai/master/fastai/core.py"""


import re
from pathlib import Path
import collections
from collections import OrderedDict
from functools import partial
import tempfile
import numpy as np
from numbers import Number
import pandas as pd
from pandas import DataFrame, Series
from typing import Any, Optional, Collection, List, Dict, Tuple
from typing import Union, Callable, Iterable


AnnealFunc = Callable[[Number, Number, float], Number]
ArgStar = Collection[Any]
BatchSamples = Collection[Tuple[Collection[int], int]]
DataFrameOrChunks = Union[DataFrame, pd.io.parsers.TextFileReader]
FilePathList = Collection[Path]
Floats = Union[float, Collection[float]]
ImgLabel = str
ImgLabels = Collection[ImgLabel]
IntsOrStrs = Union[int, Collection[int], str, Collection[str]]
KeyFunc = Callable[[int], int]
KWArgs = Dict[str, Any]
ListOrItem = Union[Collection[Any], int, float, str]
ListRules = Collection[Callable[[str], str]]
ListSizes = Collection[Tuple[int, int]]
NPArrayableList = Collection[Union[np.ndarray, list]]
NPArrayList = Collection[np.ndarray]
NPArrayMask = np.ndarray
NPImage = np.ndarray
OptDataFrame = Optional[DataFrame]
OptListOrItem = Optional[ListOrItem]
OptRange = Optional[Tuple[float, float]]
OptStrTuple = Optional[Tuple[str, str]]
OptStats = Optional[Tuple[np.ndarray, np.ndarray]]
PathOrStr = Union[Path, str]
Point = Tuple[float, float]
Points = Collection[Point]
SplitArrayList = List[Tuple[np.ndarray, np.ndarray]]
StartOptEnd = Union[float, Tuple[float, float]]
StrList = Collection[str]
Tokens = Collection[Collection[str]]
OptStrList = Optional[StrList]
np.set_printoptions(precision=6, threshold=50, edgeitems=4, linewidth=120)


def is_listy(x: Any) -> bool: return isinstance(x, (tuple, list))


def is_tuple(x: Any) -> bool: return isinstance(x, tuple)


def is_dict(x: Any) -> bool: return isinstance(x, dict)


def is_pathlike(x: Any) -> bool: return isinstance(x, (str, Path))


def noop(x): return x


def chunks(l: Collection, n: int) -> Iterable:
    "Yield successive `n`-sized chunks from `l`."
    for i in range(0, len(l), n): yield l[i:i + n]


def recurse(func: Callable, x: Any, *args, **kwargs) -> Any:
    if is_listy(x): return [recurse(func, o, *args, **kwargs) for o in x]
    if is_dict(x):  return {k: recurse(func, v, *args, **kwargs) for k, v in x.items()}
    return func(x, *args, **kwargs)


def first_el(x: Any) -> Any:
    "Recursively get the first element of `x`."
    if is_listy(x): return first_el(x[0])
    if is_dict(x):  return first_el(x[list(x.keys())[0]])
    return x


def to_int(b: Any) -> Union[int, List[int]]:
    "Recursively convert `b` to an int or list/dict of ints; raises exception if not convertible."
    return recurse(lambda x: int(x), b)


def ifnone(a: Any, b: Any) -> Any:
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a


def is1d(a: Collection) -> bool:
    "Return `True` if `a` is one-dimensional"
    return len(a.shape) == 1 if hasattr(a, 'shape') else len(np.array(a).shape) == 1


def uniqueify(x: Series, sort: bool = False) -> List:
    "Return sorted unique values of `x`."
    res = list(OrderedDict.fromkeys(x).keys())
    if sort: res.sort()
    return res


def idx_dict(a):
    "Create a dictionary value to index from `a`."
    return {v: k for k, v in enumerate(a)}


def find_classes(folder: Path) -> FilePathList:
    "List of label subdirectories in imagenet-style `folder`."
    classes = [d for d in folder.iterdir()
               if d.is_dir() and not d.name.startswith('.')]
    assert (len(classes) > 0)
    return sorted(classes, key=lambda d: d.name)


def arrays_split(mask: NPArrayMask, *arrs: NPArrayableList) -> SplitArrayList:
    "Given `arrs` is [a,b,...] and `mask`index - return[(a[mask],a[~mask]),(b[mask],b[~mask]),...]."
    assert all([len(arr) == len(arrs[0]) for arr in arrs]), 'All arrays should have same length'
    mask = array(mask)
    return list(zip(*[(a[mask], a[~mask]) for a in map(np.array, arrs)]))


def random_split(valid_pct: float, *arrs: NPArrayableList) -> SplitArrayList:
    "Randomly split `arrs` with `valid_pct` ratio. good for creating validation set."
    assert (valid_pct >= 0 and valid_pct <= 1), 'Validation set percentage should be between 0 and 1'
    is_train = np.random.uniform(size=(len(arrs[0]),)) > valid_pct
    return arrays_split(is_train, *arrs)


def listify(p: OptListOrItem = None, q: OptListOrItem = None):
    "Make `p` listy and the same length as `q`."
    if p is None:
        p = []
    elif isinstance(p, str):
        p = [p]
    elif not isinstance(p, Iterable):
        p = [p]
    # Rank 0 tensors in PyTorch are Iterable but don't have a length.
    else:
        try:
            a = len(p)
        except:
            p = [p]
    n = q if type(q) == int else len(p) if q is None else len(q)
    if len(p) == 1: p = p * n
    assert len(p) == n, f'List len mismatch ({len(p)} vs {n})'
    return list(p)


_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re2 = re.compile('([a-z0-9])([A-Z])')


def camel2snake(name: str) -> str:
    "Change `name` from camel to snake style."
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()


def even_mults(start: float, stop: float, n: int) -> np.ndarray:
    "Build log-stepped array from `start` to `stop` in `n` steps."
    mult = stop / start
    step = mult ** (1 / (n - 1))
    return np.array([start * (step ** i) for i in range(n)])


def extract_kwargs(names: Collection[str], kwargs: KWArgs):
    "Extract the keys in `names` from the `kwargs`."
    new_kwargs = {}
    for arg_name in names:
        if arg_name in kwargs:
            arg_val = kwargs.pop(arg_name)
            new_kwargs[arg_name] = arg_val
    return new_kwargs, kwargs


def series2cat(df: DataFrame, *col_names):
    "Categorifies the columns `col_names` in `df`."
    for c in listify(col_names): df[c] = df[c].astype('category').cat.as_ordered()


TfmList = Union[Callable, Collection[Callable]]


def recurse_eq(arr1, arr2):
    if is_listy(arr1):
        return is_listy(arr2) and len(arr1) == len(arr2) and np.all([recurse_eq(x, y) for x, y in zip(arr1, arr2)])
    else:
        return np.all(np.atleast_1d(arr1 == arr2))


def range_of(x):
    "Create a range from 0 to `len(x)`."
    return list(range(len(x)))


def arange_of(x):
    "Same as `range_of` but returns an array."
    return np.arange(len(x))


Path.ls = lambda x: list(x.iterdir())


def join_path(fname: PathOrStr, path: PathOrStr = '.') -> Path:
    "Return `Path(path)/Path(fname)`, `path` defaults to current dir."
    return Path(path) / Path(fname)


def join_paths(fnames: FilePathList, path: PathOrStr = '.') -> Collection[Path]:
    "Join `path` to every file name in `fnames`."
    path = Path(path)
    return [join_path(o, path) for o in fnames]


def loadtxt_str(path: PathOrStr) -> np.ndarray:
    "Return `ndarray` of `str` of lines of text from `path`."
    with open(path, 'r') as f: lines = f.readlines()
    return np.array([l.strip() for l in lines])


def save_texts(fname: PathOrStr, texts: Collection[str]):
    "Save in `fname` the content of `texts`."
    with open(fname, 'w') as f:
        for t in texts: f.write(f'{t}\n')


def df_names_to_idx(names: IntsOrStrs, df: DataFrame):
    "Return the column indexes of `names` in `df`."
    if not is_listy(names): names = [names]
    if isinstance(names[0], int): return names
    return [df.columns.get_loc(c) for c in names]


def index_row(a: Union[Collection, pd.DataFrame, pd.Series], idxs: Collection[int]) -> Any:
    "Return the slice of `a` corresponding to `idxs`."
    if a is None: return a
    if isinstance(a, (pd.DataFrame, pd.Series)):
        res = a.iloc[idxs]
        if isinstance(res, (pd.DataFrame, pd.Series)): return res.copy()
        return res
    return a[idxs]


def func_args(func) -> bool:
    "Return the arguments of `func`."
    code = func.__code__
    return code.co_varnames[:code.co_argcount]


def has_arg(func, arg) -> bool:
    "Check if `func` accepts `arg`."
    return arg in func_args(func)


def split_kwargs_by_func(kwargs, func):
    "Split `kwargs` between those expected by `func` and the others."
    args = func_args(func)
    func_kwargs = {a: kwargs.pop(a) for a in args if a in kwargs}
    return func_kwargs, kwargs


def array(a, dtype: type = None, **kwargs) -> np.ndarray:
    "Same as `np.array` but also handles generators. `kwargs` are passed to `np.array` with `dtype`."
    if not isinstance(a, collections.Sized) and not getattr(a, '__array_interface__', False):
        a = list(a)
    if np.int_ == np.int32 and dtype is None and is_listy(a) and len(a) and isinstance(a[0], int):
        dtype = np.int64
    return np.array(a, dtype=dtype, **kwargs)


def get_tmp_file(dir=None):
    "Create and return a tmp filename, optionally at a specific path. `os.remove` when done with it."
    with tempfile.NamedTemporaryFile(delete=False, dir=dir) as f: return f.name


def compose(funcs: List[Callable]) -> Callable:
    "Compose `funcs`"

    def compose_(funcs, x, *args, **kwargs):
        for f in listify(funcs): x = f(x, *args, **kwargs)
        return x

    return partial(compose_, funcs)


def float_or_x(x):
    "Tries to convert to float, returns x if it can't"
    try:
        return float(x)
    except:
        return x
