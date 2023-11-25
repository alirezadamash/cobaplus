from numba import njit
from typing import Any, Callable, Dict, Sequence, Tuple, Union
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch import optim

from experiment.cifar10 import ExperimentCIFAR10
from experiment.imdb import ExperimentIMDb
from experiment.mnist import ExperimentMNIST
from optimizer.adam import Adam
from optimizer.conjugate.conjugate_momentum_adam import ConjugateMomentumAdam
from optimizer.conjugate.coba import CoBA
from optimizer.conjugate.coba2 import CoBA2

Optimizer = Union[Adam, CoBA, ConjugateMomentumAdam]
OptimizerDict = Dict[str, Tuple[Any, Dict[str, Any]]]


def prepare_optimizers(lr: float, optimizer: str = None, **kwargs) -> OptimizerDict:
    types = ('HS', 'FR', 'PRP', 'DY', 'CD', 'LS')
    kw_const = dict(a=1, m=1)
    m_dict = dict(m0=1)
    a_dict = dict(a6=1+1e-6)
    optimizers = dict(
        AMSGrad_Existing=(Adam, dict(lr=lr, amsgrad=True, **kwargs)),
        Adam_Existing=(Adam, dict(lr=lr, amsgrad=False, **kwargs)),
        **{f'CoBAMSGrad2_{t}': (CoBA2, dict(lr=lr, amsgrad=True, cg_type=t)) for t in types},
        **{f'CoBAMSGrad2_{t}(const)': (CoBA2, dict(lr=lr, amsgrad=True, cg_type=t, **kw_const)) for t in types},
        AdaGrad_Existing=(optim.Adagrad, dict(lr=lr, **kwargs)),
        RMSProp_Existing=(optim.RMSprop, dict(lr=lr, **kwargs)),
    )
    return {optimizer: optimizers[optimizer]} if optimizer else optimizers


def prepare_optimizers_grid_search(lr: float, optimizer: str = None, **kwargs) -> OptimizerDict:
    types = ('HS', 'FR', 'PRP', 'DY', 'HZ')
    m_dict = dict(m2=1e-2, m3=1e-3, m4=1e-4)
    a_dict = dict(a4=1+1e-4, a5=1+1e-5, a6=1+1e-6, a7=1+1e-7)
    type_dict = dict(
        HZ=('m2', 'a4'),
        HS=('m5', 'a5'),
        FR=('m2', 'a5'),
        PRP=('m4', 'a4'),
        DY=('m3', 'a7'),
    )
    optimizers = dict(
        AMSGrad_Existing=(Adam, dict(lr=lr, amsgrad=True, **kwargs)),
        Adam_Existing=(Adam, dict(lr=lr, amsgrad=False, **kwargs)),
        **{f'CoBAMSGrad_{t}_{sm}_{sa}': (CoBA,
                                         dict(lr=lr, amsgrad=True, cg_type=t, m=m_dict[sm], a=a_dict[sa], **kwargs))
           for t, (sm, sa) in type_dict.items()},
        AdaGrad_Existing=(optim.Adagrad, dict(lr=lr, **kwargs)),
        RMSProp_Existing=(optim.RMSprop, dict(lr=lr, **kwargs)),
    )
    return {optimizer: optimizers[optimizer]} if optimizer else optimizers

@njit
def lr_warm_up(epoch: int, lr: float, t: int = 5, c: float = 1e-2):
    return ((1 - c) * epoch / t + c) * lr if epoch <= t else lr

@njit
def lr_divide(epoch: int, max_epoch: int, lr: float):
    p = epoch / max_epoch
    return lr if p < .5 else lr * 1e-1 if p < .75 else lr * 1e-2

@njit
def lr_warm_up_divide(epoch: int, max_epoch: int, lr: float, t: int = 5, c: float = 1e-2):
    return lr_warm_up(epoch, lr, t, c) if epoch <= t else lr_divide(epoch, max_epoch, lr)



def imdb(lr=1e-2, max_epoch=100, weight_decay=.0, batch_size=32, use_scheduler=False, **kwargs) -> None:
    e = ExperimentIMDb(max_epoch=max_epoch, batch_size=batch_size, **kwargs)
    e.execute(prepare_optimizers(lr=lr))


def mnist(lr=1e-3, max_epoch=100, weight_decay=.0, batch_size=32, model_name='Perceptron2', use_scheduler=False, **kwargs) -> None:
    e = ExperimentMNIST(max_epoch=max_epoch, batch_size=batch_size, model_name=model_name, scheduler=ReduceLROnPlateau if use_scheduler else None,
                        **kwargs)
    e.execute(prepare_optimizers(lr=lr))


def cifar10(max_epoch=300, lr=1e-3, weight_decay=0, batch_size=128, model_name='DenseNetBC24', num_workers=0,
            optimizer=None, use_scheduler=False, **kwargs) -> None:
    e = ExperimentCIFAR10(max_epoch=max_epoch, batch_size=batch_size, model_name=model_name,
                          kw_loader=dict(num_workers=num_workers), scheduler=LambdaLR if use_scheduler else None, kw_scheduler=dict(lr_lambda=lambda epoch: lr_warm_up(epoch, lr)), 
                          **kwargs)
    e(prepare_optimizers(lr=lr, optimizer=optimizer, weight_decay=weight_decay))


if __name__ == '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('-e', '--experiment')
    p.add_argument('-m', '--model_name')
    p.add_argument('-d', '--data_dir', default='dataset/data')
    p.add_argument('-me', '--max_epoch', type=int)
    p.add_argument('-bs', '--batch_size', type=int)
    p.add_argument('--lr', type=float)
    p.add_argument('--device')
    p.add_argument('-nw', '--num_workers', type=int)
    p.add_argument('-us', '--use_scheduler', action='store_true')
    p.add_argument('-o', '--optimizer', default=None)
    p.add_argument('-wd', '--weight_decay', default=0, type=float)
    args = p.parse_args()

    experiment = args.experiment
    kw = {k: v for k, v in vars(args).items() if k != 'experiment' and v is not None}
    print(kw)
    experiments: Dict[str, Callable] = dict(
        IMDb=imdb,
        CIFAR10=cifar10,
        MNIST=mnist
    )
    experiments[experiment](**kw)
