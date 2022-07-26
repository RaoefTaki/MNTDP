import logging
import time
from collections import OrderedDict
from functools import partial
from operator import itemgetter
from pathlib import Path

import networkx as nx
import torch
import torch.nn as nn

from ray import tune

from src.models.change_layer_llmodel import FrozenSequential
from src.models.utils import is_dummy_block, execute_step, graph_arch_details
from src.train.training import get_classic_dataloaders, train
from src.train.utils import _load_datasets
from src.utils.misc import pad_seq, get_env_url

logger = logging.getLogger(__name__)


class ExhaustiveSearch(nn.Module):
    def __init__(self, graph, tunable_modules, frozen_modules,
                 stochastic_nodes, in_node, out_node, max_new_blocks):
        super(ExhaustiveSearch, self).__init__()
        self.tunable_modules = nn.ModuleList(tunable_modules)
        self.tunable_modules.requires_grad_(True)
        self.frozen_modules = nn.ModuleList(frozen_modules)
        self.frozen_modules.requires_grad_(False)
        self.stochastic_nodes = stochastic_nodes
        self.graph = graph
        self.in_node = in_node
        self.out_node = out_node
        self.models = nn.ModuleList()
        self.models_idx = {}
        self.res = {}

        self.max_new_blocks = max_new_blocks

    def init_models(self, iteration=None):
        archs = list(nx.all_simple_paths(self.graph, self.in_node,
                                         self.out_node))
        for path in archs:
            new_model = FrozenSequential()
            last = None
            i = 0
            n_new_blocks = 0
            for node in path:
                assert node == self.in_node \
                       or node in self.graph.successors(last)
                nn_module = self.graph.nodes[node]['module']
                last = node
                if is_dummy_block(nn_module):
                    continue
                new_model.add_module(str(i), nn_module)
                if nn_module in self.frozen_modules:
                    new_model.frozen_modules_idx.append(i)
                else:
                    n_new_blocks += 1
                    # else:
                    #     nn_module.load_state_dict(self.block_inits[node])
                i += 1

            if n_new_blocks > self.max_new_blocks and len(archs) > 1:
                # print('Skipping {}'.format(path))
                continue
            # print('Adding {}'.format(path))
            new_model.n_out = self.n_out
            self.models_idx[tuple(path)] = len(self.models_idx)
            self.models.append(new_model)
        # if iteration is not None and iteration > 0:
        #     raise ValueError(archs, self.models_idx, len(self.models_idx))
        return archs

    def get_weights(self):
        weights = torch.tensor([1. if n in self._selected_path() else 0.
                                for n in self.stochastic_nodes])
        return weights

    def _selected_path(self):
        assert self.res
        all_res = map(lambda x: (x[1][2]['value'], x[0]), self.res.items())
        return max(all_res, key=itemgetter(0))[1]
        # return next(iter(self.res.keys()))

    def get_stoch_nodes(self):
        return self.stochastic_nodes

    def get_graph(self):
        return self.graph

    def nodes_to_prune(self, *args):
        return [n for n in self.stochastic_nodes
                if n not in self._selected_path()]

    # def parameters(self):
    #     if len(self.models) == 0:
    #         self.init_models()
    #     return self.models[self.models_idx[self._selected_path()]].parameters()

    def train_func(self, datasets_p, b_sizes, optim_fact, env_url, t_id, *args, **kwargs):
        # TODO: use argumenst for t_id etc to report to the tuner
        # if datasets_p['task']['data_path'][0].startswith('/net/blackorpheus/veniat/lileb/datasets/1406'):
        if datasets_p['task']['data_path'][0].startswith('/net/blackorpheus/veniat/lileb/datasets/2775'):
            kwargs['n_ep_max'] = 6
            kwargs['patience'] = 6
            p = Path('../../understood/')
            p.mkdir(parents=True, exist_ok=True)

        if not self.models:
            self.init_models(iteration=t_id)

        # raise ValueError(optim_fact)
        # ValueError: functools.partial(<function set_optim_params at 0x7f80c9f9fd30>,
        # optim_func=functools.partial(<class 'torch.optim.adam.Adam'>, weight_decay=0, lr=0.001, betas=[0.9, 0.999]),
        # optim_params=[{'architecture': 0, 'lr': 0.01, 'weight_decay': 0}], split_optims=True)

        # Create calls to train each of different models (combinations of modules), 7+1 (as in thesis), or 7 as depicted here
        # TODO: find out why discrepancy between 7+1 and 7?
        calls = []
        for path, idx in self.models_idx.items():
            model = self.models[idx]
            calls.append(partial(wrap, model=model, idx=idx,
                                 optim_fact=optim_fact, datasets_p=datasets_p,
                                 b_sizes=b_sizes, env_url=env_url, t_id=t_id, *args, **kwargs))

        # ctx = torch.multiprocessing.get_context('spawn')
        # ctx = None
        # TODO: make new branch, and make the execution of these steps here smarter, possibly using a callback or something
        # torch.multiprocessing.set_sharing_strategy('file_system')
        # raise ValueError(optim_fact, type(optim_fact), optim_fact.keywords['optim_params'][0]['architecture'], optim_fact.keywords)
        # ValueError: (functools.partial(<function set_optim_params at 0x7ef5e078a040>,
        # optim_func=functools.partial(<class 'torch.optim.adam.Adam'>, weight_decay=0, lr=0.001, betas=[0.9, 0.999]),
        # optim_params=[{'architecture': 0, 'lr': 0.01, 'weight_decay': 0}], split_optims=True),
        # <class 'functools.partial'>,
        # 0,
        # {'optim_func': functools.partial(<class 'torch.optim.adam.Adam'>, weight_decay=0, lr=0.001, betas=[0.9, 0.999]),
        # 'optim_params': [{'architecture': 0, 'lr': 0.01, 'weight_decay': 0}], 'split_optims': True})

        # raise ValueError(len(calls), len(self.models_idx), optim_fact.keywords['optim_params'][0]['architecture'])

        # Change the model ID to use depending on the ID of the task
        # At the first task, there is only 1 model so this needs to be 0 ofc
        model_id_to_use = optim_fact.keywords['optim_params'][0]['architecture'] if t_id > 0 else 0
        all_res = [calls[model_id_to_use]()]  # optim_fact.keywords['optim_params'][0]['architecture']]]
        # all_res = execute_step(calls, True, 4, ctx=ctx)
        for path, res in zip(self.models_idx.keys(), all_res):
            self.res[path] = res
        all_res = list(map(lambda x: (x[1][2]['value'], x[0]), self.res.items()))
        best_path = max(all_res, key=itemgetter(0))[1]
        _, best_metrics, best_chkpt = self.res[best_path]
        total_t = 0
        cum_best_iter = 0

        splits = ['Trainer', 'Train', 'Val', 'Test']
        metrics = ['accuracy_0']

        all_xs = []
        best_accs = list(self.res[best_path][1]['Val accuracy_0'].values())

        for i, (path, (t, logs, best)) in enumerate(self.res.items()):
            total_t += t
            cum_best_iter += best['iter']
            for m in metrics:
                for s in splits:
                    best_metrics[f'{s} {i} {m}_all'] = logs[f'{s} {m}']
                    if s == 'Val':
                        all_xs.append(list(logs[f'{s} {m}'].keys()))
        # print()
        max_len = max(map(len, all_xs))
        # print(all_xs)
        all_xs = [pad_seq(xs, max_len, xs[-1]) for xs in all_xs]
        # print(all_xs)
        # print(best_accs)
        # print(list(zip(*all_xs)))
        # print(list(zip_longest(*all_xs, fillvalue=0)))
        scaled_xs = [sum(steps) for steps in zip(*all_xs)]
        scaled_ys = pad_seq(best_accs, len(scaled_xs), best_accs[-1])
        new_metric = dict(zip(scaled_xs, scaled_ys))
        # print(new_metric)
        # print()
        best_metrics['Val accuracy_0_rescaled'] = new_metric

        # raise ValueError("best_metrics", best_metrics)
        # ValueError: ('best_metrics', defaultdict(<class 'dict'>, {'training_epoch': {0: 0.0, 1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0, 5: 2.5, 6: 3.0}, 'training_iteration': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}, 'data time': {0: 0.00167885422706604, 1: 0.09328415989875793, 2: 0.14605429023504257, 3: 0.2347925528883934, 4: 0.28458964079618454, 5: 0.3741266056895256, 6: 0.42111296206712723}, 'data time_ps': {0: 0.0016804561018943787, 1: 0.09328415989875793, 2: 0.07302714511752129, 3: 0.07826418429613113, 4: 0.07114741019904613, 5: 0.07482532113790512, 6: 0.07018549367785454}, 'forward time': {0: 0.0016821101307868958, 1: 0.12740835547447205, 2: 0.1556193083524704, 3: 0.11379259079694748, 4: 0.1357855424284935, 5: 0.11493149399757385, 6: 0.14160626381635666}, 'forward time_ps': {0: 0.0016828998923301697, 1: 0.12740835547447205, 2: 0.0778096541762352, 3: 0.11379259079694748, 4: 0.06789277121424675, 5: 0.11493149399757385, 6: 0.07080313190817833}, 'epoch time': {0: 0.001522742211818695, 1: 0.12787703424692154, 2: 0.2092898115515709, 3: 0.11408180743455887, 4: 0.18621454387903214, 5: 0.11521777510643005, 6: 0.18930372595787048}, 'epoch time_ps': {0: 0.001522742211818695, 1: 0.12787703424692154, 2: 0.2092898115515709, 3: 0.11408180743455887, 4: 0.18621454387903214, 5: 0.11521777510643005, 6: 0.18930372595787048}, 'Trainer loss': {0: nan, 1: 2.539400100708008, 2: 2.5377068519592285, 3: 2.451744318008423, 4: 2.4505931854248044, 5: 2.362548828125, 6: 2.3630423736572266}, 'Trainer accuracy_0': {0: nan, 1: 0.09765625, 2: 0.1075, 3: 0.14453125, 4: 0.135, 5: 0.16796875, 6: 0.1525}, 'Val loss': {0: 2.345941467285156, 1: 2.3436822509765625, 2: 2.3427082824707033, 3: 2.341555633544922, 4: 2.3413076782226563, 5: 2.3407740783691406, 6: 2.3393508911132814}, 'Val accuracy_0': {0: 0.1, 1: 0.1, 2: 0.11, 3: 0.075, 4: 0.09, 5: 0.1, 6: 0.105}, 'Test loss': {0: 2.3450255859375, 1: 2.3430546875, 2: 2.3424322265625, 3: 2.341641015625, 4: 2.341184765625, 5: 2.34046328125, 6: 2.338477734375}, 'Test accuracy_0': {0: 0.1041, 1: 0.0989, 2: 0.0999, 3: 0.1006, 4: 0.1, 5: 0.0996, 6: 0.0999}, 'eval time': {0: 0.7357497364282608, 1: 1.44418216496706, 2: 2.1067687794566154, 3: 2.7596073523163795, 4: 3.528386816382408, 5: 4.188270099461079, 6: 4.843980059027672}, 'eval time_ps': {0: 0.7357497364282608, 1: 0.72209108248353, 2: 0.7022562598188719, 3: 0.6899018380790949, 4: 0.7056773632764817, 5: 0.6980450165768465, 6: 0.6919971512896674}, 'total time': {0: 0.7374326065182686, 1: 1.5740334391593933, 2: 2.318043552339077, 3: 3.0850270241498947, 4: 3.925952062010765, 5: 4.7011095359921455, 6: 5.4309166595339775}, 'Trainer 0 accuracy_0_all': {0: nan, 1: 0.09765625, 2: 0.1075, 3: 0.14453125, 4: 0.135, 5: 0.16796875, 6: 0.1525}, 'Train accuracy_0': {}, 'Train 0 accuracy_0_all': {}, 'Val 0 accuracy_0_all': {0: 0.1, 1: 0.1, 2: 0.11, 3: 0.075, 4: 0.09, 5: 0.1, 6: 0.105}, 'Test 0 accuracy_0_all': {0: 0.1041, 1: 0.0989, 2: 0.0999, 3: 0.1006, 4: 0.1, 5: 0.0996, 6: 0.0999}, 'Val accuracy_0_rescaled': {0: 0.1, 1: 0.1, 2: 0.11, 3: 0.075, 4: 0.09, 5: 0.1, 6: 0.105}}))

        #  TODO: try disable to see if reported values are better now
        # tune.report(t=t_id,
        #             best_val=max(best_accs),
        #             avg_acc_val=-1,
        #             avg_acc_val_so_far=-1,
        #             avg_acc_test_so_far=-1,
        #             lca=-1,
        #             avg_acc_test=-1,
        #             test_acc=-1,
        #             duration_seconds=-1,
        #             duration_iterations=-1,
        #             duration_best_it=-1,
        #             duration_finish=-1,
        #             duration_model_creation=-1,
        #             duration_training=-1,
        #             duration_postproc=-1,
        #             duration_eval=-1,
        #             duration_sum=-1,
        #             iterations=-1,
        #             epochs=-1,
        #             # entropy=stats.pop('entropy'),
        #             new_params=-1,
        #             total_params=-1,
        #             total_steps=-1,
        #             fw_t=-1,
        #             data_t=-1,
        #             epoch_t=-1,
        #             eval_t=-1,
        #             total_t=-1,
        #             env_url=env_url,
        #             info_training=None)

        self.models[self.models_idx[best_path]].load_state_dict(best_chkpt['state_dict'])
        best_chkpt['cum_best_iter'] = cum_best_iter
        return total_t, best_metrics, best_chkpt, (
            self.models, len(self.models), len(self.models_idx.items()), self.models_idx.items())

    def forward(self, input):
        if not self.models:
            self.init_models()
        # Assert that we are at test time
        assert len(self.models) == 1, len(self.models)
        return self.models[0](input)

    def get_frozen_model(self):
        if not self.models:
            self.init_models()
        assert len(self.models) == 1, len(self.models)
        return self.models[0]

    def get_top_archs(self, n=1):
        res = list(map(lambda x: (x[0], x[1][2]['value']), self.res.items()))
        res = sorted(res, key=itemgetter(1), reverse=True)
        return OrderedDict(res[:n])

    def arch_repr(self):
        return graph_arch_details(self.graph)


def wrap(*args, idx=None, uid=None, optim_fact, datasets_p, b_sizes, env_url=None, t_id=-1, **kwargs):
    # TODO: somehow it doesn't enter this function the second time round. Idk why
    # if t_id != 0:
    #     raise ValueError("INTERCEPT. t_id:", t_id)

    model = kwargs['model']
    optim = optim_fact(model=model)
    datasets = _load_datasets(**datasets_p)
    train_loader, eval_loaders = get_classic_dataloaders(datasets, b_sizes, 0)
    if hasattr(model, 'train_loader_wrapper'):
        train_loader = model.train_loader_wrapper(train_loader)

    res = train(*args, train_loader=train_loader, eval_loaders=eval_loaders,
                optimizer=optim, env_url=env_url, t_id=t_id, **kwargs)
    # logger.warning('{}=Received option {} results'.format(uid, idx))
    return res
