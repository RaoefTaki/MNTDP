import logging
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from copy import deepcopy
from functools import partial
from os import path

import numpy as np
import pandas
import ray
import torch
import visdom
from ray import tune
from ray.tune import CLIReporter
from ray.tune.logger import JsonLogger, CSVLogger
import ray.tune.utils
from ray.tune.schedulers import ASHAScheduler
from torchvision.transforms import transforms

from src.datasets.TensorDataset import MyTensorDataset
from src.experiments.base_experiment import BaseExperiment
from src.models.ExhaustiveSearch import ExhaustiveSearch
from src.models.utils import execute_step
from src.train.ignite_utils import _prepare_batch
from src.train.training import train, get_classic_dataloaders
from src.train.utils import set_dropout, set_optim_params, \
    _load_datasets, evaluate_on_tasks, evaluate
from src.utils.log_observer import initialize_tune_report_arguments
from src.utils.memory_buffer import MemoryBuffer
from src.utils.misc import get_env_url, fill_matrix, \
    get_training_vis_conf
from src.utils.plotting import update_summary, plot_tasks_env_urls, \
    plot_heatmaps, \
    plot_trajectory, list_top_archs, process_final_results

visdom.logger.setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

IMAGES_PER_MB = (1000000/(3*32*32*4))  # The number of images that fit in one MB of memory. To be rounded down after possible multiplication of the number of MBs used

class StreamTuningExperiment(BaseExperiment):
    def run(self):
        if self.task_gen.concept_pool.attribute_similarities is not None:
            attr_sim = self.task_gen.concept_pool.attribute_similarities
            self.main_viz.heatmap(attr_sim,
                                  opts={'title': 'Attribute similarities'})
        if self.plot_tasks:
            self.task_gen.concept_pool.draw_tree(viz=self.main_viz,
                                                 title='Full tree')
            self.task_gen.concept_pool.draw_attrs(viz=self.main_viz)
            self.task_gen.concept_pool.plot_concepts(self.main_viz)

        self.init_tasks()
        self.init_sims()
        self.clean_tasks()
        if not self.stream_setting:
            self.init_models(True)
        else:
            details = self.init_models(False)
            logger.info('Architecture details for the first models:')
            for learner, det in details.items():
                logger.info(f'{learner}: {det} ({sum(det.values())}, '
                            f'{4 * sum(det.values()) / 1e6})')
        self.init_plots()

        logger.info("General dashboard: {}".format(get_env_url(self.main_viz)))
        logger.info('Tasks: {}'.format(get_env_url(self.task_env)))
        # if self.use_ray and not self.use_processes:
        #     if self.redis_address and not self.local_mode:
        #         ray.init(redis_address=self.redis_address)
        #     else:
        #         logger.warning('Launching a new ray cluster')
        #         ray.init(object_store_memory=int(1e7), include_webui=True,
        #                  local_mode=self.local_mode, num_gpus=0)

        # Since there is only 1 model in runs that we consider, we run the training call without parallel processing
        train_calls = []
        for model_name, ll_model in self.ll_models.items():
            vis_params = [vis_params[model_name]
                          for vis_params in self.training_envs]
            params = dict(learner=ll_model,
                          stream=self.task_gen.stream_infos(True),
                          task_level_tuning=self.val_per_task,
                          learner_name=model_name,
                          # exp_name=self.exp_name,
                          vis_params=vis_params,
                          plot_all=self.plot_all,
                          batch_sizes=self.batch_sizes,
                          n_it_max=self.n_it_max,
                          n_ep_max=self.n_ep_max,
                          augment_data=self.augment_data,
                          normalize=self.normalize,
                          schedule_mode=self.schedule_mode,
                          patience=self.patience,
                          # grace_period=self.grace_period,
                          num_hp_samplings=self.num_hp_samplings,
                          device=self.device,
                          log_steps=self.log_steps,
                          log_epoch=self.log_epoch,
                          exp_dir=self.exp_dir,
                          lca=self.lca,
                          single_pass=self.single_pass,
                          stream_setting=self.stream_setting,
                          split_optims=self.split_optims,
                          # use_ray=self.use_ray,
                          # use_ray_logging=self.use_ray_logging,
                          local_mode=self.local_mode,
                          redis_address=self.redis_address,
                          seed=self.seed
                          )
            train_calls.append(partial(tune_learner_on_stream, **params))

        # Perform the training and obtain the results
        res = {'PSSN-search-6-fw': train_calls[0]()}

        # results_array = execute_step(train_calls, self.use_processes, ctx=ctx)
        # res = dict(zip(self.ll_models.keys(), results_array)) # TODO: somehow here extra lines are included from tune_report functions

        summ = process_final_results(self.main_viz, res, self.exp_name,
                                     self.visdom_conf, self.task_envs_str,
                                     len(self.task_gen.task_pool),
                                     self.best_task_envs_str, self.val_per_task,
                                     self.visdom_traces_folder)

        plot_tasks_env_urls(self.task_envs_str, self.main_viz, 'all')
        plot_tasks_env_urls(self.best_task_envs_str, self.main_viz, 'best')
        self.save_traces()

        res_py = {k: [itm.to_dict('list') for itm in v] for k, v in res.items()}
        # res_2 = {k: [pandas.DataFrame(itm) for itm in v] for k, v in res_py.items()}

        # for (k1, v1), (k2, v2) in zip(res.items(), res_2.items()):
        # assert k1 == k2
        # print([i1.equals(i2) for i1, i2 in zip(v1, v2)])
        logger.info(f'Args {" ".join(sys.argv[2:])}')
        print(pandas.DataFrame(summ).set_index('model'))
        return [res_py, self.task_gen.stream_infos(full=False)]


def tune_learner_on_stream(learner, learner_name, task_level_tuning,
                           stream, redis_address, local_mode, num_hp_samplings,
                           vis_params, exp_dir, seed, **training_params):
    """
    Returns 2 dataframes:
     - The first one contains information about the best trajectory and
     contains as many rows as there are tasks. Each row corresponding to the
     model trained on the corresponding task in the best trajectory.
      - The second contains one row per hyper-parameters combination. Each
      Row corresponds contains information about the results on all tasks for
      this specific hp combination. Note that, *in the task-level hp optim
      settting*, this DF is useful to investigate the behaviors of specific
      trainings, but rows *DOES NOT* correspond to actual trajectories.
    """

    exp_name = os.path.basename(exp_dir)
    init_path = path.join(exp_dir, 'model_initializations', learner_name)
    torch.save(learner, init_path)
    config = {**learner.get_search_space(),
              'training-params': training_params,
              'tasks': stream,
              'vis_params': vis_params,
              # 'learner': learner,
              'learner_path': init_path,
              'task_level_tuning': task_level_tuning,
              # 'env': learner_name
              'seed': seed
              }

    def trial_name_creator(trial):
        return learner_name
        # return '{}_{}'.format(learner_name, trial.trial_id)

    reporter = CLIReporter(max_progress_rows=10, max_report_frequency=60)
    # reporter.add_metric_column('avg_acc_val')
    reporter.add_metric_column('avg_acc_val_so_far', 'avg_val')
    reporter.add_metric_column('avg_acc_test_so_far', 'avg_test')
    reporter.add_metric_column('total_params')
    # reporter.add_metric_column('fw_t')
    # reporter.add_metric_column('data_t')
    # reporter.add_metric_column('eval_t')
    # reporter.add_metric_column('epoch_t')
    reporter.add_metric_column('duration_model_creation', 'creat_t')
    reporter.add_metric_column('duration_training', 'train_t')
    reporter.add_metric_column('duration_postproc', 'pp_t')
    reporter.add_metric_column('duration_finish', 'fin_t')
    reporter.add_metric_column('duration_eval', 'ev_t')
    reporter.add_metric_column('duration_sum', 'sum_t')
    reporter.add_metric_column('duration_seconds', 'step_t')
    reporter.add_metric_column('total_t')
    reporter.add_metric_column('t')

    ray_params = dict(
        loggers=[JsonLogger, CSVLogger],
        name=learner_name,
        resources_per_trial=learner.ray_resources,
        num_samples=num_hp_samplings,
        local_dir=exp_dir,
        verbose=1,
        progress_reporter=reporter,
        trial_name_creator=trial_name_creator,
        max_failures=3,
    )
    envs = []
    all_val_accs = defaultdict(list)
    all_test_accs = defaultdict(list)
    total_iterations_so_far_per_task = []  # The number of iterations in total conducted, so far, for each encountered task
    if task_level_tuning:
        best_trials_df = []
        config['ray_params'] = ray_params
        config['local_mode'] = local_mode
        config['redis_address'] = redis_address
        # Call the training etc. function
        analysis, selected, total_iterations_so_far_per_task = train_on_tasks(config)
        for t_id, (task, task_an) in enumerate(zip(stream, analysis)):
            # envs.append([])
            for trial_n, t in enumerate(task_an.trials):
                if len(envs) <= trial_n:
                    envs.append([])
                env = '{}_Trial_{}_{}_{}'.format(exp_name, t, t.experiment_tag,
                                                 task['descriptor'])
                envs[trial_n].append(env)
                if selected[t_id] == t.experiment_tag:
                    all_val_accs[t.experiment_tag].append(
                        '<span style="font-weight:bold">{}</span>'.format(
                            t.last_result[f'Val_T{t_id}']))
                else:
                    all_val_accs[t.experiment_tag].append(
                        t.last_result[f'Val_T{t_id}'])
                all_test_accs[t.experiment_tag].append(
                    t.last_result[f'Test_T{t_id}']
                )

            # Accommodate for trials which do not have a path, i.e. filter these out, since these were stopped by a
            # scheduler
            best_trial = max(task_an.trials,
                             key=lambda trial: (trial.last_result['path'] != -1, trial.last_result['avg_acc_val_so_far']))

            df = task_an.trial_dataframes[best_trial.logdir]
            best_trials_df.append(df)

        return_df = pandas.concat(best_trials_df, ignore_index=True)
        analysis = analysis[-1]
        results = sorted(analysis.trials, reverse=True,
                         key=lambda trial: (trial.last_result['path'] != -1, trial.last_result['avg_acc_val_so_far']))
    else:
        if not ray.is_initialized():
            if local_mode:
                ray.init(local_mode=local_mode)
            else:
                ray.init(redis_address)
                # logging_level=logging.DEBUG)
        ray_params['config'] = config
        analysis = tune.run(train_on_tasks, **ray_params)

        results = sorted(analysis.trials, reverse=True,
                         key=lambda trial: trial.last_result['avg_acc_val_so_far'])
        for t in results:
            envs.append([])
            for task in stream:
                env = '{}_Trial_{}_{}_{}'.format(exp_name, t, t.experiment_tag,
                                                 task['descriptor'])
                envs[-1].append(env)
        return_df = analysis.trial_dataframes[results[0].logdir]
    # Get only the last one per task t in case there are multiple:
    return_df = return_df.groupby('t').tail(1).reset_index()
    return_df['duration_iterations'] = total_iterations_so_far_per_task

    summary = {
        'model': [t.experiment_tag for t in results],
        'Avg acc Val': [t.last_result['avg_acc_val'] for t in results],
        'Acc Val': [all_val_accs[t.experiment_tag] for t in results],
        'Avg acc Test': [t.last_result['avg_acc_test'] for t in results],
        'Acc Test': [all_test_accs[t.experiment_tag] for t in results],
        'Params': [t.last_result['total_params'] for t in results],
        'Steps': [total_iterations_so_far_per_task[len(total_iterations_so_far_per_task)-1] for t in results], #[t.last_result['total_steps'] for t in results], # TODO: make sure this is correct in the Visdom 'updates' too
        'paths': [t.logdir for t in results],
        'evaluated_params': [t.evaluated_params for t in results],
        'envs': envs
    }
    summary = pandas.DataFrame(summary)
    # pandas.set_option('display.max_colwidth', None)
    # raise ValueError("return_df:", return_df, "return_df.columns:", return_df.columns, "len(return_df.index):", len(return_df.index),
    #                  "summary:", summary, "summary.columns:", summary.columns, "len(summary.index):", len(summary.index))
    #     [12 rows x 45 columns], 'return_df.columns:', Index(['t', 'best_val', 'avg_acc_val', 'avg_acc_val_so_far',
    #                                                          'avg_acc_test_so_far', 'lca', 'avg_acc_test', 'test_acc',
    #                                                          'duration_seconds', 'duration_iterations', 'duration_best_it',
    #                                                          'duration_finish', 'duration_model_creation', 'duration_training',
    #                                                          'duration_postproc', 'duration_eval', 'duration_sum', 'iterations',
    #                                                          'epochs', 'new_params', 'total_params', 'total_steps', 'fw_t', 'data_t',
    #                                                          'epoch_t', 'eval_t', 'total_t', 'env_url', 'info_training',
    #                                                          'time_this_iter_s', 'done', 'timesteps_total', 'episodes_total',
    #                                                          'training_iteration', 'trial_id', 'experiment_id', 'date', 'timestamp',
    #                                                          'time_total_s', 'pid', 'hostname', 'node_ip', 'time_since_restore',
    #                                                          'timesteps_since_restore', 'iterations_since_restore'],
    #                                                         dtype='object'), 'len(return_df.index):', 12, 'summary:',                                 model  ...                                                                                                                                                                                                                                                                                                                                                                                                                envs
    # 0    4_0_lr=0.01,0_weight_decay=1e-05  ...                                      [1_Trial_PSSN-search-6-fw_0_0_lr=0.01,0_weight_decay=0_md-T0, 1_Trial_PSSN-search-6-fw_0_0_lr=0.01,0_weight_decay=0_md-T1, 1_Trial_PSSN-search-6-fw_0_0_lr=0.01,0_weight_decay=0_md-T2, 1_Trial_PSSN-search-6-fw_0_0_lr=0.01,0_weight_decay=0_md-T3, 1_Trial_PSSN-search-6-fw_0_0_lr=0.01,0_weight_decay=0_md-T4, 1_Trial_PSSN-search-6-fw_0_0_lr=0.01,0_weight_decay=0_md-T5]
    # 1   2_0_lr=0.01,0_weight_decay=0.0001  ...                                [1_Trial_PSSN-search-6-fw_1_0_lr=0.001,0_weight_decay=0_md-T0, 1_Trial_PSSN-search-6-fw_1_0_lr=0.001,0_weight_decay=0_md-T1, 1_Trial_PSSN-search-6-fw_1_0_lr=0.001,0_weight_decay=0_md-T2, 1_Trial_PSSN-search-6-fw_1_0_lr=0.001,0_weight_decay=0_md-T3, 1_Trial_PSSN-search-6-fw_1_0_lr=0.001,0_weight_decay=0_md-T4, 1_Trial_PSSN-search-6-fw_1_0_lr=0.001,0_weight_decay=0_md-T5]
    # 2        0_0_lr=0.01,0_weight_decay=0  ...        [1_Trial_PSSN-search-6-fw_2_0_lr=0.01,0_weight_decay=0.0001_md-T0, 1_Trial_PSSN-search-6-fw_2_0_lr=0.01,0_weight_decay=0.0001_md-T1, 1_Trial_PSSN-search-6-fw_2_0_lr=0.01,0_weight_decay=0.0001_md-T2, 1_Trial_PSSN-search-6-fw_2_0_lr=0.01,0_weight_decay=0.0001_md-T3, 1_Trial_PSSN-search-6-fw_2_0_lr=0.01,0_weight_decay=0.0001_md-T4, 1_Trial_PSSN-search-6-fw_2_0_lr=0.01,0_weight_decay=0.0001_md-T5]
    # 3   5_0_lr=0.001,0_weight_decay=1e-05  ...  [1_Trial_PSSN-search-6-fw_3_0_lr=0.001,0_weight_decay=0.0001_md-T0, 1_Trial_PSSN-search-6-fw_3_0_lr=0.001,0_weight_decay=0.0001_md-T1, 1_Trial_PSSN-search-6-fw_3_0_lr=0.001,0_weight_decay=0.0001_md-T2, 1_Trial_PSSN-search-6-fw_3_0_lr=0.001,0_weight_decay=0.0001_md-T3, 1_Trial_PSSN-search-6-fw_3_0_lr=0.001,0_weight_decay=0.0001_md-T4, 1_Trial_PSSN-search-6-fw_3_0_lr=0.001,0_weight_decay=0.0001_md-T5]
    # 4  3_0_lr=0.001,0_weight_decay=0.0001  ...              [1_Trial_PSSN-search-6-fw_4_0_lr=0.01,0_weight_decay=1e-05_md-T0, 1_Trial_PSSN-search-6-fw_4_0_lr=0.01,0_weight_decay=1e-05_md-T1, 1_Trial_PSSN-search-6-fw_4_0_lr=0.01,0_weight_decay=1e-05_md-T2, 1_Trial_PSSN-search-6-fw_4_0_lr=0.01,0_weight_decay=1e-05_md-T3, 1_Trial_PSSN-search-6-fw_4_0_lr=0.01,0_weight_decay=1e-05_md-T4, 1_Trial_PSSN-search-6-fw_4_0_lr=0.01,0_weight_decay=1e-05_md-T5]
    # 5       1_0_lr=0.001,0_weight_decay=0  ...        [1_Trial_PSSN-search-6-fw_5_0_lr=0.001,0_weight_decay=1e-05_md-T0, 1_Trial_PSSN-search-6-fw_5_0_lr=0.001,0_weight_decay=1e-05_md-T1, 1_Trial_PSSN-search-6-fw_5_0_lr=0.001,0_weight_decay=1e-05_md-T2, 1_Trial_PSSN-search-6-fw_5_0_lr=0.001,0_weight_decay=1e-05_md-T3, 1_Trial_PSSN-search-6-fw_5_0_lr=0.001,0_weight_decay=1e-05_md-T4, 1_Trial_PSSN-search-6-fw_5_0_lr=0.001,0_weight_decay=1e-05_md-T5]
    #
    # [6 rows x 10 columns], 'summary.columns:', Index(['model', 'Avg acc Val', 'Acc Val', 'Avg acc Test', 'Acc Test', 'Params',
    #                                                   'Steps', 'paths', 'evaluated_params', 'envs'],
    #                                                  dtype='object'), 'len(summary.index):', 6)
    # print("Summary results:")
    # print(summary)

    return return_df, summary


def train_on_tasks(config):
    """Config can either be the sampled configuration given by ray during a run
    or all the parameters including thos to pass to ray under the 'ray_config'
    key"""
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    print("[TEST] Start training")

    tasks = config.pop('tasks')

    task_vis_params = config.pop('vis_params')

    # all_stats = []
    transfer_matrix = defaultdict(list)
    total_steps = 0

    total_iterations_so_far_per_task = []

    if 'learner' in config:
        learner = config.pop('learner')
    else:
        learner_path = config.pop('learner_path')
        learner = torch.load(learner_path)
    task_level_tuning = config.pop('task_level_tuning')
    if task_level_tuning:
        ray_params = config.pop('ray_params')
        local_mode = config.pop('local_mode')
        redis_address = config.pop('redis_address')
        all_analysis = []
        selected_tags = []

    # # TODO: try to see if data sample saving is doable
    number_of_MB_in_memory = 5
    memory_size = math.floor(IMAGES_PER_MB * number_of_MB_in_memory)
    memory_buffer = MemoryBuffer(memory_size=memory_size)

    task_counter = 0
    for t_id, (task, vis_p) in enumerate(zip(tasks, task_vis_params)):
        # todo sync transfer matrix
        static_params = dict(
            t_id=t_id, task=task, tasks=tasks, vis_p=vis_p,
            transfer_matrix=transfer_matrix, total_steps=total_steps
        )

        print("[TEST] Current task:", t_id)

        if task_level_tuning:
            if not ray.is_initialized():
                if local_mode:
                    ray.init(local_mode=local_mode)
                else:
                    ray.init(redis_address,
                             log_to_driver=False,
                             logging_level=logging.ERROR)

            config['static_params'] = static_params
            config['learner_path'] = learner_path
            config['seed'] += t_id

            # raise ValueError("config:", config)
            # ValueError: ('config:', {'learner-params': {'arch_loss_coef': 0, 'entropy_coef': 0, 'split_training': True},
            # 'optim': [{'lr': {'grid_search': [0.01, 0.001]}, 'weight_decay': {'grid_search': [0, 0.0001, 1e-05]}}],
            # 'training-params': {'plot_all': True, 'batch_sizes': [256, 1024], 'n_it_max': None, 'n_ep_max': 3,
            # 'augment_data': True, 'normalize': True, 'schedule_mode': None, 'patience': 300, 'device': 'cuda',
            # 'log_steps': [1, 2, 3, 4, 5, 10], 'log_epoch': True, 'lca': 5, 'single_pass': False, 'stream_setting': True,
            # 'split_optims': True}, 'seed': 212952927, 'tune_report': <function report at 0x7f1511059ca0>,
            # 'static_params': {'t_id': 0, 'task': {'data_path': ['/home/TUE/s167139/data/veniat/lileb/datasets/1/md-T0_train.pth',
            # '/home/TUE/s167139/data/veniat/lileb/datasets/1/md-T0_val.pth',
            # '/home/TUE/s167139/data/veniat/lileb/datasets/1/md-T0_test.pth'],
            # 'split_names': ['Train', 'Val', 'Test'], 'id': 0, 'x_dim': [3, 32, 32], 'n_classes': [10], 'descriptor': 'md-T0',
            # 'full_descr': "10-way classification created by Mixed<['split', 'data']> ([400, 200, 10000] samples): \n\t identity->Id \n\t-('cifar10 frog',)\n\t-('cifar10 cat',)\n\t-('cifar10 dog',)\n\t-('cifar10 deer',)\n\t-('cifar10 truck',)\n\t-('cifar10 ship',)\n\t-('cifar10 airplane',)\n\t-('cifar10 horse',)\n\t-('cifar10 automobile',)\n\t-('cifar10 bird',)", 'loss_fn': <function loss at 0x7f142810bf70>, 'statistics': {'mean': [tensor(0.4904), tensor(0.4801), tensor(0.4415)], 'std': [tensor(0.2467), tensor(0.2408), tensor(0.2599)]}}, 'tasks': [{'data_path': ['/home/TUE/s167139/data/veniat/lileb/datasets/1/md-T0_train.pth', '/home/TUE/s167139/data/veniat/lileb/datasets/1/md-T0_val.pth', '/home/TUE/s167139/data/veniat/lileb/datasets/1/md-T0_test.pth'], 'split_names': ['Train', 'Val', 'Test'], 'id': 0, 'x_dim': [3, 32, 32], 'n_classes': [10], 'descriptor': 'md-T0', 'full_descr': "10-way classification created by Mixed<['split', 'data']> ([400, 200, 10000] samples): \n\t identity->Id \n\t-('cifar10 frog',)\n\t-('cifar10 cat',)\n\t-('cifar10 dog',)\n\t-('cifar10 deer',)\n\t-('cifar10 truck',)\n\t-('cifar10 ship',)\n\t-('cifar10 airplane',)\n\t-('cifar10 horse',)\n\t-('cifar10 automobile',)\n\t-('cifar10 bird',)", 'loss_fn': <function loss at 0x7f142810bf70>, 'statistics': {'mean': [tensor(0.4904), tensor(0.4801), tensor(0.4415)], 'std': [tensor(0.2467), tensor(0.2408), tensor(0.2599)]}}, {'data_path': ['/home/TUE/s167139/data/veniat/lileb/datasets/1/md-T1_train.pth', '/home/TUE/s167139/data/veniat/lileb/datasets/1/md-T1_val.pth', '/home/TUE/s167139/data/veniat/lileb/datasets/1/md-T1_test.pth'], 'split_names': ['Train', 'Val', 'Test'], 'id': 1, 'x_dim': [3, 32, 32], 'n_classes': [10], 'descriptor': 'md-T1', 'full_descr': "10-way classification created by Mixed<['split', 'data']> ([400, 200, 9786] samples): \n\t identity->Id \n\t-('mnist 7 - seven',)\n\t-('mnist 4 - four',)\n\t-('mnist 6 - six',)\n\t-('mnist 0 - zero',)\n\t-('mnist 3 - three',)\n\t-('mnist 5 - five',)\n\t-('mnist 1 - one',)\n\t-('mnist 8 - eight',)\n\t-('mnist 2 - two',)\n\t-('mnist 9 - nine',)", 'loss_fn': <function loss at 0x7f142810bf70>, 'statistics': {'mean': [tensor(0.1334), tensor(0.1334), tensor(0.1334)], 'std': [tensor(0.2913), tensor(0.2913), tensor(0.2913)]}}, {'data_path': ['/home/TUE/s167139/data/veniat/lileb/datasets/1/md-T2_train.pth', '/home/TUE/s167139/data/veniat/lileb/datasets/1/md-T2_val.pth', '/home/TUE/s167139/data/veniat/lileb/datasets/1/md-T2_test.pth'], 'split_names': ['Train', 'Val', 'Test'], 'id': 2, 'x_dim': [3, 32, 32], 'n_classes': [10], 'descriptor': 'md-T2', 'full_descr': "10-way classification created by Mixed<['split', 'data']> ([400, 200, 400] samples): \n\t identity->Id \n\t-('dtd lined',)\n\t-('dtd frilly',)\n\t-('dtd meshed',)\n\t-('dtd smeared',)\n\t-('dtd striped',)\n\t-('dtd matted',)\n\t-('dtd studded',)\n\t-('dtd woven',)\n\t-('dtd freckled',)\n\t-('dtd blotchy',)", 'loss_fn': <function loss at 0x7f142810bf70>, 'statistics': {'mean': [tensor(0.5172), tensor(0.4563), tensor(0.4023)], 'std': [tensor(0.2501), tensor(0.2302), tensor(0.2420)]}}, {'data_path': ['/home/TUE/s167139/data/veniat/lileb/datasets/1/md-T3_train.pth', '/home/TUE/s167139/data/veniat/lileb/datasets/1/md-T3_val.pth', '/home/TUE/s167139/data/veniat/lileb/datasets/1/md-T3_test.pth'], 'split_names': ['Train', 'Val', 'Test'], 'id': 3, 'x_dim': [3, 32, 32], 'n_classes': [10], 'descriptor': 'md-T3', 'full_descr': "10-way classification created by Mixed<['split', 'data']> ([400, 200, 10000] samples): \n\t identity->Id \n\t-('fashion-mnist Bag',)\n\t-('fashion-mnist Ankle boot',)\n\t-('fashion-mnist Sandal',)\n\t-('fashion-mnist Dress',)\n\t-('fashion-mnist T-shirt/top',)\n\t-('fashion-mnist Pullover',)\n\t-('fashion-mnist Trouser',)\n\t-('fashion-mnist Sneaker',)\n\t-('fashion-mnist Coat',)\n\t-('fashion-mnist Shirt',)", 'loss_fn': <function loss at 0x7f142810bf70>, 'statistics': {'mean': [tensor(0.2806), tensor(0.2806), tensor(0.2806)], 'std': [tensor(0.3335), tensor(0.3335), tensor(0.3335)]}}, {'data_path': ['/home/TUE/s167139/data/veniat/lileb/datasets/1/md-T4_train.pth', '/home/TUE/s167139/data/veniat/lileb/datasets/1/md-T4_val.pth', '/home/TUE/s167139/data/veniat/lileb/datasets/1/md-T4_test.pth'], 'split_names': ['Train', 'Val', 'Test'], 'id': 4, 'x_dim': [3, 32, 32], 'n_classes': [10], 'descriptor': 'md-T4', 'full_descr': "10-way classification created by Mixed<['split', 'data']> ([400, 200, 10000] samples): \n\t identity->Id \n\t-('svhn 9 - nine',)\n\t-('svhn 0 - zero',)\n\t-('svhn 7 - seven',)\n\t-('svhn 4 - four',)\n\t-('svhn 5 - five',)\n\t-('svhn 3 - three',)\n\t-('svhn 8 - eight',)\n\t-('svhn 6 - six',)\n\t-('svhn 1 - one',)\n\t-('svhn 2 - two',)", 'loss_fn': <function loss at 0x7f142810bf70>, 'statistics': {'mean': [tensor(0.4580), tensor(0.4634), tensor(0.4897)], 'std': [tensor(0.2029), tensor(0.2061), tensor(0.2003)]}}, {'data_path': ['/home/TUE/s167139/data/veniat/lileb/datasets/1/md-T5_train.pth', '/home/TUE/s167139/data/veniat/lileb/datasets/1/md-T5_val.pth', '/home/TUE/s167139/data/veniat/lileb/datasets/1/md-T5_test.pth'], 'split_names': ['Train', 'Val', 'Test'], 'id': 5, 'x_dim': [3, 32, 32], 'n_classes': [10], 'descriptor': 'md-T5', 'full_descr': "10-way classification created by Mixed<['split', 'data']> ([4000, 2000, 10000] samples): \n\t identity->Id \n\t-('cifar10 frog',)\n\t-('cifar10 cat',)\n\t-('cifar10 dog',)\n\t-('cifar10 deer',)\n\t-('cifar10 truck',)\n\t-('cifar10 ship',)\n\t-('cifar10 airplane',)\n\t-('cifar10 horse',)\n\t-('cifar10 automobile',)\n\t-('cifar10 bird',)", 'loss_fn': <function loss at 0x7f142810bf70>, 'statistics': {'mean': [tensor(0.4903), tensor(0.4824), tensor(0.4472)], 'std': [tensor(0.2471), tensor(0.2424), tensor(0.2607)]}}], 'vis_p': {'env': '1_PSSN-search-6-fw_md-T0', 'log_to_filename': '/home/TUE/s167139/data/veniat/lileb/visdom_traces/1/1_PSSN-search-6-fw_md-T0', 'server': 'localhost', 'port': 8097, 'offline': True}, 'transfer_matrix': defaultdict(<class 'list'>, {}), 'total_steps': 0},
            # 'learner_path': '/home/TUE/s167139/data/veniat/lileb/ray_results/1/model_initializations/PSSN-search-6-fw'})

            # Perform Ray HPO for 3 criteria: learning rate, weight decay, architecture (7+1 possibilities)
            # First define the possibilities for each criteria
            nr_of_architectures = 7 if t_id > 0 else 1  # TODO: CURRENTLY RANDOMLY PICKING AN ARCHITECTURE TO USE, JUST TO TEST IT ON VM
            config['optim'] = [{'architecture': {'grid_search': [random.randint(0, nr_of_architectures-1)]}, 'lr': {'grid_search': [0.01]}, 'weight_decay': {'grid_search': [0]}}]
            # config['optim'] = [{'architecture': {'grid_search': list(range(nr_of_architectures))}, 'lr': config['optim'][0]['lr'], 'weight_decay': config['optim'][0]['weight_decay']}]
            # 'optim' [{'architecture': {'grid_search': [0, 1, 2, 3, 4, 5, 6]}, 'lr': {'grid_search': [0.01, 0.001]}, 'weight_decay': {'grid_search': [0, 0.0001, 1e-05]}}]

            # Next define the amount of parallelism, as per the original MNTDP program
            division_factor_per_gpu = 4.0
            config['division_factor_per_gpu'] = division_factor_per_gpu
            ray_params['resources_per_trial'] = {'cpu': 0.5, 'gpu': 0}  # TODO: somehow this changes per run if set dynamically?
            # raise ValueError(ray_params)
            # ValueError: {'loggers': [<class 'ray.tune.logger.JsonLogger'>, <class 'ray.tune.logger.CSVLogger'>],
            # 'name': 'PSSN-search-6-fw', 'resources_per_trial': {'cpu': 0.5, 'gpu': 0.25}, 'num_samples': 1,
            # 'local_dir': '/home/TUE/s167139/data/veniat/lileb/ray_results/1', 'verbose': 1,
            # 'progress_reporter': <ray.tune.progress_reporter.CLIReporter object at 0x7fcceeca2460>,
            # 'trial_name_creator': <function tune_learner_on_stream.<locals>.trial_name_creator at 0x7fcceee28f70>,
            # 'max_failures': 3}

            # Define the scheduler for ASHA
            # asha_scheduler = ASHAScheduler(
            #     time_attr='epoch_of_report_T' + str(t_id),
            #     metric='best_val_T' + str(t_id),
            #     mode='max',
            #     max_t=config['training-params']['n_ep_max'] + 1,  # Represents infinity; will never be reached in MNTDP
            #     grace_period=10,
            #     reduction_factor=3,
            #     brackets=1)

            analysis = tune.run(train_t, config=config, **ray_params)  # scheduler=asha_scheduler,
            all_analysis.append(analysis)

            # TODO: consider only using tune_report in the same location, i.e. in the train script. See if that changes things perhaps/
            # TODO: also print out the **accs, **stats of tune_report maybe

            def get_key(trial):
                # return trial.last_result['avg_acc_val_so_far']
                return trial.last_result['best_val']

            print("Len(analysis):", len(analysis.trials), "analysis.trials:", list(map(get_key, analysis.trials)))

            best_trial = max(analysis.trials, key=get_key)
            # Changed the total nr of iterations to accommodate for this new approach
            total_iterations_for_this_task = 0
            # TODO: does this also consider stopped trials?
            for trial in analysis.trials:
                # print(trial.trial_id)
                # print(trial.status)
                # print(trial.last_result['iteration_of_report'])
                # print("---")
                if trial != best_trial:
                    trial_path = trial.logdir
                    shutil.rmtree(trial_path)
                total_iterations_for_this_task += trial.last_result['iteration_of_report']
            nr_of_iterations_to_add = total_iterations_so_far_per_task[t_id - 1] + total_iterations_for_this_task if t_id >= 1 else total_iterations_for_this_task
            total_iterations_so_far_per_task.append(nr_of_iterations_to_add)
            # am = np.argmax(list(map(get_key, analysis.trials)))
            # print("BEST IS {}: {}".format(am, best_trial.last_result['avg_acc_val']))

            # t = best_trial.last_result['duration_iterations']
            # Changed the total nr of iterations to accomodate for this new approach, so total_steps is essentially not used anymore
            total_steps = best_trial.last_result['total_steps']
            selected_tags.append(best_trial.experiment_tag)
            best_learner_path = os.path.join(best_trial.logdir, 'learner.pth')
            learner = torch.load(best_learner_path, map_location='cpu')
            shutil.rmtree(best_trial.logdir)

            # Save the learner
            torch.save(learner, learner_path)

            print("[TEST] Iterations for task:", t_id, "= ", total_iterations_for_this_task)
            print("[TEST] Iterations in total so far:", total_iterations_so_far_per_task[t_id])
            print("[TEST] best_trial:", best_trial, "selected_tags:", selected_tags, "best_trial.last_result:", best_trial.last_result)
            print("[TEST] best_trial's arch_scores:", best_trial.last_result["arch_scores"])  # self.arch_scores[task_id]['knn']
            print("[TEST] Finished learning on task:", t_id)

            # Backward transfer
            print("[TEST] Trying for backward transfer now based on task:", t_id)
            try_for_backward_transfer(memory_buffer=memory_buffer, task_id=t_id, task=task, learner=learner,
                                      training_params=config['training-params'])
            print("[TEST] Completed trying for backward transfer on task:", t_id)  # TODO: RESULTS SHORTLY

            # Save samples of the current task to the memory buffer
            print("[TEST] Save samples to memory")
            save_samples_to_memory(memory_buffer, t_id, task)

            print("[TEST] Finished everything for task:", t_id)

            # print(type(analysis))
            # print(analysis)
            # if task_counter == 1:
            #     exit(0)
        else:
            rescaled, t, metrics, b_state_dict, \
            stats = train_single_task(config=deepcopy(config), learner=learner,
                                      **static_params)

        # all_stats.append(stats)
        # update_rescaled(list(rescaled.values()), list(rescaled.keys()), tag,
        #                  g_task_vis, False)
        task_counter += 1

    print("[TEST] End training")

    if task_level_tuning:
        print("[TEST] len(all_analysis):", len(all_analysis), "selected_tags:", selected_tags)
        return all_analysis, selected_tags, total_iterations_so_far_per_task
    else:
        save_path = path.join(tune.get_trial_dir(), 'learner.pth')
        logger.info('Saving {} to {}'.format(learner, save_path))
        torch.save(learner, save_path)

def try_for_backward_transfer(memory_buffer=None, task_id=None, task=None, learner=None, training_params=None):
    if memory_buffer is None or task_id is None or task is None or learner is None or training_params is None:
        raise ValueError('Some arguments are None or not supplied')

    if memory_buffer.nr_of_observed_data_samples == 0 or task_id == 0:
        return

    # Get the settings for transforming and normalizing the data
    transforms, normalize = get_transform_normalize(training_params, task)  # TODO: needed?

    # Get the unique labels of the current task (c_t)
    c_t_val_dataset = get_datasets_of_task(task, transforms=None, normalize=None)[1]
    c_t_labels = c_t_val_dataset.tensors[1].tolist()
    c_t_labels = [item for sublist in c_t_labels for item in sublist]
    # print("c_t_labels:", c_t_labels)
    # print("c_t_val_dataset.tensors:")
    # print(c_t_val_dataset.tensors)

    # Get the model of the current task, which was just created
    c_t_model = learner.get_model(task_id=task_id)

    # Get the validation score
    c_t_val_dataset = _load_datasets(task, 'Val', normalize=normalize)[0]
    c_t_c_m_acc = evaluate(c_t_model, c_t_val_dataset, training_params['batch_sizes'][1], training_params['device'])
    print("c_t_c_m_acc:", c_t_c_m_acc)
    c_t_EVAL_dataset = _load_datasets(task, 'Test', normalize=normalize)[0]
    c_t_c_m_EVAL_acc = evaluate(c_t_model, c_t_EVAL_dataset, training_params['batch_sizes'][1], training_params['device'])
    print("c_t_c_m_EVAL_acc:", c_t_c_m_EVAL_acc)

    # For the currently added/created network, evaluate which past task, based on the saved data samples, has the same
    # labels as the current task, and gets higher avg accuracy than on its own network TODO: check if this can actually work or not
    for p_t_id in range(task_id):
        print("p_t_id:", p_t_id)
        # Get all data samples of the past task
        p_t_samples = memory_buffer.get_samples(p_t_id)
        p_t_labels = set([sample[1] for sample in p_t_samples])
        # print(len(memory_buffer.memory))
        # print(memory_buffer.memory)
        # print(p_t_labels)

        # Get evaluation data samples of the past task, just for comprehension purposes
        p_t_EVAL_dataset = _load_datasets(task, 'Test', normalize=normalize)[0]

        # Check if the past samples' labels are all included in the labels of the current task
        if not p_t_labels.issubset(c_t_labels):
            continue

        # Convert data samples to tensors
        p_t_samples_tensor, p_t_labels_tensor = convert_memory_samples_to_tensors(memory_samples=p_t_samples, memory_size=memory_buffer.memory_size)
        p_t_tensor = MyTensorDataset(p_t_samples_tensor, p_t_labels_tensor, transforms=None)

        # print("type(p_t_samples):")
        # print(type(p_t_samples))
        # print("p_t_samples:")
        # print(p_t_samples)
        # print("type(p_t_labels):")
        # print(type(p_t_labels))
        # print("p_t_labels:")
        # print(p_t_labels)
        # print("---")
        # print("len(p_t_samples_tensor):")
        # print(len(p_t_samples_tensor))
        # print("p_t_samples_tensor:")
        # print(p_t_samples_tensor)
        # print("len(p_t_labels_tensor):")
        # print(len(p_t_labels_tensor))
        # print("p_t_labels_tensor:")
        # print(p_t_labels_tensor)
        # print("p_t_tensor.tensors:")
        # print(p_t_tensor.tensors)

        # Get the past model
        p_t_model = learner.get_model(task_id=p_t_id)

        # Evaluate the past samples on the past model
        p_t_p_m_acc = evaluate(p_t_model, p_t_tensor, training_params['batch_sizes'][1], training_params['device'])
        print("Score of the past samples on the past model:", p_t_p_m_acc)
        # Evaluate the past samples eval dataset on the past model
        p_t_p_m_EVAL_acc = evaluate(p_t_model, p_t_EVAL_dataset, training_params['batch_sizes'][1], training_params['device'])
        print("EVAL score of the past samples on the past model:", p_t_p_m_EVAL_acc)

        # Evaluate the past samples on the current model
        p_t_c_m_acc = evaluate(c_t_model, p_t_tensor, training_params['batch_sizes'][1], training_params['device'])
        print("Score of the past samples on the current model:", p_t_c_m_acc)
        # Evaluate the past samples eval dataset on the current model
        p_t_c_m_EVAL_acc = evaluate(c_t_model, p_t_EVAL_dataset, training_params['batch_sizes'][1], training_params['device'])
        print("EVAL score of the past samples on the current model:", p_t_c_m_EVAL_acc)

        # Evaluate the current samples on the past model
        c_t_p_m_acc = evaluate(p_t_model, c_t_val_dataset, training_params['batch_sizes'][1], training_params['device'])
        print("Score of the current samples on the past model:", c_t_p_m_acc)
        # Evaluate the current samples eval dataset on the past model
        c_t_p_m_EVAL_acc = evaluate(p_t_model, c_t_EVAL_dataset, training_params['batch_sizes'][1], training_params['device'])
        print("EVAL score of the past samples on the current model:", c_t_p_m_EVAL_acc)

        # Print the outcome
        if p_t_c_m_acc > p_t_p_m_acc:
            print("!!! Score of the past samples on the current model > on past model. Can enable for BW transfer")
        if c_t_p_m_acc > c_t_c_m_acc:
            print("!!! Score of the current samples on the past model > on current model. Can enable for more transfer. This case should be rare")
    # res = defaultdict(lambda: defaultdict(list))
    # for t_id, task in enumerate(tqdm(tasks, desc='Evaluation on tasks',
    #                                  leave=False, disable=True)):
    #     t_id = t_id if cur_task is None else min(t_id, cur_task)
    #     eval_model = ll_model.get_model(task_id=t_id)
    #     for split in splits:
    #         split_dataset = _load_datasets(task, split, normalize=normalize)[0]
    #         acc, conf_mat = evaluate(eval_model, split_dataset, batch_size,
    #                                  device)
    #         res[split]['accuracy'].append(acc)
    #         res[split]['confusion'].append(conf_mat)
    #     eval_model.cpu()
    #     torch.cuda.empty_cache()
    # # TODO: what?
    # pass
    # # TODO IMPLEMENT, CHECK

def get_transform_normalize(training_params=None, task=None):
    if training_params is None or task is None:
        raise ValueError('Some arguments are None or not supplied')

    augment_data = training_params['augment_data']
    transformations = []
    if augment_data:
        transformations.extend([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor()
        ])
    t_trans = [[] for _ in range(len(task['split_names']))]
    t_trans[0] = transformations.copy()
    normalize = training_params['normalize']
    return t_trans, normalize

def convert_memory_samples_to_tensors(memory_samples=None, memory_size=None):
    if memory_samples is None or memory_size is None:
        raise ValueError('Some arguments are None or not supplied')
    samples_tensor = [entry[0] for entry in memory_samples]
    saved_labels = [entry[1] for entry in memory_samples]

    samples_tensor = torch.stack(samples_tensor)
    labels_tensor = torch.zeros(len(saved_labels), 1).int()

    for i in range(len(saved_labels)):
        labels_tensor[i][0] = int(saved_labels[i])
    return samples_tensor, labels_tensor

def get_datasets_of_task(task=None, transforms=None, normalize=None):
    if task is None:
        raise ValueError('Some arguments are None or not supplied')

    # Get the datasets of the current task
    datasets_p = dict(task=task, transforms=None, normalize=None)
    datasets = _load_datasets(**datasets_p)
    return datasets

def save_samples_to_memory(memory_buffer=None, task_id=None, task=None):
    if memory_buffer is None or task_id is None or task is None:
        raise ValueError('Some arguments are None or not supplied')

    # Get the datasets of the current task
    datasets = get_datasets_of_task(task, transforms=None, normalize=None)

    # (Try to) add each data sample of the current task to the memory buffer
    for i, data_sample in enumerate(datasets[0].tensors[0]):
        label = torch.index_select(datasets[0].tensors[1], 0, torch.tensor([i])).tolist()[0][0]
        memory_buffer.observe_sample(data_sample, task_id, label)

    # Print info about the current memory contents for clarity
    print("Task ID:", task_id)
    print("Memory:")
    total_entries = 0
    memory_display_dict = {}
    for key in memory_buffer.memory:
        nr_entries = len(memory_buffer.memory[key])
        memory_display_dict[key] = nr_entries
        total_entries += nr_entries
        print(key, nr_entries)
    print("total_entries:", total_entries)
    print("-----")

def train_t(config):
    # As per https://docs.ray.io/en/latest/tune/tutorials/tune-resources.html:
    # Occasionally, you may run into GPU memory issues when running a new trial.
    # This may be due to the previous trial not cleaning up its GPU state fast enough. Use this:
    division_factor_per_gpu = config.pop('division_factor_per_gpu')
    # tune.utils.wait_for_gpu(target_util=1-(1/division_factor_per_gpu))

    # This function does not allow for printing to be seen in the output files
    seed = config.pop('seed')
    static_params = config.pop('static_params')

    torch.backends.cudnn.enabled = True
    if static_params['t_id'] == 0:
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    else:
        torch.backends.cudnn.deterministic = False

    if 'PSSN' in tune.get_trial_name() or static_params['t_id'] == 0:
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    if 'learner' in config:
        learner = config.pop('learner')
    else:
        learner_path = config.pop('learner_path')
        learner = torch.load(learner_path)

    rescaled, t, metrics, b_state_dict, stats = train_single_task(config=config, learner=learner,
                                                                  **static_params)

    # If the results are valid: save them
    if rescaled != -1 and t != -1 and metrics != -1 and b_state_dict != -1 and stats != -1:
        learner_save_path = os.path.join(tune.get_trial_dir(), 'learner.pth')
        # raise ValueError(learner_save_path)
        torch.save(learner, learner_save_path)


def train_single_task(t_id, task, tasks, vis_p, learner, config, transfer_matrix,
                      total_steps):
    training_params = config.pop('training-params')
    learner_params = config.pop('learner-params', {})
    assert 'model-params' not in config, "Can't have model-specific " \
                                         "parameters while tuning at the " \
                                         "stream level."

    if learner_params:
        learner.set_h_params(**learner_params)

    # TODO: probably define shared memory somewhere so that progress values can be kept and saved to perform LC extrapolation

    batch_sizes = training_params.pop('batch_sizes')
    # optim_func = training_params.pop('optim_func')
    optim_func = learner.optim_func
    optim_params = config.pop('optim')
    schedule_mode = training_params.pop('schedule_mode')
    split_optims = training_params.pop('split_optims')

    # optim = set_optim_params(optim_func, optim_params, model, split_optims)
    optim_fact = partial(set_optim_params,
                         optim_func=optim_func,
                         optim_params=optim_params,
                         split_optims=split_optims)

    # Create a dictionary with keywords used for tune.report with initialized -1 values. This is needed because the keywords
    # supplied to the first tune.report call are set in stone and no new ones can be added afterwards, I observed
    evaluation_splits = ['Val', 'Test']
    tune_report_arguments_initialized = initialize_tune_report_arguments(tasks, evaluation_splits)

    # In case it is the first task, we only have 1 unique architecture, so terminate all others
    if t_id == 0 and optim_fact.keywords['optim_params'][0]['architecture'] != 0:
        current_task_best_val_time_attr = {'best_val_T' + str(t_id): -1, 'epoch_of_report_T' + str(t_id): -1}
        tune.report(t=t_id,
                    best_val=-1,
                    duration_best_it=-1,
                    iteration_of_report=0,
                    epoch_of_report=-1,
                    **current_task_best_val_time_attr,
                    **tune_report_arguments_initialized)
        return -1, -1, -1, -1, -1

    dropout = config.pop('dropout') if 'dropout' in config else None

    stream_setting = training_params.pop('stream_setting')
    plot_all = training_params.pop('plot_all')
    lca_n = training_params.pop('lca')

    if plot_all:
        vis_p = get_training_vis_conf(vis_p, tune.get_trial_dir())
        # print('NEW vis: ', vis_p)
        task_vis = visdom.Visdom(**vis_p)
        # env = [env[0], env[-1]]
        # vis_p['env'] = '_'.join(env)
        # vis_p['log_to_filename'] = os.path.join(vis_logdir, vis_p['env'])
        # g_task_vis = visdom.Visdom(**vis_p)

        logger.info(get_env_url(task_vis))
    else:
        task_vis = None
    env_url = get_env_url(vis_p)
    # if t_id != 0:
    #     raise ValueError("t_id:", t_id, "env_url:", env_url, "vis_p:", vis_p)
    # ValueError: ('t_id:', 0, 'env_url:', 'http://localhost:8097/env/1_Trial_PSSN-search-6-fw_1_0_lr=0.001,0_weight_decay=0_md-T0', 'vis_p:', {'env': '1_Trial_PSSN-search-6-fw_1_0_lr=0.001,0_weight_decay=0_md-T0', 'log_to_filename': '/home/TUE/s167139/data/veniat/lileb/visdom_traces/1/1_Trial_PSSN-search-6-fw_1_0_lr=0.001,0_weight_decay=0_md-T0', 'server': 'localhost', 'port': 8097, 'offline': True})
    # ValueError: ('t_id:', 0, 'env_url:', 'http://localhost:8097/env/1_Trial_PSSN-search-6-fw_2_0_lr=0.01,0_weight_decay=0.0001_md-T0', 'vis_p:', {'env': '1_Trial_PSSN-search-6-fw_2_0_lr=0.01,0_weight_decay=0.0001_md-T0', 'log_to_filename': '/home/TUE/s167139/data/veniat/lileb/visdom_traces/1/1_Trial_PSSN-search-6-fw_2_0_lr=0.01,0_weight_decay=0.0001_md-T0', 'server': 'localhost', 'port': 8097, 'offline': True})
    # ... etc
    # ValueError: ('t_id:', 1, 'env_url:', 'http://localhost:8097/env/1_Trial_PSSN-search-6-fw_1_0_lr=0.001,0_weight_decay=0_md-T1', 'vis_p:', {'env': '1_Trial_PSSN-search-6-fw_1_0_lr=0.001,0_weight_decay=0_md-T1', 'log_to_filename': '/home/TUE/s167139/data/veniat/lileb/visdom_traces/1/1_Trial_PSSN-search-6-fw_1_0_lr=0.001,0_weight_decay=0_md-T1', 'server': 'localhost', 'port': 8097, 'offline': True})
    # ... etc
    # TODO: try to fix the error that occurs at approx. line 346 in stream_tuning.py, put tune_reporter back in at end of this func
    # TODO: and try to see if can run s_test fully with good/expected results

    t_trans, normalize = get_transform_normalize(training_params, task)
    training_params.pop('augment_data')
    training_params.pop('normalize')

    datasets_p = dict(task=task,
                      transforms=t_trans,
                      normalize=normalize)
    datasets = _load_datasets(**datasets_p)
    train_loader, eval_loaders = get_classic_dataloaders(datasets,
                                                         batch_sizes)

    assert t_id == task['id']

    # TODO: this below should maybe not be ran concurrently, preferably it should be ran once beforehand to save time
    start1 = time.time()
    model = learner.get_model(task['id'], x_dim=task['x_dim'],
                              n_classes=task['n_classes'],
                              descriptor=task['descriptor'],
                              dataset=eval_loaders[:2])
    model_creation_time = time.time() - start1
    # raise ValueError("[TEST] Memory currently in the GPU cache:", torch.cuda.memory_allocated())
    # 0 in memory if it crashes in learner.getmodel(...)
    # Crashes with information:
    # raise ValueError(task['id'], optim_fact)
    # ValueError: (1, functools.partial(<function set_optim_params at 0x7ee4133a0040>,
    # optim_func=functools.partial(<class 'torch.optim.adam.Adam'>,
    # weight_decay=0, lr=0.001, betas=[0.9, 0.999]),
    # optim_params=[{'architecture': 5, 'lr': 0.01, 'weight_decay': 0}], split_optims=True))
    # TODO: check to see if the model actually grows over time
    # if t_id > 0:
    #     raise ValueError("learner:", learner, "model.models:", model.models, "model.models_idx:", model.models_idx, "model.get_graph():", model.get_graph())

    loss_fn = task['loss_fn']
    training_params['loss_fn'] = loss_fn

    prepare_batch = _prepare_batch
    if hasattr(model, 'prepare_batch_wrapper'):
        prepare_batch = model.prepare_batch_wrapper(prepare_batch, t_id)

    if hasattr(model, 'loss_wrapper'):
        training_params['loss_fn'] = \
            model.loss_wrapper(training_params['loss_fn'])

    # if hasattr(model, 'backward_hook'):
    #     training_params[]

    # if schedule_mode == 'steps':
    #     lr_scheduler = torch.optim.lr_scheduler.\
    #         MultiStepLR(optim[0], milestones=[25, 40])
    # elif schedule_mode == 'cos':
    #     lr_scheduler = torch.optim.lr_scheduler.\
    #         CosineAnnealingLR(optim[0], T_max=200, eta_min=0.001)
    # elif schedule_mode is None:
    #     lr_scheduler = None
    # else:
    #     raise NotImplementedError()
    if dropout is not None:
        set_dropout(model, dropout)

    assert not config, config
    start2 = time.time()
    # TODO, marker: training of models conducted in this function
    rescaled, t, metrics, b_state_dict, info_training = train_model(model, datasets_p,
                                                                    batch_sizes, optim_fact,
                                                                    prepare_batch, task,
                                                                    train_loader, eval_loaders,
                                                                    training_params, env_url,
                                                                    tune_report_arguments_initialized, config)

    training_time = time.time() - start2
    start3 = time.time()
    if not isinstance(model, ExhaustiveSearch):
        # todo Handle the state dict loading uniformly for all learners RN only
        # the exhaustive search models load the best state dict after training
        model.load_state_dict(b_state_dict['state_dict'])

    iterations = list(metrics.pop('training_iteration').values())
    epochs = list(metrics.pop('training_epoch').values())

    assert len(iterations) == len(epochs)
    index = dict(epochs=epochs, iterations=iterations)
    update_summary(index, task_vis, 'index', 0.5)

    grouped_xs = dict()
    grouped_metrics = defaultdict(list)
    grouped_legends = defaultdict(list)
    for metric_n, metric_v in metrics.items():
        split_n = metric_n.split()
        if len(split_n) < 2:
            continue
        name = ' '.join(split_n[:-1])
        grouped_metrics[split_n[-1]].append(list(metric_v.values()))
        grouped_legends[split_n[-1]].append(name)
        if split_n[-1] in grouped_xs:
            if len(metric_v) > len(grouped_xs[split_n[-1]]):
                longer_xs = list(metric_v.keys())
                assert all(a == b for a, b in zip(longer_xs,
                                                  grouped_xs[split_n[-1]]))
                grouped_xs[split_n[-1]] = longer_xs
        else:
            grouped_xs[split_n[-1]] = list(metric_v.keys())

    for (plot_name, val), (_, legends) in sorted(zip(grouped_metrics.items(),
                                                     grouped_legends.items())):
        assert plot_name == _
        val = fill_matrix(val)
        if len(val) == 1:
            val = np.array(val[0])
        else:
            val = np.array(val).transpose()
        x = grouped_xs[plot_name]
        task_vis.line(val, X=x, win=plot_name,
                      opts={'title': plot_name, 'showlegend': True,
                            'width': 500, 'legend': legends,
                            'xlabel': 'iterations', 'ylabel': plot_name})

    avg_data_time = list(metrics['data time_ps'].values())[-1]
    avg_forward_time = list(metrics['forward time_ps'].values())[-1]
    avg_epoch_time = list(metrics['epoch time_ps'].values())[-1]
    avg_eval_time = list(metrics['eval time_ps'].values())[-1]
    total_time = list(metrics['total time'].values())[-1]

    entropies, ent_legend = [], []
    for metric_n, metric_v in metrics.items():
        if metric_n.startswith('Trainer entropy'):
            entropies.append(list(metric_v.values()))
            ent_legend.append(metric_n)

    if entropies:
        task_vis.line(np.array(entropies).transpose(), X=iterations,
                      win='ENT',
                      opts={'title': 'Arch entropy', 'showlegend': True,
                            'width': 500, 'legend': ent_legend,
                            'xlabel': 'Iterations', 'ylabel': 'Loss'})

    if hasattr(learner, 'arch_scores') and hasattr(learner, 'get_top_archs'):
        update_summary(learner.arch_scores[t_id], task_vis, 'scores')
        archs = model.get_top_archs(5)
        list_top_archs(archs, task_vis)

    if 'training_archs' in metrics:
        plot_trajectory(model.ssn.graph, metrics['training_archs'],
                        model.ssn.stochastic_node_ids, task_vis)

    postproc_time = time.time() - start3
    start4 = time.time()
    save_path = tune.get_trial_dir()
    finish_res = learner.finish_task(datasets[0], t_id,
                                     task_vis, save_path)
    finish_time = time.time() - start4

    start5 = time.time()
    eval_tasks = tasks
    # eval_tasks = tasks[:t_id + 1] if stream_setting else tasks
    evaluation = evaluate_on_tasks(eval_tasks, learner, batch_sizes[1],
                                   training_params['device'],
                                   evaluation_splits, normalize,
                                   cur_task=t_id)
    assert evaluation['Val']['accuracy'][t_id] == b_state_dict['value']

    stats = {}
    eval_time = time.time() - start5

    stats.update(finish_res)

    # test_accs = metrics['Test accuracy_0']
    # if not test_accs:
    #     lca = np.float('nan')
    # else:
    #     if len(test_accs) <= lca_n:
    #         last_key = max(test_accs.keys())
    #         assert len(test_accs) == last_key + 1, \
    #             f"Can't compute LCA@{lca_n} if steps were skipped " \
    #             f"(got {list(test_accs.keys())})"
    #         test_accs = test_accs.copy()
    #         last_acc = test_accs[last_key]
    #         for i in range(last_key + 1, lca_n + 1):
    #             test_accs[i] = last_acc
    #     lca = np.mean([test_accs[i] for i in range(lca_n + 1)])
    lca = -1

    accs = {}
    key = 'accuracy'
    # logger.warning(evaluation)
    for split in evaluation.keys():
        transfer_matrix[split].append(evaluation[split][key])
        for i in range(len(tasks)):
            split_acc = evaluation[split][key]
            if i < len(split_acc):
                accs['{}_T{}'.format(split, i)] = split_acc[i]
            else:
                accs['{}_T{}'.format(split, i)] = float('nan')
    plot_heatmaps(list(transfer_matrix.keys()),
                  list(map(fill_matrix, transfer_matrix.values())),
                  task_vis)

    # logger.warning(t_id)
    # logger.warning(transfer_matrix)

    avg_val = np.mean(evaluation['Val']['accuracy'])
    avg_val_so_far = np.mean(evaluation['Val']['accuracy'][:t_id + 1])
    avg_test = np.mean(evaluation['Test']['accuracy'])
    avg_test_so_far = np.mean(evaluation['Test']['accuracy'][:t_id + 1])

    step_time_s = time.time() - start1
    step_sum = model_creation_time + training_time + postproc_time + \
               finish_time + eval_time
    best_it = b_state_dict.get('cum_best_iter', b_state_dict['iter'])
    # TODO: add parameters to be reported here?
    # tune_report is already called inside ExhaustiveSearch. When reporting multiple times to tune_report for 1 trial,
    # clean it up so that only the last one remains before plotting.py is ran
    # if t_id > 0:
    #     raise ValueError("info_training['path']:", info_training['path'])
    current_task_best_val_time_attr = {'best_val_T' + str(t_id): b_state_dict['value'], 'epoch_of_report_T' + str(t_id): training_params['n_ep_max']}
    tune.report(t=t_id,
                best_val=b_state_dict['value'],
                iteration_of_report=t,  # Infinite, since we just use this as a counter for the scheduler
                epoch_of_report=training_params['n_ep_max'],  # Infinite, same reason
                avg_acc_val=avg_val,
                avg_acc_val_so_far=avg_val_so_far,
                avg_acc_test_so_far=avg_test_so_far,
                lca=lca,
                avg_acc_test=avg_test,
                test_acc=evaluation['Test']['accuracy'][t_id],
                duration_seconds=step_time_s,
                duration_iterations=t,
                duration_best_it=best_it,
                duration_finish=finish_time,
                duration_model_creation=model_creation_time,
                duration_training=training_time,
                duration_postproc=postproc_time,
                duration_eval=eval_time,
                duration_sum=step_sum,
                iterations=iterations,
                epochs=epochs,
                # entropy=stats.pop('entropy'),
                new_params=learner.new_params(t_id),
                total_params=learner.n_params(t_id),
                total_steps=total_steps + t,
                fw_t=round(avg_forward_time * 1000) / 1000,
                data_t=round(avg_data_time * 1000) / 1000,
                epoch_t=round(avg_epoch_time * 1000) / 1000,
                eval_t=round(avg_eval_time * 1000) / 1000,
                total_t=round(total_time * 1000) / 1000,
                env_url=get_env_url(vis_p),
                # info_training=info_training,
                path=info_training['path'],
                used_architecture_id=info_training['params']['architecture'],
                arch_scores=learner.arch_scores,
                **current_task_best_val_time_attr,
                **accs, **stats)
    return rescaled, t, metrics, b_state_dict, stats


def train_model(model, datasets_p, batch_sizes, optim_fact, prepare_batch,
                task, train_loader, eval_loaders, training_params, env_url, tune_report_arguments_initialized, config):
    if hasattr(model, 'train_func'):
        assert not config, config
        f = model.train_func
        t, metrics, b_state_dict, info_training = f(datasets_p=datasets_p,
                                                    b_sizes=batch_sizes,
                                                    optim_fact=optim_fact,
                                                    # lr_scheduler=lr_scheduler,
                                                    # viz=task_vis,
                                                    prepare_batch=prepare_batch,
                                                    split_names=task['split_names'],
                                                    env_url=env_url,
                                                    t_id=task['id'],
                                                    tune_report_arguments_initialized=tune_report_arguments_initialized,
                                                    # viz=task_vis,
                                                    **training_params)
        rescaled = list(
            filter(lambda itm: 'rescaled' in itm[0], metrics.items()))
        rescaled = rescaled[0][1]
    else:
        optim = optim_fact(model=model)
        if hasattr(model, 'train_loader_wrapper'):
            train_loader = model.train_loader_wrapper(train_loader)
        t, metrics, b_state_dict = train(model, train_loader, eval_loaders,
                                         optimizer=optim,
                                         # lr_scheduler=lr_scheduler,
                                         # viz=task_vis,
                                         prepare_batch=prepare_batch,
                                         split_names=task['split_names'],
                                         # viz=task_vis,
                                         **training_params)
        info_training = None
        rescaled = metrics['Val accuracy_0']

    return rescaled, t, metrics, b_state_dict, info_training
