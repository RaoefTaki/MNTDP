import collections
import logging
from typing import Dict, List, Optional

import numpy as np

from ray.tune import trial_runner
from ray.tune.result import DEFAULT_METRIC
from ray.tune.trial import Trial
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler

from pylrpredictor.curvefunctions import model_defaults, vap, pow3, loglog_linear, dr_hill_zero_background, \
    log_power, pow4, mmf, exp4, janoschek, weibull, ilog2
from pylrpredictor.curvemodels import MCMCCurveModel
from pylrpredictor.ensemblecurvemodel import CurveEnsemble
import numpy as np

logger = logging.getLogger(__name__)


class LearningCurveExtrapolationScheduler(FIFOScheduler):
    """Implements a scheduler based on learning curve extrapolation as described in this paper:

    https://www.ijcai.org/Proceedings/15/Papers/487.pdf

    Currently only checked with mode == max

    Args:
        time_attr (str): The training result attr to use for comparing time.
            Note that you can pass in something non-temporal such as
            `training_iteration` as a measure of progress, the only requirement
            is that the attribute should increase monotonically. Should be some
            sort of epoch number for LCE.
        metric (str): The training result objective value attribute. Stopping
            procedures will use this attribute. If None but a mode was passed,
            the `ray.tune.result.DEFAULT_METRIC` will be used per default.
        mode (str): One of {min, max}. Determines whether objective is
            minimizing or maximizing the metric attribute.
        grace_period (float): Only stop trials at least this old in time.
            The mean will only be computed from this time onwards. The units
            are the same as the attribute named by `time_attr`.
        check_epoch (int): The epoch at which the LC extrapolation check,
            and possible stopping, is performed
        certainty (float): The level of certainty with which the extrapolation
            is conducted.
    """

    def __init__(self,
                 time_attr: str = "time_total_s",
                 metric: Optional[str] = None,
                 mode: Optional[str] = None,
                 grace_period: float = 1,
                 check_epoch: int = 30,
                 extrapolated_epoch: int = 300,
                 certainty: float = 0.95):
        FIFOScheduler.__init__(self)
        self._stopped_trials = set()
        self._grace_period = grace_period
        self._check_epoch = check_epoch
        self._extrapolated_epoch = extrapolated_epoch
        self._certainty = certainty
        self._metric = metric
        self._worst = 0
        self._compare_op = None

        self._mode = mode
        if mode:
            assert mode in ["min", "max"], "`mode` must be 'min' or 'max'."
            self._worst = float("-inf") if self._mode == "max" else float(
                "inf")
            self._compare_op = max if self._mode == "max" else min

        self._best_obtained_value = (None, self._worst)  # Stores a tuple of (Trial, obtained_performance)
        self._obtained_values = {}  # Stores per trial a dict of obtained values per reported epoch number
        self._time_attr = time_attr
        self._last_pause = collections.defaultdict(lambda: float("-inf"))
        self._results = collections.defaultdict(list)

    def set_search_properties(self, metric: Optional[str],
                              mode: Optional[str]) -> bool:
        if self._metric and metric:
            return False
        if self._mode and mode:
            return False

        if metric:
            self._metric = metric
        if mode:
            self._mode = mode

        self._worst = float("-inf") if self._mode == "max" else float("inf")
        self._compare_op = max if self._mode == "max" else min

        if self._metric is None and self._mode:
            # If only a mode was passed, use anonymous metric
            self._metric = DEFAULT_METRIC

        return True

    def on_trial_add(self, trial_runner: "trial_runner.TrialRunner",
                     trial: Trial):
        if not self._metric or not self._worst or not self._compare_op:
            raise ValueError(
                "{} has been instantiated without a valid `metric` ({}) or "
                "`mode` ({}) parameter. Either pass these parameters when "
                "instantiating the scheduler, or pass them as parameters "
                "to `tune.run()`".format(self.__class__.__name__, self._metric,
                                         self._mode))

        self._obtained_values[trial.trial_id] = {}

        super(LearningCurveExtrapolationScheduler, self).on_trial_add(trial_runner, trial)

    def on_trial_result(self, trial_runner: "trial_runner.TrialRunner",
                        trial: Trial, result: Dict) -> str:
        """Callback for early stopping.

        This stopping rule stops a running trial if the extrapolated objective value at epoch 300 of the trial is
        expected to be strictly worse, with 95% certainty, than the best objective value obtained.
        It is asynchronous, i.e. it completes all trials until a certain epoch X before performing the LC extrapolation
        and removing bad ones based on the best value obtained thus far.
        """
        epoch = result[self._time_attr]
        resulting_val = result[self._metric]

        # Update the obtained values and the best obtained value so far
        self._obtained_values[trial.trial_id][epoch] = resulting_val
        if resulting_val > self._best_obtained_value[1]:
            self._best_obtained_value = (trial, resulting_val)

        if self._time_attr not in result or self._metric not in result:
            return TrialScheduler.CONTINUE

        if epoch < self._grace_period:
            return TrialScheduler.CONTINUE

        # Pause each trial if it's at a check epoch, and see if the expected extrapolated performance is, with 95% certainty,
        # strictly worse than the current best extrapolated or actually obtained performance, at epoch 300
        if epoch % self._check_epoch == 0:
            # Depending on the expectation of surpassing, either stop or continue
            if not self._lce_surpass(trial, epoch):
                action = TrialScheduler.STOP
            else:
                action = TrialScheduler.CONTINUE
        else:
            action = TrialScheduler.CONTINUE

        return action

    def _lce_surpass(self, trial, epoch):
        # Build the LC extrapolator model from scratch and check whether with 95% certainty we will not reach the
        # maximum atained performance so far. Outputs whether it will surpass (True) the best obtained result so far or
        # not (False)
        # First collect the obtained performance metrics for this model
        metric_list = list(self._obtained_values[trial.trial_id].values())

        x_values = np.array(list(range(1, epoch + 1)))
        y_values = np.array(metric_list)

        # Fit the LC model
        lc_model = self._initialize_lc_extrapolation_model()
        lc_model.fit(x_values, y_values)

        # Extrapolate and calculate the probabilities
        current_end_posterior_prob = lc_model.posterior_prob_x_greater_than(self._extrapolated_epoch, self._best_obtained_value[1])
        return not(current_end_posterior_prob <= 1 - self._certainty)

    def _initialize_lc_extrapolation_model(self):
        lc_functions = {'pow4': pow4}
        lc_models = []
        for key, value in lc_functions.items():
            temp_lc_model = MCMCCurveModel(function=lc_functions[key],
                                           default_vals=model_defaults[key])
            lc_models.append(temp_lc_model)
        lc_model = CurveEnsemble(lc_models)
        return lc_model

    def on_trial_complete(self, trial_runner: "trial_runner.TrialRunner",
                          trial: Trial, result: Dict):
        self._results[trial].append(result)

    # def choose_trial_to_run(
    #         self, trial_runner: "trial_runner.TrialRunner") -> Optional[Trial]:
    #     try:
    #         for trial in trial_runner.get_trials():
    #             if trial.status == Trial.PENDING and trial_runner.has_resources(trial.resources) and self._can_proceed_next_run(trial):
    #                 return trial
    #         for trial in trial_runner.get_trials():
    #             if trial.status == Trial.PAUSED and trial_runner.has_resources(trial.resources) and self._can_proceed_next_run(trial):
    #                 return trial
    #     except:
    #         raise ValueError(trial_runner, type(trial_runner), trial_runner.get_trials())
    #     return None
