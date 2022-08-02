import collections
import logging
from typing import Dict, List, Optional

import numpy as np

from ray.tune import trial_runner
from ray.tune.result import DEFAULT_METRIC
from ray.tune.trial import Trial
from ray.tune.schedulers.trial_scheduler import TrialScheduler
from src.train.ray_schedulers import FIFOScheduler

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

        self._best_trial = (None, self._worst)  # Stores a tuple of (Trial, performance)
        self._trials_nr_of_checks = {}  # Stores the number of checks performed, in this scheduler, for each trial
        self._nr_of_checks = 0
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

        self._trials_nr_of_checks[trial.trial_id] = 0

        # super(LearningCurveExtrapolationScheduler, self).on_trial_add(trial_runner, trial)

    def on_trial_result(self, trial_runner: "trial_runner.TrialRunner",
                        trial: Trial, result: Dict) -> str:
        """Callback for early stopping.

        This stopping rule stops a running trial if the extrapolated objective value at epoch 300 of the trial is
        expected to be strictly worse, with 95% certainty, than the best objective value obtained.
        It is asynchronous, i.e. it completes all trials until a certain epoch X before performing the LC extrapolation
        and removing bad ones based on the best value obtained thus far.
        """
        if self._time_attr not in result or self._metric not in result:
            return TrialScheduler.CONTINUE

        # Stop if not found. Apparently something went wrong
        if trial.trial_id not in self._trials_nr_of_checks:
            return TrialScheduler.STOP

        # Pause each trial if it's at a check epoch, and update the best found trial if applicable
        epoch = result[self._time_attr]
        resulting_val = result[self._metric]
        if epoch % self._check_epoch == 0:
            self._trials_nr_of_checks[trial.trial_id] += 1
            action = TrialScheduler.PAUSE
        else:
            action = TrialScheduler.CONTINUE

        # Check if all trials have been checked the same number of times. If so, update the global check counter
        current_nr_of_checks = self._trials_nr_of_checks[trial.trial_id]
        if all(self._trials_nr_of_checks[trial.trial_id] == current_nr_of_checks for trial in trial_runner.get_trials()):
            self._nr_of_checks += 1

        return action

        # TODO: just to see if it works, try stopping every trial at 1 epoch
        # TODO: then in next github push only resuming the single best trial

        # if result[self._metric] == -1:
        #     return action
        #
        # # Pause each trial if it's at a check epoch, and update the best found trial if applicable
        # epoch = result[self._time_attr]
        # resulting_val = result[self._metric]
        # if epoch % self._check_epoch == 0:
        #     # Update if applicable
        #     if self._compare_op(resulting_val, self._best_trial[1]) == resulting_val:
        #         self._best_trial = (trial, resulting_val)
        #
        #     # Set the action to pause
        #     action = TrialScheduler.PAUSE  # TODO: STOP
        #
        # return action

        # # In case all trials have been paused, perform the LC extrapolation
        # if all(trial.status == Trial.PAUSED for trial in trial_runner.get_trials()):
        #     # LCE for each trial
        #     # TODO: stop the others and resume the best one
        #     pass
        # else:
        #     return action
        #
        # # TODO: clean self._trials if new epochs starts etc
        #
        # trial_runner.trial_executor.unpause_trial(trial)
        #
        # if trial in self._stopped_trials:
        #     assert not self._hard_stop
        #     # Fall back to FIFO
        #     return TrialScheduler.CONTINUE
        #
        # time = result[self._time_attr]
        # self._results[trial].append(result)
        #
        # if time < self._grace_period:
        #     return TrialScheduler.CONTINUE
        #
        # trials = self._trials_beyond_time(time)
        # trials.remove(trial)
        #
        # # if len(trials) < self._min_samples_required:
        # #     action = self._on_insufficient_samples(trial_runner, trial, time)
        # #     if action == TrialScheduler.PAUSE:
        # #         self._last_pause[trial] = time
        # #         action_str = "Yielding time to other trials."
        # #     else:
        # #         action_str = "Continuing anyways."
        # #     logger.debug(
        # #         "MedianStoppingRule: insufficient samples={} to evaluate "
        # #         "trial {} at t={}. {}".format(
        # #             len(trials), trial.trial_id, time, action_str))
        # #     return action
        #
        # median_result = self._median_result(trials, time)
        # best_result = self._best_result(trial)
        # logger.debug("Trial {} best res={} vs median res={} at t={}".format(
        #     trial, best_result, median_result, time))
        #
        # if self._compare_op(median_result, best_result) != best_result:
        #     logger.debug("LearningCurveExtrapolationScheduler: early stopping {}".format(trial))
        #     self._stopped_trials.add(trial)
        #     if self._hard_stop:
        #         return TrialScheduler.STOP
        #     else:
        #         return TrialScheduler.PAUSE
        # else:
        #     return TrialScheduler.CONTINUE

    def on_trial_complete(self, trial_runner: "trial_runner.TrialRunner",
                          trial: Trial, result: Dict):
        self._results[trial].append(result)

    def _can_proceed_next_run(self, trial):
        # Check to see if all trials have been processed, and whether this trial is thus ready to proceed
        return self._trials_nr_of_checks[trial.trial_id] == self._nr_of_checks

    # def choose_trial_to_run(
    #         self, trial_runner: "trial_runner.TrialRunner") -> Optional[Trial]:
    #     for trial in trial_runner.get_trials():
    #         if (trial.status == Trial.PENDING
    #                 and trial_runner.has_resources(trial.resources)):
    #             return trial
    #     for trial in trial_runner.get_trials():
    #         if (trial.status == Trial.PAUSED
    #                 and trial_runner.has_resources(trial.resources)):
    #             return trial
    #     return None

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
