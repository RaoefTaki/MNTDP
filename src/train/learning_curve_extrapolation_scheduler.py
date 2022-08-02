import collections
import logging
from typing import Dict, List, Optional

import numpy as np

from ray.tune import trial_runner
from ray.tune.result import DEFAULT_METRIC
from ray.tune.trial import Trial
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler

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
                 reward_attr: Optional[str] = None,
                 metric: Optional[str] = None,
                 mode: Optional[str] = None,
                 grace_period: float = 60.0,
                 check_epoch: int = 30,
                 certainty: float = 0.95):
        if reward_attr is not None:
            mode = "max"
            metric = reward_attr
            logger.warning(
                "`reward_attr` is deprecated and will be removed in a future "
                "version of Tune. "
                "Setting `metric={}` and `mode=max`.".format(reward_attr))
        FIFOScheduler.__init__(self)
        self._stopped_trials = set()
        self._grace_period = grace_period
        self._check_epoch = check_epoch
        self._certainty = certainty
        self._metric = metric
        self._best_trial = (None, None)  # Stores a tuple of (Trial, performance)
        self._compare_op = None

        self._mode = mode
        if mode:
            assert mode in ["min", "max"], "`mode` must be 'min' or 'max'."
            self._worst = float("-inf") if self._mode == "max" else float(
                "inf")
            self._compare_op = max if self._mode == "max" else min

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

        super(LearningCurveExtrapolationScheduler, self).on_trial_add(trial_runner, trial)

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
        # TODO: just to see if it works, try stopping every trial at 1 epoch
        # TODO: then in next github push only resuming the single best trial
        action = None

        # Pause each trial if it's at a check epoch, and update the best found trial if applicable
        epoch = result[self._time_attr]
        if epoch % self._check_epoch == 0:
            # Update if applicable
            if result[self._metric] > self._best_trial[1]:
                self._best_trial = (trial, result[self._metric])

            # Set the action to pause
            action = TrialScheduler.PAUSE  # TODO: STOP

        return action

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

    def debug_string(self) -> str:
        return "Using LearningCurveExtrapolationScheduler: num_stopped={}.".format(
            len(self._stopped_trials))

    # def _on_insufficient_samples(self,
    #                              trial_runner: "trial_runner.TrialRunner",
    #                              trial: Trial, time: float) -> str:
    #     pause = time - self._last_pause[trial] > self._min_time_slice
    #     pause = pause and [
    #         t for t in trial_runner.get_trials()
    #         if t.status in (Trial.PENDING, Trial.PAUSED)
    #     ]
    #     return TrialScheduler.PAUSE if pause else TrialScheduler.CONTINUE

    def _trials_beyond_time(self, time: float) -> List[Trial]:
        trials = [
            trial for trial in self._results
            if self._results[trial][-1][self._time_attr] >= time
        ]
        return trials

    def _median_result(self, trials: List[Trial], time: float):
        return np.median([self._running_mean(trial, time) for trial in trials])

    def _running_mean(self, trial: Trial, time: float) -> np.ndarray:
        results = self._results[trial]
        # TODO(ekl) we could do interpolation to be more precise, but for now
        # assume len(results) is large and the time diffs are roughly equal
        scoped_results = [
            r for r in results
            if self._grace_period <= r[self._time_attr] <= time
        ]
        return np.mean([r[self._metric] for r in scoped_results])

    def _best_result(self, trial):
        results = self._results[trial]
        return self._compare_op([r[self._metric] for r in results])
