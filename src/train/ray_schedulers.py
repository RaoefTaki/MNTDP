from typing import Dict, Optional

from ray.tune import trial_runner
from ray.tune.result import DEFAULT_METRIC
from ray.tune.trial import Trial
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler

class FIFOCheckWaitScheduler(TrialScheduler):
    """Simple scheduler that just runs trials in submission order."""

    def on_trial_add(self, trial_runner: "trial_runner.TrialRunner",
                     trial: Trial):
        pass

    def on_trial_error(self, trial_runner: "trial_runner.TrialRunner",
                       trial: Trial):
        pass

    def on_trial_result(self, trial_runner: "trial_runner.TrialRunner",
                        trial: Trial, result: Dict) -> str:
        return TrialScheduler.CONTINUE

    def on_trial_complete(self, trial_runner: "trial_runner.TrialRunner",
                          trial: Trial, result: Dict):
        pass

    def on_trial_remove(self, trial_runner: "trial_runner.TrialRunner",
                        trial: Trial):
        pass

    def choose_trial_to_run(
            self, trial_runner: "trial_runner.TrialRunner") -> Optional[Trial]:
        for trial in trial_runner.get_trials():
            if (trial.status == Trial.PENDING
                    and trial_runner.has_resources(trial.resources)):
                return trial
        for trial in trial_runner.get_trials():
            if (trial.status == Trial.PAUSED
                    and trial_runner.has_resources(trial.resources)):
                return trial
        return None

    def debug_string(self) -> str:
        return "Using FIFO scheduling algorithm."
