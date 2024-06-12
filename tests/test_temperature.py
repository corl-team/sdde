import math
from unittest import TestCase, main

import torch

from openood.pipelines.temperature import TemperatureWrapper


class TestTemperature(TestCase):
    def test_temperature_tuning(self):
        n = 100
        nc = 5
        torch.manual_seed(0)
        for fraction in [0.6, 0.75, 0.9]:
            with torch.no_grad():
                error_targets = torch.randint(0, nc, [n])
                error_mask = torch.rand(n) > fraction
                n_errors = error_mask.float().sum().item()
                targets = error_targets.clone()
                targets[error_mask] = (error_targets[error_mask] + 1) % nc
                logits = torch.nn.functional.one_hot(error_targets).float()
                a = 1 - n_errors / n
                gt_temp = 1 / math.log(a * (nc - 1) / (1 - a))
            temp = TemperatureWrapper._tune_temperature(logits, targets, tol=1e-4)
            print(temp, gt_temp)
            self.assertAlmostEqual(temp, gt_temp, delta=1e-3)


    @staticmethod
    def _likelihood(probs, targets):
        return -torch.nn.functional.cross_entropy(probs.log(), targets).item()


if __name__ == "__main__":
    main()
