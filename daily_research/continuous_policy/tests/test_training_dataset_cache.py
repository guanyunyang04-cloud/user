import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import pandas as pd

from daily_research.continuous_policy import train_policy
from daily_research.continuous_policy.training_dataset_cache import (
    build_training_dataset_fingerprint,
    load_training_dataset_cache,
    save_training_dataset_cache,
)


class TrainingDatasetCacheTest(unittest.TestCase):
    def test_training_dataset_cache_round_trips_all_constructed_training_sets(self) -> None:
        sample_frame = pd.DataFrame(
            {
                "stock": ["A", "B"],
                "date": ["2026-05-13", "2026-05-13"],
                "action_label": ["reduce", "open"],
                "current_weight": [0.20, 0.0],
            }
        )
        daily_frame = pd.DataFrame({"date": ["2026-05-13"], "gross_exposure_target": [0.80]})
        teacher_summary = {"label_preset": "holdcash_v3", "train_sample_rows": 2}
        spec = {"pool": "liquid500", "label_preset": "holdcash_v3", "budget_objective": "result_value_v10"}

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            saved = save_training_dataset_cache(
                cache_root=root,
                spec=spec,
                sample_frame=sample_frame,
                daily_frame=daily_frame,
                teacher_summary=teacher_summary,
            )
            loaded = load_training_dataset_cache(cache_root=root, spec=spec)

        self.assertIsNotNone(loaded)
        assert loaded is not None
        pd.testing.assert_frame_equal(loaded.sample_frame, sample_frame)
        pd.testing.assert_frame_equal(loaded.daily_frame, daily_frame)
        self.assertEqual(loaded.teacher_summary, teacher_summary)
        self.assertEqual(loaded.cache_key, saved.cache_key)
        self.assertTrue(str(saved.cache_dir).endswith(saved.cache_key))

    def test_training_dataset_cache_fingerprint_is_stable_and_invalidates_changed_specs(self) -> None:
        left = {"b": 2, "a": {"x": 1}}
        same = {"a": {"x": 1}, "b": 2}
        changed = {"a": {"x": 2}, "b": 2}

        self.assertEqual(build_training_dataset_fingerprint(left), build_training_dataset_fingerprint(same))
        self.assertNotEqual(build_training_dataset_fingerprint(left), build_training_dataset_fingerprint(changed))

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            save_training_dataset_cache(
                cache_root=root,
                spec=left,
                sample_frame=pd.DataFrame({"x": [1]}),
                daily_frame=pd.DataFrame({"d": [1]}),
                teacher_summary={"ok": True},
            )

            self.assertIsNotNone(load_training_dataset_cache(cache_root=root, spec=same))
            self.assertIsNone(load_training_dataset_cache(cache_root=root, spec=changed))

    def test_train_policy_exposes_reusable_training_dataset_cache_by_default(self) -> None:
        parser = train_policy.build_parser()
        args = parser.parse_args([])
        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }

        self.assertIn("--training-dataset-cache-mode", option_strings)
        self.assertEqual(args.training_dataset_cache_mode, "auto")

    def test_train_policy_exposes_prepare_only_mode_for_reusable_dataset_builds(self) -> None:
        parser = train_policy.build_parser()
        args = parser.parse_args(["--prepare-only"])
        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }

        self.assertIn("--prepare-only", option_strings)
        self.assertTrue(args.prepare_only)


if __name__ == "__main__":
    unittest.main()
