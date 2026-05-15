import unittest

from daily_research.continuous_policy import evaluate_policy, export_action_panel, train_policy


class LakeEvaluatorCliContractTest(unittest.TestCase):
    def test_evaluate_policy_parser_accepts_lake(self) -> None:
        args = evaluate_policy.build_parser().parse_args(
            [
                "--data-source",
                "lake",
                "--lake-dataset-id",
                "policy_input_bundle__fixture",
                "--data-lake-root",
                "H:/lake",
            ]
        )

        self.assertEqual(args.data_source, "lake")
        self.assertEqual(args.lake_dataset_id, "policy_input_bundle__fixture")
        self.assertEqual(args.data_lake_root, "H:/lake")

    def test_export_action_panel_parser_accepts_lake(self) -> None:
        args = export_action_panel.build_parser().parse_args(
            [
                "--data-source",
                "lake",
                "--lake-dataset-id",
                "policy_input_bundle__fixture",
                "--data-lake-root",
                "H:/lake",
            ]
        )

        self.assertEqual(args.data_source, "lake")
        self.assertEqual(args.lake_dataset_id, "policy_input_bundle__fixture")
        self.assertEqual(args.data_lake_root, "H:/lake")

    def test_train_policy_parser_accepts_lake_without_changing_strict_gold_default(self) -> None:
        args = train_policy.build_parser().parse_args(
            [
                "--data-source",
                "lake",
                "--lake-dataset-id",
                "policy_input_bundle__fixture",
                "--data-lake-root",
                "H:/lake",
            ]
        )

        self.assertEqual(args.data_source, "lake")
        self.assertEqual(args.lake_dataset_id, "policy_input_bundle__fixture")
        self.assertEqual(args.data_lake_root, "H:/lake")
        self.assertEqual(args.training_dataset_id, "")


if __name__ == "__main__":
    unittest.main()
