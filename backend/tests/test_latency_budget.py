import unittest

from app.pipeline.latency_budget import LatencyTracker, STAGE_BUDGETS, TOTAL_BUDGET


class LatencyTrackerTests(unittest.TestCase):
    def test_empty_report_has_all_stages(self):
        tracker = LatencyTracker()
        report = tracker.report()
        self.assertEqual(set(report), set(STAGE_BUDGETS))
        self.assertTrue(all(item["count"] == 0 for item in report.values()))

    def test_record_and_report_metrics(self):
        tracker = LatencyTracker()
        tracker.record("llm_inference", 100)
        tracker.record("llm_inference", 200)
        tracker.record("unknown_stage", 999)
        report = tracker.report()["llm_inference"]
        self.assertEqual(report["count"], 2)
        self.assertEqual(report["min_ms"], 100)
        self.assertEqual(report["max_ms"], 200)
        self.assertEqual(tracker.latest()["llm_inference"], 200)

    def test_measurements_are_bounded(self):
        tracker = LatencyTracker()
        for value in range(1005):
            tracker.record("response_encode", value)
        self.assertEqual(tracker.report()["response_encode"]["count"], 1000)
        self.assertEqual(tracker.latest()["response_encode"], 1004)

    def test_total_budget_is_derived_from_stage_budgets(self):
        self.assertEqual(TOTAL_BUDGET["p50_ms"], sum(item["p50_ms"] for item in STAGE_BUDGETS.values()))
        self.assertEqual(TOTAL_BUDGET["p95_ms"], sum(item["p95_ms"] for item in STAGE_BUDGETS.values()))


if __name__ == "__main__":
    unittest.main()