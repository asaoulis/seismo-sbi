import unittest
from datetime import datetime, timedelta


def get_continuous_regions(event_start_end_times, start_time, end_time):
    continuous_regions = []
    gaps = []

    # Sort the list of start and end datetime pairs
    sorted_ranges = sorted(event_start_end_times, key=lambda x: x[0])

    # Initialize the start and end times with the first event
    current_start, current_end = sorted_ranges[0]
    continuous_regions.append((start_time, current_start))
    # Iterate through the sorted_ranges
    for start, end in sorted_ranges[1:]:
        if start > current_end:
            continuous_regions.append((current_end, start))
            gaps.append((current_start, current_end))
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    gaps.append((current_start, current_end))
    continuous_regions.append((current_end, end_time))

    # Append the last continuous region and any remaining gap
    return continuous_regions, gaps

class TestGetContinuousRegions(unittest.TestCase):

    def test_continuous_regions(self):
        event_start_end_times = [
            (datetime(2021, 7, 5, 8, 0), datetime(2021, 7, 5, 10, 0)),
            (datetime(2021, 7, 5, 11, 30), datetime(2021, 7, 5, 12, 30)),
            (datetime(2021, 7, 5, 14, 0), datetime(2021, 7, 5, 16, 0)),
            # ... more time ranges ...
        ]
        start_time = datetime(2021, 7, 5, 7, 0)
        end_time = datetime(2021, 7, 5, 18, 0)

        continuous_regions, _ = get_continuous_regions(event_start_end_times, start_time, end_time)

        # Expected continuous regions
        expected_continuous_regions = [
            (start_time, datetime(2021, 7, 5, 8, 0)),
            (datetime(2021, 7, 5, 10, 0), datetime(2021, 7, 5, 11, 30)),
            (datetime(2021, 7, 5, 12, 30), datetime(2021, 7, 5, 14, 0)),
            (datetime(2021, 7, 5, 16, 0), end_time)
        ]

        self.assertEqual(continuous_regions, expected_continuous_regions)

    def test_overlapping_continuous_regions(self):
        event_start_end_times = [
            (datetime(2021, 7, 5, 8, 0), datetime(2021, 7, 5, 10, 0)),
            (datetime(2021, 7, 5, 11, 30), datetime(2021, 7, 5, 12, 30)),
            (datetime(2021, 7, 5, 11, 45), datetime(2021, 7, 5, 13, 30)),
            (datetime(2021, 7, 5, 14, 0), datetime(2021, 7, 5, 16, 0)),
            # ... more time ranges ...
        ]
        start_time = datetime(2021, 7, 5, 7, 0)
        end_time = datetime(2021, 7, 5, 18, 0)

        continuous_regions, _ = get_continuous_regions(event_start_end_times, start_time, end_time)

        # Expected continuous regions
        expected_continuous_regions = [
            (start_time, datetime(2021, 7, 5, 8, 0)),
            (datetime(2021, 7, 5, 10, 0), datetime(2021, 7, 5, 11, 30)),
            (datetime(2021, 7, 5, 13, 30), datetime(2021, 7, 5, 14, 0)),
            (datetime(2021, 7, 5, 16, 0), end_time)
        ]

        self.assertEqual(continuous_regions, expected_continuous_regions)

    def test_gaps(self):
        event_start_end_times = [
            (datetime(2021, 7, 5, 8, 0), datetime(2021, 7, 5, 10, 0)),
            (datetime(2021, 7, 5, 11, 30), datetime(2021, 7, 5, 12, 30)),
            (datetime(2021, 7, 5, 14, 0), datetime(2021, 7, 5, 16, 0)),
            # ... more time ranges ...
        ]
        start_time = datetime(2021, 7, 5, 7, 0)
        end_time = datetime(2021, 7, 5, 18, 0)

        _, gaps = get_continuous_regions(event_start_end_times, start_time, end_time)
        # Expected gaps
        expected_gaps = [
            (datetime(2021, 7, 5, 8, 0), datetime(2021, 7, 5, 10, 0)),
            (datetime(2021, 7, 5, 11, 30), datetime(2021, 7, 5, 12, 30)),
            (datetime(2021, 7, 5, 14, 0), datetime(2021, 7, 5, 16, 0))
        ]

        self.assertEqual(gaps, expected_gaps)

    def test_overlapping_gaps(self):
        event_start_end_times = [
            (datetime(2021, 7, 5, 8, 0), datetime(2021, 7, 5, 10, 0)),
            (datetime(2021, 7, 5, 11, 30), datetime(2021, 7, 5, 12, 30)),
            (datetime(2021, 7, 5, 11, 45), datetime(2021, 7, 5, 13, 30)),
            (datetime(2021, 7, 5, 14, 0), datetime(2021, 7, 5, 16, 0)),
            # ... more time ranges ...
        ]
        start_time = datetime(2021, 7, 5, 7, 0)
        end_time = datetime(2021, 7, 5, 18, 0)

        _, gaps = get_continuous_regions(event_start_end_times, start_time, end_time)
        # Expected gaps
        expected_gaps = [
            (datetime(2021, 7, 5, 8, 0), datetime(2021, 7, 5, 10, 0)),
            (datetime(2021, 7, 5, 11, 30), datetime(2021, 7, 5, 13, 30)),
            (datetime(2021, 7, 5, 14, 0), datetime(2021, 7, 5, 16, 0))
        ]

        self.assertEqual(gaps, expected_gaps)

if __name__ == '__main__':
    unittest.main()
