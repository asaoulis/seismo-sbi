

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
    continuous_regions.append((current_end, end_time))

    # Append the last continuous region and any remaining gap
    return continuous_regions, gaps