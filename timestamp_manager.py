


class Timestamp_manager(object):
    def __init__(self):
        self._start_times = []
        self._end_times = []

    @property
    def start_times(self):
        return self._start_times

    @property
    def end_times(self):
        return self._end_times


    @start_times.setter
    def start_times(self, value):
        idx, start_time = value
        self._start_times.append([idx, start_time])

    @end_times.setter
    def end_times(self, value):
        idx, end_time = value
        self._end_times.append([idx, end_time])


    def get_time_diff_every_n_inputs(self, n_inputs):
        print('timestamp manager... ')
        start_times = sorted(self._start_times, key=lambda x: x[1], reverse=False)
        end_times = sorted(self._end_times, key=lambda  x: x[1], reverse=False)

        print('start time: ', start_times)
        print('end time: ', end_times)
        idx = 0
        batch_start_time = 0
        batch_end_time = 0
        for start_time, end_time in zip(start_times, end_times):
            if batch_start_time == 0:
                batch_start_time = start_time

            batch_end_time = end_time

            if (idx + 1) % n_inputs == 0:
                print(batch_end_time[1] - batch_start_time[1])
                batch_start_time = 0


            idx = idx + 1

        print('...')
        print('tatol time: ', end_times[-1][1] - start_times[0][1])

    def clearAll(self):
        self._start_times = []
        self._end_times = []

