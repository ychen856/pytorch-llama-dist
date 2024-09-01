import gc
import math

class Calcualte_opt(object):

    def __init__(self):
        self._start_idx = 0
        self._end_idx = 0
        self._end_idx_buff = 0
        self._statisitc_period = 10

        self._client_comp_statistics = []
        self._server_comp_statistics = []
        self._comm_statistics = []
        self._max_end_idx = 0
        self._last_opt_calc_time = math.inf
        self._outgoint_count = 0
        self._incoming_count = 0
        self._steady_state = False

    @property
    def start_idx(self):
        return self._start_idx

    @property
    def end_idx(self):
        return self._end_idx

    @property
    def end_idx_buff(self):
        return self._end_idx_buff

    @property
    def statistic_period(self):
        return self._statisitc_period

    @property
    def client_comp_statistics(self):
        return self._client_comp_statistics

    @property
    def server_comp_statistics(self):
        return self._server_comp_statistics

    @property
    def comm_statistics(self):
        return self._comm_statistics

    @property
    def max_end_idx(self):
        return self._max_end_idx

    @property
    def outgoint_count(self):
        return self._outgoint_count

    @property
    def incoming_count(self):
        return self._incoming_count

    @property
    def last_opt_calc_time(self):
        return self.last_opt_calc_time

    @property
    def steady_state(self):
        return self._steady_state

    @client_comp_statistics.setter
    def client_comp_statistics(self, value): #[end_idx, buff_end_idx, comp_time]
        end_idx, buff_end_idx, comp_time = value
        self._client_comp_statistics.append([end_idx, buff_end_idx, comp_time])

    @server_comp_statistics.setter
    def server_comp_statistics(self, value): #[start_idx, comp_time]
        start_idx, comp_time = value
        self._server_comp_statistics.append([start_idx, comp_time])

    @comm_statistics.setter
    def comm_statistics(self, value):
        self._comm_statistics.append(value)

    @max_end_idx.setter
    def max_end_idx(self, end_idx):
        self._max_end_idx = max(self._max_end_idx, end_idx)

    @outgoint_count.setter
    def outgoint_count(self, value):
        self._outgoint_count = value

    @incoming_count.setter
    def incoming_count(self, value):
        self._incoming_count = value

    @start_idx.setter
    def start_idx(self, value):
        self._start_idx = value

    @end_idx.setter
    def end_idx(self, value):
        self._end_idx = value

    @end_idx_buff.setter
    def end_idx_buff(self, value):
        self._end_idx_buff = value

    @statistic_period.setter
    def statistic_period(self, value):
        self._statisitc_period = value

    @steady_state.setter
    def steady_state(self, value):
        self._steady_state = value

    def calclate_opt(self):
        print('do opt')
        #print('FFFFFFFFFFFFFFFFFFFF: ', self._client_comp_statistics)
        #print('ZZZZZZZZZZZZZZZZZZZZ: ', self._server_comp_statistics)
        client_comp_time_temp = sorted(self._client_comp_statistics[:len(self._server_comp_statistics)], key=lambda x: x[0])
        server_comp_time_temp = self._server_comp_statistics

        #print('fffffffffffffffffff: ', client_comp_time_temp)
        #print('zzzzzzzzzzzzzzzzzzz: ', server_comp_time_temp)

        client_end_idx = client_comp_time_temp[0][0]
        avg_client_comp_time = 0
        avg_server_comp_time = 0
        opt_comp_time = math.inf
        opt_splitting_point = 0
        client_count = 0
        server_count = 0
        for i in range(0, len(client_comp_time_temp)):
            if client_end_idx == client_comp_time_temp[i][0]:
                #print('clientPPPPPP: ', client_comp_time_temp[i])
                client_count = client_count + 1
                avg_client_comp_time = avg_client_comp_time + client_comp_time_temp[i][2]


                server_count = 0
            else:
                for j in range(0, len(server_comp_time_temp)):
                    if server_comp_time_temp[j][0] == client_end_idx + 1:
                        #print('serverVVVV: ', server_comp_time_temp[j])
                        server_count = server_count + 1
                        avg_server_comp_time = avg_client_comp_time + server_comp_time_temp[j][1]
                #print('client count: ', client_count)
                #print('server count: ', server_count)
                #print('+++ end idx: ', client_end_idx)
                #print('+++ time: ', (avg_client_comp_time/ client_count + avg_server_comp_time/ server_count))
                if client_count > 0 and server_count > 0 and (avg_client_comp_time/client_count + avg_server_comp_time / server_count) < opt_comp_time:
                    opt_splitting_point = client_end_idx
                    opt_comp_time = avg_client_comp_time/client_count + avg_server_comp_time/server_count

                client_end_idx = client_comp_time_temp[i][0]
                avg_client_comp_time = client_comp_time_temp[i][2]
                client_count = 1

        min_client_comp_time = 10000
        opt_buff_idx = 0
        for i in range(0, len(client_comp_time_temp)):
            if opt_splitting_point == client_comp_time_temp[i][0]:
                if client_comp_time_temp[i][2] < min_client_comp_time:
                    min_client_comp_time = client_comp_time_temp[i][2]
                    opt_buff_idx = client_comp_time_temp[i][1]

        self._client_comp_statistics = self._client_comp_statistics[len(client_comp_time_temp) :]
        self._server_comp_statistics = self._client_comp_statistics[len(server_comp_time_temp) :]
        #self.comm_statistics = [max(len(self._server_comp_statistics), 10) :]

        self._end_idx = opt_splitting_point
        self._end_idx_buff = opt_buff_idx

        #print('last opt: ', self._last_opt_calc_time)
        #print('opt: ', opt_comp_time)
        if self._last_opt_calc_time * 1.5 < opt_comp_time:
            self._statisitc_period = max(10, self._statisitc_period - 4)
        elif self._last_opt_calc_time * 1.3 > opt_comp_time:
            self._statisitc_period = min(300, self._statisitc_period + 8)

        #self._last_opt_calc_time = min(self._last_opt_calc_time, opt_comp_time)
        self._last_opt_calc_time = opt_comp_time


        gc.collect()
        #print('opt splitting point: ', opt_splitting_point)
        #print('statisitc period: ', self._statisitc_period)
        if self._statisitc_period > 20:
            self._steady_state = True


        return opt_splitting_point, opt_buff_idx, self._statisitc_period
