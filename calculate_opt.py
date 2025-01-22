import gc
import math
import time


def find_row(table, idx, value):
    new_table = []
    for row in table:
        if row[idx] == value:
            new_table.append(row)
    return new_table

class Calcualte_opt(object):

    def __init__(self):
        self._start_idx = 0
        self._end_idx = 0
        self._end_idx_buff = 0
        #self._layer_amout = 0
        self._statisitc_period = 10

        self._client_comp_statistics = []
        self._gateway_comp_statistics = []
        self._server_comp_statistics = []
        self._comm_statistics = []
        self._max_end_idx = 0
        self._max_layer_amount = 0
        self._last_opt_calc_time = math.inf
        self._outgoint_count = 0
        self._incoming_count = 0
        self._steady_state = False

        self._gateway_opt_table = []    #[[gateway_start_idx, gateway_end_idx, opt_gateway_layer_amount, opt_buff_idx, opt_comp_time], [], ...]

    @property
    def start_idx(self):
        return self._start_idx

    @property
    def end_idx(self):
        return self._end_idx

    @property
    def end_idx_buff(self):
        return self._end_idx_buff

    '''@property
    def layer_amount(self):
        return self._layer_amout'''

    @property
    def statistic_period(self):
        return self._statisitc_period

    @property
    def client_comp_statistics(self):
        return self._client_comp_statistics

    @property
    def gateway_comp_statistics(self):
        return self._gateway_comp_statistics

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
    def max_layer_amount(self):
        return self._max_layer_amount

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

    @property
    def gateway_opt_table(self):
        return self._gateway_opt_table

    @client_comp_statistics.setter
    def client_comp_statistics(self, value): #[end_idx, buff_end_idx, comp_time]
        end_idx, buff_end_idx, comp_time = value
        self._client_comp_statistics.append([end_idx, buff_end_idx, comp_time])

    @gateway_comp_statistics.setter
    def gateway_comp_statistics(self, value): #[start_idx, end_dix, buff_end_idx, comp_time]
        start_idx, end_idx, layer_amount, buff_end_idx, comp_time = value
        self._gateway_comp_statistics.append([start_idx, end_idx, layer_amount, buff_end_idx, comp_time])

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

    @max_layer_amount.setter
    def max_layer_amount(self, value):
        self._max_layer_amount = max(self._max_layer_amount, value)

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

    '''@layer_amount.setter
    def layer_amount(self, value):
        self._layer_amout = value'''

    @statistic_period.setter
    def statistic_period(self, value):
        self._statisitc_period = value

    @steady_state.setter
    def steady_state(self, value):
        self._steady_state = value

    @gateway_opt_table.setter
    def gateway_opt_table(self, value):
        start_idx, end_idx, layer_amount, buff_idx, comp_time = value
        self._gateway_opt_table.append([start_idx, end_idx, layer_amount, buff_idx, comp_time])


    def calclate_opt(self):
        print('do opt')
        #print('FFFFFFFFFFFFFFFFFFFF: ', self._client_comp_statistics)
        #print('ZZZZZZZZZZZZZZZZZZZZ: ', self._server_comp_statistics)
        client_comp_time_temp = sorted(self._client_comp_statistics[:len(self._server_comp_statistics)], key=lambda x: x[0])
        server_comp_time_temp = self._server_comp_statistics

        '''for client in client_comp_time_temp:
            print('CLIENT SIDE!!!: ', client)
        for server in server_comp_time_temp:
            print('SERVER SIDE!!!: ', server)'''

        #print('fffffffffffffffffff: ', client_comp_time_temp)
        #print('zzzzzzzzzzzzzzzzzzz: ', server_comp_time_temp)

        client_end_idx = client_comp_time_temp[0][0]
        avg_client_comp_time = 0
        avg_server_comp_time = 0
        opt_comp_time = math.inf
        opt_splitting_point = 0
        client_count = 0
        server_count = 0
        i = 0
        #for i in range(0, len(client_comp_time_temp)):
        while i < len(client_comp_time_temp):
            if client_end_idx == client_comp_time_temp[i][0]:
                #print('clientPPPPPP: ', client_comp_time_temp[i])
                client_count = client_count + 1
                avg_client_comp_time = avg_client_comp_time + client_comp_time_temp[i][2]

                server_count = 0
                i = i + 1
            else:
                for j in range(0, len(server_comp_time_temp)):
                    #print('is match??')
                    #print('client data + 1: ', client_end_idx + 1)
                    #print('server data: ', server_comp_time_temp[j])
                    if server_comp_time_temp[j][0] == client_end_idx + 1:
                        #print('serverVVVV: ', server_comp_time_temp[j])
                        server_count = server_count + 1
                        avg_server_comp_time = avg_server_comp_time + server_comp_time_temp[j][1]
                #print('client count: ', client_count)
                #print('server count: ', server_count)
                #print('+++ end idx: ', client_end_idx)
                #print('+++ time: ', (avg_client_comp_time/ client_count + avg_server_comp_time/ server_count))
                if client_count > 0 and server_count > 0 and (avg_client_comp_time/client_count + avg_server_comp_time / server_count) < opt_comp_time:
                    print('avg time: ', avg_client_comp_time/client_count + avg_server_comp_time / server_count)
                    opt_splitting_point = client_end_idx
                    opt_comp_time = avg_client_comp_time/client_count + avg_server_comp_time/server_count

                client_end_idx = client_comp_time_temp[i][0]
                #avg_client_comp_time = client_comp_time_temp[i][2]
                avg_client_comp_time = 0
                avg_server_comp_time = 0
                client_count = 0

        for j in range(0, len(server_comp_time_temp)):
            #print('is match??')
            #print('client data + 1: ', client_end_idx + 1)
            #print('server data: ', server_comp_time_temp[j])
            if server_comp_time_temp[j][0] == client_end_idx + 1:
                #print('serverVVVV: ', server_comp_time_temp[j])
                server_count = server_count + 1
                avg_server_comp_time = avg_server_comp_time + server_comp_time_temp[j][1]
        print('client count: ', client_count)
        print('server count: ', server_count)
        print('avg client: ', avg_client_comp_time)
        print('avg server: ', avg_server_comp_time)
        # print('+++ end idx: ', client_end_idx)
        # print('+++ time: ', (avg_client_comp_time/ client_count + avg_server_comp_time/ server_count))
        if client_count > 0 and server_count > 0 and (avg_client_comp_time / client_count + avg_server_comp_time / server_count) < opt_comp_time:
            print('avg time: ', avg_client_comp_time/client_count + avg_server_comp_time / server_count)
            opt_splitting_point = client_end_idx
            opt_comp_time = avg_client_comp_time / client_count + avg_server_comp_time / server_count

        min_client_comp_time = 10000
        opt_buff_idx = 0
        for i in range(0, len(client_comp_time_temp)):
            if opt_splitting_point == client_comp_time_temp[i][0]:
                if client_comp_time_temp[i][2] < min_client_comp_time:
                    min_client_comp_time = client_comp_time_temp[i][2]
                    opt_buff_idx = client_comp_time_temp[i][1]

        self._client_comp_statistics = self._client_comp_statistics[len(client_comp_time_temp) :]
        self._server_comp_statistics = self._server_comp_statistics[len(server_comp_time_temp) :]
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


    def calclate_opt_gateway(self, start_idx):
        print('do opt')
        #print('FFFFFFFFFFFFFFFFFFFF: ', self._client_comp_statistics)
        #print('ZZZZZZZZZZZZZZZZZZZZ: ', self._server_comp_statistics)
        gateway_comp_time_temp = sorted(self._gateway_comp_statistics[:len(self._server_comp_statistics)], key=lambda x: x[0])
        server_comp_time_temp = self._server_comp_statistics

        #print('fffffffffffffffffff: ', gateway_comp_time_temp)
        #print('zzzzzzzzzzzzzzzzzzz: ', server_comp_time_temp)

        gateway_start_idx = gateway_comp_time_temp[0][0]
        avg_gateway_comp_time = 0
        avg_server_comp_time = 0
        opt_comp_time = math.inf
        opt_gateway_layer_amount = 0
        opt_splitting_point = 0
        client_count = 0
        server_count = 0

        i = 0
        while i < len(gateway_comp_time_temp):
            gateway_start_idx = gateway_comp_time_temp[i][0]
            print('gateway_start_idx: ', gateway_start_idx)
            gateway_sub_list = find_row(gateway_comp_time_temp, 0, gateway_start_idx)
            print('gateway sub list: ', gateway_sub_list)
            gateway_sub_list_temp = sorted(gateway_sub_list, key=lambda x: x[1])
            gateway_end_idx = gateway_sub_list_temp[0][1]
            for j in range(0, len(gateway_sub_list_temp)):
                if gateway_end_idx == gateway_comp_time_temp[j][1]:
                    print('clientPPPPPP: ', client_comp_time_temp[i])
                    client_count = client_count + 1
                    avg_gateway_comp_time = avg_gateway_comp_time + gateway_comp_time_temp[i][3]

                    server_count = 0
                else:
                    for k in range(0, len(server_comp_time_temp)):
                        if server_comp_time_temp[k][0] == gateway_end_idx + 1:
                            print('serverVVVV: ', server_comp_time_temp[j])
                            server_count = server_count + 1
                            avg_server_comp_time = avg_server_comp_time + server_comp_time_temp[k][1]
                    print('client count: ', client_count)
                    print('server count: ', server_count)
                    #print('+++ end idx: ', client_end_idx)
                    #print('+++ time: ', (avg_client_comp_time/ client_count + avg_server_comp_time/ server_count))
                    if client_count > 0 and server_count > 0 and (avg_gateway_comp_time/client_count + avg_server_comp_time / server_count) < opt_comp_time:
                        print('avg: ', avg_gateway_comp_time/client_count + avg_server_comp_time / server_count)
                        opt_gateway_layer_amount = gateway_end_idx - gateway_start_idx
                        opt_comp_time = avg_gateway_comp_time/client_count + avg_server_comp_time/server_count

                    gateway_end_idx = gateway_comp_time_temp[j][1]
                    avg_gateway_comp_time = gateway_comp_time_temp[j][3]
                    client_count = 1

                min_gateway_comp_time = 10000
                opt_buff_idx = 0
                for n in range(0, len(gateway_sub_list_temp)):
                    if gateway_end_idx == gateway_sub_list_temp[n][1]:
                        if gateway_sub_list_temp[n][4] < min_gateway_comp_time:
                            min_gateway_comp_time = gateway_sub_list_temp[n][4]
                            opt_buff_idx = gateway_sub_list_temp[n][3]

                #rint('opt table: ', self._gateway_opt_table)
                list_idx = 0
                for opt_list in self._gateway_opt_table:
                    #print('m: ', list_idx)
                    if self._gateway_opt_table[list_idx][0] == gateway_start_idx and self._gateway_opt_table[list_idx][1] == gateway_end_idx:
                        #print('pop')
                        self._gateway_opt_table.pop(list_idx)
                        list_idx = list_idx - 1

                    list_idx = list_idx + 1

                self.gateway_opt_table = [gateway_start_idx, gateway_end_idx, opt_gateway_layer_amount, opt_buff_idx, opt_comp_time]

            '''i = i + 1
            gateway_start_idx = gateway_comp_time_temp[i + 1][0]
            print('gateway_start_idx: ', gateway_start_idx)'''

            i = i + len(gateway_sub_list)
            #print('next idx: ', i)

            time.sleep(5)

        self._gateway_comp_statistics = self._gateway_comp_statistics[len(gateway_comp_time_temp) :]
        self._server_comp_statistics = self._server_comp_statistics[len(server_comp_time_temp) :]
        #self.comm_statistics = [max(len(self._server_comp_statistics), 10) :]

        #self._end_idx = opt_splitting_point
        #self._end_idx_buff = opt_buff_idx

        print('opt table: ', self._gateway_opt_table)
        #print('start idx: ', start_idx)
        opt_row = find_row(self._gateway_opt_table, 0, start_idx)

        # the opt data haven't been discovered
        if len(opt_row) == 0:
            # return start_idx + opt_gateway_layer_amount, opt_buff_idx, self._statisitc_period
            return start_idx + 2, start_idx + 5, self._statisitc_period

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


        #[gateway_start_idx, gateway_end_idx, opt_gateway_layer_amount, opt_buff_idx, opt_comp_time] = find_row(self._gateway_opt_table, start_idx)

        #return start_idx + opt_gateway_layer_amount, opt_buff_idx, self._statisitc_period
        #print('FFFFFFFFFFFFFFFFF: ', opt_row)
        return start_idx + opt_row[0][2], opt_row[0][3], self._statisitc_period