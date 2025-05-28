import collections
import math
from datetime import datetime, timedelta

class PerformanceDataStore:
    def __init__(self, max_records_per_type=20, statisitc_period=10):
        """
        Initializes the CommunicationDataStore.

        Args:
            max_records_per_type (int): The maximum number of records to store
                                        for each unique (start_index, end_index) pair.
        """
        # Main storage for complete records (client + server data), keyed by (start_index, end_index)
        self.data_storage = collections.defaultdict(collections.deque)
        self.max_records_per_type = max_records_per_type

        # Temporary storage for pending client data.
        # Key: client_end_index (since this is what we have)
        # Value: List of client records with that end_index, as multiple client events might share an end_index
        self.pending_client_data = collections.defaultdict(collections.deque)

        # Temporary storage for pending server data.
        # Key: server_start_index (since this is what we have)
        # Value: List of server records with that start_index
        self.pending_server_data = collections.defaultdict(collections.deque)

        self.optimal_latency_history = math.inf


        self._start_idx = 0
        self._end_idx = 0
        self._end_idx_buff = 0
        self._max_end_idx = 0
        self._statisitc_period = statisitc_period
        self._outgoing_count = 0
        self._incoming_count = 0
        self._new_record_count = 0
        self._steady_state = False


    @property
    def start_idx(self):
        return self._start_idx

    @start_idx.setter
    def start_idx(self, value):
        self._start_idx = value

    @property
    def end_idx(self):
        return self._end_idx

    @end_idx.setter
    def end_idx(self, value):
        self._end_idx = value

    @property
    def end_idx_buff(self):
        return self._end_idx_buff

    @end_idx_buff.setter
    def end_idx_buff(self, value):
        self._end_idx_buff = value

    @property
    def max_end_idx(self):
        return self._max_end_idx

    @max_end_idx.setter
    def max_end_idx(self, end_idx):
        self._max_end_idx = max(self._max_end_idx, end_idx)

    @property
    def statistic_period(self):
        return self._statisitc_period

    @statistic_period.setter
    def statistic_period(self, value):
        self._statisitc_period = value

    @property
    def outgoing_count(self):
        return self._outgoing_count

    @outgoing_count.setter
    def outgoing_count(self, value):
        self._outgoing_count = value

    @property
    def incoming_count(self):
        return self._incoming_count

    @incoming_count.setter
    def incoming_count(self, value):
        self._incoming_count = value

    @property
    def new_record_count(self):
        return self._new_record_count

    @new_record_count.setter
    def new_record_count(self, value):
        self._new_record_count = value

    @property
    def steady_state(self):
        return self._steady_state

    @steady_state.setter
    def steady_state(self, value):
        self._steady_state = value

    def _add_complete_record_to_main_storage(self, complete_record):
        """
        Helper to manage adding a complete record to the main data_storage,
        respecting the max_records_per_type limit for its (start_index, end_index) key.
        """
        start_idx = complete_record["start_index"]
        end_idx = complete_record["end_index"]
        key = (start_idx, end_idx)

        if len(self.data_storage[key]) < self.max_records_per_type:
            self.data_storage[key].append(complete_record)
        else:
            self.data_storage[key].popleft()  # Remove the oldest
            self.data_storage[key].append(complete_record)

        self.new_record_count = self.new_record_count + 1

        #print('matched data: ', self.data_storage)
        print(f"[{datetime.now()}] Combined record added to main storage for key {key}. Current count: {len(self.data_storage[key])}")

    def add_client_info(self, timestamp, client_end_index, end_index_buffer, client_computation_time):
        """
        Stores client-side information and attempts to form complete records
        by matching with pending server data based on the linking rule
        (server_start_idx == client_end_index + 1) and oldest timestamp preference.

        Args:
            timestamp (datetime): The timestamp when the data was recorded on the client.
            client_end_index (int): The end index collected by the client.
            end_index_buffer (any): The end index buffer data from client.
            client_computation_time (float): The client's computation time.
        """
        client_data = {
            "timestamp": timestamp,
            "client_end_index": client_end_index,
            "end_index_buffer": end_index_buffer,
            "client_computation_time": client_computation_time,
        }
        # print(f"[{datetime.now()}] Client: Received data (client_end_idx={client_end_index}, timestamp={timestamp}). Checking for server match...")

        server_match_start_index_key = client_end_index + 1

        matched_server_record = None
        if server_match_start_index_key in self.pending_server_data and self.pending_server_data[
            server_match_start_index_key]:
            oldest_timestamp = datetime.max
            oldest_server_rec_to_match = None

            # Find the oldest matching server record
            for server_rec in self.pending_server_data[server_match_start_index_key]:
                if server_rec["timestamp"] < oldest_timestamp:
                    oldest_timestamp = server_rec["timestamp"]
                    oldest_server_rec_to_match = server_rec

            if oldest_server_rec_to_match:
                matched_server_record = oldest_server_rec_to_match
                # Now remove the specific matched record from the deque
                # Create a new deque containing all elements EXCEPT the matched_server_record
                new_deque = collections.deque(
                    [rec for rec in self.pending_server_data[server_match_start_index_key]
                     if rec is not matched_server_record]  # Use 'is' for identity comparison
                )
                self.pending_server_data[server_match_start_index_key] = new_deque

        if matched_server_record:
            complete_record = {
                "timestamp": client_data["timestamp"],
                "start_index": matched_server_record["start_index"],
                "end_index": matched_server_record["end_index"],
                "client_end_index": client_data["client_end_index"],
                "end_index_buffer": client_data["end_index_buffer"],
                "client_computation_time": client_data["client_computation_time"],
                "server_computation_time": matched_server_record["server_computation_time"],
                "communication_time": matched_server_record["communication_time"],
            }
            self._add_complete_record_to_main_storage(complete_record)
        else:
            self.pending_client_data[client_end_index].append(client_data)
            # print(f"[{datetime.now()}] Client: No server match for client_end_idx={client_end_index}. Stored as pending.")

    def add_server_info(self, timestamp, server_start_index, server_end_index, server_computation_time,
                        communication_time):
        """
        Stores server-side information and attempts to form complete records
        by matching with pending client data based on the linking rule
        (server_start_idx == client_end_index + 1) and oldest timestamp preference.

        Args:
            timestamp (datetime): The timestamp when the data was recorded on the server.
            server_start_index (int): The starting index from server.
            server_end_index (int): The ending index from server.
            server_computation_time (float): The server's computation time.
            communication_time (float): The communication time.
        """
        server_data = {
            "timestamp": timestamp,
            "start_index": server_start_index,
            "end_index": server_end_index,
            "server_computation_time": server_computation_time,
            "communication_time": communication_time,
        }
        # print(f"[{datetime.now()}] Server: Received data (start_idx={server_start_index}, end_idx={server_end_index}, timestamp={timestamp}). Checking for client match...")

        client_match_end_index_key = server_start_index - 1

        print('pending client data: ', self.pending_client_data)
        matched_client_record = None
        if client_match_end_index_key in self.pending_client_data and self.pending_client_data[
            client_match_end_index_key]:
            oldest_timestamp = datetime.max
            oldest_client_rec_to_match = None

            # Find the oldest matching client record
            for client_rec in self.pending_client_data[client_match_end_index_key]:
                if client_rec["timestamp"] < oldest_timestamp:
                    oldest_timestamp = client_rec["timestamp"]
                    oldest_client_rec_to_match = client_rec

            if oldest_client_rec_to_match:
                matched_client_record = oldest_client_rec_to_match
                # Create a new deque containing all elements EXCEPT the matched_client_record
                new_deque = collections.deque(
                    [rec for rec in self.pending_client_data[client_match_end_index_key]
                     if rec is not matched_client_record]  # Use 'is' for identity comparison
                )
                self.pending_client_data[client_match_end_index_key] = new_deque

        if matched_client_record:
            complete_record = {
                "timestamp": matched_client_record["timestamp"],
                "start_index": server_data["start_index"],
                "end_index": server_data["end_index"],
                "client_end_index": matched_client_record["client_end_index"],
                "end_index_buffer": matched_client_record["end_index_buffer"],
                "client_computation_time": matched_client_record["client_computation_time"],
                "server_computation_time": server_data["server_computation_time"],
                "communication_time": server_data["communication_time"],
            }
            self._add_complete_record_to_main_storage(complete_record)
        else:
            self.pending_server_data[server_start_index].append(server_data)
            # print(f"[{datetime.now()}] Server: No client match for server_start_idx={server_start_index}. Stored as pending.")




    def get_data(self, start_index, end_index):
        """
        Retrieves all stored complete data for a specific (start_index, end_index) pair.
        """
        key = (start_index, end_index)
        return list(self.data_storage.get(key, collections.deque()))

    def get_all_data(self):
        """
        Retrieves all complete data stored in the system, grouped by (start_index, end_index).
        """
        return {k: list(v) for k, v in self.data_storage.items()}

    def get_new_record_count(self):
        return self.new_record_count

    def get_num_pending_client_items(self):
        return sum(len(deque) for deque in self.pending_client_data.values())

    def get_num_pending_server_items(self):
        return sum(len(deque) for deque in self.pending_server_data.values())

    def get_pending_client_items_by_end_index(self, end_index):
        return list(self.pending_client_data.get(end_index, collections.deque()))

    def get_pending_server_items_by_start_index(self, start_index):
        return list(self.pending_server_data.get(start_index, collections.deque()))

    def get_pending_data_status(self):
        status = {
            "pending_client": {},
            "pending_server": {},
            "total_pending_client": self.get_num_pending_client_items(),
            "total_pending_server": self.get_num_pending_server_items(),
        }
        for k, v in self.pending_client_data.items():
            status["pending_client"][k] = len(v)
        for k, v in self.pending_server_data.items():
            status["pending_server"][k] = len(v)
        return status

    def get_total_stored_records(self):
        """
        Returns the total number of complete records stored across all (start_index, end_index) sets.
        """
        total_records = 0
        for deque_of_records in self.data_storage.values():
            total_records += len(deque_of_records)
        return total_records

def calculate_opt(data_store: PerformanceDataStore):
    """
    Calculates the average of client computation time, server computation time,
    and communication time for each (server_start_index, server_end_index) set, and
    returns the set with the minimal total average latency, along with convergence status.

    Args:
        data_store (CommunicationDataStore): An instance of the CommunicationDataStore
                                             containing the collected data.

    Returns:
        tuple or None: A tuple (
            start_index,
            end_index,
            minimal_total_latency,
            is_converging: bool,
            latency_diff: float or None
        ) if complete data is available, otherwise None.
    """
    all_data = data_store.get_all_data()
    if not all_data:
        return None

    min_latency = float('inf')
    current_optimal_set = None

    for (start_idx, end_idx), records in all_data.items():
        if not records:
            continue

        total_client_time = 0.0
        total_server_time = 0.0
        total_communication_time = 0.0
        num_valid_records = 0

        for record in records:
            if (record["client_computation_time"] is not None and
                record["server_computation_time"] is not None and
                record["communication_time"] is not None):
                total_client_time += record["client_computation_time"]
                total_server_time += record["server_computation_time"]
                total_communication_time += record["communication_time"]
                num_valid_records += 1

        if num_valid_records == 0:
            continue

        avg_client_time = total_client_time / num_valid_records
        avg_server_time = total_server_time / num_valid_records
        avg_communication_time = total_communication_time / num_valid_records

        total_average_latency = avg_client_time + avg_server_time + avg_communication_time

        if total_average_latency < min_latency:
            min_latency = total_average_latency
            current_optimal_set = (start_idx, end_idx, min_latency)

    if current_optimal_set:
        if data_store.optimal_latency_history *  1.1 < min_latency:
            data_store._statisitc_period = max(10, math.floor(data_store._statisitc_period * 2 / 3))
        elif data_store.optimal_latency_history *  1.1 > min_latency:
            data_store._statisitc_period = min(100, data_store._statisitc_period + 6)

        data_store.optimal_latency_history = min_latency

    if data_store._statisitc_period > 20:
        data_store._steady_state = True

    data_store.max_records_per_type = data_store.statistic_period + 10

    data_store._new_record_count = 0
    print('statistic period: ', data_store._statisitc_period)
    return current_optimal_set[0] - 1, current_optimal_set[0] + 1, data_store._statisitc_period

if __name__ == "__main__":
    # Test with convergence history size of 3 and threshold of 0.01
    data_store = PerformanceDataStore(max_records_per_type=2, convergence_history_size=3, convergence_threshold=0.01)

    print("--- Simulating data arrival over time to observe convergence (Corrected Logic) ---")

    # Round 1: Initial data
    print("\n--- Round 1 ---")
    data_store.add_client_info(datetime.now() - timedelta(seconds=10), 100, "buf_A_1_old", 0.05) # Client 100, old
    data_store.add_client_info(datetime.now() - timedelta(seconds=9), 100, "buf_A_1_new", 0.06) # Client 100, new
    data_store.add_server_info(datetime.now() - timedelta(seconds=8), 101, 500, 0.1, 0.03) # Server 101-500. Matches old client 100.
    # Expected: (101,500) combined with "buf_A_1_old". "buf_A_1_new" remains pending.

    data_store.add_client_info(datetime.now() - timedelta(seconds=7), 200, "buf_B_1", 0.06)
    data_store.add_server_info(datetime.now() - timedelta(seconds=6), 201, 501, 0.12, 0.04)
    # Expected: (201,501) combined with "buf_B_1".

    opt_result_1 = calculate_opt(data_store)
    if opt_result_1:
        start_idx, end_idx, min_latency, is_converging, diff = opt_result_1
        print(f"Optimal (S_idx, E_idx): ({start_idx}, {end_idx}), Latency: {min_latency:.4f}, Converging: {is_converging}, Diff: {diff:.4f if diff is not None else 'N/A'}")
    print(f"Optimal Latency History: {[f'{l:.4f}' for l in data_store.get_optimal_latency_history()]}")
    print(f"Pending status: {data_store.get_pending_data_status()}")


    # Round 2: Data brings a slightly lower optimal latency
    print("\n--- Round 2 ---")
    data_store.add_client_info(datetime.now() - timedelta(seconds=5), 300, "buf_C_1_old", 0.03)
    data_store.add_client_info(datetime.now() - timedelta(seconds=4), 300, "buf_C_1_new", 0.035)
    data_store.add_server_info(datetime.now() - timedelta(seconds=3), 301, 502, 0.07, 0.01) # Latency ~ 0.11 (new optimal candidate)
    # Expected: (301,502) combined with "buf_C_1_old". "buf_C_1_new" remains pending.

    opt_result_2 = calculate_opt(data_store)
    if opt_result_2:
        start_idx, end_idx, min_latency, is_converging, diff = opt_result_2
        print(f"Optimal (S_idx, E_idx): ({start_idx}, {end_idx}), Latency: {min_latency:.4f}, Converging: {is_converging}, Diff: {diff:.4f if diff is not None else 'N/A'}")
    print(f"Optimal Latency History: {[f'{l:.4f}' for l in data_store.get_optimal_latency_history()]}")
    print(f"Pending status: {data_store.get_pending_data_status()}")


    # Round 3: Data brings another optimal latency, now history is full, check convergence
    print("\n--- Round 3 ---")
    data_store.add_client_info(datetime.now() - timedelta(seconds=2), 400, "buf_D_1", 0.02)
    data_store.add_server_info(datetime.now() - timedelta(seconds=1), 401, 503, 0.06, 0.005) # Latency ~ 0.085 (potentially new optimal)

    opt_result_3 = calculate_opt(data_store)
    if opt_result_3:
        start_idx, end_idx, min_latency, is_converging, diff = opt_result_3
        print(f"Optimal (S_idx, E_idx): ({start_idx}, {end_idx}), Latency: {min_latency:.4f}, Converging: {is_converging}, Diff: {diff:.4f if diff is not None else 'N/A'}")
    print(f"Optimal Latency History: {[f'{l:.4f}' for l in data_store.get_optimal_latency_history()]}")
    print(f"Pending status: {data_store.get_pending_data_status()}")


    # Round 4: Simulate stability/convergence
    print("\n--- Round 4 (Simulating Convergence) ---")
    # This should yield a very similar optimal latency
    data_store.add_client_info(datetime.now() + timedelta(seconds=0), 400, "buf_D_2", 0.021)
    data_store.add_server_info(datetime.now() + timedelta(seconds=1), 401, 503, 0.059, 0.006) # Latency ~ 0.086 (very close to previous optimal)

    opt_result_4 = calculate_opt(data_store)
    if opt_result_4:
        start_idx, end_idx, min_latency, is_converging, diff = opt_result_4
        print(f"Optimal (S_idx, E_idx): ({start_idx}, {end_idx}), Latency: {min_latency:.4f}, Converging: {is_converging}, Diff: {diff:.4f if diff is not None else 'N/A'}")
    print(f"Optimal Latency History: {[f'{l:.4f}' for l in data_store.get_optimal_latency_history()]}")
    print(f"Pending status: {data_store.get_pending_data_status()}")

    # Test with no data
    print("\n--- Testing with no data ---")
    empty_data_store = PerformanceDataStore()
    empty_opt = calculate_opt(empty_data_store)
    print(f"Result for empty store: {empty_opt}")
