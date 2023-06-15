import random
from typing import List, Tuple

from shapely.geometry import Point


class BaseStation:
    def __init__(
        self,
        bs_id: int,
        pos: Tuple[float, float],
        bw: float,
        freq: float,
        tx: float,
        height: float,
    ):
        # BS ID should be final, i.e., BS ID must be unique
        # NOTE: cannot use Final typing due to support for Python 3.7
        self.bs_id = bs_id
        self.x, self.y = pos
        self.bw = bw  # in Hz
        self.frequency = freq  # in MHz
        # self.tx_power = tx  # in dBm
        self.tx_power = random.randint(30, 40)  # dB
        self.height = height  # in m
        self.edge_servers = []

    @property
    def coords(self):
        return (self.x, self.y)

    @property
    def point(self):
        return Point(int(self.x), int(self.y))

    def add_edge_server(self, es_id: int):
        self.edge_servers.append(es_id)

    def __str__(self):
        return f"BS: {self.bs_id}"


class UserEquipment:
    def __init__(
        self,
        ue_id: int,
        velocity: float,
        snr_tr: float,
        noise: float,
        height: float,
    ):
        # UE ID should be final, i.e., UE ID must be unique
        # NOTE: cannot use Final typing due to support for Python 3.7
        self.ue_id = ue_id
        self.velocity: float = velocity
        self.snr_threshold = snr_tr
        self.noise = noise
        # self.noise = round(random.uniform((1e-9) - (1e-10), (1e-9) + (1e-10)), 11)
        self.height = height

        self.x: float = None
        self.y: float = None
        self.stime: int = None
        self.extime: int = None
        self.task = None
        self.current_sp = None

    @property
    def point(self):
        return Point(int(self.x), int(self.y))

    @property
    def coords(self):
        return (self.x, self.y)

    def generate_task(self, new=True):
        if new or self.task == None:
            self.task = Task(
                self.ue_id,
                random.randint(1, 16),
                random.randint(1, 128),
                random.randint(1, 200),
            )
        return self.task

    def __str__(self):
        return f"UE: {self.ue_id}"


class EdgeInfrastructureProvider:
    def __init__(self, inp_id) -> None:
        self.inp_id = inp_id
        self.bundle = None
        self.edge_servers = []

    def offer_bundle(self, new=False):
        if self.bundle == None or new:
            self.bundle = {
                "storage": random.randint(1, 1000),  # in GB
                "vCPU": random.randint(1, 416),  # in vCPU
            }
        return self.bundle


class EdgeServer:
    def __init__(self, es_id: int, inp, loc_x: float, loc_y: float) -> None:
        self.es_id = es_id
        self.inp = inp
        self.bs_id = None
        self.loc_x = loc_x
        self.loc_y = loc_y
        self.bundle = None

    def offer_bundle(self):
        self.bundle = self.inp.offer_bundle()
        return self.bundle

    def choose_bid_winner(self, bids: List[Tuple[int, int]]):
        # Pay attention to the case of equal bids
        return max(bids)[1]

    def __str__(self) -> str:
        return f"ES: {self.es_id}"


class ServiceProvider:
    def __init__(self, sp_id: int, budget: float, U: int, R: int, subscription_fee):
        self.sp_id = sp_id
        self.Budget = budget
        self.U = U  # maximum number of user to consider at a timestamp
        self.R = R  # maximum number of bundle offers to consider for a timeslot
        self.subscription_fee = subscription_fee
        self.users = []

    def subscribe(self, ue):
        self.Budget += self.subscription_fee
        self.users.append(ue)

    def bid(self):
        # greedy
        # a3c
        pass


class Task:
    def __init__(
        self,
        ue_id: int,
        computing_req: int,
        data_req: int,
        latency_req: float,
    ):
        self.ue_id = ue_id
        self.computing_req = computing_req
        self.data_req = data_req
        self.latency_req = latency_req

    def __str__(self):
        return f"Task: (ue: {self.ue_id}, req: ({self.computing_req, self.data_req, self.latency_req}))"
