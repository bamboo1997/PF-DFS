import pickle

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import HyperBand

from sampler.utils import suggest_host, suggest_run_id


class HyperBandSampler(object):

    def __init__(
        self,
        cfg,
    ):

        self.cfg = cfg
        self.setup_server()

        if cfg.objective.name in ["lcbench", "nb301"]:
            from worker.lcbench_worker import LCBenchWorker as Worker
        elif cfg.objective.name == "nb201":
            from worker.nb201_worker import NB201Worker as Worker
        elif self.cfg.objective.name == "dnns":
            from worker.dnn_worker import DNNsWorker as Worker

            self.cfg.sampler.n_parallel = 1

        # Start local worker
        self.workers = []
        for i in range(self.cfg.sampler.n_parallel):
            w = Worker(
                self.cfg,
                run_id=self.run_id,
                host=self.host,
                nameserver=self.ns_host,
                nameserver_port=self.ns_port,
                id=i,
            )
            w.run(background=True)
            self.workers.append(w)

        self.sampler = HyperBand(
            configspace=w.cs,
            run_id=self.run_id,
            host=self.host,
            nameserver=self.ns_host,
            nameserver_port=self.ns_port,
            result_logger=self.result_logger,
            min_budget=cfg.sampler.min_budget,
            max_budget=cfg.sampler.max_budget,
        )

    def setup_server(self):
        # Define runid
        self.run_id = suggest_run_id(self.cfg)

        # Define name server
        self.host = suggest_host(self.cfg)
        self.NS = hpns.NameServer(
            run_id=self.run_id,
            host=self.host,
            port=None,
        )
        # Start a nameserver
        self.ns_host, self.ns_port = self.NS.start()

        # Construct logger
        self.result_logger = hpres.json_result_logger(
            directory=self.cfg.out_dir, overwrite=True
        )

    def run(self):
        res = self.sampler.run(n_iterations=self.cfg.sampler.n_trials)

        with open(self.cfg.out_dir + "results.pkl", "wb") as fh:
            pickle.dump(res, fh)

        self.sampler.shutdown(shutdown_workers=True)
        for w in self.workers:
            w.shutdown()
            w.results_logger.show_results()
        self.NS.shutdown()
