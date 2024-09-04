import os
import random, ast
import torch
import numpy as np
import pandas as pd
import pandas as pd
from rdkit.Chem import AllChem as Chem


class ForceField(torch.nn.Module):
    def __init__(self, dataset, job_details, phi0_flag=True):
        super(ForceField, self).__init__()
        self.learning_rate = job_details.learning_rate
        self.phi0_flag = phi0_flag

        # Bond
        num_types = len(dataset.batches[0].get_data("bond", "type").unique())
        top_target = np.array([])
        types = np.array([])
        unique_types = dataset.batches[0].get_data("bond", "type").unique().tolist()
        top_per_type = {}
        k_per_type = {}
        for batch in dataset.batches:
            top_target = np.concatenate(
                (top_target, batch.get_data("bond", "b").squeeze(1).tolist())
            )
            types = np.concatenate(
                (types, batch.get_data("bond", "type").squeeze(1).tolist())
            )

        for cur_type in unique_types:
            inx = np.where(types == cur_type)[0]
            top_per_type[cur_type] = (
                (
                    1
                    - (
                        random.random() * 2 * job_details.top_perturb_hyp
                        - job_details.top_perturb_hyp
                    )
                )
                * top_target[inx].mean(0)
            ) ** 0.5
            k_per_type[cur_type] = (
                (
                    (
                        1
                        - (
                            random.random() * 2 * job_details.top_perturb_hyp
                            - job_details.top_perturb_hyp
                        )
                    )
                    * 300
                )
                ** 0.5
            ) / 10

        self.k_bond = torch.tensor(
            list(k_per_type.values()), requires_grad=True, dtype=torch.float
        )
        self.b0 = torch.tensor(
            list(top_per_type.values()), requires_grad=True, dtype=torch.float
        )  # b0 is in sqrt root of actual equilibrium
        print("initial bond lengths:", self.b0.pow(2))
        print("initial bond stiffness:", (self.k_bond * 10).pow(2))

        # Angle
        num_types = len(dataset.batches[0].get_data("angle", "type").unique())
        top_target = np.array([])
        types = np.array([])
        unique_types = dataset.batches[0].get_data("angle", "type").unique().tolist()
        top_per_type = {}
        k_per_type = {}
        for batch in dataset.batches:
            top_target = np.concatenate(
                (top_target, batch.get_data("angle", "theta").squeeze(1).tolist())
            )
            types = np.concatenate(
                (types, batch.get_data("angle", "type").squeeze(1).tolist())
            )

        for cur_type in unique_types:
            inx = np.where(types == cur_type)[0]
            top_per_type[cur_type] = (
                (
                    1
                    - (
                        random.random() * 2 * job_details.top_perturb_hyp
                        - job_details.top_perturb_hyp
                    )
                )
                * top_target[inx].mean(0)
            ) ** 0.5
            k_per_type[cur_type] = (
                (
                    (
                        1
                        - (
                            random.random() * 2 * job_details.top_perturb_hyp
                            - job_details.top_perturb_hyp
                        )
                    )
                    * 50
                )
                ** 0.5
            ) / 10

        self.k_angle = torch.tensor(
            list(k_per_type.values()), requires_grad=True, dtype=torch.float
        )
        self.theta0 = torch.tensor(
            list(top_per_type.values()), requires_grad=True, dtype=torch.float
        )  # b0 is in sqrt root of actual equilibrium

        # Dihedral
        num_types = len(dataset.batches[0].get_data("dihedral", "type").unique())
        self.k1 = torch.tensor(
            ((torch.zeros(num_types) + 0.01) / 20).tolist(),
            requires_grad=True,
            dtype=torch.float,
        )
        self.k2 = torch.tensor(
            ((torch.zeros(num_types) + 0.01) / 20).tolist(),
            requires_grad=True,
            dtype=torch.float,
        )
        self.k3 = torch.tensor(
            ((torch.zeros(num_types) + 0.01) / 20).tolist(),
            requires_grad=True,
            dtype=torch.float,
        )
        self.k4 = torch.tensor(
            ((torch.zeros(num_types) + 0.01) / 20).tolist(),
            requires_grad=True,
            dtype=torch.float,
        )

        # Improper
        num_types = len(dataset.batches[0].get_data("improper", "type").unique())
        self.k_improper = torch.tensor(
            ((torch.zeros(num_types) + 0.00) / 1).tolist(),
            requires_grad=True,
            dtype=torch.float,
        )

        # Pair
        num_types = len(dataset.batches[0].get_data("node", "type").unique())

        if job_details.learn_charge_flag:
            Q_target = np.array([])
            types = np.array([])
            unique_types = dataset.batches[0].get_data("node", "type").unique().tolist()
            q_per_type = {}
            for batch in dataset.batches:
                Q_target = np.concatenate(
                    (Q_target, batch.get_data("node", "q").squeeze(1).tolist())
                )
                types = np.concatenate(
                    (types, batch.get_data("node", "type").squeeze(1).tolist())
                )
            for cur_type in unique_types:
                inx = np.where(types == cur_type)[0]
                q_per_type[cur_type] = (
                    (
                        1
                        - (
                            random.random() * 2 * job_details.top_perturb_hyp
                            - job_details.top_perturb_hyp
                        )
                    )
                    * Q_target[inx].mean(0)
                    * job_details.ion_sc_hyp
                )
        path = os.path.join(job_details.TEMPLATEDIR, "template.params")
        df = pd.read_csv(path, index_col=0)

        self.num_types = num_types

        _param = {}
        for name in ["sigma", "epsilon", "charge"]:
            _param[name] = torch.tensor(list(df[name].fillna(0))).to(torch.float)

        type_map = {}
        to_type = df["types"].to_dict()
        for name in _param.keys():  # ['sigma', 'epsilon', 'charge']
            print(name)
            to_custom_type = -1 * torch.ones(num_types).to(torch.long)
            for custom_type in to_type.keys():
                if np.isnan(df[name][custom_type]):
                    continue
                for t in ast.literal_eval(to_type[custom_type]):
                    to_custom_type[t] = custom_type
            type_map[name] = to_custom_type.view(-1, 1)
        self.type_map = type_map

        self._sigma = torch.tensor(
            _param["sigma"].tolist(), requires_grad=True, dtype=torch.float
        )
        self._epsilon = torch.tensor(
            _param["epsilon"].tolist(), requires_grad=True, dtype=torch.float
        )
        self._charge = torch.tensor(
            _param["charge"].tolist(), requires_grad=True, dtype=torch.float
        )
        self.sigma = torch.tensor(
            ((torch.zeros(num_types) + 1.0) / 1).tolist(),
            requires_grad=True,
            dtype=torch.float,
        )
        self.epsilon = torch.tensor(
            ((torch.zeros(num_types) + 0.1) / 1).tolist(),
            requires_grad=True,
            dtype=torch.float,
        )
        if job_details.learn_charge_flag:
            self.charge = torch.tensor(
                list(q_per_type.values()), requires_grad=True, dtype=torch.float
            )
        else:
            self.charge = torch.tensor(
                (torch.zeros(num_types)).tolist(), requires_grad=True, dtype=torch.float
            )
        print("initial charges:", self.charge)

        self.set_params()

        self.learned_params = {
            "k_bond": [],
            "b0": [],
            "k_angle": [],
            "theta0": [],
            "k_bond_grad": [],
            "b0_grad": [],
            "k_angle_grad": [],
            "theta0_grad": [],
            "k1": [],
            "k2": [],
            "k3": [],
            "k4": [],
            "k_improper": [],
            "_sigma": [],
            "_epsilon": [],
            "_charge": [],
            "sigma": [],
            "epsilon": [],
            "charge": [],
            "sigma_grad": [],
            "epsilon_grad": [],
            "charge_grad": [],
        }

    def set_params(self):
        self.params = {
            "k_bond": self.k_bond.tolist(),
            "b0": self.b0.tolist(),
            "k_angle": self.k_angle.tolist(),
            "theta0": self.theta0.tolist(),
            "k1": self.k1.tolist(),
            "k2": self.k2.tolist(),
            "k3": self.k3.tolist(),
            "k4": self.k4.tolist(),
            "k_improper": self.k_improper.tolist(),
            "_sigma": self._sigma.tolist(),
            "_epsilon": self._epsilon.tolist(),
            "_charge": self._charge.tolist(),
            "sigma": self.sigma.tolist(),
            "epsilon": self.epsilon.tolist(),
            "charge": self.charge.tolist(),
        }

    def train(self, device):
        self.k_bond = torch.tensor(
            self.params["k_bond"], requires_grad=True, dtype=torch.float, device=device
        )
        self.b0 = torch.tensor(
            self.params["b0"], requires_grad=True, dtype=torch.float, device=device
        )
        self.k_angle = torch.tensor(
            self.params["k_angle"], requires_grad=True, dtype=torch.float, device=device
        )
        self.theta0 = torch.tensor(
            self.params["theta0"], requires_grad=True, dtype=torch.float, device=device
        )
        self.k1 = torch.tensor(
            self.params["k1"], requires_grad=True, dtype=torch.float, device=device
        )
        self.k2 = torch.tensor(
            self.params["k2"], requires_grad=True, dtype=torch.float, device=device
        )
        self.k3 = torch.tensor(
            self.params["k3"], requires_grad=True, dtype=torch.float, device=device
        )
        self.k4 = torch.tensor(
            self.params["k4"], requires_grad=True, dtype=torch.float, device=device
        )
        self.k_improper = torch.tensor(
            self.params["k_improper"],
            requires_grad=True,
            dtype=torch.float,
            device=device,
        )
        self._sigma = torch.tensor(
            self.params["_sigma"], requires_grad=True, dtype=torch.float, device=device
        )
        self._epsilon = torch.tensor(
            self.params["_epsilon"],
            requires_grad=True,
            dtype=torch.float,
            device=device,
        )
        self._charge = torch.tensor(
            self.params["_charge"], requires_grad=True, dtype=torch.float, device=device
        )
        self.sigma = torch.tensor(
            self.params["sigma"], requires_grad=True, dtype=torch.float, device=device
        )
        self.epsilon = torch.tensor(
            self.params["epsilon"], requires_grad=True, dtype=torch.float, device=device
        )
        self.charge = torch.tensor(
            self.params["charge"], requires_grad=True, dtype=torch.float, device=device
        )
        self.optimizer = torch.optim.Adam(
            [
                self.k_bond,
                self.b0,
                self.k_angle,
                self.theta0,
                self.k1,
                self.k2,
                self.k3,
                self.k4,
                self.k_improper,
                self.sigma,
                self.charge,
                self.epsilon,
            ],
            lr=self.learning_rate,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, "min"
        )

    def update_learned_params(self):
        self.learned_params["k_bond"].append(self.k_bond.tolist())
        self.learned_params["b0"].append(self.b0.tolist())
        self.learned_params["k_angle"].append(self.k_angle.tolist())
        self.learned_params["theta0"].append(self.theta0.tolist())
        self.learned_params["k_improper"].append(self.k_improper.tolist())
        self.learned_params["k1"].append(self.k1.tolist())
        self.learned_params["k2"].append(self.k2.tolist())
        self.learned_params["k3"].append(self.k3.tolist())
        self.learned_params["k4"].append(self.k4.tolist())
        self.learned_params["_sigma"].append(self._sigma.tolist())
        self.learned_params["_epsilon"].append(self._epsilon.tolist())
        self.learned_params["_charge"].append(self._charge.tolist())
        self.learned_params["sigma"].append(self.sigma.tolist())
        self.learned_params["epsilon"].append(self.epsilon.tolist())
        self.learned_params["charge"].append(self.charge.tolist())
        self.learned_params["k_bond_grad"].append(self.k_bond.grad.tolist())
        self.learned_params["b0_grad"].append(self.b0.grad.tolist())
        self.learned_params["k_angle_grad"].append(self.k_angle.grad.tolist())
        self.learned_params["theta0_grad"].append(self.theta0.grad.tolist())
        self.learned_params["sigma_grad"].append(self.sigma.grad.tolist())
        self.learned_params["epsilon_grad"].append(self.epsilon.grad.tolist())
        self.learned_params["charge_grad"].append(self.charge.grad.tolist())

    def bond(self, batch, job_details, train_flag=False, energy_flag=False):
        topology_dict = batch.GetTopology("bond", create_graph=True)
        types = topology_dict["type"]
        num = topology_dict["num"]
        index = topology_dict["index"]
        graph_index = topology_dict["graph_index"]
        b = topology_dict["b"]
        db = topology_dict["db"].view(-1, 2, 3)
        K_BOND = (self.k_bond * 10).pow(2)
        B0 = self.b0.pow(2)
        if len(types) == 0:
            E_bond = torch.zeros(len(num), 1).to(torch.float).to(types.device)
            F = 0.0
        else:
            if energy_flag:
                E_bond = K_BOND[types] * (b - B0[types]).pow(2)
                E_bond = (
                    torch.zeros(len(num))
                    .to(types.device)
                    .scatter_add(0, graph_index.view(-1), E_bond.view(-1))
                    .view(-1, 1)
                )
                return E_bond
            else:
                F = torch.zeros(batch.N, 3).to(types.device)
                f = 2 * K_BOND[types] * (b - B0[types]).pow(1)
                index = index.view(-1, 1).expand(2 * len(index), 3)
                if len(index) != 0:
                    f = (f.unsqueeze(1) * db).view(-1, 3)
                    F = F - torch.zeros_like(F).scatter_add(0, index, f)
        if len(batch._data["bond"]["type"]) == 0:
            batch._data["bond"]["k2"] = torch.tensor([]).view(-1, 1)
            batch._data["bond"]["b0"] = torch.tensor([]).view(-1, 1)
        else:
            batch._data["bond"]["k2"] = K_BOND[types].view(-1, 1)
            batch._data["bond"]["b0"] = B0[types].view(-1, 1)
        return F

    def angle(self, batch, job_details, train_flag=False, energy_flag=False):
        topology_dict = batch.GetTopology("angle", create_graph=True)
        types = topology_dict["type"]
        num = topology_dict["num"]
        index = topology_dict["index"]
        graph_index = topology_dict["graph_index"]
        theta = topology_dict["theta"]
        dtheta = topology_dict["dtheta"].view(-1, 3, 3)
        K_ANGLE = (self.k_angle * 10).pow(2)
        THETA0 = self.theta0.pow(2)
        if len(types) == 0:
            E_angle = torch.zeros(len(num), 1).to(torch.float).to(types.device)
            F = 0.0
        else:
            if energy_flag:
                E_angle = K_ANGLE[types] * (theta - THETA0[types]).pow(2)
                E_angle = (
                    torch.zeros(len(num))
                    .to(types.device)
                    .scatter_add(0, graph_index.view(-1), E_angle.view(-1))
                    .view(-1, 1)
                )
                return E_angle
            else:
                F = torch.zeros(batch.N, 3).to(types.device)
                f = 2 * K_ANGLE[types] * (theta - THETA0[types]).pow(1)
                index = index.view(-1, 1).expand(3 * len(index), 3)
                if len(index) != 0:
                    f = (f.unsqueeze(1) * dtheta).view(-1, 3)
                    F = F - torch.zeros_like(F).scatter_add(0, index, f)

        if len(batch._data["angle"]["type"]) == 0:
            batch._data["angle"]["k2"] = torch.tensor([]).view(-1, 1)
            batch._data["angle"]["theta0"] = torch.tensor([]).view(-1, 1)
        else:
            batch._data["angle"]["k2"] = K_ANGLE[types].view(-1, 1)
            batch._data["angle"]["theta0"] = THETA0[types].view(-1, 1)
        return F

    def get_dihedral_params(self, job_details):
        K1 = 20 * self.k1
        K2 = 20 * self.k2
        K3 = 20 * self.k3
        K4 = 20 * self.k4
        return K1, K2, K3, K4

    def dihedral(self, batch, job_details, train_flag=False, energy_flag=False):
        K1 = 20 * self.k1
        K2 = 20 * self.k2
        K3 = 20 * self.k3
        K4 = 20 * self.k4
        topology_dict = batch.GetTopology("dihedral", create_graph=True)
        types = topology_dict["type"]
        num = topology_dict["num"]
        index = topology_dict["index"]
        graph_index = topology_dict["graph_index"]
        phi = topology_dict["phi"]
        # dphi = topology_dict['dphi'].view(-1,4,3)
        # sin_phi = topology_dict['sin_phi']
        # sin_2phi = topology_dict['sin_2phi']
        # sin_3phi = topology_dict['sin_3phi']
        # sin_4phi = topology_dict['sin_4phi']

        if len(types) == 0:
            E_dihedral = torch.zeros(len(num), 1).to(torch.float).to(types.device)
            F = 0.0
        else:
            E_dihedral = 0.0
            E_dihedral = E_dihedral + 0.5 * K1[types] * (
                1 + topology_dict["cos_phi"].to(job_details.device)
            )
            E_dihedral = E_dihedral + 0.5 * K2[types] * (
                1 - topology_dict["cos_2phi"].to(job_details.device)
            )
            E_dihedral = E_dihedral + 0.5 * K3[types] * (
                1 + topology_dict["cos_3phi"].to(job_details.device)
            )
            E_dihedral = E_dihedral + 0.5 * K4[types] * (
                1 - topology_dict["cos_4phi"].to(job_details.device)
            )
            E_dihedral = (
                torch.zeros(len(num))
                .to(types.device)
                .scatter_add(0, graph_index.view(-1), E_dihedral.view(-1))
                .view(-1, 1)
            )
            if energy_flag == True:
                return E_dihedral
            F = -1.0 * torch.autograd.grad(
                E_dihedral.sum(), batch.get_data("node", "xyz"), create_graph=True
            )[0].to(job_details.device)

        if len(batch._data["dihedral"]["type"]) == 0:
            batch._data["dihedral"]["k1"] = torch.tensor([]).view(-1, 1)
            batch._data["dihedral"]["k2"] = torch.tensor([]).view(-1, 1)
            batch._data["dihedral"]["k3"] = torch.tensor([]).view(-1, 1)
            batch._data["dihedral"]["k4"] = torch.tensor([]).view(-1, 1)
        else:
            batch._data["dihedral"]["k1"] = K1[types].view(-1, 1)
            batch._data["dihedral"]["k2"] = K2[types].view(-1, 1)
            batch._data["dihedral"]["k3"] = K3[types].view(-1, 1)
            batch._data["dihedral"]["k4"] = K4[types].view(-1, 1)
        return F

    def improper(self, batch, job_details, train_flag=False, energy_flag=False):
        topology_dict = batch.GetTopology("improper", create_graph=True)
        types = topology_dict["type"]
        num = topology_dict["num"]
        index = topology_dict["index"]
        graph_index = topology_dict["graph_index"]
        chi = topology_dict["chi"].to(job_details.device)
        dchi = topology_dict["dchi"].view(-1, 4, 3)
        K_IMPROPER = 100 * self.k_improper.pow(2)
        if len(types) == 0:
            E_improper = torch.zeros(len(num), 1).to(torch.float).to(types.device)
            F = 0.0
        else:
            E_improper = K_IMPROPER[types] * chi.pow(2)
            E_improper = (
                torch.zeros(len(num))
                .to(types.device)
                .scatter_add(0, graph_index.view(-1), E_improper.view(-1))
                .view(-1, 1)
            )
            if energy_flag:
                return E_improper
            F = -1.0 * torch.autograd.grad(
                E_improper.sum(), batch.get_data("node", "xyz"), create_graph=True
            )[0].to(job_details.device)

        if len(batch._data["improper"]["type"]) == 0:
            batch._data["improper"]["k2"] = torch.tensor([]).view(-1, 1)
        else:
            batch._data["improper"]["k2"] = K_IMPROPER[types].view(-1, 1)
        return F

    def pair(self, batch, job_details, energy_flag=False):
        _SIGMA = self._sigma
        _EPSILON = self._epsilon
        _CHARGE = self._charge
        SIGMA = self.sigma.pow(2)
        EPSILON = self.epsilon
        CHARGE = self.charge

        param = {}
        param["sigma"] = SIGMA
        param["epsilon"] = EPSILON
        param["charge"] = CHARGE

        _param = {}
        _param["sigma"] = _SIGMA
        _param["epsilon"] = _EPSILON
        _param["charge"] = _CHARGE

        topology_dict = batch.GetTopology("pair", create_graph=True)
        types = topology_dict["type"]
        base_types = topology_dict["base_type"]
        num = topology_dict["num"]
        index = topology_dict["index"]
        graph_index = topology_dict["graph_index"]
        inv_D = topology_dict["inv_D"].to(job_details.device)
        inv_D_2 = topology_dict["inv_D_2"].to(job_details.device)
        prefactor = topology_dict["f_1_4"]
        C = topology_dict["C"]

        _types = (
            torch.arange(self.num_types)
            .to(torch.long)
            .view(-1, 1)
            .to(job_details.device)
        )
        PARAM = {}
        for name in param.keys():
            custom_types = self.type_map[name][_types.view(-1)].to(job_details.device)
            custom_type_mask = (custom_types < 0).to(torch.float)
            PARAM[name] = 0.0
            PARAM[name] = PARAM[name] + _param[name][custom_types] * (
                1 - custom_type_mask
            )
            PARAM[name] = PARAM[name] + param[name][_types] * custom_type_mask
        #         print(PARAM['epsilon'])

        if len(types) == 0:
            E_pair = torch.zeros(len(num), 1).to(torch.float).to(job_details.device)
            F = 0.0
        else:
            # geometric mixing
            SIGMA_MIXED = (PARAM["sigma"][types]).prod(1).pow(0.5).view(-1, 1)
            EPSILON_MIXED = (PARAM["epsilon"][types]).prod(1).pow(0.5).view(-1, 1)
            Q = PARAM["charge"][types]
            CHARGE_PRODUCT = Q.prod(1).view(-1, 1)
            D_inv_scal = SIGMA_MIXED * inv_D
            D_inv_scal_3 = D_inv_scal.pow(3)
            D_inv_scal_6 = D_inv_scal_3.pow(2)

            E_LJ = (
                4
                * EPSILON_MIXED
                * ((SIGMA_MIXED * inv_D).pow(12) - (SIGMA_MIXED * inv_D).pow(6))
            )
            E_coulomb = 332.0636 * CHARGE_PRODUCT * inv_D
            E_pair = 0.0
            E_pair = E_pair + E_LJ * prefactor
            E_pair = E_pair + E_coulomb * prefactor
            E_pair = (
                torch.zeros(len(num))
                .to(job_details.device)
                .scatter_add(0, graph_index.view(-1), E_pair.view(-1))
                .view(-1, 1)
            )
            if energy_flag:
                return E_pair
            F = -1.0 * torch.autograd.grad(
                E_pair.sum(), batch.get_data("node", "xyz"), create_graph=True
            )[0].to(job_details.device)

        types = batch.get_data("node", "type")
        batch._data["node"]["sigma"] = (PARAM["sigma"][types]).view(-1, 1)
        batch._data["node"]["epsilon"] = (PARAM["epsilon"][types]).view(-1, 1)
        batch._data["node"]["charge"] = (PARAM["charge"][types]).view(-1, 1)
        return F

    def get_charge_loss(self, batch, job_details):
        _CHARGE = self._charge
        CHARGE = self.charge
        Q_target = batch.get_data("node", "q")
        types = batch.get_data("node", "type")
        Q_MSE = 0

        if job_details.dip_hyp:
            multipole_target = torch.cat(
                (
                    batch.get_data("prop", "multipole_x"),
                    batch.get_data("prop", "multipole_y"),
                    batch.get_data("prop", "multipole_z"),
                ),
                dim=1,
            )
        param = {}
        param["charge"] = CHARGE
        _param = {}
        _param["charge"] = _CHARGE

        _types = (
            torch.arange(self.num_types).to(torch.long).view(-1, 1).to(CHARGE.device)
        )
        PARAM = {}
        for name in param.keys():
            custom_types = self.type_map[name][_types.view(-1)].to(_types.device)
            custom_type_mask = (custom_types < 0).to(torch.float)
            PARAM[name] = 0.0
            PARAM[name] = PARAM[name] + _param[name][custom_types] * (
                1 - custom_type_mask
            )
            PARAM[name] = PARAM[name] + param[name][_types] * custom_type_mask

        c = batch.get_data("node", "component").to(torch.long)
        c = list(c.split(batch.get_data("num", "node").view(-1).tolist()))
        for i in range(len(c)):
            c[i] = c[i].unique(return_counts=True)[1]
        c = torch.cat(c).tolist()

        # Charge loss between current AutoBADDIE atom charges and avg DFT atom charges
        # this will be the same thing every molecule...
        Q = PARAM["charge"][types].squeeze(1)
        both_Q = torch.cat([Q_target, Q], dim=1)
        if job_details.ion_sc_hyp != 1.0:
            split_Q = list(both_Q.split(c))  # split into molecules
            lens = torch.LongTensor([len(cur) for cur in split_Q])
            unique_lens = lens.unique()
            both_Q = []
            for cur in unique_lens:
                first_inx = np.where(lens == cur)[0][0]
                if cur == 1 or cur == 15:
                    both_Q.append(
                        torch.cat(
                            [
                                split_Q[first_inx][:, [0]] * job_details.ion_sc_hyp,
                                split_Q[first_inx][:, [1]],
                            ],
                            dim=1,
                        )
                    )
                else:
                    both_Q.append(
                        torch.cat(
                            [split_Q[first_inx][:, [0]], split_Q[first_inx][:, [1]]],
                            dim=1,
                        )
                    )
            both_Q = torch.cat(both_Q, 0)
        Q_MSE = (
            Q_MSE
            + (both_Q[:, [0]] - both_Q[:, [1]]).pow(2).mean() * job_details.atom_hyp
        )  # MSE multiplied by hyperparameter

        # Charge loss from per-molecule net charge (-1 for TFSI, 0 for polymer, +1 for [Li+], etc)
        both_Q = torch.cat([Q_target, Q], dim=1)  # get unscaled DFT charges
        both_Q = list(both_Q.split(c))  # split into molecules
        for i in range(len(both_Q)):  # This adds to get the total charge per molecule
            both_Q[i] = both_Q[i].sum(0)
        both_Q = torch.stack(both_Q)
        both_Q[:, [0]] = (
            both_Q[:, [0]].round() * job_details.ion_sc_hyp
        )  # round the target total molecule charge to integer molecular charge
        Q_MSE = (
            Q_MSE
            + (both_Q[:, [0]] - both_Q[:, [1]]).pow(2).mean() * job_details.mol_hyp
        )  # MSE multiplied by hyperparameter

        # multipole difference loss (try to match x,y,z components of conformer diopole to the database values)
        if job_details.dip_hyp:
            Q_MSE = (
                Q_MSE
                + (multipole - multipole_target).pow(2).sum(1).pow(0.5).mean()
                * job_details.dip_hyp
            )

        # charge regularization loss
        if job_details.creg_hyp != None:
            Q_MSE = Q_MSE + PARAM["charge"].pow(2).sum() * job_details.creg_hyp

        return Q_MSE
