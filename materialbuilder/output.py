import torch, mdtraj
import numpy as np
import pandas as pd


def write_lammps_data_file(
    batch, filename, job_details, phi0_flag=True, style="OPLS", single_mol=False
):
    smiles = job_details.to_parameterize
    if style == "compass":
        params = {}
        params["bond"] = ["b0", "k_b2", "k_b3", "k_b4"]
        params["angle"] = ["theta0", "k_a2", "k_a3", "k_a4", "k_bb", "k_ba0", "k_ba1"]
        params["dihedral"] = [
            "k1",
            "k2",
            "k3",
            "phi1",
            "phi2",
            "phi3",
            "k_mt1",
            "k_mt2",
            "k_mt3",
            "k_bct10",
            "k_bct20",
            "k_bct30",
            "k_bct11",
            "k_bct21",
            "k_bct31",
            [],
        ]
        params["improper"] = ["k2"]
        params["node"] = ["mass", "epsilon", "sigma"]
    elif style == "OPLS":
        params = {}
        params["bond"] = ["k2", "b0"]
        params["angle"] = ["k2", "theta0"]
        params["dihedral"] = ["k1", "k2", "k3", "k4"]
        # params["improper"] = ["k2"]
        params["node"] = ["mass", "epsilon", "sigma"]
    else:
        raise Exception("Only compass and OPLS styles are supported.")

    node_types = batch.get_data("node", "type").view(-1).tolist()

    learned_params = {}

    top = "node"
    if top in batch._data.keys():
        df = batch.get_dataframe(top)
        # this was a trick to add Li to TFSI single molecules, but no longer needed
        # if single_mol and smiles == 'O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F':
        #     #index, base_type, type (make very large to be a unique atom type), degree, atomic_num, mass,
        #     #onehot(oh), oh_r1, oh_r2, oh_r3, oh_r4, sigma, epsilon, charge, xyz
        #     df.loc[len(df.index)] = [15, 1, 99999999999, 0, 3, 6.94,
        #                              [], [], [], [], [], 1.635, 0.3727, job_details.param_sc_hyp, [2.5, 2.5, 2.5]]
        df_full_nodes = df.copy()
        df = df[~df["type"].duplicated()]
        df["new_type"] = (1 + torch.arange(len(df))).tolist()
        df = df.set_index("type")
        learned_params[top] = df

    top = "bond"
    if top in batch._data.keys():
        df = batch.get_dataframe(top)
        if len(df) != 0:
            df = df[~df["type"].duplicated()]
            df["new_type"] = (1 + torch.arange(len(df))).tolist()
            df = df.set_index("type")
            learned_params[top] = df

    top = "angle"
    if top in batch._data.keys():
        df = batch.get_dataframe(top)
        if len(df) != 0:
            df_reversed = df.copy(deep=True)
            for column in ["index", "b", "bond"]:
                df_reversed[column] = df_reversed[column].apply(lambda x: x[::-1])
            df = pd.concat([df, df_reversed], ignore_index=True)
            df["node"] = df["index"].apply(lambda x: [node_types[i] for i in x])
            #             new_types = torch.tensor(df['node'].values.tolist())
            #             new_types = new_types.unique(dim=0, return_inverse=True)[1]
            #             new_types = new_types.tolist()
            # repeat the angle types twice in order to have the "reversed index" have the same angle type
            new_types = (
                batch.get_data("angle", "type")
                .unique(dim=0, return_inverse=True)[1]
                .tolist()
                * 2
            )
            #             import ipdb
            #             ipdb.set_trace()
            df["new_type"] = new_types
            df = df[~df["new_type"].duplicated()]
            df["new_type"] = (1 + torch.arange(len(df))).tolist()
            learned_params[top] = df
            bond_indices = (
                torch.tensor(learned_params["angle"]["bond"].values.tolist())
                .view(-1, 2)
                .to(torch.long)
            )
            b0 = batch.get_data("bond", "b0").view(-1)
            b0 = b0[bond_indices]
            learned_params[top]["b0"] = b0.tolist()

    top = "dihedral"
    if top in batch._data.keys():
        df = batch.get_dataframe(top)
        if len(df) != 0:
            df_reversed = df.copy(deep=True)
            for column in ["index", "b", "bond", "theta", "angle"]:
                df_reversed[column] = df_reversed[column].apply(lambda x: x[::-1])
            df = pd.concat([df, df_reversed], ignore_index=True)
            df["node"] = df["index"].apply(lambda x: [node_types[i] for i in x])
            new_types = torch.tensor(df["node"].values.tolist())
            new_types = new_types.unique(dim=0, return_inverse=True)[1]
            new_types = new_types.tolist()
            df["new_type"] = new_types
            df = df[~df["new_type"].duplicated()]
            df["new_type"] = (1 + torch.arange(len(df))).tolist()
            learned_params[top] = df
            bond_indices = (
                torch.tensor(learned_params["dihedral"]["bond"].values.tolist())
                .view(-1, 3)
                .to(torch.long)
            )
            b0 = batch.get_data("bond", "b0").view(-1)
            b0 = b0[bond_indices]
            learned_params[top]["b0"] = b0.tolist()
            angle_indices = (
                torch.tensor(learned_params["dihedral"]["angle"].values.tolist())
                .view(-1, 2)
                .to(torch.long)
            )
            theta0 = batch.get_data("angle", "theta0").view(-1)
            theta0 = theta0[angle_indices]
            learned_params[top]["theta0"] = theta0.tolist()

    top = "improper"
    if top in batch._data.keys():
        df = batch.get_dataframe(top)
        if len(df) != 0:
            df_reversed = df.copy(deep=True)
            for column in ["index", "theta", "angle"]:
                df_reversed[column] = df_reversed[column].apply(lambda x: x[::-1])
            df = pd.concat([df, df_reversed], ignore_index=True)
            df["node"] = df["index"].apply(lambda x: [node_types[i] for i in x])
            new_types = torch.tensor(df["node"].values.tolist())
            new_types = new_types.unique(dim=0, return_inverse=True)[1]
            new_types = new_types.tolist()
            df["new_type"] = new_types
            df = df[~df["new_type"].duplicated()]
            df["new_type"] = (1 + torch.arange(len(df))).tolist()
            learned_params[top] = df
            angle_indices = (
                torch.tensor(learned_params["improper"]["angle"].values.tolist())
                .view(-1, 3)
                .to(torch.long)
            )
            theta0 = batch.get_data("angle", "theta0").view(-1)
            theta0 = theta0[angle_indices]
            learned_params[top]["theta0"] = theta0.tolist()

    types = {}
    for top in ["node", "bond", "angle", "dihedral", "improper"]:
        if top in learned_params.keys():  # batch._data.keys():
            if top == "node":
                types[top] = df_full_nodes["type"].tolist()
            else:
                types[top] = batch.get_data(top, "type").view(-1).tolist()
    # if single_mol and smiles == 'O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F':
    #     xyz = torch.tensor(df_full_nodes['xyz'])
    #     charges = df_full_nodes['charge'].tolist()
    else:
        xyz = batch.get_data("node", "xyz")
        charges = batch.get_data("node", "charge").view(-1).tolist()
    atomic_nums = batch.get_data("node", "atomic_num").view(-1).tolist()
    N = len(atomic_nums)
    # if single_mol and smiles == 'O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F': #add lithium to TFSI for charge neutrality
    #     N+=1
    f = open(filename, "w+")
    tab = " "
    newline = "\n"
    f.write("LAMMPS data file generated by AutoBADDIE for {}.".format(smiles))
    f.write(newline)
    f.write(newline)
    f.write(str(N) + tab + "atoms" + newline)
    for top in ["bond", "angle", "dihedral", "improper"]:
        if top in learned_params.keys():  # batch._data.keys():
            if len(types[top]) == 0:
                continue
            f.write(str(len(types[top])) + tab + "{}s".format(top) + newline)
    f.write(newline)
    f.write(str(learned_params["node"].shape[0]) + tab + "atom types" + newline)
    for top in ["bond", "angle", "dihedral", "improper"]:
        if top in learned_params.keys():  # batch._data.keys():
            # print(top)
            if learned_params[top].shape[0] == 0:
                continue
            f.write(
                str(learned_params[top].shape[0])
                + tab
                + "{} types".format(top)
                + newline
            )
    f.write(newline)
    multiplier = 100
    if single_mol:
        f.write(
            format(-50, "<+13.5f")
            + tab
            + format(50, "<+13.5f")
            + tab
            + "xlo xhi"
            + newline
        )
        f.write(
            format(-50, "<+13.5f")
            + tab
            + format(50, "<+13.5f")
            + tab
            + "ylo yhi"
            + newline
        )
        f.write(
            format(-50, "<+13.5f")
            + tab
            + format(50, "<+13.5f")
            + tab
            + "zlo zhi"
            + newline
        )
    else:
        f.write(
            format(xyz[:, 0].min().item() * multiplier, "<+13.5f")
            + tab
            + format(xyz[:, 0].max().item() * multiplier, "<+13.5f")
            + tab
            + "xlo xhi"
            + newline
        )
        f.write(
            format(xyz[:, 1].min().item() * multiplier, "<+13.5f")
            + tab
            + format(xyz[:, 1].max().item() * multiplier, "<+13.5f")
            + tab
            + "ylo yhi"
            + newline
        )
        f.write(
            format(xyz[:, 2].min().item() * multiplier, "<+13.5f")
            + tab
            + format(xyz[:, 2].max().item() * multiplier, "<+13.5f")
            + tab
            + "zlo zhi"
            + newline
        )
    f.write(newline)
    f.write("Masses" + newline)
    f.write(newline)
    for i in range(learned_params["node"].shape[0]):
        line = ""
        line += str(int(learned_params["node"]["new_type"].iloc[i])) + tab
        line += format(learned_params["node"]["mass"].iloc[i], "<+13.5f") + tab
        f.write(line)
        f.write(newline)
    f.write(newline)
    top = "pair"
    if top in batch._data.keys():
        f.write("Pair Coeffs" + newline)
        f.write(newline)
        for i in range(learned_params["node"].shape[0]):
            line = ""
            line += str(int(learned_params["node"]["new_type"].iloc[i])) + tab
            line += format(learned_params["node"]["epsilon"].iloc[i], "<+13.5f") + tab
            line += format(learned_params["node"]["sigma"].iloc[i], "<+13.5f") + tab
            f.write(line)
            f.write(newline)
        f.write(newline)
    top = "bond"
    # if top in batch._data.keys() and learned_params[top].shape[0] != 0:
    if top in learned_params.keys() and learned_params[top].shape[0] != 0:
        f.write("Bond Coeffs" + newline)
        f.write(newline)
        for i in range(learned_params[top].shape[0]):
            line = ""
            line += str(int(learned_params[top]["new_type"].iloc[i])) + tab
            for param in params[top]:
                line += format(learned_params[top][param].iloc[i], "<+13.5f") + tab
            f.write(line)
            f.write(newline)
        f.write(newline)
    top = "angle"

    # if top in batch._data.keys() and learned_params[top].shape[0] != 0:
    if top in learned_params.keys() and learned_params[top].shape[0] != 0:
        f.write("Angle Coeffs" + newline)
        f.write(newline)
        for i in range(learned_params[top].shape[0]):
            line = ""
            line += str(int(learned_params[top]["new_type"].iloc[i])) + tab
            if style == "compass":
                _params = ["theta0", "k_a2", "k_a3", "k_a4"]
            elif style == "OPLS":
                _params = ["k2", "theta0"]
            for param in _params:
                val = learned_params[top][param].iloc[i]
                if param == "theta0":
                    val = val * 180 / 3.1415926535
                line += format(val, "<+13.5f") + tab
            f.write(line)
            f.write(newline)
        f.write(newline)
    top = "dihedral"
    # if top in batch._data.keys() and learned_params[top].shape[0] != 0:
    if top in learned_params.keys() and learned_params[top].shape[0] != 0:
        f.write("Dihedral Coeffs" + newline)
        f.write(newline)
        for i in range(learned_params[top].shape[0]):
            line = ""
            line += str(int(learned_params[top]["new_type"].iloc[i])) + tab
            if style == "compass":
                _params = ["k1", "phi1", "k2", "phi2", "k3", "phi3"]
            elif style == "OPLS":
                _params = ["k1", "k2", "k3", "k4"]
            for param in _params:
                if param in ["phi1", "phi2", "phi3"]:
                    if phi0_flag:
                        val = learned_params[top][param].iloc[i]
                        val = val * 180 / 3.1415926535
                        line += format(val, "<+13.5f") + tab
                    else:
                        line += format(0.0, "<+13.5f") + tab
                else:
                    line += format(learned_params[top][param].iloc[i], "<+13.5f") + tab
            f.write(line)
            f.write(newline)
        f.write(newline)
    top = "improper"
    # if top in batch._data.keys() and learned_params[top].shape[0] != 0:
    if top in learned_params.keys() and learned_params[top].shape[0] != 0:
        f.write("Improper Coeffs" + newline)
        f.write(newline)
        for i in range(learned_params[top].shape[0]):
            line = ""
            line += str(int(learned_params[top]["new_type"].iloc[i])) + tab
            line += format(learned_params[top]["k2"].iloc[i], "<+13.5f") + tab
            line += format(0.0, "<+13.5f") + tab
            f.write(line)
            f.write(newline)
        f.write(newline)
    if style == "compass":
        top = "angle"
        if (
            top in batch._data.keys()
            and top in learned_params.keys()
            and learned_params[top].shape[0] != 0
        ):
            f.write("BondBond Coeffs" + newline)
            f.write(newline)
            for i in range(learned_params[top].shape[0]):
                line = ""
                line += str(int(learned_params[top]["new_type"].iloc[i])) + tab
                line += format(learned_params[top]["k_bb"].iloc[i], "<+13.5f") + tab
                line += format(learned_params[top]["b0"].iloc[i][0], "<+13.5f") + tab
                line += format(learned_params[top]["b0"].iloc[i][1], "<+13.5f") + tab
                f.write(line)
                f.write(newline)
            f.write(newline)
            f.write("BondAngle Coeffs" + newline)
            f.write(newline)
            for i in range(learned_params[top].shape[0]):
                line = ""
                line += str(int(learned_params[top]["new_type"].iloc[i])) + tab
                line += format(learned_params[top]["k_ba0"].iloc[i], "<+13.5f") + tab
                line += format(learned_params[top]["k_ba1"].iloc[i], "<+13.5f") + tab
                line += format(learned_params[top]["b0"].iloc[i][0], "<+13.5f") + tab
                line += format(learned_params[top]["b0"].iloc[i][1], "<+13.5f") + tab
                f.write(line)
                f.write(newline)
            f.write(newline)
        top = "dihedral"
        if (
            top in batch._data.keys()
            and top in learned_params.keys()
            and learned_params[top].shape[0] != 0
        ):
            f.write("MiddleBondTorsion Coeffs" + newline)
            f.write(newline)
            for i in range(learned_params[top].shape[0]):
                line = ""
                line += str(int(learned_params[top]["new_type"].iloc[i])) + tab
                line += format(learned_params[top]["k_mt1"].iloc[i], "<+13.5f") + tab
                line += format(learned_params[top]["k_mt2"].iloc[i], "<+13.5f") + tab
                line += format(learned_params[top]["k_mt3"].iloc[i], "<+13.5f") + tab
                line += format(learned_params[top]["b0"].iloc[i][1], "<+13.5f") + tab
                f.write(line)
                f.write(newline)
            f.write(newline)
            f.write("EndBondTorsion Coeffs" + newline)
            f.write(newline)
            for i in range(learned_params[top].shape[0]):
                line = ""
                line += str(int(learned_params[top]["new_type"].iloc[i])) + tab
                line += format(learned_params[top]["k_bct10"].iloc[i], "<+13.5f") + tab
                line += format(learned_params[top]["k_bct20"].iloc[i], "<+13.5f") + tab
                line += format(learned_params[top]["k_bct30"].iloc[i], "<+13.5f") + tab
                line += format(learned_params[top]["k_bct11"].iloc[i], "<+13.5f") + tab
                line += format(learned_params[top]["k_bct21"].iloc[i], "<+13.5f") + tab
                line += format(learned_params[top]["k_bct31"].iloc[i], "<+13.5f") + tab
                line += format(learned_params[top]["b0"].iloc[i][0], "<+13.5f") + tab
                line += format(learned_params[top]["b0"].iloc[i][2], "<+13.5f") + tab
                f.write(line)
                f.write(newline)
            f.write(newline)

            f.write("AngleTorsion Coeffs" + newline)
            f.write(newline)
            for i in range(learned_params[top].shape[0]):
                line = ""
                line += str(int(learned_params[top]["new_type"].iloc[i])) + tab
                line += format(learned_params[top]["k_at10"].iloc[i], "<+13.5f") + tab
                line += format(learned_params[top]["k_at20"].iloc[i], "<+13.5f") + tab
                line += format(learned_params[top]["k_at30"].iloc[i], "<+13.5f") + tab
                line += format(learned_params[top]["k_at11"].iloc[i], "<+13.5f") + tab
                line += format(learned_params[top]["k_at21"].iloc[i], "<+13.5f") + tab
                line += format(learned_params[top]["k_at31"].iloc[i], "<+13.5f") + tab
                line += (
                    format(
                        learned_params[top]["theta0"].iloc[i][0] * 180 / 3.1415926535,
                        "<+13.5f",
                    )
                    + tab
                )
                line += (
                    format(
                        learned_params[top]["theta0"].iloc[i][1] * 180 / 3.1415926535,
                        "<+13.5f",
                    )
                    + tab
                )
                f.write(line)
                f.write(newline)
            f.write(newline)
            f.write("AngleAngleTorsion Coeffs" + newline)
            f.write(newline)
            for i in range(learned_params[top].shape[0]):
                line = ""
                line += str(int(learned_params[top]["new_type"].iloc[i])) + tab
                line += format(learned_params[top]["k_aat"].iloc[i], "<+13.5f") + tab
                line += (
                    format(
                        learned_params[top]["theta0"].iloc[i][0] * 180 / 3.1415926535,
                        "<+13.5f",
                    )
                    + tab
                )
                line += (
                    format(
                        learned_params[top]["theta0"].iloc[i][1] * 180 / 3.1415926535,
                        "<+13.5f",
                    )
                    + tab
                )
                f.write(line)
                f.write(newline)
            f.write(newline)
            f.write("BondBond13 Coeffs" + newline)
            f.write(newline)
            for i in range(learned_params[top].shape[0]):
                line = ""
                line += str(int(learned_params[top]["new_type"].iloc[i])) + tab
                line += format(learned_params[top]["k_b1b3"].iloc[i], "<+13.5f") + tab
                line += format(learned_params[top]["b0"].iloc[i][0], "<+13.5f") + tab
                line += format(learned_params[top]["b0"].iloc[i][2], "<+13.5f") + tab
                f.write(line)
                f.write(newline)
            f.write(newline)
        top = "improper"
        if (
            top in batch._data.keys()
            and top in learned_params.keys()
            and learned_params[top].shape[0] != 0
        ):
            f.write("AngleAngle Coeffs" + newline)
            f.write(newline)
            for i in range(learned_params[top].shape[0]):
                line = ""
                line += str(int(learned_params[top]["new_type"].iloc[i])) + tab
                line += format(learned_params[top]["k_aa1"].iloc[i], "<+13.5f") + tab
                line += format(learned_params[top]["k_aa2"].iloc[i], "<+13.5f") + tab
                line += format(learned_params[top]["k_aa3"].iloc[i], "<+13.5f") + tab
                line += (
                    format(
                        learned_params[top]["theta0"].iloc[i][0] * 180 / 3.1415926535,
                        "<+13.5f",
                    )
                    + tab
                )
                line += (
                    format(
                        learned_params[top]["theta0"].iloc[i][1] * 180 / 3.1415926535,
                        "<+13.5f",
                    )
                    + tab
                )
                line += (
                    format(
                        learned_params[top]["theta0"].iloc[i][2] * 180 / 3.1415926535,
                        "<+13.5f",
                    )
                    + tab
                )
                f.write(line)
                f.write(newline)
            f.write(newline)
    f.write("Atoms" + newline)
    f.write(newline)
    for i in range(len(types["node"])):
        line = ""
        line += str(i + 1) + tab
        # if single_mol and smiles == 'O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F' and i == range(len(types['node']))[-1]:
        #     line += str(2) + tab
        # else:
        #     line += str(1) + tab
        line += str(1) + tab
        line += str(int(learned_params["node"]["new_type"].loc[types["node"][i]])) + tab
        line += format(charges[i], "<+13.5f") + tab
        line += format(xyz[i, 0].tolist(), "<+13.5f") + tab
        line += format(xyz[i, 1].tolist(), "<+13.5f") + tab
        line += format(xyz[i, 2].tolist(), "<+13.5f") + tab
        f.write(line)
        f.write(newline)
    f.write(newline)
    for top in ["bond", "angle", "dihedral", "improper"]:
        if top in batch._data.keys():
            index = batch.get_data(top, "index").tolist()
            batchtypes = batch.get_data(top, "type").view(-1).tolist()
            if len(index) == 0:
                continue
            f.write("{}s".format(top.capitalize()) + newline)
            f.write(newline)
            for i in range(len(types[top])):
                line = ""
                line += str(i + 1) + tab
                if top == "bond":
                    t = learned_params[top]["new_type"].loc[types[top][i]].item()
                if top == "angle":
                    node_indices = index[i]
                    node_types = [types["node"][j] for j in node_indices]
                    mask = learned_params[top]["node"].apply(lambda x: x == node_types)
                    maskbatch = learned_params[top]["type"].apply(
                        lambda x: x == batchtypes[i]
                    )
                    #                     import ipdb
                    #                     ipdb.set_trace()
                    t = learned_params[top]["new_type"][maskbatch].item()
                if top in ["dihedral", "improper"]:
                    node_indices = index[i]
                    node_types = [types["node"][j] for j in node_indices]

                    def compare(x, node_types):
                        return x == node_types

                    mask = learned_params[top]["node"].apply(
                        lambda x: compare(x, node_types)
                    )
                    t = learned_params[top]["new_type"][mask].item()
                line += str(int(t)) + tab
                for j in range(len(index[i])):
                    line += str(index[i][j] + 1) + tab
                f.write(line)
                f.write(newline)
        f.write(newline)
    f.close()


def write_traj(atomic_nums, infile, outfile):
    trajectory = mdtraj.formats.XYZTrajectoryFile(infile).read()
    if atomic_nums == None:
        z = np.zeros(trajectory.shape[1]).tolist() * trajectory.shape[0]
    else:
        z = np.array(atomic_nums).tolist() * trajectory.shape[0]
    trajectory = np.concatenate(
        (np.array(z).reshape(trajectory.shape[0], trajectory.shape[1], 1), trajectory),
        axis=2,
    )
    file = open(outfile, "w")
    atom_no = trajectory.shape[1]
    for i, frame in enumerate(trajectory):
        frame[:, -3:] -= frame[:, -3].mean(0)
        file.write(str(atom_no) + "\n")
        file.write("Atoms. Timestep: " + str(i) + "\n")
        for atom in frame:
            if atom.shape[0] == 4:
                try:
                    file.write(
                        str(int(atom[0]))
                        + " "
                        + str(atom[1])
                        + " "
                        + str(atom[2])
                        + " "
                        + str(atom[3])
                        + "\n"
                    )
                except:
                    file.write(
                        str(atom[0])
                        + " "
                        + str(atom[1])
                        + " "
                        + str(atom[2])
                        + " "
                        + str(atom[3])
                        + "\n"
                    )
            elif atom.shape[0] == 3:
                file.write(
                    "1"
                    + " "
                    + str(atom[0])
                    + " "
                    + str(atom[1])
                    + " "
                    + str(atom[2])
                    + "\n"
                )
            else:
                raise ValueError("The trajectory you provided has the wrong format.")
    file.close()


def write_traj_pablo(atomic_nums, infile, outfile, freq=1, elem=""):
    trajectory = mdtraj.formats.XYZTrajectoryFile(infile).read()
    if atomic_nums == None:
        z = np.zeros(trajectory.shape[1]).tolist() * trajectory.shape[0]
    else:
        z = np.array(atomic_nums).tolist() * trajectory.shape[0]
    trajectory = np.concatenate(
        (np.array(z).reshape(trajectory.shape[0], trajectory.shape[1], 1), trajectory),
        axis=2,
    )
    file = open(outfile, "w")
    atom_no = trajectory.shape[1]
    for i, frame in enumerate(trajectory):
        frame[:, -3:] -= frame[:, -3].mean(0)
        file.write(str(atom_no) + "\n")
        file.write("Atoms. Timestep: " + str(i * freq) + "\n")
        for atom in frame:
            if atom.shape[0] == 4:
                try:
                    #                     file.write(str(int(atom[0])) + " " + str(atom[1]) + " " + str(atom[2]) + " " + str(atom[3]) + "\n")
                    file.write(
                        "{:d} {:.3f} {:.3f} {:.3f}\n".format(*[int(atom[0]), *atom[1:]])
                    )
                except:
                    file.write(
                        str(atom[0])
                        + " "
                        + str(atom[1])
                        + " "
                        + str(atom[2])
                        + " "
                        + str(atom[3])
                        + "\n"
                    )
            #                 if i==0:
            #                     if int(atom[0]) == elem:
            #                         atom_list.append(atom_counter)
            #                     atom_counter += 1
            elif atom.shape[0] == 3:
                file.write(
                    "1"
                    + " "
                    + str(atom[0])
                    + " "
                    + str(atom[1])
                    + " "
                    + str(atom[2])
                    + "\n"
                )
            else:
                raise ValueError("The trajectory you provided has the wrong format.")
    file.close()


#     return atom_list


def get_atom_list_pablo(atomic_nums, infile, elem=""):
    with open(xyzfile.path, "r") as fxyz:
        xyz = []
        numAtoms = line
        for line in fxyz:
            if line.startswith(numAtoms):
                next(fxyz)
                xyz.append([])
                continue
            else:
                values = line.split()
                values_floats = [float(cell) for cell in values[1:]]
                xyz[-1].append(values_floats)
                continue
    #     xyz = numpy.transpose(numpy.array(xyz))
    return xyz
    return atom_list
