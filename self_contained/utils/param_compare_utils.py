import sys
import pickle
import time
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from self_contained.forcefields.Forcefield_pcff_linHar_nocrossnoanhar import (
    MIN_BOND_DIS,
    MAX_BOND_DIS,
    MIN_ANGLE_DIS,
    MAX_ANGLE_DIS,
)

# TRAIN_BASE = '/home/gridsan/pleon/proj/repos/AutoBADDIEPipeline/train'
FS_TO_NS = 1 / 1e6


def param_compare(job_details, condition):
    if "onlyglyme" in job_details.date:
        base_concentration = "glyme4"
        plotting_coeffs = [
            [
                "Bond",
                "Angle",
                "Dihedral",
                "BondBond",
                "BondAngle",
                "MiddleBondTorsion",
                "EndBondTorsion",
                "AngleAngleTorsion",
                "AngleTorsion",
            ]
        ]
    elif "onlytfsi" in job_details.date:
        base_concentration = "tfsi"
        plotting_coeffs = [["Bond", "Angle", "Dihedral"]]
    elif "addsaltmult" in job_details.date:
        return
    else:
        base_concentration = "glyme4litfsi"
        plotting_coeffs = [["Bond", "Angle", "Dihedral"]]

    print("beginning this molecule:", base_concentration, "\n\n")
    coeffs = get_coefficients_from_datafile(job_details, condition, base_concentration)
    if "pcff" in job_details.job_name:
        LITERATURE_CONDITION = (
            "230223_notrain_lit_start_linHar_reg0_topreg0_a0_m1000_i0_LR_0.01"
        )
    else:
        LITERATURE_CONDITION = "b_220930_a100_m100_i0_LR_01"
    bench_coeffs = get_coefficients_from_datafile(
        job_details, LITERATURE_CONDITION, base_concentration
    )
    plot_equilibrium_distances_and_stiffnesses_parity(
        job_details, condition, coeffs, bench_coeffs
    )
    if "onlyglyme" in job_details.date:
        plot_crossterms_parity(job_details, condition, coeffs, bench_coeffs)


def get_coefficients_from_datafile(
    job_details, condition, base_concentration, equil_flag=False
):
    coeffs = {}
    WORKDIR = f"{job_details.train_autopath}/{job_details.job_name}/{condition}"
    if equil_flag:
        dataname = f'{base_concentration}_0_i{str(job_details.param_sc_hyp).split(".")[-1]}.data'
    else:
        dataname = f'{condition}_{base_concentration}_SCALED{str(job_details.param_sc_hyp).split(".")[-1]}.data'
    print("loading data for", dataname)
    with open(f"{WORKDIR}/{dataname}", "r") as datafile:
        lines = datafile.read().split("\n")
    read_flag = False
    for line in lines:
        if "Coeffs" in line:
            values = line.split()
            coeffs[values[0]] = []
            read_flag = True
            charge_flag = False
            continue
        elif "Atoms" in line:
            coeffs["Charge"] = {}
            read_flag = True
            charge_flag = True
            continue
        if read_flag:
            if charge_flag:
                if line == "" and len(coeffs["Charge"].keys()) > 0:
                    read_flag = False
                    charge_flag = False
                    continue
                elif line == "":
                    continue
                values = line.split()
                if values[2] not in coeffs["Charge"].keys():
                    coeffs["Charge"][values[2]] = float(values[3])
            else:
                if line == "" and len(coeffs[values[0]]) > 0:
                    coeffs[values[0]] = np.array(coeffs[values[0]], dtype=float)
                    read_flag = False
                    continue
                elif line == "":
                    continue
                coeffs[values[0]].append(line.split()[1:])
    return coeffs


def plot_equilibrium_distances_and_stiffnesses_parity(
    job_details, condition, coeffs, bench_coeffs
):
    fig, ax = plt.subplots(2, 3, figsize=(16, 12))
    plt_counter = 0
    for curkey in ["Bond", "Angle", "Charge", "BondK", "AngleK", "DihedralK"]:
        # for curkey in []:
        if curkey == "Charge":
            un_auto = np.array(list(coeffs[curkey].values()))
            un = np.array(list(bench_coeffs[curkey].values()))
            print("AutoBADDIE charges:", un_auto)
            print("literature charges:", un)
        elif curkey.endswith("K"):
            curkey = curkey[:-1]
            if curkey == "Dihedral":
                inx = [0, 2, 4]
            else:  # BondK, AngleK
                inx = [1, 2, 3]
            un_auto, un_inx = np.unique(
                coeffs[curkey][:, inx], axis=0, return_index=True
            )
            # un_auto = un_auto.squeeze()
            un = bench_coeffs[curkey][un_inx]
            un = un[:, inx]
            rmse = np.power(un_auto - un, 2).mean() ** 0.5
            mae = np.abs(un_auto - un).mean()
            spearman, pvalue = stats.spearmanr(un_auto.flatten(), un.flatten())
            print(
                "mae: {:0.2f}, rmse: {:0.2f}, spearman: {:0.2f}".format(
                    mae, rmse, spearman
                )
            )
            ax[plt_counter // 3, plt_counter % 3].plot(
                np.arange(un_auto.min() * 0.9 - 1, un_auto.max() * 1.1 + 1),
                np.arange(un_auto.min() * 0.9 - 1, un_auto.max() * 1.1 + 1),
                linestyle="dotted",
                color="k",
                label="spearman={:0.2f}\nmae={:0.2f}\nrmse={:0.2f}".format(
                    spearman, mae, rmse
                ),
            )
            if curkey in ["Bond", "Angle"]:
                for curinx in range(len(inx)):
                    try:
                        ax[plt_counter // 3, plt_counter % 3].scatter(
                            un_auto[:, curinx],
                            un[:, curinx],
                            label=f"{curinx+2}",
                            alpha=0.5,
                        )
                    except:
                        import ipdb

                        ipdb.set_trace()
            else:
                for curinx in range(len(inx)):
                    ax[plt_counter // 3, plt_counter % 3].scatter(
                        un_auto[:, curinx],
                        un[:, curinx],
                        label=f"{curinx+1}",
                        alpha=0.5,
                    )
            ax[plt_counter // 3, plt_counter % 3].grid()
            ax[plt_counter // 3, plt_counter % 3].set_xlabel("AutoBADDIE parameter")
            ax[plt_counter // 3, plt_counter % 3].set_ylabel("Literature parameter")
            ax[plt_counter // 3, plt_counter % 3].set_title(f"{curkey} stiffness")
            ax[plt_counter // 3, plt_counter % 3].legend()
            plt_counter = plt_counter + 1
            continue
        else:
            inx = [0]
            un_auto, un_inx = np.unique(
                coeffs[curkey][:, inx], axis=0, return_index=True
            )
            un_auto = un_auto.squeeze()
            un = bench_coeffs[curkey][un_inx, inx]

        rmse = np.power(un_auto - un, 2).mean() ** 0.5
        mae = np.abs(un_auto - un).mean()
        spearman, pvalue = stats.spearmanr(un_auto.flatten(), un.flatten())
        print(
            "mae: {:0.2f}, rmse: {:0.2f}, spearman: {:0.2f}".format(mae, rmse, spearman)
        )

        ax[plt_counter // 3, plt_counter % 3].plot(
            np.arange(un_auto.min() * 0.9 - 1, un_auto.max() * 1.1 + 1),
            np.arange(un_auto.min() * 0.9 - 1, un_auto.max() * 1.1 + 1),
            linestyle="dotted",
            color="k",
            label="spearman={:0.2f}\nmae={:0.2f}\nrmse={:0.2f}".format(
                spearman, mae, rmse
            ),
        )
        ax[plt_counter // 3, plt_counter % 3].scatter(un_auto, un, alpha=0.5)

        ax[plt_counter // 3, plt_counter % 3].grid()
        ax[plt_counter // 3, plt_counter % 3].set_xlabel("AutoBADDIE parameter")
        ax[plt_counter // 3, plt_counter % 3].set_ylabel("Literature parameter")
        if curkey == "Charge":
            ax[plt_counter // 3, plt_counter % 3].set_title(
                f"{str(job_details.param_sc_hyp)}*{curkey}"
            )
            ax[plt_counter // 3, plt_counter % 3].set_ylim(-1.5, 1)
            ax[plt_counter // 3, plt_counter % 3].set_xticks(np.arange(-1.5, 1.51, 0.5))
            ax[plt_counter // 3, plt_counter % 3].set_yticks(np.arange(-1.5, 1.51, 0.5))
        else:
            ax[plt_counter // 3, plt_counter % 3].set_title(
                f"{curkey} equilibrium position"
            )
        ax[plt_counter // 3, plt_counter % 3].legend()
        plt_counter = plt_counter + 1

    plt.tight_layout()
    plt.savefig(
        f"{job_details.train_autopath}/{job_details.job_name}/{condition}/train/parity_harmonic.png"
    )
    plt.savefig(
        f"{job_details.train_autopath}/{job_details.job_name}/{condition}/train/parity_harmonic.jpg"
    )
    plt.close()


def plot_crossterms_parity(job_details, condition, coeffs, bench_coeffs):
    fig, ax = plt.subplots(2, 3, figsize=(16, 12))
    plt_counter = 0
    for curkey in [
        "BondBond",
        "BondAngle",
        "MiddleBondTorsion",
        "EndBondTorsion",
        "AngleAngleTorsion",
        "AngleTorsion",
    ]:
        if curkey == "Dihedral":
            inx = [0, 2, 4]
        elif (
            curkey == "BondBond"
            or curkey == "AngleAngleTorsion"
            or curkey == "BondBond13"
            or curkey == "Improper"
        ):
            inx = [0]
        elif curkey == "BondAngle":
            inx = [0, 1]
        elif curkey == "MiddleBondTorsion" or curkey == "AngleAngle":
            inx = [0, 1, 2]
        elif curkey == "EndBondTorsion" or curkey == "AngleTorsion":
            inx = [0, 1, 2, 3, 4, 5]
        else:
            inx = [1, 2, 3]
        un_auto, un_inx = np.unique(coeffs[curkey][:, inx], axis=0, return_index=True)
        un_auto = un_auto.squeeze()
        un = bench_coeffs[curkey][un_inx]
        un = un[:, inx]
        rmse = np.power((un_auto - un), 2).mean() ** 0.5
        mae = np.abs(un_auto - un).mean()
        spearman, pvalue = stats.spearmanr(un_auto.flatten(), un.flatten())
        print(
            "mae: {:0.2f}, rmse: {:0.2f}, spearman: {:0.2f}".format(mae, rmse, spearman)
        )
        ax[plt_counter // 3, plt_counter % 3].plot(
            np.arange(un_auto.min() * 0.9 - 1, un_auto.max() * 1.1 + 1),
            np.arange(un_auto.min() * 0.9 - 1, un_auto.max() * 1.1 + 1),
            linestyle="dotted",
            color="k",
            label="spearman={:0.2f}\nmae={:0.2f}\nrmse={:0.2f}".format(
                spearman, mae, rmse
            ),
        )
        if len(inx) > 1:
            if curkey in ["Bond", "Angle"]:
                for curinx in range(len(inx)):
                    ax[plt_counter // 3, plt_counter % 3].scatter(
                        un_auto[:, curinx],
                        un[:, curinx],
                        label=f"{curinx+2}",
                        alpha=0.5,
                    )
            else:
                for curinx in range(len(inx)):
                    ax[plt_counter // 3, plt_counter % 3].scatter(
                        un_auto[:, curinx],
                        un[:, curinx],
                        label=f"{curinx+1}",
                        alpha=0.5,
                    )
        else:
            ax[plt_counter // 3, plt_counter % 3].scatter(un_auto, un, alpha=0.5)
        ax[plt_counter // 3, plt_counter % 3].set_xlabel("AutoBADDIE parameter")
        ax[plt_counter // 3, plt_counter % 3].set_ylabel("Literature parameter")
        ax[plt_counter // 3, plt_counter % 3].set_title(f"{curkey} stiffness")
        ax[plt_counter // 3, plt_counter % 3].grid()
        ax[plt_counter // 3, plt_counter % 3].legend()
        plt_counter = plt_counter + 1
    plt.tight_layout()
    plt.savefig(
        f"{job_details.train_autopath}/{job_details.job_name}/{condition}/train/parity_crossterms.png"
    )
    plt.savefig(
        f"{job_details.train_autopath}/{job_details.job_name}/{condition}/train/parity_crossterms.jpg"
    )
    plt.close()


def plot_param_evolution(job_details, condition, forcefield):
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    plt_counter = 0
    if "k_b21" in forcefield.learned_params.keys():
        param_kb21 = np.array(forcefield.learned_params["k_b21"]).transpose(1, 0)
        param_kb22 = np.array(forcefield.learned_params["k_b22"]).transpose(1, 0)
        KB2 = param_kb21 + param_kb22
        B0 = (param_kb21 * MIN_BOND_DIS + param_kb22 * MAX_BOND_DIS) / (KB2)
        param_ka21 = np.array(forcefield.learned_params["k_a21"]).transpose(1, 0)
        param_ka22 = np.array(forcefield.learned_params["k_a22"]).transpose(1, 0)
        KA2 = param_ka21 + param_ka22
        THETA0 = (
            (param_ka21 * MIN_ANGLE_DIS + param_ka22 * MAX_ANGLE_DIS)
            / (KA2)
            * 180
            / np.pi
        )
    else:
        if "opls" in job_details.job_name:
            K2 = np.array(forcefield.learned_params["k2"]).transpose(1, 0) * 20
            KB2 = (
                np.array(forcefield.learned_params["k_bond"]).transpose(1, 0) * 10
            ) ** 2
            B0 = np.array(forcefield.learned_params["b0"]).transpose(1, 0) ** 2
            KA2 = (
                np.array(forcefield.learned_params["k_angle"]).transpose(1, 0) * 10
            ) ** 2
            THETA0 = (
                (np.array(forcefield.learned_params["theta0"]) ** 2).transpose(1, 0)
                * 180
                / np.pi
            )
        else:
            K2 = np.array(forcefield.learned_params["k2"]).transpose(1, 0)
            KB2 = (
                np.array(forcefield.learned_params["k_b2"]).transpose(1, 0) * 10
            ).pow(2)
            B0 = np.array(forcefield.learned_params["b0"]).transpose(1, 0) ** 2
            KA2 = (
                np.array(forcefield.learned_params["k_a2"]).transpose(1, 0) * 10
            ).pow(2)
            THETA0 = (
                (np.array(forcefield.learned_params["theta0"]) ** 2).transpose(1, 0)
                * 180
                / np.pi
            )
    # for i, curkey in enumerate(['k1', 'k2', 'k3']):
    # names = ['k1', 'k2', 'k3']
    names = ["A2", "THETA0", "k2"]
    # for i, curkey in enumerate(['k1', 'k2', 'k3']):
    for i, param in enumerate(
        [
            KA2,
            THETA0,
            K2,
        ]
    ):
        # param = np.array(forcefield.learned_params[curkey]).transpose(1,0)
        for j, cur in enumerate(param):
            if (
                (names[i] == "THETA0" and cur[0] <= 0)
                or (names[i] == "A2" and cur[-1].round() == 50)
                or (names[i] == "k2" and cur[0] == 0)
            ):
                continue
            ax[plt_counter // 3, plt_counter % 3].plot(cur.tolist(), label=f"{j}")
        ax[plt_counter // 3, plt_counter % 3].set_xlabel("parameter update")
        ax[plt_counter // 3, plt_counter % 3].set_ylabel("parameter")
        ax[plt_counter // 3, plt_counter % 3].set_title(names[i])
        plt_counter = plt_counter + 1

    CHARGE = np.array(forcefield.learned_params["charge"]).transpose(1, 0)
    names = ["B2", "B0", "charge"]
    for i, param in enumerate([KB2, B0, CHARGE]):
        for j, cur in enumerate(param):
            if (
                (names[i] == "B2" and cur[-1].round() == 300)
                or (names[i] == "B0" and cur[0] == -10)
                or (names[i] == "B0" and cur[0] == 0)
                or (names[i] == "charge" and np.isnan(cur[0]))
            ):
                continue
            ax[plt_counter // 3, plt_counter % 3].plot(
                cur.tolist(), label=f"{j}", color=f"C{j}"
            )
        ax[plt_counter // 3, plt_counter % 3].set_xlabel("parameter update")
        ax[plt_counter // 3, plt_counter % 3].set_ylabel("parameter")
        ax[plt_counter // 3, plt_counter % 3].set_title(names[i])
        plt_counter = plt_counter + 1
    fig.legend(
        *ax[0, 0].get_legend_handles_labels(),
        bbox_to_anchor=(1, 0.5),
        loc="center left",
    )
    plt.tight_layout()

    # print('saving to:', f'{TRAIN_BASE}/{job_details.job_name}/{condition}/train/param_evolutions.jpg')
    if not os.path.exists(
        f"{job_details.self_contained_base}/../train/{job_details.job_name}/{condition}/train/"
    ):
        os.makedirs(
            f"{job_details.self_contained_base}/../train/{job_details.job_name}/{condition}/train/"
        )
        print(
            "created:",
            f"{job_details.self_contained_base}/../train/{job_details.job_name}/{condition}/train/",
        )
    plt.savefig(
        f"{job_details.self_contained_base}/../train/{job_details.job_name}/{condition}/train/param_evolutions.jpg"
    )
    plt.close()


def plot_param_evolution_cross(job_details, condition, forcefield):
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    plt_counter = 0
    names1 = ["BondBond", "Dihedral", "MiddleBondTorsion"]
    names2 = ["EndBondTorsion", "AngleAngleTorsion", "AngleTorsion"]
    learned = forcefield.learned_params
    bbs = [np.array(learned["k_b1b3"])]
    diheds = [np.array(learned["k1"]), np.array(learned["k2"]), np.array(learned["k3"])]
    mbts = [
        np.array(learned["k_mt1"]),
        np.array(learned["k_mt2"]),
        np.array(learned["k_mt3"]),
    ]
    ebts = [
        np.array(learned["k_bct10"]),
        np.array(learned["k_bct20"]),
        np.array(learned["k_bct30"]),
        np.array(learned["k_bct11"]),
        np.array(learned["k_bct21"]),
        np.array(learned["k_bct31"]),
    ]
    aats = [np.array(learned["k_aat"])]
    ats = [
        np.array(learned["k_at10"]),
        np.array(learned["k_at20"]),
        np.array(learned["k_at30"]),
        np.array(learned["k_at11"]),
        np.array(learned["k_at21"]),
        np.array(learned["k_at31"]),
    ]
    # for i, curkey in enumerate(['k1', 'k2', 'k3']):
    print("starting crosstermevolution")
    for i, param in enumerate([bbs, diheds, mbts]):
        # param = np.array(forcefield.learned_params[curkey]).transpose(1,0)
        for j, cur in enumerate(param):
            for this in cur.transpose(1, 0):
                ax[plt_counter // 3, plt_counter % 3].plot(
                    this.tolist(), label=f"{j}", color=f"C{j}"
                )
        ax[plt_counter // 3, plt_counter % 3].set_xlabel("parameter update")
        ax[plt_counter // 3, plt_counter % 3].set_ylabel("parameter")
        ax[plt_counter // 3, plt_counter % 3].set_title(names1[i])
        plt_counter = plt_counter + 1

    for i, param in enumerate([ebts, aats, ats]):
        for j, cur in enumerate(param):
            for this in cur.transpose(1, 0):
                ax[plt_counter // 3, plt_counter % 3].plot(
                    this.tolist(), label=f"{j}", color=f"C{j}"
                )
        ax[plt_counter // 3, plt_counter % 3].set_xlabel("parameter update")
        ax[plt_counter // 3, plt_counter % 3].set_ylabel("parameter")
        ax[plt_counter // 3, plt_counter % 3].set_title(names2[i])
        plt_counter = plt_counter + 1
    fig.legend(
        *ax[0, 0].get_legend_handles_labels(),
        bbox_to_anchor=(1, 0.5),
        loc="center left",
    )
    plt.tight_layout()

    # print('saving to:', f'{TRAIN_BASE}/{job_details.job_name}/{condition}/train/param_evolutions.jpg')
    if not os.path.exists(
        f"{job_details.self_contained_base}/../train/{job_details.job_name}/{condition}/train/"
    ):
        os.makedirs(
            f"{job_details.self_contained_base}/../train/{job_details.job_name}/{condition}/train/"
        )
        print(
            "created:",
            f"{job_details.self_contained_base}/../train/{job_details.job_name}/{condition}/train/",
        )
    plt.savefig(
        f"{job_details.self_contained_base}/../train/{job_details.job_name}/{condition}/train/param_evolutions_cross.jpg"
    )
    plt.close()
