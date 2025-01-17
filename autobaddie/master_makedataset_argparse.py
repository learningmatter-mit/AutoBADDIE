import os
import argparse
import torch
import pandas as pd
import _pickle as pickle
import pandas as pd
import json
import argparse
import autobaddie.utils.train as train
import autobaddie.utils.parameterize as parameterize


def none_or_str(value):
    if value == "None":
        return None
    return value


parser = argparse.ArgumentParser(description="Parse for AutoBADDIE training")
parser.add_argument("--job_name", type=str, action="store", nargs="?")
parser.add_argument("--condition", type=str, action="store", nargs="?")
parser.add_argument("--date", type=str, action="store", nargs="?")
parser.add_argument("--param_json", type=str, action="store", nargs="?")
parser.add_argument("--E_hyp", type=float, default=100.0, nargs="?")
parser.add_argument("--dih_reg_hyp", type=float, default=100.0, nargs="?")
parser.add_argument("--top_reg_hyp", type=float, default=100.0, nargs="?")
parser.add_argument("--no_train_flag", dest="train_flag", action="store_false")
parser.add_argument("--train_flag", dest="train_flag", action="store_true")
parser.add_argument("--glymecond", type=none_or_str, action="store", nargs="?")
parser.add_argument("--tfsicond", type=none_or_str, action="store", nargs="?")
parser.add_argument("--licond", type=none_or_str, action="store", nargs="?")
parser.add_argument("--training_ff", type=none_or_str, action="store", nargs="?")
parser.add_argument("--gen", type=none_or_str, action="store", nargs="?")
parser.add_argument("--other_cond_for_dih", type=str, action="store", nargs="?")
parser.add_argument("--dih_label", type=str, action="store", nargs="?")
parser.add_argument("--other_dih_label", type=str, action="store", nargs="?")
parser.add_argument("--autobaddie_base", type=str, action="store", nargs="?")
parser.add_argument("--provided_charges", type=float, action="store", nargs="+")
parser.add_argument("--training_data_path", type=str, action="store", nargs="?")
parser.add_argument("--train_autopath", type=str, action="store", nargs="?")
args = parser.parse_args()

train_flag = args.train_flag
AUTOBADDIE_BASE = args.autobaddie_base
if not train_flag:
    job_name = args.job_name
    condition = args.condition

myarray = []
# if args.dih_reg_hyp > 1e0:
if 50 > 1e0:
    date = f"{args.date}_reg{args.dih_reg_hyp}_topreg{args.top_reg_hyp}"
else:
    date = f"{args.date}_reg{str(args.dih_reg_hyp).split('.')[-1]}_topreg{args.top_reg_hyp}"
if train_flag:
    with open(
        f"{AUTOBADDIE_BASE}/training_params/{args.param_json}.json", "r"
    ) as jsonFile:
        lines = jsonFile.read().split("\n")
        for line in lines:
            if "date" in line:
                line += '"' + date + '",'
            elif '"job_name' in line:
                line += '"' + args.job_name + '",'
            elif '"energy_hyp' in line:
                line += f" {args.E_hyp},"
            elif '"reg_hyp' in line:
                line += f" {args.dih_reg_hyp},"
            elif '"top_reg_hyp' in line:
                line += f" {args.top_reg_hyp},"
            elif '"provided_charges' in line:
                line += f" {args.provided_charges},"
            myarray.append(line)
    with open(
        f"{AUTOBADDIE_BASE}/training_params/cur_job_details_pretrain.json", "w"
    ) as jsonFile:
        jsonFile.write("\n".join(myarray))
    with open(
        f"{AUTOBADDIE_BASE}/training_params/cur_job_details_pretrain.json", "r"
    ) as jsonFile:
        job_details = json.load(jsonFile)
    job_details["training_data_path"] = args.training_data_path
    job_details["train_autopath"] = args.train_autopath
    job_details["autobaddie_base"] = AUTOBADDIE_BASE
    job_details["generation"] = args.gen
    job_details["glymecond"] = args.glymecond
    job_details["tfsicond"] = args.tfsicond
    job_details["licond"] = args.licond
    if "onlytfsi" in job_details["date"]:
        job_details["training_ff"] = args.training_ff
        job_details["other_condition_for_dih"] = args.other_cond_for_dih
        job_details["dih_label"] = args.dih_label
        job_details["other_dih_label"] = args.other_dih_label
        if "startprev" in job_details["date"]:
            job_details["tfsicond"] = args.tfsicond
    elif "onlyglyme" in job_details["date"]:
        job_details["training_ff"] = args.training_ff
        job_details["other_condition_for_dih"] = args.other_cond_for_dih
        job_details["dih_label"] = args.dih_label
        job_details["other_dih_label"] = args.other_dih_label
    elif "addsaltmultnotrain" in job_details["date"]:
        job_details["training_ff"] = args.training_ff
        job_details["other_condition_for_dih"] = args.other_cond_for_dih
        job_details["dih_label"] = args.dih_label
        job_details["other_dih_label"] = args.other_dih_label
else:
    with open(
        f"{job_details['autobaddie_base']}/../training_results/{job_details['job_name']}/{job_details['condition']}/job_details.json",
        "r",
    ) as jsonFile:
        job_details = json.load(jsonFile)
        try:
            job_details.description
        except:
            job_details["description"] = None
logtext = ""

# begin program
print("beginning the program")
print("job name:", job_details["job_name"])
print("AUTOBADDIE_BASE:", job_details["autobaddie_base"])
#             #-----------------------------------------------train and val-----------------------------------------------------------
job_details, condition = train.assign_job_details(job_details)
job_details.mode = "train"
torch.set_default_device(job_details.device)
print("condition:", condition)

# getting single datasets
job_details.TEMPLATEDIR = os.path.join(
    os.path.abspath("."), job_details.job_name, "template"
)
path = os.path.join(job_details.TEMPLATEDIR, "template.py")
template_dataset = torch.load(path, weights_only=False)
if job_details.anion_smiles:
    single_anion_dataset, single_cation_dataset, single_solv_dataset = [], [], []
    print("beginning to get the single mol datasets")
    for cur_smi in job_details.template_smiles.split("."):
        if cur_smi in job_details.anion_smiles:
            # start by creating the anion and cation dataset in order to know what are their types for scaling
            single_anion_dataset.append(
                parameterize.get_dataset_from_smiles(
                    cur_smi, job_details, condition, template_dataset, "cpu"
                )
            )
        elif cur_smi in job_details.cation_smiles:
            single_cation_dataset.append(
                parameterize.get_dataset_from_smiles(
                    cur_smi, job_details, condition, template_dataset, "cpu"
                )
            )
        else:
            single_solv_dataset.append(
                parameterize.get_dataset_from_smiles(
                    cur_smi, job_details, condition, template_dataset, "cpu"
                )
            )
else:
    single_anion_dataset, single_cation_dataset, single_solv_dataset = None, None, []
    if "pretrain" in job_details.date or "onlyglyme" in job_details.date:
        cur_smi = job_details.template_smiles.split(".")[0]
        single_solv_dataset.append(
            parameterize.get_dataset_from_smiles(
                cur_smi, job_details, condition, template_dataset, "cpu"
            )
        )
    else:
        for cur_smi in job_details.template_smiles.split("."):
            single_solv_dataset.append(
                parameterize.get_dataset_from_smiles(
                    cur_smi, job_details, condition, template_dataset, "cpu"
                )
            )

dataset_path_base = os.path.join(job_details.WORKDIR, condition)
dump_path = os.path.join(
    dataset_path_base, "{}_{}_dataset_solv.pkl".format(condition, job_details.mode)
)
pickle.dump(single_solv_dataset, open(dump_path, "wb"))
dump_path = os.path.join(
    dataset_path_base, "{}_{}_dataset_anion.pkl".format(condition, job_details.mode)
)
pickle.dump(single_anion_dataset, open(dump_path, "wb"))
dump_path = os.path.join(
    dataset_path_base, "{}_{}_dataset_cation.pkl".format(condition, job_details.mode)
)
pickle.dump(single_cation_dataset, open(dump_path, "wb"))


print("creating train dataset")
device = torch.device(job_details.device)
# device = torch.device("cpu")

# add ensemble_to_smiles
df = pd.read_csv(job_details.training_data_path + "/file_contents.csv")
ensemble_to_smiles = [
    df[df.species_id == species_id].smiles.to_list()[0]
    for species_id in df.species_id.unique()
]
job_details["ensemble_to_smiles"] = ensemble_to_smiles
with open(
    os.path.join(job_details.WORKDIR, condition, "job_details.json"), "w"
) as jsonFile:
    json.dump(job_details, jsonFile, indent=1)


if not os.path.exists(os.path.join(job_details.WORKDIR, condition, "train")):
    os.makedirs(os.path.join(job_details.WORKDIR, condition, "train"))
    print("created:", os.path.join(job_details.WORKDIR, condition, "train"))
if not os.path.exists(
    os.path.join(
        dataset_path_base, "{}_{}_dataset.pkl".format(condition, job_details.mode)
    )
):
    # if True:
    print(job_details.template_smiles)
    path = job_details.training_data_path
    dataset, dataset_val, dataset_test, template_dataset, job_details = (
        train.create_train_and_val_df_file(
            path, job_details, condition, template_dataset, charge=True
        )
    )
else:
    dataset = pickle.load(
        open(f"{dataset_path_base}/{condition}_{job_details.mode}_dataset.pkl", "rb")
    )

print(
    "number of geometries in training dataset:",
    dataset._data["prop"]["geom_id"].shape[0],
)
print("done making datasets!")
