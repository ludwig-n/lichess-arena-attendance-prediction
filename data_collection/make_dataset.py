import json
import pathlib

import sklearn.model_selection

import preprocessing


in_dir = "data/tournament_info"
out_dir = "data/tournament_dataset"

tournaments = []
for file in sorted(pathlib.Path(in_dir).glob("*.ndjson")):
    tournaments += [json.loads(line) for line in file.read_text().splitlines()]

df = preprocessing.api_objects_to_dataframe(tournaments)
df_train, df_test = sklearn.model_selection.train_test_split(df, test_size=0.1, random_state=27, stratify=df.freq)

# We don't have any true float features, so we dump all "floats" as ints
save_args = {"sep": "\t", "index": False, "float_format": lambda f: str(int(f))}
df.to_csv(f"{out_dir}/all.tsv", **save_args)
df_train.to_csv(f"{out_dir}/train.tsv", **save_args)
df_test.to_csv(f"{out_dir}/test.tsv", **save_args)
