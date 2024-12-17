import pathlib

import sklearn.model_selection

import feature_engineering.transform


in_dir = "data/tournament_info"
out_dir = "data/tournament_dataset"

lines = []
for file in sorted(pathlib.Path(in_dir).glob("*.ndjson")):
    lines += file.read_text().splitlines()

df = feature_engineering.transform.json_list_to_dataframe(lines)
df_train, df_test = sklearn.model_selection.train_test_split(df, test_size=0.1, random_state=27, stratify=df.freq)

# We don't have any true float features, so we dump all "floats" as ints
save_args = {"sep": "\t", "index": False, "float_format": lambda f: str(int(f))}
df.to_csv(f"{out_dir}/all.tsv", **save_args)
df_train.to_csv(f"{out_dir}/train.tsv", **save_args)
df_test.to_csv(f"{out_dir}/test.tsv", **save_args)
