# from utils import
import os
from tqdm import tqdm
import codecs
import youtokentome
data_folder = "data"
min_length = 1
max_length = 400
max_length_ratio = 20.
retain_case = True


print("\nLearning BPE...")
youtokentome.BPE.train(data=os.path.join(data_folder, "all_text"), vocab_size=10000,
                       model=os.path.join(data_folder, "bpe.model"))

# Load BPE model
print("\nLoading BPE model...")
bpe_model = youtokentome.BPE(model=os.path.join(data_folder, "bpe.model"))

# Re-read English, German
print("\nRe-reading single files...")
with codecs.open(os.path.join(data_folder, "train.z"), "r", encoding="utf-8") as f:
    z = f.read().split("\n")
with codecs.open(os.path.join(data_folder, "train.en"), "r", encoding="utf-8") as f:
    en = f.read().split("\n")

# Filter
print("\nFiltering...")
pairs = list()
for en, de in tqdm(zip(z, en), total=len(z)):
    en_tok = bpe_model.encode(en, output_type=youtokentome.OutputType.ID)
    de_tok = bpe_model.encode(de, output_type=youtokentome.OutputType.ID)
    len_en_tok = len(en_tok)
    len_de_tok = len(de_tok)
    if min_length < len_en_tok < max_length and \
            min_length < len_de_tok < max_length and \
            1. / max_length_ratio <= len_de_tok / len_en_tok <= max_length_ratio:
        pairs.append((en, de))
    else:
        continue
print("\nNote: %.2f per cent of en-de pairs were filtered out based on sub-word sequence length limits." % (100. * (
        len(z) - len(pairs)) / len(z)))

# Rewrite files
english, german = zip(*pairs)
print("\nRe-writing filtered sentences to single files...")
os.remove(os.path.join(data_folder, "train.z"))
os.remove(os.path.join(data_folder, "train.en"))
# os.remove(os.path.join(data_folder, "train.ende"))
with codecs.open(os.path.join(data_folder, "train.z"), "w", encoding="utf-8") as f:
    f.write("\n".join(english))
with codecs.open(os.path.join(data_folder, "train.en"), "w", encoding="utf-8") as f:
    f.write("\n".join(german))
del english, german, bpe_model, pairs

print("\n...DONE!\n")