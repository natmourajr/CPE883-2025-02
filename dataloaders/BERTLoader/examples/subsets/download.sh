mkdir -p subsets
cd subsets
for i in $(seq -w 0 20); do
  wget "https://huggingface.co/datasets/Skylion007/openwebtext/resolve/main/subsets/urlsf_subset${i}.tar"
done