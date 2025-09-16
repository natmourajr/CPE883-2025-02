import datasets

base_url = "https://huggingface.co/datasets/Skylion007/openwebtext/resolve/main/subsets/urlsf_subset{:02d}.tar"
urls = [base_url.format(i) for i in range(21)]
print(urls)

ds = datasets.load_dataset(path = "Skylion007/openwebtext", data_files=urls, split="train")
print(next(iter(ds)))