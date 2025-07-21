eval_only = False # if True, script exits right after the first eval

init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'

bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer

weight_decay = 1e-1

grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.

dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler


out_dir = 'out-shakespeare-char'
eval_interval = 50 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1

# Training parameters
batch_size = 32
learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 3000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta1 = 0.9
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small


warmup_iters = 100 # not super necessary potentially


device = 'cpu'  # options 'cuda', 'mps', 'cpu' 
compile = False 

# Model parameters
block_size = 128
n_layer = 2
n_head = 2
n_embd = 128
dropout = 0.1
model_name="bert"
kan=False

# Dataloader parameters
tokenizer = 'char'
dataset_name = 'shakespeare'
split_ratio = {"train": 0.9, "val": 0.1, "test": 0.0}
retokenize=True

