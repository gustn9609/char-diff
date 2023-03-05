from dataloader import QQPLoader, QTLoader

args = parse_args()

local_rank = int(os.environ['LOCAL_RANK'])
device = torch.device("cuda", local_rank)

torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=9600))
set_seed(args)

if dist.get_rank() == 0:
    log_dir = './logs'
    fitlog.set_log_dir(log_dir)
    # fitlog.commit(__file__)
    # fitlog.add_hyper(args)
    # fitlog.add_hyper_in_file(__file__)

Dataloaders = {
    'qqp': QQPLoader,
    'QT': QTLoader,
}

Loader = Dataloaders[args.task_name]

# save_path = f'./model_name_{args.model_name_or_path}_taskname_{args.task_name}_lr_{args.lr}_seed_{args.seed}_numsteps_{args.num_steps}_sample_{args.sample_strategy}_schedule_{args.schedule}_hybridlambda_{args.hybrid_lambda}_wordfreqlambda_{args.word_freq_lambda}_fromscratch_{args.from_scratch}_timestep_{args.timestep}_ckpts'
if args.model_name_or_path in ['bert-base-uncased', 'bert-large-uncased']:
    model_cls = BertForMaskedLM
    cfg_cls = BertConfig
    tok_cls = BertTokenizer
elif args.model_name_or_path in ['roberta-base']:
    model_cls = RobertaForMaskedLM
    cfg_cls = RobertaConfig
    tok_cls = RobertaTokenizer
else:
    raise NotImplementedError

tokenizer = tok_cls.from_pretrained(args.model_name_or_path)
# word_freq = torch.zeros(tokenizer.vocab_size)
# assert word_freq.size(0) == tokenizer.vocab_size


# def word_freq_preprocess_fn(wf):
    # wf = wf + 1
    # wf = wf.log()
    # wf = wf / wf.max()

    # range: 0 - 1
    # return wf


# word_freq = word_freq_preprocess_fn(word_freq)

# word_freq[tokenizer.pad_token_id] = 0.  # stable training

if args.sample_strategy == 'Categorical':
    sample_cls = Categorical()
elif args.sample_strategy == 'wwm':
    sample_cls = WholeWordMasking(tokenizer)
else:
    raise ValueError

diffusion_schedule = diffusion_condition.create_discrete_diffusion_schedule(args.schedule, num_steps=args.num_steps)
diffusion_instance = diffusion_condition.MaskDiffusion(
    dim=tokenizer.vocab_size,
    schedule=diffusion_schedule,
    tokenizer=tokenizer,
    sample_cls=sample_cls,
    # word_freq=word_freq,
    # word_freq_lambda=args.word_freq_lambda,
    device=device
)

if args.load_step > 0:
    ckpt = torch.load(os.path.join(save_path, f'{args.load_step}.th'))
cfg = cfg_cls.from_pretrained(args.model_name_or_path)
cfg.overall_timestep = diffusion_instance.num_steps

if args.from_scratch:
    model = model_cls(cfg).to(device)
elif args.load_step <= 0:
    model = model_cls.from_pretrained(args.model_name_or_path, config=cfg).to(device)
else:
    model = model_cls(cfg).to(device)
    model.load_state_dict(ckpt['model'])

model = DDP(model, device_ids=[local_rank], output_device=local_rank)
optimizer = AdamW(model.parameters(), lr=args.lr)
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=10000)

train_data, dev_data = Loader(tokenizer=tokenizer).my_load(splits=['train', 'validation'])
