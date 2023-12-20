class BaseTrainer():
    def __init__(self, args, device) -> None:
        self.args = args
        self.device = device
        
        self.config = {
            "vocab_size": 12,
            "max_len": 6,
            "hidden_size": args.input_dim,
            "num_head": 4,
        }
        
        self.exp_name = args.save.strip("/").replace(args.base_path.strip("/"), "").replace("_", "").replace("/", "_").strip("_")
        self.data_dir = os.path.join(
            args.base_path, "processed_data", "toy-add", f"tn{args.train_num}-dn{args.dev_num}-r{args.ratio_1_2}", f"{args.seed}-{args.seed_data}")
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.model = ToyTransformer(self.config).to(device)
        
        if args.load_toy_data is None:
            torch.save(self.model.state_dict(), os.path.join(self.data_dir, "model_init.pt"))
        else:
            self.model.load_state_dict(torch.load(os.path.join(self.data_dir, "model_init.pt")))
        
        self.optimizer = SGD(self.model.parameters(), lr=args.lr)
    
        self.train_data, self.dev_data, self.test_data = self.get_data()
        self.train_data = self.reform_data(self.train_data).to(device)
        self.dev_data = self.reform_data(self.dev_data).to(device)
        self.test_data = self.reform_data(self.test_data).to(device)
    
        print("train data size: {} | dev data size: {} | test data size: {}".format(
            self.train_data.size(), self.dev_data.size(), self.test_data.size()))

    def get_model()