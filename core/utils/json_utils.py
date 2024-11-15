def generate_json_config(args):
    return {
        "base_model": args.model,
        "dataset": args.dataset,  # Dataset used for training
        "subset_size": args.subset_size,  # Size of the subset
        "image_encoder_name": args.image_encoder_name,  # Name of the image encoder
        "optimizer": args.optimizer,  # Optimization algorithm
        "batch_size": args.batch_size,  # Number of samples in each batch
        "learning_rate": args.lr,  # Learning rate for the model
        "global_rounds": args.global_rounds,  # Number of global rounds
        "local_epochs": args.local_epochs,  # Number of local epochs
        "warm_up": args.warm_up,  # Warm up epochs
        "seed": args.seed,  # Seed for random number generator
    }
