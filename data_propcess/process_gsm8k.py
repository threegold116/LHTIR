import datasets








if __name__ == "__main__":
    train_file_path = "/share/home/sxjiang/myproject/LHTIR/data/gsm8k/train.parquet"
    
    dataset = datasets.load_dataset("parquet", data_files=train_file_path)["train"]
    
    pass