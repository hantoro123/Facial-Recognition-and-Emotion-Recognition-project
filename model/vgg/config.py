config = {
    "files": {
        "time_save": "/home/KDT-admin/work/ssm/training_time.csv",  #시간저장
        # "img_path":"/home/KDT-admin/work/data6_last_1000_300", 
        "img_path":"/home/KDT-admin/work/data4_small_200_100", 
        "model_name": "VGG_test4" #
    },
    "model_params": {
        "hidden_dim": 32,
        "use_dropout": True,
    },
    "train_params": {
        "batch_size": 256, 
        "shuffle": True,
        "learning_rate" : 0.001,
        "device": "cpu",
        "num_epochs": 2,
        "pbar": True,
    },
    "train": True,
    "validation": False,
    "class_to_idx" : {
        '분노': 0,
        '중립': 1,
        '기쁨': 2,
        '당황': 3,
        '불안': 4,
        '상처': 5,
        '슬픔': 6
                    },
}