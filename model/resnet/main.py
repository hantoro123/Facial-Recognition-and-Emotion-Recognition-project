from torchvision import models 
# from torchinfo import summary
import torchvision.transforms as transforms
from train import do_train, do_evaluate
from plot import one_item_plot
from dataset import LoadData, LoadTestData
from utils import Multi_confusion, confusion_Matrix


def main(cfg):
    device = cfg.get('device')
    print(f'avaiable device : %s' % device)

    # 파라미터 지정하기
    file = cfg.get('files')
    trn_param = cfg.get('trn_params')
    batch_size = trn_param.get('batch_size')
    lr = trn_param.get('lr')
    epoch = trn_param.get('epoch')
    trn_root = file.get('trn_root')
    val_root = file.get('val_root')
    tst_root = file.get('tst_root')

    criterion = trn_param.get('criterion')

    # 데이터 불러오기
    train_dl, val_dl = LoadData(trn_root, val_root, batch_size)
    
    model = cfg.get('model')

    # 학습시키기

    trn_loss, trn_acc, model_trn = do_train(model, train_dl, criterion, lr, epoch, device=device)

    # 평가하기

    val_loss, val_acc = do_evaluate(model_trn, val_dl, device, criterion)

    # 출력하기

    one_item_plot(trn_loss, val_loss, 'loss', epoch, lr, batch_size)
    one_item_plot(trn_acc, val_acc, 'acc', epoch, lr, batch_size)

    #테스트와의 평가

    tst_dl = LoadTestData(tst_root, batch_size)
    pred, true_label, tst_losses, tst_acces = Multi_confusion(model_trn, tst_dl, device, criterion)

    pred_flat = [item for sublist in pred for item in sublist]
    true_label_flat = [item for sublist in true_label for item in sublist]
    
    print("train_test_loss_acc compare graph")
    one_item_plot(trn_loss, tst_losses, 'loss', epoch, lr, batch_size)
    one_item_plot(trn_acc, tst_acces, 'acc', epoch, lr, batch_size)
    confusion_Matrix(pred_flat, true_label_flat)


def get_args_parser(add_help=True):
  import argparse
  
  parser = argparse.ArgumentParser(description="?", add_help=add_help)
  parser.add_argument("-c", "--config", default="./config.py", type=str, help="configuration file")

  return parser

if __name__ == "__main__":
  args = get_args_parser().parse_args()
  exec(open(args.config).read())
  main(config)
