import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from models import Autoformer, DLinear, TimeLLM

from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"
def inverse_transform_single_feature(scaler, data, target_index=0):
    """
    安全地對單一變數做 inverse_transform，避免多欄 scaler 報錯。
    支援 batch 預測資料 shape: (B, T, 1)
    """
    shape = data.shape
    data = data.reshape(-1)  # 扁平化為 1D

    # 從 scaler 取出對應特徵的 mean / std
    mean = scaler.mean_[target_index]
    std = scaler.scale_[target_index]

    # 反標準化
    data = (data * std) + mean

    return data.reshape(shape)

import deepspeed.comm.comm as ds_comm
def fake_mpi_discovery(*args, **kwargs):
    print("[INFO] Skipping mpi4py (patched)")
    return
ds_comm.mpi_discovery = fake_mpi_discovery
from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content

parser = argparse.ArgumentParser(description='Time-LLM')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, DLinear]')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model') # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768


# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--custom_features', type=str, default=None, help='comma-separated list of custom features to use')

args = parser.parse_args()
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
accelerator = Accelerator()

def plot_prediction(past, pred, true, title='BTC Forecast', save_path=None):

            plt.figure(figsize=(12, 5))
            plt.plot(np.arange(len(past)), past, label='Past Input (Observed)', color='blue')
            plt.plot(np.arange(len(past), len(past) + len(true)), true, label='Ground Truth', color='green')
            plt.plot(np.arange(len(past), len(past) + len(pred)), pred, label='Prediction', color='orange', linestyle='--')
            plt.title(title)
            plt.xlabel('Time (hours)')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()

for ii in range(args.itr):
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.task_name, args.model_id, args.model, args.data, args.features,
        args.seq_len, args.label_len, args.pred_len, args.d_model, args.n_heads,
        args.e_layers, args.d_layers, args.d_ff, args.factor, args.embed, args.des, ii)

    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    if args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    else:
        model = TimeLLM.Model(args).float()

    path = os.path.join(args.checkpoints, setting + '-' + args.model_comment)
    args.content = load_content(args)
    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)

    checkpoint_file = os.path.join(path, 'checkpoint')
    if args.is_training == 0:
        if os.path.exists(checkpoint_file):
            accelerator.print(f"[INFO] Loading model from checkpoint: {checkpoint_file}")
            model.load_state_dict(torch.load(checkpoint_file, map_location=accelerator.device))
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

        model = accelerator.prepare(model)
        criterion = nn.MSELoss()
        mae_metric = nn.L1Loss()
        test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
        accelerator.print(f"[Inference] Test Loss: {test_loss:.4f}, MAE: {test_mae_loss:.4f}")
        exit()

    # Training mode from here on
    time_now = time.time()
    train_steps = len(train_loader)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)
    trained_parameters = [p for p in model.parameters() if p.requires_grad]
    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=args.pct_start,
            epochs=args.train_epochs,
            max_lr=args.learning_rate)

    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()

    train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        train_loader, vali_loader, test_loader, model, model_optim, scheduler)

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []
        model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)

            if args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    if isinstance(outputs, tuple): outputs = outputs[0]
                    f_dim = -1 if args.features == 'MS' else 0
                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -args.pred_len:, f_dim:]
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if isinstance(outputs, tuple): outputs = outputs[0]
                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:]
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                accelerator.print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                accelerator.print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                iter_count = 0
                time_now = time.time()

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                accelerator.backward(loss)
                model_optim.step()

            if args.lradj == 'TST':
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()

        accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        vali_loss, vali_mae_loss = vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric)
        test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
        accelerator.print(f"Epoch: {epoch + 1} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f} MAE Loss: {test_mae_loss:.7f}")
        # 🔽 🔽 🔽 插入視覺化畫圖區塊 🔽 🔽 🔽
        # =================== Visualization (after training epoch) ====================
            # 即時畫圖
        model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                if i > 0:
                    break  # 只畫第一 batch
                batch_x = batch_x.float().to(accelerator.device)
                batch_y = batch_y.float().to(accelerator.device)
                batch_x_mark = batch_x_mark.float().to(accelerator.device)
                batch_y_mark = batch_y_mark.float().to(accelerator.device)

                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(accelerator.device)
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)

                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                f_dim = -1 if args.features == 'MS' else 0
                pred = outputs[:, -args.pred_len:, f_dim:].detach().cpu().numpy()
                true = batch_y[:, -args.pred_len:, f_dim:].detach().cpu().numpy()
                past = batch_x[:, :, f_dim:].detach().cpu().numpy()

                # 反標準化
                test_data = test_loader.dataset
                pred_shape = pred.shape
                true_shape = true.shape
                past_shape = past.shape

                pred = inverse_transform_single_feature(test_data.scaler, pred, target_index=0)
                true = inverse_transform_single_feature(test_data.scaler, true, target_index=0)
                past = inverse_transform_single_feature(test_data.scaler, past, target_index=0)


                # 儲存成每個 epoch 專屬圖檔
                save_name = f"forecast_result_epoch{epoch+1}.png"
                plot_prediction(
                    past=past[0].squeeze(),
                    pred=pred[0].squeeze(),
                    true=true[0].squeeze(),
                    title=f"Epoch {epoch+1} Forecast: {args.seq_len}h → {args.pred_len}h",
                    save_path=save_name
                )
                print(f"📊 Saved real-time forecast chart: {save_name}")
                

        # 🔼 🔼 🔼 結束畫圖區塊 🔼 🔼 🔼
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            break

        if args.lradj != 'TST':
            if args.lradj == 'COS':
                scheduler.step()
                accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                if epoch == 0:
                    args.learning_rate = model_optim.param_groups[0]['lr']
                    accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)
        else:
            accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
                

        


accelerator.wait_for_everyone()
if accelerator.is_local_main_process:
    path = './checkpoints'
    # del_files(path)  # optional clean-up
    accelerator.print('success delete checkpoints')
    # 🎬 合成訓練預測動畫

if accelerator.is_local_main_process:
    ...
    accelerator.print('success delete checkpoints')

# 🎯 產出長時段預測圖
def generate_long_prediction_plot(model, test_loader, args, accelerator, save_path="long_forecast.png", max_points=10000):
    import matplotlib.pyplot as plt
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        count = 0
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float()

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if isinstance(outputs, tuple): outputs = outputs[0]
            f_dim = -1 if args.features == 'MS' else 0

            pred = outputs[:, -args.pred_len:, f_dim:].detach().cpu().numpy()
            true = batch_y[:, -args.pred_len:, f_dim:].detach().cpu().numpy()

            preds.append(pred)
            trues.append(true)

            count += pred.shape[0] * pred.shape[1]
            if count >= max_points:
                break

    preds = np.concatenate(preds, axis=0).reshape(-1)
    trues = np.concatenate(trues, axis=0).reshape(-1)

    # 反標準化
    test_data = test_loader.dataset
    preds = inverse_transform_single_feature(test_data.scaler, preds.reshape(-1, 1), target_index=0).reshape(-1)
    trues = inverse_transform_single_feature(test_data.scaler, trues.reshape(-1, 1), target_index=0).reshape(-1)


    # 畫圖
    plt.figure(figsize=(20, 6))
    plt.plot(trues, label="Ground Truth", color='green', linewidth=1)
    plt.plot(preds, label="Prediction", color='orange', linestyle='--', linewidth=1)
    plt.legend()
    plt.title(f"📈 Long-term Forecast (first {len(preds)} hours)")
    plt.xlabel("Time (hours)")
    plt.ylabel("Price (USD)")
    plt.grid(True)
    plt.savefig(save_path)
    print(f"✅ Saved long forecast plot to {save_path}")

generate_long_prediction_plot(model, test_loader, args, accelerator, save_path="long_forecast.png", max_points=10000)

import glob
from PIL import Image

image_files = sorted(
    glob.glob("forecast_result_epoch*.png"),
    key=lambda x: int(x.split("epoch")[1].split(".")[0])
)

if len(image_files) > 1:
    images = [Image.open(img) for img in image_files]
    images[0].save(
        "training_forecast.gif",
        save_all=True,
        append_images=images[1:],
        duration=1000,  # 每張圖顯示 1 秒
        loop=0
    )
    accelerator.print("🎞️ Saved training animation as: training_forecast.gif")
else:
    accelerator.print("⚠️ Not enough forecast images to create animation.")

import pandas as pd
from utils.tools import NewsRetriever

# Load news data from cryptonews.csv
def load_news_data(news_file):
    news_df = pd.read_csv(news_file)
    news_list = []
    for _, row in news_df.iterrows():
        news_list.append({
            "content": row["content"],
            "metadata": {
                "date": row["date"],
                "title": row["title"],
                "sentiment": row["sentiment"],
                "source": row["source"],
                "topic": row["topic"]
            }
        })
    return news_list

# Initialize NewsRetriever and add news data
news_file = "cryptonews.csv"
news_data = load_news_data(news_file)
retriever = NewsRetriever()
retriever.add_news(news_data)

# Example query for testing
query = "Bitcoin price prediction"
retrieved_news = retriever.search(query, top_k=3)
print("Retrieved News:", retrieved_news)