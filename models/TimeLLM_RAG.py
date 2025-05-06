import datetime
from math import sqrt

import torch
import torch.nn as nn
import pandas as pd

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize
from utils.retriever import Retriever

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        


        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The dataset is a CSV file that likely contains historical Bitcoin price data. The "H" in the filename might indicate that the data is organized on an hourly basis. The file may include columns such as date, open price, high price, low price, close price, and possibly trading volume or other related metrics.'

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)
        self.retriever = Retriever(
            index_path="./dataset/cryptonews",
            csv_path="./dataset/cryptonews.csv"  # 第一次用來建立索引
        )

        # self.retriever = Retriever(index_path="./dataset/cryptonews.csv")
        
        
        self.news_window_hours = configs.seq_len
        
        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, timestamp=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, timestamp)
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, timestamp=None):
        # Step 0: Pre-process timestamps
        if timestamp is not None:
            if isinstance(timestamp, torch.Tensor):
                timestamp = timestamp.cpu().numpy()
            current_times = [pd.to_datetime(ts) for ts in timestamp]
        else:
            current_times = [None] * x_enc.shape[0]

        # Step 1: Normalize & prepare patch embedding
        x_enc = self.normalize_layers(x_enc, 'norm')
        B, T, N = x_enc.size()  # B=批次, T=時間步, N=變量數
        
        prompts = []
        retrieve_dates = []
        
        for b in range(B):
            # 取出單個 batch 的數據
            x_enc_b = x_enc[b]  # [T, N]
            
            # 為這個 batch 的所有變量計算統計數據
            # 方案1：計算每個變量的統計，然後取平均
            x_enc_b_reshaped = x_enc_b.T.unsqueeze(-1)  # [N, T, 1]
            stats = self._calculate_statistics(x_enc_b_reshaped)
            
            # 使用所有變量的平均統計數據
            min_val = stats['min_values'].mean().item()
            max_val = stats['max_values'].mean().item()
            median_val = stats['medians'].mean().item()
            trend = stats['trends'].mean().item()
            
            # 對於 lags，可以取第一個變量的，或計算平均 lags
            lags = stats['lags'][0].tolist()  # 使用第一個變量的 lags
            
            prompt = self._generate_prompt(
                min_val, max_val, median_val, trend, lags, current_times[b]
            )
            prompts.append(prompt)

            sample_datetime = self._get_sample_datetime(x_mark_enc[b], timestamp=current_times[b])
            retrieve_dates.append(sample_datetime.strftime("%Y-%m-%d %H:%M:%S"))

        # Step 3: Batch retrieve
        retrieved_results_batch = self.retriever.batch_retrieve(
            prompts=prompts,
            date_strs=retrieve_dates,
            top_k=self.top_k,
            window_size_hours=self.seq_len
        )

        # Step 4: Format RAG inputs
        retrieved_texts = []
        for i in range(B):
            try:
                result = retrieved_results_batch[i]
                if isinstance(result, list) and len(result) > 0:
                    combined = "\n\n".join([f"{news['title']}. {news['text']}" for _, news in result])
                else:
                    raise ValueError("Empty or invalid result")
            except Exception as e:
                #print(f"[RAG - Sample {i}] Fallback triggered due to: {str(e)}")
                combined = "No relevant news found."
            retrieved_texts.append(combined)

        # Step 5: Tokenize prompts and news
        prompt_tokens = self.tokenizer(
            prompts, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).input_ids.to(x_enc.device)
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_tokens)

        retrieved_tokens = self.tokenizer(
            retrieved_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).input_ids.to(x_enc.device)
        rag_embeddings = self.llm_model.get_input_embeddings()(retrieved_tokens)

        # Step 6: Patch + Reprogramming
        x_enc_patch = x_enc.permute(0, 2, 1).contiguous()  # [B, N, T]
        enc_out, n_vars = self.patch_embedding(x_enc_patch)
        
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        # Step 7-9: 保持不變
        assert (
            prompt_embeddings.shape[0]
            == rag_embeddings.shape[0]
            == enc_out.shape[0]
        ), f"Batch size mismatch: prompt {prompt_embeddings.shape}, rag {rag_embeddings.shape}, enc {enc_out.shape}"

        model_input = torch.cat([prompt_embeddings, enc_out], dim=1)
        
        dec_out = self.llm_model(inputs_embeds=model_input).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = self._post_process_output(dec_out, n_vars)
        return dec_out
    def _calculate_statistics(self, x):
        """Calculate statistical features of input sequence"""
        min_values = torch.min(x, dim=1)[0]
        max_values = torch.max(x, dim=1)[0]
        medians = torch.median(x, dim=1).values
        lags = self.calcute_lags(x)
        trends = x.diff(dim=1).sum(dim=1)
        
        return {
            'min_values': min_values,
            'max_values': max_values,
            'medians': medians,
            'lags': lags,
            'trends': trends
        }

    def _generate_prompt(self, min_val, max_val, median_val, trend, lags, current_time):
        """Generate prompt with statistical information"""
        return (
            f"<|start_prompt|>Dataset description: {self.description}"
            f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
            f"Current time: {current_time if current_time else 'unknown'}; "
            f"Input statistics: min value {min_val:.4f}, "
            f"max value {max_val:.4f}, "
            f"median value {median_val:.4f}, "
            f"the trend of input is {'upward' if trend > 0 else 'downward'}, "
            f"top 5 lags are : {lags}<|<end_prompt>|>"
        )

    
    def _post_process_output(self, dec_out, n_vars):
        """Post-process model output into final predictions"""
        dec_out = dec_out.reshape(-1, n_vars, dec_out.shape[-2], dec_out.shape[-1])
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        return self.normalize_layers(dec_out, 'denorm')

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags

    def _get_sample_datetime(self, x_mark_enc_sample, timestamp=None):
        """
        x_mark_enc_sample: Tensor of shape [seq_len, num_time_features]
        timestamp: Optional datetime object

        Returns a single datetime object representing the sample's time
        """
        from datetime import datetime
        import pandas as pd
        import numpy as np

        if timestamp is not None:
            if isinstance(timestamp, str):
                return pd.to_datetime(timestamp).to_pydatetime()
            elif isinstance(timestamp, pd.Timestamp):
                return timestamp.to_pydatetime()
            elif isinstance(timestamp, datetime):
                return timestamp
            elif isinstance(timestamp, (list, np.ndarray)) and len(timestamp) > 0:
                return pd.to_datetime(timestamp[0]).to_pydatetime()
            else:
                raise ValueError(f"Unsupported timestamp format: {type(timestamp)}")

        # 如果沒有 timestamp，則從 x_mark_enc 取出第一個時間戳
        time_array = x_mark_enc_sample[0].detach().cpu().numpy().astype(int)
        if len(time_array) >= 4:
            year, month, day, hour = time_array[:4]
            return datetime(year, month, day, hour)
        else:
            raise ValueError(f"Insufficient time features: {time_array}")



class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding