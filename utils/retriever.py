import torch
import numpy as np
import faiss
import pickle
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from datetime import datetime

class Retriever:
    def __init__(self, index_path=None, csv_path=None, top_k=3, model_name='all-MiniLM-L6-v2'):
        self.top_k = top_k
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.index_to_id = {}
        self.reference_data = {}
        self.embeddings = None
        self.df = None

        if index_path:
            index_file = f"{index_path}_faiss.index"
            metadata_file = f"{index_path}_metadata.pkl"
            embedding_file = f"{index_path}_embeddings.npy"

            if os.path.exists(index_file) and os.path.exists(metadata_file) and os.path.exists(embedding_file):
                print(f"[Retriever] Loading existing index from {index_path}")
                self.load_index(index_path)
            elif csv_path:
                print(f"[Retriever] Index not found, building from {csv_path}")
                self.build_index(csv_path, index_path)
                self.load_index(index_path)
            else:
                raise FileNotFoundError("No index found and csv_path not provided to build a new one.")
        else:
            raise ValueError("You must provide a valid index_path.")
        

    def build_index(self, csv_path, save_index_path):
        df = pd.read_csv(csv_path)
        df['datetime'] = pd.to_datetime(df['date'], errors='coerce')
        self.df = df.copy()  # store full DF for filtering

        reference_data = {}
        texts = []
        processed_ids = []

        print(f"[INFO] Processing {len(df)} news items...")

        real_idx = 0
        for idx, row in df.iterrows():
            # 選擇 text 或 title 作為檢索內容
            text_content = row.get('text', '')
            if not text_content or pd.isna(text_content) or len(str(text_content).strip()) < 10:
                text_content = row.get('title', '')
            if not text_content or pd.isna(text_content) or len(str(text_content).strip()) < 10:
                print(f"[WARN] Skipping row {idx} due to empty/short content.")
                continue

            # 檢查 datetime 合法性
            datetime_obj = row['datetime']
            if pd.isnull(datetime_obj):
                print(f"[WARN] Skipping row {idx} due to invalid datetime.")
                continue

            # 建立 reference 資料
            reference_data[str(real_idx)] = {
                'date': datetime_obj.strftime('%Y-%m-%d'),
                'datetime': np.datetime64(datetime_obj),
                'sentiment': row.get('sentiment', ''),
                'source': row.get('source', ''),
                'subject': row.get('subject', ''),
                'text': row.get('text', ''),
                'title': row.get('title', '')
            }


            texts.append(str(text_content).strip())
            processed_ids.append(str(real_idx))
            real_idx += 1

        print(f"[INFO] Processed {len(texts)} valid news items...")

        # 生成向量
        embeddings = self.model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        np.save(f"{save_index_path}_embeddings.npy", embeddings)

        # 建立 FAISS index
        cpu_index = faiss.IndexFlatIP(self.dimension)
        cpu_index.add(embeddings)

        # 儲存 index 和 metadata
        faiss.write_index(cpu_index, f"{save_index_path}_faiss.index")
        with open(f"{save_index_path}_metadata.pkl", 'wb') as f:
            pickle.dump({
                'reference_data': reference_data,
                'index_to_id': {i: pid for i, pid in enumerate(processed_ids)},
                'dimension': self.dimension
            }, f)

        # 儲存處理過的 dataframe
        self.df.to_csv(f"{save_index_path}_df.csv", index=False)
        
        print(f"[INFO] Index built and saved at {save_index_path}")


    def load_index(self, index_path):
        cpu_index = faiss.read_index(f"{index_path}_faiss.index")
        with open(f"{index_path}_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
            self.reference_data = metadata['reference_data']
            self.index_to_id = metadata['index_to_id']
            self.dimension = metadata['dimension']

        self.embeddings = np.load(f"{index_path}_embeddings.npy")

        # Load dataframe used for date filtering
        self.df = pd.read_csv(f"{index_path}_df.csv")
        self.df['datetime'] = pd.to_datetime(self.df['date'], errors='coerce')

        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            print("FAISS index loaded to GPU.")
        else:
            self.index = cpu_index
            print("FAISS index loaded to CPU.")

        # Initialize numpy arrays for fast batch_retrieve
        num_items = len(self.reference_data)
        self.datetimes_np = np.empty(num_items, dtype='datetime64[ns]')
        self.titles_np = np.empty(num_items, dtype=object)
        self.texts_np = np.empty(num_items, dtype=object)

        for idx_str, data in self.reference_data.items():
            idx = int(idx_str)
            dt = data.get("datetime")
            if isinstance(dt, str):
                dt = pd.to_datetime(dt, errors="coerce")
            elif isinstance(dt, pd.Timestamp):
                dt = dt.to_datetime64()
            self.datetimes_np[idx] = dt
            self.titles_np[idx] = data.get("title", "")
            self.texts_np[idx] = data.get("text", "")



    def retrieve(self, query_text, date_str, top_k=None, return_details=False, window_size_hours=24):
        if top_k is None:
            top_k = self.top_k

        query_embedding = self.model.encode(query_text, convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Convert date_str to datetime
        current_time = pd.to_datetime(date_str)

        # Filter out future news
        current_time = pd.to_datetime(date_str)
        start_time = current_time - pd.Timedelta(hours=window_size_hours)

        valid_indices = []
        for i, data_id in self.index_to_id.items():
            dt = self.reference_data[data_id].get('datetime', None)

            if isinstance(dt, np.ndarray):
                dt = dt.item()
            elif isinstance(dt, pd.Timestamp):
                dt = dt.to_pydatetime()

            if isinstance(dt, datetime) and start_time <= dt < current_time:
                valid_indices.append(i)


        if not valid_indices:
            return "No valid historical news before the given time."

        # Create temporary index from filtered embeddings
        embeddings_subset = self.embeddings[valid_indices]
        filtered_index = faiss.IndexFlatIP(self.dimension)
        filtered_index.add(embeddings_subset)

        similarities, indices = filtered_index.search(query_embedding.reshape(1, -1), min(top_k, len(valid_indices)))
        original_indices = [valid_indices[idx] for idx in indices[0] if idx >= 0]

        filtered_results = []
        for i, idx in enumerate(original_indices):
            data_id = self.index_to_id[idx]
            news = self.reference_data[data_id]
            similarity = similarities[0][i] if i < len(similarities[0]) else 0
            filtered_results.append((similarity, news))

        if return_details:
            return filtered_results
        else:
            results_text = "Similar news found:\n"
            for rank, (similarity, news) in enumerate(filtered_results, 1):
                results_text += f"News {rank} (similarity: {similarity:.3f}):\n"
                results_text += f"- Title: {news['title']}\n"
                results_text += f"- Date: {news['date']}\n"
                results_text += f"- Source: {news['source']}\n"
                text_preview = (news['text'][:100] + "...") if len(news['text']) > 100 else news['text']
                results_text += f"- Text Preview: {text_preview}\n\n"
            return results_text if filtered_results else "No results found."
    def batch_retrieve(self, prompts, date_strs, top_k=3, window_size_hours=96):
        assert len(prompts) == len(date_strs), "Mismatched lengths for prompts and date_strs"
        results = []

        query_datetimes = np.array([np.datetime64(pd.to_datetime(dt)) for dt in date_strs])
        start_times = query_datetimes - np.timedelta64(window_size_hours, 'h')

        query_embeddings = self.model.encode(
            prompts, batch_size=64, show_progress_bar=False, convert_to_numpy=True
        )
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)

        D, I = self.index.search(query_embeddings, top_k * 100)
        

        for i in range(len(prompts)):
            indices = I[i]
            similarities = D[i]

            times = self.datetimes_np[indices]
            valid_mask = (times >= start_times[i]) & (times < query_datetimes[i])
            valid_indices = indices[valid_mask][:top_k]
            valid_similarities = similarities[valid_mask][:top_k]

            titles = self.titles_np[valid_indices]
            texts = self.texts_np[valid_indices]
            datetimes = self.datetimes_np[valid_indices]

            result = []
            for j in range(len(valid_indices)):
                result.append((
                    float(valid_similarities[j]),
                    {
                        "title": titles[j],
                        "text": texts[j],
                        "datetime": str(datetimes[j])
                    }
                ))

            #print(f"[RAG - Sample {i}] Found {len(result)} matches in time range {start_times[i]} ~ {query_datetimes[i]}")
            results.append(result)

        return results



        
    def encode(self, texts):
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        return embeddings  
    def _filter_by_time(self, date_str, window_size_hours):
        if self.df is None:
            raise ValueError("DataFrame not loaded. `_filter_by_time` needs `self.df` to be initialized.")

        current_time = pd.to_datetime(date_str)
        start_time = current_time - pd.Timedelta(hours=window_size_hours)

        filtered = self.df[
            (self.df['datetime'] >= start_time) & (self.df['datetime'] < current_time)
        ]

        return filtered.to_dict('records')
    
    def _cosine_similarity(self, vector, matrix):
        # vector: (D,), matrix: (N, D)
        return torch.nn.functional.cosine_similarity(matrix, vector.unsqueeze(0), dim=1)
        

