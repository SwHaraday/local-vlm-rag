'''
vlm-rag
'''
import os, glob, sys, codecs, shutil
import time
import json
import logging
from typing import List, Dict, Any, Optional

import torch
import numpy as np
import pickle
from PIL import Image
from pdf2image import convert_from_path
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    MultiVectorConfig, MultiVectorComparator
)

# proxyを回避するための呪文　※qdrantローカルサーバへのアクセス
os.environ["NO_PROXY"] = "localhost, 127.0.0.1/8, ::1"

# グローバルスコープにすべく先に宣言
DOCUMENT＿ROOT = 'c:/document_root' # PDF資料を保存するフォルダを作成してもらうルートフォルダ 適宜管理者が決めればよい ※スラッシュで終わらないこと！！
pdf_paths = []
img_path = None
collection_name = None # Qdrantコレクション名
url = None

# ---------- Qdrant----------
class QdrantVectorStore:
    def __init__(
        self,
        collection_name: str = collection_name,
        timeout: float = 120.0,
        url = url,
    ):
        """
        QdrantVectorStore の初期化

        Args:
            url (str): Qdrant サーバーのURL (デフォルト: http://qdrant:6333)
            api_key (Optional[str]): APIキー
            collection_name (str): コレクション名。RDBのテーブルのような概念
            timeout (float): タイムアウト秒数
            prefer_grpc (bool): RESTよりgRPCを優先して通信する（バイナリデータの転送で、高速・低レイテンシ）
            grpc_port (int): gRPCポート番号 (デフォルト: 6334)
        """
        self.collection_name = collection_name

        #QdrantClientのインスタンス化
        self.client = QdrantClient(
            timeout=timeout,
            #path=path,
            url=url,
        )
        print(f"Qdrant client initialized with URL: {url}")

    def create_collection(
        self,
        multivector_size: int = 256,
        force_recreate: bool = True,
        distance: Distance = Distance.COSINE,
        comparator: MultiVectorComparator = MultiVectorComparator.MAX_SIM,
    ) -> None:
        """
        Qdrantにコレクションを作成するメソッド（マルチベクター対応）

        Args:
            multivector_size: 埋め込みモデルが出力するベクトルの次元数
            force_recreate: Trueの場合、既存コレクションを削除して再作成
            Distance.COSINE: 検索にコサイン類似度を使用
            comparator(MultiVectorComparator.MAX_SIM): 複数のベクトルの中で一番スコアが高い(類似度が高い)ものを選ぶ

        Raises:
            Exception: コレクション作成時にエラーが発生した場合
        """
        try:
            exists = any(
                col.name == self.collection_name
                for col in self.client.get_collections().collections
            )
            if exists:
                if force_recreate:
                    print(f"Deleting existing collection: {self.collection_name}")
                    self.client.delete_collection(self.collection_name)
                else:
                    print(f"Collection {self.collection_name} already exists")
                    return
                    
            #コレクション作成メソッドの呼び出し
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=multivector_size,
                    distance=distance,
                    multivector_config=MultiVectorConfig(comparator=comparator),
                ),
            )
            print(
                f"Created collection {self.collection_name} with multivector support"
            )
        except Exception as e:
            print(f"Error creating collection: {e}")
            raise

    def _to_multivector(self, embedding: Any) -> List[List[float]]:
        """
        Qdrantにマルチベクターを登録する前処理
        Embeddingの形状が1次元でも2次元でも、2次元リストに正規化する
        """
        arr = (
            #embeddingがTensorならCPUに移動し、fp32に変換したうえでNumPy配列にする
            #Tensorでない場合、NumPy配列に変換し、精度をfp32に統一
            embedding.detach().cpu().to(torch.float32).numpy()
            if isinstance(embedding, torch.Tensor)
            else np.array(embedding, dtype=np.float32)
        )
        #1次元のベクトルの場合[0.1, 0.2]、 2次元リストにラップして返す [[0.1, 0.2]]
        if arr.ndim == 1:
            return [arr.tolist()]
        #2次元ベクトルの場合、そのままのpythonリストに変換して返す
        if arr.ndim == 2:
            return arr.tolist() 
        raise ValueError(f"Unexpected embedding shape: {arr.shape}")

    def store_embeddings(
        self,
        embeddings: List[torch.Tensor],
        metadata: List[Dict[str, Any]],
        upsert_batch_size: int = 64,
        wait: bool = False,
    ) -> None:
        """
        Qdrantへembeddingをアップサート
        """
        #embeddingsとmetadataの数が一致しているかをチェック
        assert len(embeddings) == len(metadata), "embeddings/metadata length mismatch"

        sent = 0
        buf: List[PointStruct] = []

        def flush(_points: List[PointStruct]):
            """
            バッファに溜まったポイントをQdrantに送信する内部関数
            ポイントはQdrantの基本単位で、1件のベクトルデータを表すレコード(id,ベクトル,メタデータ)。
            バッファは複数のポイントを一時的に貯めておくリスト
            """
            if not _points:
                return
            t0 = time.perf_counter()

            # バッファに溜まったポイントをQdrantに一括送信
            self.client.upsert(
                collection_name=self.collection_name,
                points=_points,
                wait=wait, #False: WAL(ログ)への追記が終わった時点で次のバッファを送る → データファイルへの書き込み、インデックス更新はQdrant内で非同期処理
            )
            dt = time.perf_counter() - t0
            print(f"Qdrant upsert {len(_points)} pts in {dt:.2f}s")

        try:
            # embeddingとmetadataを1件ずつ処理
            for i, (emb, meta) in enumerate(zip(embeddings, metadata)):
                # embeddingを2次元リストに正規化
                multivector = self._to_multivector(emb)
                # Qdrantに送信可能な形式に変換。（id, ベクトル, メタデータ)=ポイント
                pid = int(meta.get("global_page_num", i))
                buf.append(PointStruct(id=pid, vector=multivector, payload=meta))
                # バッファがbatch_sizeに達したらQdrant送信
                if len(buf) >= upsert_batch_size:
                    flush(buf)
                    sent += len(buf)
                    buf = []
            # ループ終了後に、余ったバッファがあれば送信
            if buf:
                flush(buf)
                sent += len(buf)
            print(f"Stored total {sent} embeddings in Qdrant")
        except Exception as e:
            print(f"Error storing embeddings: {e}")
            raise

    def search(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 3,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        検索クエリの埋め込みベクトルを、Qdrantが受け付ける形式（2次元リスト）に変換する。
        Qdrantで類似検索を実行して結果を整形して返す。
    
        Args:
            query_embedding: 検索クエリの埋め込みベクトル
            top_k: 上位何件を返すか（デフォルト: 3）
            score_threshold : 類似度スコアのしきい値（低スコアを除外）
        
        Notes:
            Qdrant API呼び出し時の主なパラメータ:
              - collection_name: 検索対象のコレクション名
              - query: 検索クエリベクトル（正規化済み）
              - limit: 上位件数 (top_k)
              - score_threshold: 類似度スコアのしきい値
              - with_payload=True: 結果にペイロード（メタデータ）を含める
        """
        try:
            # クエリ埋め込みをQdrantに渡せる形（2次元リスト）に正規化
            query = self._to_multivector(query_embedding)
            # Qdrantへ検索リクエストを送信
            res = self.client.query_points(
                collection_name=self.collection_name,
                query=query,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            )
            # 検索結果（id,スコア,メタデータ）を整形してPythonの辞書リストに変換
            out = [
                {"id": p.id, "score": float(p.score), "payload": p.payload}
                for p in res.points
            ]
            print(f"Found {len(out)} results")
            return out
        except Exception as e:
            print(f"Error during search: {e}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Qdrantのコレクション情報を取得
    
        Returns:
            - name: コレクション名
            - vectors_count: 登録されているベクトルの総数
            - points_count: 登録されているポイントの総数
            - status: コレクションの状態
        """
        try:
            # Qdrantからコレクション情報を取得
            info = self.client.get_collection(self.collection_name)

            # 必要な情報を辞書にまとめて返す
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status,
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {}


class MMrag():
    def __init__(self):
        self.pdf_dpi: int = 150 #300 # PDF→画像変換時の解像度 大きいほぼ回答精度が上がる傾向にあるが大きすぎるとメモリの問題がある
        self.pdf_fmt: str = "jpeg" # PDF→画像変換時のフォーマット（jpegはI/O軽量）
        self.use_pdftocairo: bool = True # pdf画像変換で、pdftocairoを利用する(高品質なレンダリング)
        self.document_mapping: Dict[str, Dict[str, int]] = {} # PDF名ごとのページ範囲マッピング
        self.use_qdrant: bool = True
        self.force_recreate_collection=True # 既存コレクションを削除して再作成
        
        self.vector_store = QdrantVectorStore(
            collection_name=collection_name,
            timeout=120.0,
            url = url,
        )


    def pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """
        PDFファイルをページごとに画像へ変換する関数

        Args:
            pdf_path (str): 入力PDFファイルのパス
            
        Returns:
            List[Image.Image]: 変換後の各ページ画像のリスト
        """
        print(f"Converting PDF to images: {pdf_path} (dpi={self.pdf_dpi})")
        
         # PDFファイルの存在チェック（なければ例外を投げる）
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        t0 = time.perf_counter()

        # pdf2imageを使ってPDFを画像に変換
        images = convert_from_path(
            pdf_path,
            dpi=self.pdf_dpi, # 解像度（dpi）
            use_pdftocairo=self.use_pdftocairo, # pdftocairoを使うか（高速・正確)
            fmt=self.pdf_fmt, #出力画像フォーマット（jpeg）
        )
        dt = time.perf_counter() - t0
        print(f"pdf2image({os.path.basename(pdf_path)}): {len(images)} pages in {dt:.2f}s")
        return images

    def load_ret_proc_pdf(self):
        retrieval_model_name: str = "vidore/colqwen2.5-v0.2" # 検索用埋め込みモデル
        # 検索用埋め込みモデルをロード（検索用）
        retrieval_model = ColQwen2_5.from_pretrained(
            retrieval_model_name,
            dtype=torch.bfloat16,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None, # FlashAttention2対応なら高速化を有効にする
        ).eval()

        # プロセッサをロード（入力画像/テキストを埋め込み入力形式に整形）
        retrieval_processor = ColQwen2_5_Processor.from_pretrained(
            retrieval_model_name,
            use_fast=True
        )

        all_images: List[Image.Image] = []
        all_metadata: List[Dict[str, Any]] = []
        start_page = 0

        for pdf_path in pdf_paths:
            print(f"Processing PDF: {pdf_path}")
            # PDFファイルの存在確認（なければスキップ）
            if not os.path.exists(pdf_path):
                print(f"PDF file not found: {pdf_path}")
                continue

            # PDFをページごとに画像へ変換
            images = self.pdf_to_images(pdf_path)
            pdf_name = os.path.basename(pdf_path)
            just_pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

            # ページ範囲を記録（ドキュメントマッピング用）
            end_page = start_page + len(images)
            self.document_mapping[pdf_name] = {
                "start_page": start_page,
                "end_page": end_page,
                "total_pages": len(images),
            }

            # ページごとのメタデータを作成
            metadata = [
                {
                    "pdf_path": pdf_path,
                    "pdf_name": codecs.encode(pdf_name, 'utf-8'),
                    "page_num": i + 1, # PDF内のページ番号（1始まり）
                    "global_page_num": start_page + i, # 全体での通しページ番号
                    "image_path": f"{just_pdf_name}_page_{i+1}.{self.pdf_fmt}", # 変換後の画像ファイル名
                }
                for i in range(len(images))
            ]

            #★★試しにすべての画像を保存してみる
            for i, img in enumerate(images):
                img.save(fr'{img_path}/{start_page + i:04}_{just_pdf_name}_page_{i+1}.{self.pdf_fmt}')

            # 全体リストに追加(複数PDF分の変換データ)
            all_images.extend(images)
            all_metadata.extend(metadata)
            start_page = end_page # 次のPDFのページ開始位置を更新
            
            print(f"Added {len(images)} pages from {pdf_name}")
        #print('all_images:', all_images)
        with open(fr"{img_path}/{collection_name}.pickle", mode="wb") as f:
            pickle.dump(all_metadata, f)

        print('len(all_images):', len(all_images))

        # 画像を埋め込みモデルの入力形式に変換するためのヘルパー関数
        def collate_fn(batch_imgs: List[Image.Image]):
            return retrieval_processor.process_images(batch_imgs)

        num_workers = 2
        batch_size = 2
        prefetch_factor = 2
        # DataLoaderの設定（画像を埋め込みモデルに効率的に流し込み）
        dl_kwargs = dict(
            batch_size=max(1, batch_size),
            shuffle=False, # 順序を固定（PDFのページ順を維持）
            collate_fn=collate_fn, # 画像を埋め込みモデルの入力形式に変換する前処理関数
            num_workers=num_workers, # データローダーのワーカー数
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0), # ワーカーを持続させる（エポックごとにワーカーを再作成しない）
        )
        if num_workers > 0:
            dl_kwargs["prefetch_factor"] = prefetch_factor # GPUに渡す次バッチを、CPUワーカーが先に用意しておく深さ。GPUを待たせないため

        dataloader = DataLoader(all_images, **dl_kwargs)
        #print('dataloader :', dataloader)

        # 行列演算の高速化設定（TF32を使った最適化）
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        # 埋め込みを溜めるリスト
        embeddings: List[torch.Tensor] = []
        t0 = time.perf_counter()

        first_batch_logged = False

        # 画像のベクトル変換
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for bi, batch_images in enumerate(tqdm(dataloader, desc="Embedding pages (opt)")):

                t_start = time.perf_counter()
                # GPUへ転送
                batch_images = {
                    k: v.to(retrieval_model.device, non_blocking=True)

                    for k, v in batch_images.items()
                }
                t_ready = time.perf_counter()

                # 画像のベクトル変換
                out = retrieval_model(**batch_images)
                t_done = time.perf_counter()

                # バッチ出力を1件ずつ分解してリストに追加
                embeddings.extend(torch.unbind(out))

                data_wait = t_ready - t_start  # データ準備にかかった時間
                compute = t_done - t_ready  # 推論計算にかかった時間
                print(f"[batch {bi:03d}] bs={out.shape[0]} data_wait={data_wait:.3f}s compute={compute:.3f}s")
                if not first_batch_logged:
                    print(f"first_batch_total_latency={t_done - t_start:.3f}s")
                    first_batch_logged = True
        total = len(embeddings)

        # 画像ベクトルをQdrantへ保存
        if self.use_qdrant and self.vector_store and total > 0:
            emb_dim = embeddings[0].shape[-1] # ベクトルの次元数を取得
            self.vector_store.create_collection(
                multivector_size=emb_dim,
                force_recreate=self.force_recreate_collection,
            )
            # 画像ベクトルとメタデータをQdrantにバッチ送信
            self.vector_store.store_embeddings(
                embeddings, all_metadata,
                upsert_batch_size=16, # Qdrantへの書き込みバッチ件数 qdrantをサーバーで運用する場合これが64とかだと書き込めない
                wait=False, #upsert_wait,
            )
            print("All embeddings stored in Qdrant successfully")

def mk_list_folders(path):
    # 指定の文字列が'/'で終わっているかチェック
    if path.endswith('/'): # 終わっていればそのまま
        path = f'{DOCUMENT＿ROOT}/{path}'
    else: # でなければ’/’を付加する
        path = f'{DOCUMENT＿ROOT}/{path}/'
    img_path = path + 'images' # PDFをが増加したデータとmetadataのpickleを保存するフォルダ
    # 既存のimagesフォルダは一旦フォルダごと消去する
    try:
        shutil.rmtree(img_path)
    except:
        pass
    os.makedirs(img_path, exist_ok=True) # フォルダの存在確認が面倒なのでmkdirではなくmakedirsを使う。
    # collection_name.txtがあると参照してしまうのでDB作成の際は一旦消去する
    file_path = f'{DOCUMENT＿ROOT}/{path}/collection_name.txt'
    if os.path.exists(file_path): # collection_name.txtが存在したら消す
        os.remove(file_path)

    pdf_paths = []
    files = glob.glob(path + '*.pdf') # フォルダ内のpdfファイルのフルパス名を取得
    for f in files:
        f = f.replace(os.path.sep, '/') # pathのセパレータを"/"に変更
        #f = f.replace('/', '\\')
        pdf_paths.append(f)
    # 下記の記述で呼び出せばPDFのリストとデータを格納するために生成したフォルダパスが受け取れる
    # img_path, pdf_paths = mk_list_folders(path)
    return img_path, pdf_paths

def name_collection(path):
    # Qdrantベクトルストアのコレクション名を参照するPDFフォルダ名として残しておく関数
    # PDFフォルダ名自体をコレクション名にしてcollection_name.txtの中に保存する。
    file_path = f'{DOCUMENT＿ROOT}/{path}/collection_name.txt'
    if os.path.exists(file_path): # collection_name.txtが存在したら
        #print('found!') debugprint
        with open(file_path, 'r') as f:
            collection_name = f.read()
    else: # collection_name.txtが無かったら
        try:
            collection_name = os.path.basename(path)
            #print('create txt file...') # debugprint
            with open(file_path, 'w') as f:
                f.write(collection_name)
        except:
            print('指定されたフォルダは存在しませんでした。やり直しください。')
            sys.exit()
    return collection_name

if __name__ == "__main__":
    path = input(f'{DOCUMENT＿ROOT}内に作成したDB化するPDFフォルダ名は？ -> ')
    url = "http://127.0.0.1:6333"
    img_path, pdf_paths = mk_list_folders(path)
    collection_name = name_collection(path)
    rag = MMrag()
    rag.load_ret_proc_pdf()
