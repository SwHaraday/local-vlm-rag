'''
GPUSOROBANによるウェビナーでのデモをローカル環境で動くようにしたプログラム
Qdrantはローカルサーバーとして常駐させているという前提でvectorstoreを保存している
あらかじめ作成されているvectorstoreを参照してクライアントPCからの質問をGradio経由で回答する。202511 yh
単一のデータベースを参照するだけでは遊んでいる時間が長くなるので、ユーザー側でデータベースを切り替えられる機能を追加した。
これで、管理する側は新しいデータベース作成や新しいPDF資料追加の時だけ対応すればよく、動かしっぱなしにできる。コスパもやや向上？　202601 yh
更に
参照元："https://github.com/JamePeng/llama-cpp-python?tab=readme-ov-file"
2026年2月現在 Officialのllamacpp-pythonがQWEN3-VLのGUFFに対応しきれていないため
JamePeng氏によるfork先からllama-cpp-pthon　version0.3.23をインストールした。
Officialとはインターフェイスが異なるが十分に試用することができる　20260213 YH

'''
import os, glob, sys, pickle, codecs
import time
from typing import List, Dict, Any, Optional

import torch
import numpy as np
from PIL import Image

#from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from transformers import LogitsProcessor, LogitsProcessorList

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    MultiVectorConfig, MultiVectorComparator
)
import gradio as gr
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen3VLChatHandler
import base64

os.environ["NO_PROXY"] = "localhost, 127.0.0.1/8, ::1" # proxy環境で自分自身にアクセスするための呪文

# ★★試用するモデル、サーバー、AIのロール設定は環境に合わせて適宜書き換えてください
# ローカルディスクに保存済みのgguf生成AIモデルを指定
MODEL_PATH = r"./Qwen3VL-8B-Instruct-Q8_0.gguf"
#MODEL_PATH = r"./Qwen3VL-8B-Instruct-Q4_K_M.gguf"
# 上のモデルに適合するmmprojのggufファイルを指定（画像を読み込ませるため）
MMPROJ_PATH = r"./mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf"
SRV_IP = '10.xxx.XXX.27' # Gradioでwebサービスする際のIP
SRV_PT = 8080 # Gradioでwebサービスする際のport
URL = '127.0.0.1:6333' # IP & port for Qdrant server
pre_role = 'あなたはAIについての博士です。' # roleに特別な役割を与えたい場合に記述。不要なら’’としておく。
# ★★

# グローバルスコープにすべく先に宣言
SETTING_VISIBLE = True
DOCUMENT＿ROOT = 'c:/document_root' # PDF資料を保存するフォルダを作成してもらうルートフォルダ 適宜管理者が決めればよい ※スラッシュで終わらないこと！！

#グローバルで使いたいが書き換える変数
pdf_paths = []
img_path = None
collection_name = "" #Qdrantコレクション名
all_collections = [] # Qdrantの登録された全てのコレクション名リスト
paths = [] # 参照資料へのリンクを許容するフォルダパス
KN = 5 # 検索した文書の最大採用数

print(f"Loading VLM model: {MODEL_PATH}")
t0 = time.perf_counter()
# guff形式のVLMモデル本体をロード（生成用）
llm = Llama(
    model_path=MODEL_PATH,
    # Set up the chat handler for Qwen3-VL, specifying the projector path
    chat_handler=Qwen3VLChatHandler(
      clip_model_path=MMPROJ_PATH,
      #force_reasoning=True,
      verbose=False,
      image_min_tokens=1024, # Note: Qwen-VL models require at minimum 1024 image tokens to function correctly on bbox grounding tasks
    ),
    n_gpu_layers=-1,
    n_ctx=20480,
    swa_full=True,
    verbose=False, #True,
)
#モデル開放エラー回避のために挿入
#余計なエラーが出ると何がどう悪いのか分からなくなるので解決しておく
#参照元：https://github.com/abetlen/llama-cpp-python/issues/1610
import atexit
@atexit.register
def free_model():
    llm._sampler.close()
    llm.close()

print(f"VLM loaded in {time.perf_counter() - t0:.2f}s")


#=====================
# 画像をbase64で変換する際に必要なファイル形式に関する定数
# Comprehensive MIME type mapping (updated as of 2025)
# Based on Pillow 10.x+ "Fully Supported" (Read & Write) formats
# Reference: IANA official media types + common real-world usage
# See: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html
_IMAGE_MIME_TYPES = {
    # Most common formats
    '.png':  'image/png',
    '.jpg':  'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif':  'image/gif',
    '.webp': 'image/webp',

    # Next-generation formats
    '.avif': 'image/avif',
    '.jp2':  'image/jp2',
    '.j2k':  'image/jp2',
    '.jpx':  'image/jp2',

    # Legacy / Windows formats
    '.bmp':  'image/bmp',
    '.ico':  'image/x-icon',
    '.pcx':  'image/x-pcx',
    '.tga':  'image/x-tga',
    '.icns': 'image/icns',

    # Professional / Scientific imaging
    '.tif':  'image/tiff',
    '.tiff': 'image/tiff',
    '.eps':  'application/postscript',
    '.dds':  'image/vnd-ms.dds',
    '.dib':  'image/dib',
    '.sgi':  'image/sgi',

    # Portable Map formats (PPM/PGM/PBM)
    '.pbm':  'image/x-portable-bitmap',
    '.pgm':  'image/x-portable-graymap',
    '.ppm':  'image/x-portable-pixmap',

    # Miscellaneous / Older formats
    '.xbm':  'image/x-xbitmap',
    '.mpo':  'image/mpo',
    '.msp':  'image/msp',
    '.im':   'image/x-pillow-im',
    '.qoi':  'image/qoi',
}

def image_to_base64_data_uri(
    file_path: str,
    *,
    fallback_mime: str = "application/octet-stream"
) -> str:
    """
    Convert a local image file to a base64-encoded data URI with the correct MIME type.

    Supports 20+ image formats (PNG, JPEG, WebP, AVIF, BMP, ICO, TIFF, etc.).

    Args:
        file_path: Path to the image file on disk.
        fallback_mime: MIME type used when the file extension is unknown.

    Returns:
        A valid data URI string (e.g., data:image/webp;base64,...).

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If reading the file fails.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")

    extension = os.path.splitext(file_path)[1].lower()
    mime_type = _IMAGE_MIME_TYPES.get(extension, fallback_mime)

    if mime_type == fallback_mime:
        print(f"Warning: Unknown extension '{extension}' for '{file_path}'. "
              f"Using fallback MIME type: {fallback_mime}")

    try:
        with open(file_path, "rb") as img_file:
            encoded_data = base64.b64encode(img_file.read()).decode("utf-8")
    except OSError as e:
        raise OSError(f"Failed to read image file '{file_path}': {e}") from e

    return f"data:{mime_type};base64,{encoded_data}"

'''
参照元：https://zenn.dev/kun432/scraps/b2cb6e607969c0
kun432氏のZennへの投稿記事
Qwen/Qwen3-VL-4B-Instruct を適用した際に回答に多くの繰り返しを含む挙動を抑制するためのクラス関数。
「外観監視AI博士」でも同じ症状が出るので読み解いて適用してみることにする。20251113Y.H
'''
class PresencePenaltyProcessor(LogitsProcessor):
    """
    Apply a presence penalty: discourage generating tokens that have already appeared
    in the generated sequence (not frequency-based, but presence-based).
    This mimics OpenAI-style presence_penalty in a simple way by subtracting a fixed
    penalty from logits of any token present at least once in the generated tokens.
    """
    def __init__(self, presence_penalty: float):
        super().__init__()
        if presence_penalty < 0:
            raise ValueError("presence_penalty must be >= 0.")
        self.presence_penalty = presence_penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # input_ids shape: (batch, cur_len)
        # scores shape: (batch, vocab_size)
        batch_size = input_ids.shape[0]
        for b in range(batch_size):
            seen = set(input_ids[b].tolist())
            if len(seen) == 0:
                continue
            # Subtract penalty from logits of seen tokens
            # Note: scores[b] is (vocab_size,)
            # Efficient masking
            indices = torch.tensor(list(seen), device=scores.device, dtype=torch.long)
            # Clamp indices to valid range just in case
            indices = indices[(indices >= 0) & (indices < scores.shape[-1])]
            if indices.numel() > 0:
                scores[b, indices] -= self.presence_penalty
        return scores

# ---------- Qdrant----------
class QdrantVectorStore:
    def __init__(
        self,
        collection_name: str = collection_name,
        timeout: float = 120.0,
        url=URL,
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

        #QdrantClientのインスタンス化
        self.client = QdrantClient(
            timeout=timeout,
            url=URL,
        )
        print(f"Qdrant client initialized with URL: {url}")

        # 既存コレクションのリストを取得
        collections_response = self.client.get_collections()
        # コレクションの名前だけを抽出
        self.all_collections = [collection.name for collection in collections_response.collections]
        print(f"Existing collections in Qdrant:{self.all_collections}")
        
        # コレクション名の先頭のモノをスタート時のコレクションにする
        collection_name = self.all_collections[0]
        self.collection_name = collection_name
        # 参照資料へのリンクを許容するフォルダパスを生成
        for col in self.all_collections:
            p = f'{DOCUMENT＿ROOT}/{col}/'
            paths.append(p)
        #print(f'paths:{paths}') #debugprint

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
                    #on_disk=True, #特に指定しなくてもpathでディスクを指定していればよい。
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
            #print(f"Found {len(out)} results")
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
        self.pdf_dpi: int = 200 #300 # PDF→画像変換時の解像度 高いほど回答精度が上がる傾向はあるが大きすぎるとメモリの問題がある
        self.pdf_fmt: str = "jpeg" # PDF→画像変換時のフォーマット（jpegはI/O軽量）
        self.use_pdftocairo: bool = True # pdf画像変換で、pdftocairoを利用する(高品質なレンダリング)
        self.document_mapping: Dict[str, Dict[str, int]] = {} # PDF名ごとのページ範囲マッピング
        self.use_qdrant: bool = True
        self.force_recreate_collection=True, # 既存コレクションを削除して再作成

        self.retrieval_model_name: str = "vidore/colqwen2.5-v0.2" # 検索用埋め込みモデル
        
        self.vector_store = QdrantVectorStore(
            timeout=120.0,
            url=URL,
        )

    def load_models(self):
        # 行列演算の高速化設定（TF32を使った最適化）
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        
        self.vlm_model = llm

        # 検索用埋め込みモデルをロード（検索用）
        self.retrieval_model = ColQwen2_5.from_pretrained(
            self.retrieval_model_name,
            dtype=torch.bfloat16,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            #attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None, # FlashAttention2対応なら高速化を有効にする
        ).eval()

        # プロセッサをロード（入力画像/テキストを埋め込み入力形式に整形）
        self.retrieval_processor = ColQwen2_5_Processor.from_pretrained(
            self.retrieval_model_name,
            use_fast=True
        )

    # Graioのradioで選択されたPDF参照先に切り替える関数 
    # PDF資料を保存しておくルートフォルダをグローバルに決めておく必要がある -> DOCUMENT＿ROOT
    def switch_document(self, collection_name):
        self.vector_store.collection_name = collection_name
        all_images: List[Image.Image] = []
        all_metadata: List[Dict[str, Any]] = []
        start_page = 0

        #ディスクに保存したmetadataの取り込み
        with open(f"{DOCUMENT＿ROOT}/{collection_name}/images/{collection_name}.pickle", mode="rb") as f:
            all_metadata = pickle.load(f)
        print(f"★★ {DOCUMENT＿ROOT}/{collection_name}/images/{collection_name}.pickle をメタデータとします。") #debugprint

        #ディスクに保存したPDF画像の取り込み
        il = glob.glob(f'{DOCUMENT＿ROOT}/{collection_name}/images/*.jpeg')
        for fn in il:
            im = Image.open(fn)
            all_images.append(im)

        # PDFから画像に変換した結果をインスタンス変数に保存（後で埋め込み生成や検索に利用）
        self.all_document_images = all_images
        self.all_document_metadata = all_metadata
        print(f"★★ フォルダ '{DOCUMENT＿ROOT}/{collection_name}/images/' にある画像がVLMに提供されます。") #debugprint

    def search_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        テキストクエリをベクトルに変換して Qdrantで類似検索し、検索結果を返す関数
        """
        if self.retrieval_model is None or self.retrieval_processor is None:
            raise ValueError("Retrieval model not loaded. Call load_retrieval() first.")
        # テキストクエリのベクトル変換
        with torch.no_grad():
            # クエリ文字列を前処理して、モデル入力形式に変換（トークナイズ）
            proc = self.retrieval_processor.process_queries([query]).to(self.retrieval_model.device)
            # 検索用埋め込みベクトルを生成
            q_emb = self.retrieval_model(**proc)
        # Qdrantに埋め込みを渡して類似検索を実行
        hits = self.vector_store.search(q_emb[0], top_k=top_k)
        results = []
        # Qdrantの検索結果を整形
        for h in hits:
            meta = h["payload"]
            page_idx = int(meta.get("global_page_num", h["id"]))
            results.append({
                "pdf_name": meta["pdf_name"],
                "pdf_path": meta["pdf_path"],
                "page_num": meta["page_num"],  # PDF内のページ番号（1始まり）
                "global_page_num": meta["global_page_num"],  # 全体ページ番号（通し番号）
                "score": h["score"],  # 類似度スコア
                "metadata": meta,  # メタ情報
                "image": self.all_document_images[page_idx] if 0 <= page_idx < len(self.all_document_images) else None, #画像
            })
        return results

    # GUFFモデルを使うときはコピー元のtransformerモデルとは違う形に回答が生成されるためそれに合わせて
    # 関数generate_answerを改変・追加している。
    def generate_answer_GGUF(
        self, question: str, search_results: List[Dict[str, Any]], max_new_tokens: int = 1000 #500
    ) -> str:
        """
        検索結果から画像を抽出して、ユーザーの質問と合わせてVLMに入力して回答を生成する
        """
        if self.vlm_model is None: # or self.vlm_processor is None:
            raise ValueError("VLM model not loaded. Call load_vlm() first.")

        # 検索結果から画像を抽出
        imgs = []
        #この部分でstring型の画像フルパスを生成
        for res in search_results:
            gpn = res["global_page_num"]
            nam = os.path.splitext(res["pdf_name"])[0]
            im_path = f'{DOCUMENT＿ROOT}/{self.vector_store.collection_name}/images/{gpn:04}_{nam}_page_{res["page_num"]}.{self.pdf_fmt}'
            #print(im_path) #debugprint
            imgs.append(im_path)

        # 画像と質問を格納する空のリストを生成
        images_messages = []

        # 抽出しておいたimgsから画像をbase64変換してdata URLとしてリストに追加する
        for path in imgs:
            data_uri = image_to_base64_data_uri(path)
            images_messages.append({"type": "image_url", "image_url": {"url": data_uri}})

        # 最後にpre-roleと質問を追加する
        images_messages.append({"type": "text", "text": f"これらの画像を参照して質問に答えてください：\n\n質問: {question}\n\n回答:"})

        presence_penalty = 1.98 #1.5 # Qwen3の繰り返し回答を回避するために設定 範囲は-2.0～2.0らしい…
        # PresencePenalty の適用
        logits_processors = LogitsProcessorList()
        if presence_penalty and presence_penalty > 0:
            logits_processors.append(PresencePenaltyProcessor(presence_penalty))

        # VLMで回答生成
        with torch.no_grad():
            res = self.vlm_model.create_chat_completion(
                messages=[
                          {"role": "system", "content": {pre_role}},
                          {"role": "user", "content": images_messages}],
                temperature=0.1, #0.7,
                top_p=0.9,
                logits_processor=logits_processors, # Qwen3の繰り返し回答を回避するために設定
            )

        return res['choices'][0]['message']['content']

class AdvancedQASystem:
    def __init__(self, rag: MMrag):
        self.rag = rag

    def answer_question(self, question: str, top_k: int = 3, max_new_tokens: int = 1000) -> Dict[str, Any]:
        """
        ユーザーの質問に対して、検索と回答生成をまとめて実行する。
        
        Args:
            question (str): ユーザーの質問（自然文）
            top_k (int): 検索で取得する上位候補数
            max_new_tokens (int): 回答生成の最大トークン数
        """
        # 関連文書を検索
        results = self.rag.search_documents(question, top_k=top_k)
        
        # 検索結果を基に回答を生成
        answer = self.rag.generate_answer_GGUF(question, results, max_new_tokens=max_new_tokens)

        # 質問文・回答文・参照したソース情報（PDF名・ページ番号・スコア）を辞書で返す
        return {
            "question": question,
            "answer": answer,
            "sources": [
                {"pdf_name": r["pdf_name"], "page_num": r["page_num"], "score": r["score"]}
                for r in results
            ],
        }

def chat(message, history, radio):
    rag.switch_document(radio) # ここで毎回radioを読んでDBを切り替えたい
    res = qa.answer_question(message, top_k=KN)
    out = res["answer"] + f"\n【参照ソース】"
    for i, src in enumerate(res["sources"]):
        p_nam = src['pdf_name'] # 'と"を使い切るのでやむを得ず一旦取り込み
        p_pag = src['page_num'] # 'と"を使い切るのでやむを得ず一旦取り込み
        #print(f'http://{SRV_IP}:{SRV_PT}/gradio_api/file={path}/{p_nam}') #debugprint
        out = out + f"\n画像{i}　-> <a href='http://{SRV_IP}:{SRV_PT}/gradio_api/file={DOCUMENT＿ROOT}/{rag.vector_store.collection_name}/{p_nam}#page={p_pag}' target='_blank'>{src['pdf_name']} ページ{src['page_num']}</a> (スコア:{src['score']:.3f})" # ファイルへのリンク有
    yield out

if __name__ == "__main__":
    rag = MMrag()
    collection_name = rag.vector_store.all_collections[0]
    rag.load_models()
    rag.switch_document(collection_name) # ディスク上の画像とmetadataをセットする

    # QAシステムをラップするクラスを用意
    qa = AdvancedQASystem(rag)
    
    print(f'★★　VLMに与えたrole：{pre_role}')

    # Gradioチャットインタフェースを作成
    with gr.Blocks(title='アングラAI博士') as demo:
        radio = gr.Radio(
        rag.vector_store.all_collections, label="使用するベクトルDB",
        value=rag.vector_store.collection_name,
        )
        gr.ChatInterface(fn=chat,
                     title='AI博士',
                     concurrency_limit=4, # 同時に4人まで接続できるように…　と追加　20250228yh
                     type='messages',
                     additional_inputs=[radio], # ここで毎回radio選択されたDBを渡す
                     #theme=gr.themes.Soft(),
                    )
    demo.queue().launch(server_name=SRV_IP, server_port=SRV_PT, allowed_paths=paths) # share=True にするとほかのPCからもアクセス可能
