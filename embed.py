import oci
import os
import json
import time
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

# OCI設定
CONFIG_PROFILE = "DEFAULT"
config = oci.config.from_file('~/.oci/config', CONFIG_PROFILE)
compartment_id = os.getenv("OCI_COMPARTMENT_ID") 
model_id = "cohere.embed-multilingual-v3.0"
generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(config=config, retry_strategy=oci.retry.NoneRetryStrategy(), timeout=(10,240))


def generate_embeddings(batch):
    embed_text_detail = oci.generative_ai_inference.models.EmbedTextDetails()
    embed_text_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=model_id)
    embed_text_detail.inputs = batch
    embed_text_detail.truncate = "NONE"
    embed_text_detail.compartment_id = compartment_id
    embed_text_detail.is_echo = False
    embed_text_detail.input_type = "SEARCH_DOCUMENT"

    embed_text_response = generative_ai_inference_client.embed_text(embed_text_detail)
        
    return embed_text_response.data.embeddings

def process(batch_size=96):
    total_processed = 0
    total_time = 0
    batch_count = 0
    all_texts = []
    all_embeddings = []

    # ファイルを読み込む
    with open('総務省FAQ.txt', 'r', encoding='utf-8') as f:
        texts = f.readlines()
    
    # バッチ処理
    for i in range(0, len(texts), batch_size):
        batch_start_time = time.time()
        batch = texts[i:i + batch_size]
        
        # 空行を除去し、テキストをクリーニング
        batch = [text.strip() for text in batch if text.strip()]
        
        # バッチサイズが0の場合はスキップ
        if len(batch) == 0:
            continue
        
        # embeddingを生成
        embeddings = generate_embeddings(batch)

        # 結果を保存用リストに追加
        all_texts.extend(batch)
        all_embeddings.extend(embeddings)

        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time
        batch_count += 1
        total_processed += len(batch)
        total_time += batch_time

        print(f"バッチ {batch_count}: {len(batch)} 件の embedding を生成しました。処理時間: {batch_time:.2f} 秒")

    # 結果をJSONファイルに保存
    result = {
        "texts": all_texts,
        "embeddings": all_embeddings
    }
    
    with open('総務省FAQ埋め込み.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return total_processed, total_time

if __name__ == "__main__":
    try:  
        start_time = time.time()
        total_processed, processing_time = process()
        end_time = time.time()

        total_time = end_time - start_time

        print(f"\n処理が完了しました。")
        print(f"合計処理件数: {total_processed} 件")
        print(f"embedding 生成時間: {processing_time:.2f} 秒")
        print(f"総処理時間: {total_time:.2f} 秒")

    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
