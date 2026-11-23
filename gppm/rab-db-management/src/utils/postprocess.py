"""
 ***********************************************************************************************************************
 * Creator     : changhi.lee
 * Create Date : 2026.09.17
 * Description : 임베딩 결과를 Milvus insert용 entity dict로 변환하는 후처리 모듈
 ***********************************************************************************************************************
 *                                                   Change History
 *********** ************ ************ ***************************************************** ************* *************
 * Change No  Change Date  Change User               Change Remark                               CR No       CTS Req No
 *********** ************ ************ ***************************************************** ************* *************
 *    1       2026.09.17   changhi.lee                  최초 작성
 *********** ************ ************ ***************************************************** ************* *************
"""

import json

def get_database_input_entity_list(embeded_chunks):
    keys = ['chunk_text', 'metadata', 'tokenized_text', 'dense_embedding', 'chunk_caption', 'chunk_caption_match']
    return [postprocess_for_database(*args) for args in zip(*(embeded_chunks[key] for key in keys))]

def postprocess_for_database(text, metadata, text_for_sparse, dense_embeddings, captions, captions_match):
    database_entity = {
        'text': text,
        'text_for_sparse': text_for_sparse,
        'dense': dense_embeddings[0],
    }
    if len(dense_embeddings) >= 2:
        for i in range(1, len(dense_embeddings)):
            database_entity[f'dense_local_{i}'] = dense_embeddings[i]
    metadata['image_info'] = add_caption_to_metadata(metadata['image_info'], captions, captions_match)
    database_entity.update(metadata)

    return database_entity

def add_caption_to_metadata(metadata, caption, caption_match):
    assert len(metadata) == len(caption_match)
    assert len(caption) == sum(caption_match)

    caption_iter = iter(caption)
    new_metadata = []
    for idx, (data, needs_caption) in enumerate(zip(metadata, caption_match)):
        new_data = data.copy()
        if needs_caption:
            new_data['caption'] = next(caption_iter)
            new_data['caption_idx'] = idx
        new_metadata.append(new_data)

    return json.dumps(new_metadata, ensure_ascii=False)