import random

from candidate_generation import candidate_generator


def recommendation(data, index, content_idx):

    if content_idx == -1:  # PICK 10 Random Contents !
        contents = random.sample(range(index.ntotal), 10)
        print("#############################################")
        print("################# 추천콘텐츠 ################")
        print("#############################################")
        print(data.loc[contents, ["vod_num", "date", "title", "time"]])

    else:  # PICK TOP 10 Contents !
        query_emb = index.reconstruct(content_idx)
        _, similar_contents_idx = candidate_generator(index, query_emb, 11)

        print("#############################################")
        print("################# 현재콘텐츠 ################")
        print("#############################################")
        print(content_idx)
        print(data.loc[content_idx, ["vod_num", "date", "title", "time"]])
        print("#############################################")
        print("################# 추천콘텐츠 ################")
        print("#############################################")
        print(
            data.loc[
                similar_contents_idx.flatten()[1:],
                ["vod_num", "date", "title", "time"],
            ]
        )

    content_idx = int(input("🛒 다음 보고 싶은 콘텐츠 인덱스 >>>"))
    return recommendation(data, index, content_idx)
