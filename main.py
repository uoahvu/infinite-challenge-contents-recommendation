from data_loader import data_preprocess
from candidate_generation import ContentEmbedding, candidate_generator
from pairwise_ranking import PairwiseRanking, pair_preference_survey

import torch

if __name__ == "__main__":
    data = data_preprocess()

    print("CONTENT EMBEDDING ...")
    hidden_size = 64
    num_season = data["season"].nunique()
    embedding_model = ContentEmbedding(hidden_size, num_season)
    index, content_embedding = embedding_model.inference(data)
    print("content_embedding", content_embedding, content_embedding.shape)

    print("SEARCH SIMILAR CONTENTS ...")
    k = 20
    query_idx = 4
    query_emb = index.reconstruct(query_idx)
    distance, similar_contents_idx = candidate_generator(index, query_emb, k)

    print(data.loc[query_idx])
    print(data.loc[similar_contents_idx.flatten()])
    print(distance)

    print("PAIR-WISE PREFERENCE SURVEY ...")
    preference = pair_preference_survey(data, index)

    print("FINE TUNING ...")
    num_epochs = 100
    num_contents = content_embedding.size(0)
    embedding_dim = content_embedding.size(1)
    ranking_model = PairwiseRanking(num_epochs, num_contents, embedding_dim, index)
    optimizer = torch.optim.Adam(ranking_model.parameters(), lr=0.001)
    updated_index = ranking_model.train_model(preference, optimizer)

    print("RECOMMENDATION ...")
    query_idx = 4
    query_emb2 = updated_index.reconstruct(query_idx)
    distance, similar_contents_idx = candidate_generator(updated_index, query_emb2, k)
    # print(data.loc[query_idx])
    print(data.loc[similar_contents_idx.flatten()])
    print(distance)
