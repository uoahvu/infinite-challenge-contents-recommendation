from data_loader import data_preprocess
from candidate_generation import ContentEmbedding, candidate_generator
from pairwise_ranking import PairwiseRanking, pair_preference_survey

if __name__ == "__main__":
    data = data_preprocess()

    print("CONTENT EMBEDDING ...")
    hidden_size = 64
    num_season = data["season"].nunique()
    embedding_model = ContentEmbedding(hidden_size, num_season)
    index, content_embedding = embedding_model.inference(data)
    print("content_embedding", content_embedding)

    print("SEARCH SIMILAR CONTENTS ...")
    k = 5
    query_idx = 4
    query_emb = index.reconstruct(query_idx)
    _, similar_contents_idx = candidate_generator(index, query_emb, k)

    # print(data.loc[query_idx])
    # print(data.loc[similar_contents_idx.flatten()])

    print("PAIR-WISE PREFERENCE SURVEY ...")
    preference = pair_preference_survey(data, index)
    print(preference)

    print("FINE TUNING ...")
    num_epochs = 10
    ranking_model = PairwiseRanking(num_epochs, index)
    ranking_model.train_model(preference)
    print("RECOMMENDATION ...")
    # TODO
