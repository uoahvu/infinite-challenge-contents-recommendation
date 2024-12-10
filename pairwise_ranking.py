import torch
import torch.nn as nn
import numpy as np
from itertools import combinations
import random
import faiss


def clustering(index, num_clusters):
    embedding_dim = index.d
    kmeans = faiss.Kmeans(embedding_dim, 10)

    vectors = np.array([index.reconstruct(i) for i in range(index.ntotal)])
    kmeans.train(vectors)
    D, I = kmeans.index.search(vectors, 1)

    clusters = {i: [] for i in range(num_clusters)}
    for idx, cluster_id in enumerate(I.flatten()):
        clusters[cluster_id].append(idx)

    return clusters


def pair_preference_survey(data, index):
    num_clusters = 10
    clusters = clustering(index, num_clusters)

    cluster_pairs = list(combinations(range(num_clusters), 2))
    sample_pairs = random.sample(cluster_pairs, 10)

    result = []
    print("🎯 더 재밌게 본 영상을 알려주세요 !")
    print("｢    둘  다  별로  ! => 0")
    print("     1이 더 좋아요 ! => 1     ")
    print("     2이 더 좋아요 ! => 2     ")
    print("     둘  다 좋아요 ! => 3    ｣")
    for pair in sample_pairs:
        content1 = random.choice(clusters[pair[0]])
        content2 = random.choice(clusters[pair[1]])
        print("#############################################")
        print("1)", data.loc[content1, "title"])
        print("VS")
        print("2)", data.loc[content2, "title"])
        print("#############################################")

        user_input = int(input())

        result.append([content1, content2, user_input])

    return result


class PairwiseRanking(nn.Module):
    def __init__(self, num_epochs, num_contents, embedding_dim, index):
        super(PairwiseRanking, self).__init__()
        self.num_epochs = num_epochs
        self.index = index
        self.k = 200
        self.embedding_dim = embedding_dim

        self.embeddings = nn.Embedding(num_contents, self.embedding_dim)
        for i in range(index.ntotal):
            self.embeddings.weight.data[i] = torch.tensor(index.reconstruct(i))

    def forward(self, triplet_set):
        anchors, positives, negatives = [], [], []
        for x in triplet_set:
            anchor_id = [[x[0]] * self.k]

            positive_id = x[1]
            positive_vector = self.index.reconstruct(positive_id)
            _, positive_ids = self.index.search(positive_vector.reshape(1, -1), self.k)

            negative_id = x[2]
            negative_vector = self.index.reconstruct(negative_id)
            _, negative_ids = self.index.search(negative_vector.reshape(1, -1), self.k)

            anchors.extend(anchor_id[0])
            positives.extend(positive_ids[0])
            negatives.extend(negative_ids[0])

        anchor_embeddings = self.embeddings(torch.tensor(anchors, dtype=torch.long))
        positive_embeddings = self.embeddings(torch.tensor(positives, dtype=torch.long))
        negative_embeddings = self.embeddings(torch.tensor(negatives, dtype=torch.long))

        return anchor_embeddings, positive_embeddings, negative_embeddings

    def train_model(self, preference_data, optimizer):
        total_loss = 0
        triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
        triplet_set = self._generate_triplet(preference_data)

        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            anchor, positive, negative = self(triplet_set)
            loss = triplet_loss(anchor, positive, negative)
            # total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f"EPOCH ...({epoch+1}/{self.num_epochs})")

        self._update_index()
        return self.index

    def _generate_triplet(self, preference_data):
        triplet_set = []
        for pref in preference_data:
            if pref[2] == 1 or pref[2] == 2:
                print([pref[pref[2] - 1], pref[pref[2] - 1], pref[[1, 0][pref[2] - 1]]])
                triplet_set.append(
                    [pref[pref[2] - 1], pref[pref[2] - 1], pref[[1, 0][pref[2] - 1]]]
                )
                continue

            elif pref[2] == 3:
                anchor_id = pref[0]
                positive_id = pref[1]
                negative_id = random.choice(
                    [
                        x
                        for x in range(self.index.ntotal)
                        if x != anchor_id and x != positive_id
                    ]
                )
                triplet_set.append([anchor_id, positive_id, negative_id])
                continue

        return triplet_set

    def _update_index(self):
        updated_index = faiss.IndexFlatL2(self.embedding_dim)

        updated_embeddings = self.embeddings.weight.detach().cpu().numpy()
        updated_index.add(updated_embeddings)

        self.index = updated_index
