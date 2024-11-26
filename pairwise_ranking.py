import torch
import torch.nn as nn
from itertools import combinations
import random
import faiss
from candidate_generation import candidate_generator


def pair_preference_survey(data, index):
    pairs = list(combinations(range(index.ntotal), 2))
    sample_pairs = random.sample(pairs, 10)
    result = []
    print("🎯 더 재밌게 본 영상을 알려주세요 !")
    print("   둘 다 별로 ! => 0")
    print("1이 더 좋아요 ! => 1")
    print("2이 더 좋아요 ! => 2")
    print(" 둘 다 좋아요 ! => 3")
    for pair in sample_pairs:
        print("#############################################")
        print("1)", data.loc[pair[0], "title"])
        print("VS")
        print("2)", data.loc[pair[1], "title"])
        print("#############################################")

        user_input = int(input())

        result.append([pair[0], pair[1], user_input])

    return result


class PairwiseRanking(nn.Module):
    def __init__(self, num_epochs, index):
        super(PairwiseRanking, self).__init__()
        self.num_epochs = num_epochs
        self.index = index
        self.k = 10

    def forward(self, x):
        for pref in x:
            if pref[2] == 1:
                _, I = candidate_generator(
                    self.index, self.index.reconstruct(pref[0]), self.k
                )

        anchor_id = pref[0]
        anchor_vector = self.index.reconstruct(pref[0])
        _, positive_ids = self.index.search(anchor_vector.reshape(1, -1), self.k)

        _, negative_ids = self.index.search(-anchor_vector.reshape(1, -1), self.k)

        anchor = self.embeddings(torch.tensor(anchor_id, dtype=torch.long))
        positive = self.embeddings(torch.tensor(positive_ids, dtype=torch.long))
        negative = self.embeddings(torch.tensor(negative_ids, dtype=torch.long))

        return anchor, positive, negative

    def train_model(self, x, optimizer):
        triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)

        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            anchor, positive, negative = self(x)
            loss = triplet_loss(anchor, positive, negative)
            total_loss += loss.item()

            # Backward 및 Optimizer 업데이트
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return self.index
