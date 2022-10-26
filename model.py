import torch
import numpy as np
import utils
import scipy.sparse as sp

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.utils import InputType
from recbole.model.layers import MLPLayers


class LightSAGE(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(LightSAGE, self).__init__(config, dataset)
        self.config = config

        if self.config['dummy']:  # two dummy nodes
            self.n_users = self.n_users + 1
            self.n_items = self.n_items + 1

        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.adj_mat = self.get_adj_mat()

        self.latent_dim = self.config['embedding_size']  # 64
        self.n_layers = self.config['n_layers']  # 2
        self.neighbour_nums = self.config['neighbour_nums']
        self.reg_weight = config['reg_weight']

        if not self.config['light']:
            self.GNNlayers = torch.nn.ModuleList()
            for layer in range(self.n_layers):
                self.GNNlayers.append(MLPLayers([self.latent_dim]))

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)

        self.final_embedding_user = None
        self.final_embedding_item = None

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['final_embedding_user', 'final_embedding_item']

        print(f"LightSAGE is already to go")

    def get_adj_mat(self):
        if self.config['dummy']:
            dummy = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32).tolil()
            dummy[:-1, :-1] = self.interaction_matrix.tolil()
            dummy[-1, :] = 1
            dummy[:, -1] = 1

            adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            adj_mat[:self.n_users, self.n_users:] = dummy
            adj_mat[self.n_users:, :self.n_users] = dummy.T
            adj_mat = adj_mat.tocsr()
        else:
            adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = self.interaction_matrix.tolil()
            user_sum = R.sum(axis=1)
            item_sum = R.sum(axis=0)
            super_value = item_sum.max()
            super_indexes = np.argwhere(item_sum > super_value - 1)

            for row in np.argwhere(user_sum < 1):
                R[row[0], super_indexes] = 1

            adj_mat[:self.n_users, self.n_users:] = R
            adj_mat[self.n_users:, :self.n_users] = R.T
            adj_mat = adj_mat.tocsr()
        return adj_mat

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        init_users_emb = self.embedding_user(user)
        init_pos_emb = self.embedding_item(pos_item)
        init_neg_emb = self.embedding_item(neg_item)
        final_users_emb = self.final_embedding_user[user]
        final_pos_emb = self.final_embedding_item[pos_item]
        final_neg_emb = self.final_embedding_item[neg_item]

        reg_loss = (1 / 2) * (init_users_emb.norm(2).pow(2) +
                              init_pos_emb.norm(2).pow(2) +
                              init_neg_emb.norm(2).pow(2)) / float(len(user))

        pos_scores = torch.mul(final_users_emb, final_pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(final_users_emb, final_neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss + reg_loss*self.reg_weight

    def net(self, hidden):
        lamda = self.config['concat_lamda']
        for layer in range(self.n_layers):
            next_hidden = []
            for hop in range(self.n_layers - layer):
                src_node_features = hidden[hop]
                src_node_num = len(src_node_features)
                neighbor_node_features = hidden[hop + 1].view((src_node_num, self.neighbour_nums[hop], -1))

                aggr_neighbor = neighbor_node_features.mean(dim=1)
                h = lamda * src_node_features + (1.0 - lamda) * aggr_neighbor

                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0]

    def net1(self, hidden):
        for layer in range(self.n_layers):
            next_hidden = []
            for hop in range(self.n_layers - layer):
                src_node_features = hidden[hop]
                src_node_num = len(src_node_features)
                neighbor_node_features = hidden[hop + 1].view((src_node_num, self.neighbour_nums[hop], -1))

                aggr_neighbor = neighbor_node_features.mean(dim=1)
                h = self.GNNlayers[layer](src_node_features + aggr_neighbor)

                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0]

    def forward(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_features = torch.cat([users_emb, items_emb])
        all_final_features = all_features.clone().to(self.config['device'])

        all_indexes = np.arange(self.n_users + self.n_items)
        for (batch_id, (batch_indexes)) in enumerate(
                utils.minibatch(all_indexes, batch_size=self.config['train_batch_size'])
        ):
            sampling_index_result = utils.rw_samling(
                self.config['rw_args'],
                self.adj_mat,
                batch_indexes,
                self.neighbour_nums,
            )

            sampling_feature_result = [all_features[idx] for idx in sampling_index_result]
            if self.config['light']:
                all_final_features[batch_indexes] = self.net(sampling_feature_result)
            else:
                all_final_features[batch_indexes] = self.net1(sampling_feature_result)
            if self.config['l2norm']:
                all_final_features[batch_indexes] = torch.nn.functional.normalize(all_final_features[batch_indexes], p=2, dim=1)

        self.final_embedding_user = all_final_features[:self.n_users]
        self.final_embedding_item = all_final_features[self.n_users:]

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        all_users = self.final_embedding_user
        all_items = self.final_embedding_item
        if self.config['dummy']:
            all_users = all_users[0:-1]
            all_items = all_items[0:-1]
        users_emb = all_users[user]
        items_emb = all_items[item]
        rating = torch.matmul(users_emb, items_emb.t())
        return rating

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        all_users = self.final_embedding_user
        all_items = self.final_embedding_item
        if self.config['dummy']:
            all_users = all_users[0:-1]
            all_items = all_items[0:-1]
        users_emb = all_users[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(users_emb, all_items.transpose(0, 1))

        return scores.view(-1)
