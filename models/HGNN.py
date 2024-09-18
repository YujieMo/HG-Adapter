import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from embedder import embedder
from utils.process import GCN_DAE, get_random_mask, get_loss_reconstruction, pairwise_distance, encode_onehot,GNNNodeEva

VERY_SMALL_NUMBER = 1e-12
INF = 1e20

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)


class train(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args
        self.adapter_net = adapter(self.args).to(self.args.device)
        self.optimizer = torch.optim.Adam(self.adapter_net.parameters(), lr=self.args.lr)

    def training(self):
        self.features = self.features.to(self.args.device)
        hom_after = torch.load('pre_trained_embedding/emb_hom_{}.pt'.format(self.args.dataset))
        het_after = torch.load('pre_trained_embedding/embs_het_{}.pt'.format(self.args.dataset))
        hom_before = torch.load('pre_trained_embedding/x_emb_{}.pt'.format(self.args.dataset))
        het_before = torch.load('pre_trained_embedding/vec_list_{}.pt'.format(self.args.dataset))
        best_val_mac = final_test_mac = final_test_mic = 0
        self.soft = torch.nn.Softmax(dim=1)
        for epoch in tqdm(range(self.args.nb_epochs)):
            self.adapter_net.train()
            self.optimizer.zero_grad()
            prediction, A, loss_rec, loss_margin = self.adapter_net(self.features, hom_before.detach(),
                                                                hom_after.detach(), het_after.detach(), het_before)
            embedding_train = prediction[self.train_idx]
            class_center = torch.stack([torch.mean(embedding_train[torch.argmax(self.new_labels[self.train_idx], dim=-1) == j], dim=0) for j in
                                        range(self.args.nb_classes)])

            similarity = torch.matmul(prediction, class_center.T)
            logp = F.log_softmax(similarity, dim=1)
            index_test = torch.cat([self.test_idx, self.val_idx], dim=0).to(self.args.device)
            loss_contrastive = F.nll_loss(logp[self.train_idx], torch.argmax(self.new_labels[self.train_idx], dim=-1)) \
                         + self.args.lamda * F.nll_loss(logp[index_test], torch.argmax(torch.mm(A, self.new_labels)[index_test], dim=-1))
            loss = loss_contrastive + self.args.eta * loss_rec + self.args.mu * loss_margin #+ 0.1 * loss_intra
            loss.backward(retain_graph=True)
            self.optimizer.step()
            if self.args.custom_key == "Node":
                if self.args.upload_pa == True:
                    logp = torch.load('saved_model/prediction_{}.pt'.format(self.args.dataset))
                val_f1_macro, val_f1_micro = GNNNodeEva(logp, self.adapter_net, self.labels, self.val_idx)
                test_f1_macro, test_f1_micro = GNNNodeEva(logp, self.adapter_net, self.labels, self.test_idx)
                if val_f1_macro > best_val_mac:
                    best_val_mac = val_f1_macro
                    final_test_mac, final_test_mic = test_f1_macro, test_f1_micro
                    # torch.save(logp, 'saved_model/prediction_{}.pt'.format(self.args.dataset))
        print('\t[Classification] mac_fi, mic_f1: {:.4f} | {:.4f}'.format(final_test_mac, final_test_mic))

        return final_test_mac, final_test_mic

class adapter(torch.nn.Module):
    def __init__(self, args):
        super(adapter, self).__init__()
        self.args = args
        self.bnn = nn.ModuleDict()
        self.fc = nn.ModuleDict()
        self.dropout = nn.Dropout(self.args.dropout)
        self.model_dae = GCN_DAE(self.args.ft_size, self.args.out_ft, self.args.ft_size, 2, self.args.dropout_cla,
                            self.args.dropout_adj, self.args.ft_size, self.args.k, 0).to(self.args.device)
        for layer in range(self.args.num_layer):
            for i in range(args.prompt_num):
                if args.bottleneck_dim > 0:
                    self.adapter = torch.nn.Sequential(
                        torch.nn.Linear(args.out_ft if i > 0 else args.out_ft, args.bottleneck_dim),
                        torch.nn.Linear(args.bottleneck_dim, args.out_ft),
                        torch.nn.ReLU(),
                    )
                else:
                    self.adapter = torch.nn.Sequential(
                        torch.nn.Linear(args.out_ft, args.out_ft),  # args.hid_units
                    )
        for layer in range(self.args.num_layer):
            for i in range(args.prompt_num):
                if args.bottleneck_dim_2 > 0:
                    self.adapter_2 = torch.nn.Sequential(
                        torch.nn.Linear(args.hid_units if i > 0 else args.hid_units, args.bottleneck_dim_2),

                        torch.nn.Linear(args.bottleneck_dim_2, args.out_ft),
                        torch.nn.ReLU(),
                    )
                else:
                    self.adapter_2 = torch.nn.Sequential(
                        torch.nn.Linear(args.hid_units, args.out_ft),  # args.hid_units
                    )
        self.answering = torch.nn.Sequential(nn.Linear(self.args.out_ft * 2, self.args.nb_classes))
        self.w_list = nn.ModuleList([nn.Linear(self.args.out_ft, 1, bias=False) for _ in range(3)])
        self.att_act1 = nn.Tanh()
        self.att_act2 = nn.Softmax(dim=-1)

    def forward(self, features, hom_before, hom_after, het_after, het_before):
        margin_loss = 0
        for i in range(self.args.prompt_num):
            F_emb = self.adapter(hom_before)
        mask = get_random_mask(features[0: self.args.node_num], 10, 10, self.args).to(self.args.device)
        loss_rec, adj = get_loss_reconstruction(self.model_dae, features[0: self.args.node_num], mask, hom_before,
                                                    self.args.dataset, self.args)
        F_emb = torch.mm(adj, F_emb)
        hom_emb = self.args.alpha * F_emb + hom_after  #
        embs1 = torch.zeros((self.args.node_size, self.args.out_ft)).to(self.args.device)
        score_list = []
        if self.args.dataset == "DBLP":
            vec = het_before[1]
            n = 'a'
        elif self.args.dataset in ["ACM", "Aminer"] :
            vec = het_before[0]
            n = 'p'
        elif self.args.dataset == "Yelp":
            vec = het_before[0]
            n = 'b'
        vec_list_1 = []
        vec_ori = []
        for i in range(len(vec)):
            vec_list_1.append(self.adapter_2(vec[i]))
            vec_ori.append(self.adapter_2(vec[i]))
        h_combine_list = []
        for i, h in enumerate(vec_ori):
            h = self.w_list[i](h)
            h_combine_list.append(h)
        score = torch.cat(h_combine_list, -1)
        score = self.att_act1(score)
        score = self.att_act2(score)
        score = torch.unsqueeze(score, -1)
        h = torch.stack(vec_list_1, dim=1)
        m = score * h
        score_list.append(score)
        v_summary = torch.sum(m, dim=1) #vec_list_1[1] #torch.sum(h, dim=1) #vec_list_1[1] #
        embs1[self.args.node_cnt[n]] = v_summary

        M_emb = embs1[0:self.args.node_num]
        embs_het_new = het_after + self.args.beta * M_emb
        h_concat = []
        h_concat.append(embs_het_new)
        h_concat.append(hom_emb)
        h_concat = torch.cat(h_concat, 1)
        # h_sum = embs_het_new_ori + F_emb
        out = self.answering(h_concat)
        seudo = torch.argmax(out[:, :self.args.nb_classes],dim=1)
        margin = torch.nn.MarginRankingLoss(margin=self.args.margin, reduce=False)
        if self.args.dataset in ["DBLP"]:
            embedding_new = F_emb
        else:
            embedding_new = hom_after
        # if epoch >= 0:
        cluster_center = torch.stack([torch.mean(embedding_new[seudo == j], dim=0) for j in
                                      range(self.args.nb_classes)])  # Shape: (num_clusters, embedding_dim)
        cluster_emb = cluster_center[seudo]
        s_p = F.pairwise_distance(v_summary, cluster_emb)
        s_n_list = []
        idx_list = []
        for i in range(self.args.nb_classes-1):
            idx_list.append((seudo+i+1)%self.args.nb_classes)
        for h_n in idx_list:
            s_n = F.pairwise_distance(v_summary, cluster_emb[h_n])
            s_n_list.append(s_n)
        margin_label = -1 * torch.ones_like(s_p)
        for s_n in s_n_list:
            margin_loss += (margin(s_p, s_n, margin_label)).mean()
        return out, adj, loss_rec, margin_loss



