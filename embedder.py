import torch
import torch.nn.functional as F
import numpy as np
from utils import process, data_load
class embedder:
    def __init__(self, args):
        args.gpu_num_ = args.gpu_num
        if args.gpu_num_ == -1:
            args.device = 'cpu'
        else:
            args.device = torch.device("cuda:" + str(args.gpu_num_) if torch.cuda.is_available() else "cpu") # "cuda:0" #
        if args.dataset == "ACM":
            features, feature_distance, node_rel, edge_dict, labels, node_idx = data_load.load_ACM(args)
            args.nb_classes = len(set(np.array(labels[0])[:, 1]))
        if args.dataset == "Yelp":
            features, feature_distance, node_rel, edge_dict, labels, node_idx = data_load.load_Yelp(args)
            args.nb_classes = len(set(np.array(labels[0])[:, 1]))
        if args.dataset == "DBLP":
            features, feature_distance, node_rel, edge_dict, labels, node_idx = data_load.load_DBLP(args)
            args.nb_classes = len(set(np.array(labels[0])[:, 1]))
        if args.dataset == "Aminer":
            features, feature_distance, node_rel, edge_dict, labels, args.train_idx, args.val_idx, args.test_idx, node_idx = data_load.load_Aminer(args)
            args.nb_classes = int(max(labels)-min(labels)+1)

        if args.dataset in ["ACM", "Yelp", "DBLP", "Aminer"]:
            subgraph = {}
            for nt, rels in node_rel.items():
                rel_list = []
                for rel in rels:
                    s, t = rel.split('-')
                    if args.dataset == "Yelp":
                        e = edge_dict[rel][node_idx[s], :][:, node_idx[t]]
                    else:
                        e = edge_dict[rel][0][node_idx[s], :][:, node_idx[t]]
                    e = process.normalize_adj(e)
                    e = process.sparse_to_tuple(e)
                    rel_list.append(torch.sparse_coo_tensor(torch.LongTensor(e[0]), torch.FloatTensor(e[1]), torch.Size(e[2])))
                subgraph[nt] = rel_list
            args.nt_rel = node_rel
            args.node_cnt = node_idx
            args.node_type = list(args.node_cnt)
            args.ft_size = features.shape[1]
            args.node_size = features.shape[0]

        if args.dataset in ['ACM', 'Yelp']:
            self.train_idx = torch.LongTensor(labels[0][:, 0])
            self.val_idx = torch.LongTensor(labels[1][:, 0])
            self.test_idx = torch.LongTensor(labels[2][:, 0])
            label_list = list(range(args.node_num))
            for i, j in enumerate(self.train_idx):
                label_list[j] = int(labels[0][i][1])
            for i, j in enumerate(self.val_idx):
                label_list[j] = int(labels[1][i][1])
            for i, j in enumerate(self.test_idx):
                label_list[j] = int(labels[2][i][1])
            self.labels = torch.FloatTensor(label_list)
            self.labels = torch.FloatTensor(process.encode_onehot(self.labels)).to(args.device)
        if args.dataset == 'DBLP':
            self.train_idx = torch.LongTensor(np.array(labels[0])[:, 0])
            self.val_idx = torch.LongTensor(np.array(labels[1])[:, 0])
            self.test_idx = torch.LongTensor(np.array(labels[2])[:, 0])
            self.labels = torch.cat(
                [torch.FloatTensor(np.array(labels[i])[:, 1]) for i in range(3)])  # .to(self.args.device)
            self.labels = torch.FloatTensor(process.encode_onehot(self.labels)).to(args.device)
        if args.dataset == 'Aminer':
            self.labels = torch.FloatTensor(process.encode_onehot(labels)).to(args.device)
            self.train_idx = args.train_idx
            self.val_idx = args.val_idx
            self.test_idx = args.test_idx
        new_labels = self.labels.clone()
        if args.dataset in ['ACM', 'Yelp']:
            new_labels[self.test_idx] = torch.FloatTensor([0, 0, 0]).to(args.device)
            new_labels[self.val_idx] = torch.FloatTensor([0, 0, 0]).to(args.device)
        elif args.dataset in ['DBLP', 'Aminer']:
            new_labels[self.test_idx] = torch.FloatTensor([0, 0, 0, 0]).to(args.device)
            new_labels[self.val_idx] = torch.FloatTensor([0, 0, 0, 0]).to(args.device)

        self.features = features
        self.args = args
        self.new_labels = new_labels

        print("Dataset: %s" % args.dataset)
        print("learning rate: %s" % args.lr)
        if args.gpu_num == "cpu":
            print("use cpu")
        else:
            print("use cuda")
