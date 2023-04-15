import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size):
        """初始化参数：
            vocab_size:字典的大小
            emb_size:词向量的维数
            hidden_size：隐向量的维数
            out_size:标注的种类
        """
        super(BiLSTM, self).__init__()
        # 一个简单的查找表（lookup table），存储固定字典和大小的词嵌入。
        # 此模块通常用于存储单词嵌入并使用索引检索它们(类似数组)。模块的输入是一个索引列表，输出是相应的词嵌入。
        # num_embeddings - 词嵌入字典大小，即一个字典里要有多少个词。
        # embedding_dim - 每个词嵌入向量的大小。
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.bilstm = nn.LSTM(emb_size, hidden_size,
                              batch_first=True,
                              bidirectional=True)
        # 设置网络中的全连接层的，需要注意在二维图像处理的任务中，全连接层的输入与输出一般都设置为二维张量，形状通常为[batch_size, size]
        self.lin = nn.Linear(2*hidden_size, out_size)

    def forward(self, sents_tensor, lengths):
        emb = self.embedding(sents_tensor)  # [B, L, emb_size]
        # 当采用RNN训练序列样本数据时，会面临序列样本数据长短不一的情况。比如做NLP任务、语音处理任务时，每个句子或语音序列的长度经常是不相同。
        # 难道要一个序列一个序列的喂给网络进行训练吗？这显然是行不通的。为了更高效的进行batch处理，就需要对样本序列进行填充，保证各个样本长度相同，在
        # PyTorch里面使用函数pad_sequence对序列进行填充。填充之后的样本序列，虽然长度相同了，但是序列里面可能填充了很多无效值0 ，将填充值0喂给
        # RNN进行forward计算，不仅浪费计算资源，最后得到的值可能还会存在误差。因此在将序列送给RNN进行处理之前，需要采用pack_padded_sequence进行压缩，
        # 压缩掉无效的填充值。序列经过RNN处理之后的输出仍然是压紧的序列，需要采用pad_packed_sequence把压紧的序列再填充回来，便于进行后续的处理。
        packed = pack_padded_sequence(emb, lengths, batch_first=True)
        rnn_out, _ = self.bilstm(packed)
        # rnn_out:[B, L, hidden_size*2]
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)

        scores = self.lin(rnn_out)  # [B, L, out_size]

        return scores

    def test(self, sents_tensor, lengths, _):
        """第三个参数不会用到，加它是为了与BiLSTM_CRF保持同样的接口"""
        logits = self.forward(sents_tensor, lengths)  # [B, L, out_size]
        _, batch_tagids = torch.max(logits, dim=2)

        return batch_tagids
