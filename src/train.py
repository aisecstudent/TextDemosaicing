# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset

import json, time, pickle, csv, re, os, gc, logging, zlib, orjson, joblib
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from tensorboardX import SummaryWriter

from reformer_pytorch import Reformer, ReformerLM
from reformer_pytorch.reformer_pytorch import FixedPositionalEmbedding

input_dir = '../input/'
model_dir = '../model/'
log_dir = '../logs/'
PAD_index, BOS_index, EOS_index, UNK_index = 0, 1, 2, 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


vocab = json.load(open(input_dir + 'vocab.json', 'r'))
vocab_inverse = {vocab[k]:k for k in vocab}
class ReviewDataset(Dataset):
    def __init__(self, file_path, encoder_max_len=1584, decoder_max_len=67):
        super().__init__()
        self.file_path = file_path
        self.encoder_max_len = encoder_max_len
        self.decoder_max_len = decoder_max_len
        self.docs = []
        self.load_json()
    def __len__(self):
        return len(self.docs)
    def pad_image(self, arr):
        arr =  ((255 - np.asarray(arr)) / 255.0).astype(np.float32)
        mosaic = np.zeros((self.encoder_max_len, 32), dtype=np.float32)
        height, width = arr.shape
        mosaic[:width, :min(32, height)] = arr.T[:, :min(32, height)]
        return mosaic
    def pad_txt(self, arr):
        arr = [vocab.get('[BOS]')] + [vocab.get(i, 5) for i in arr] + [vocab.get('[EOS]')]
        arr = arr + [0] * (self.decoder_max_len - len(arr))
        return arr
    def load_json(self):
        with open(input_dir + self.file_path, 'r') as file_in:
            for line in file_in:
                line = orjson.loads(line.strip())
                mosaic = np.asarray(line['mosaic'], dtype=np.uint8)
                txt = line['txt']
                self.docs.append([txt, mosaic])
    def __getitem__(self, index):
        txt, mosaic = self.docs[index]
        mosaic = self.pad_image(mosaic)
        txt = self.pad_txt(txt)
        return torch.tensor(txt, dtype=torch.long), torch.tensor(mosaic, dtype=torch.float)

class CRNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=300):
        super(CRNN, self).__init__()
        self.in_channels = in_channels
        hidden_size = 150

        self.cnn_struct = ((32, ), (64, ), (128, 128), (256, 256), (256, )) 
        self.cnn_paras = ((3, 1, 1), (3, 1, 1),
                          (3, 1, 1), (3, 1, 1), (2, 1, 0)) 

        self.pool_struct = ((2, 2), (2, 2), (2, 1), (2, 1), None) 

        self.batchnorm = (False, False, False, True, False) 
        self.cnn = self._get_cnn_layers()
        self.rnn1 = nn.LSTM(self.cnn_struct[-1][-1], hidden_size, bidirectional=True)
        self.rnn2 = nn.LSTM(hidden_size*2, hidden_size, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self._initialize_weights()
    
    def forward(self, x):           # input: height=32, width>=100
        x = self.cnn(x)             # batch, channel=512, height=1, width>=24
        x = x.squeeze(2)            # batch, channel=512, width>=24
        x = x.permute(2, 0, 1)      # width>=24, batch, channel=512
        x = self.rnn1(x)[0]         # length=width>=24, batch, channel=256*2
        x = self.dropout(x)
        x = self.rnn2(x)[0]         # length=width>=24, batch, channel=256*2
        x = x.transpose(0, 1)
        return x

    def _get_cnn_layers(self):
        cnn_layers = []
        in_channels = self.in_channels
        for i in range(len(self.cnn_struct)):
            for out_channels in self.cnn_struct[i]:
                cnn_layers.append(
                    nn.Conv2d(in_channels, out_channels, *(self.cnn_paras[i])))
                if self.batchnorm[i]:
                    cnn_layers.append(nn.BatchNorm2d(out_channels))
                cnn_layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
            if (self.pool_struct[i]):
                cnn_layers.append(nn.MaxPool2d(self.pool_struct[i]))
        return nn.Sequential(*cnn_layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class TransformerModel(nn.Module):
    def __init__(self, model_args={}):
        super().__init__()
        self.model_dim = model_args.get('model_dim', 300)
        self.max_encoder_len = model_args.get('max_encoder_len', 1604)
        self.max_decoder_len = model_args.get('max_decoder_len', 66)
        self.vocab_size = model_args.get('vocab_size', 4000)

        self.encoder = CRNN()

        self.decoder = ReformerLM(
            num_tokens = self.vocab_size,
            dim = self.model_dim,
            depth = 2,
            heads = 1,
            bucket_size = 233,
            ff_dropout=0.2,
            causal = True,
            max_seq_len = self.max_decoder_len
        )
        if model_args.get('decoder_embedding', None) is not None:
            self.decoder.token_emb = nn.Embedding.from_pretrained(model_args['decoder_embedding'], freeze=False)
        else:
            self.decoder.token_emb = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=300, padding_idx=0)

    def forward(self, x, yi):
        x = x.unsqueeze(-1).transpose(1, 3)
        enc_keys = self.encoder(x) 
        input_mask = yi.ne(0).bool()
        yo = self.decoder(yi, keys=enc_keys, input_mask=input_mask)

        return yo


logging.basicConfig(filename=log_dir+'train1116.log', filemode="a", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
model_args = {'decoder_embedding': torch.load(input_dir + 'word2vec.torch')}


def cal_performance(pred, gold, trg_pad_idx, smoothing=False):

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).mean()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx) #, reduction='mean'
    return loss

def patch_trg(trg, pad_idx):
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold

def load_train_dataset(dataset, part=0):
    indices = list(range(len(dataset)))
    fold = []
    kf = KFold(n_splits=5, shuffle=True, random_state=2020)
    for train_index, valid_index in kf.split(indices):
        fold.append([train_index, valid_index])
    
    train_set, valid_set = Subset(dataset, fold[part][0]), Subset(dataset, fold[part][1])

    return train_set, valid_set

def train(dataset, part=0):
    train_set, valid_set = load_train_dataset(dataset=dataset, part=part)

    writer = SummaryWriter(log_dir + '/tensorbord1207/%d' % part) 
    model = TransformerModel(model_args).to(device) 

    optim = torch.optim.Adam(model.parameters(), lr=5e-3)

    valid_loader = DataLoader(valid_set, batch_size=40, num_workers=4, shuffle=False, drop_last=True)
    count = 0  
    PAD = torch.tensor(PAD_index).to(device)
    
    for epoch in range(20):
        torch.manual_seed(epoch)  
        train_loader = DataLoader(train_set, batch_size=40, num_workers=4, shuffle=True, drop_last=True, worker_init_fn=np.random.seed(epoch))
        model.train()

        for batch in tqdm(train_loader, total=len(train_loader), desc='  - (Training)   '):
            tgt, src = batch
            src_seq = src.to(device)
        
            trg_seq, gold = map(lambda x: x.to(device), patch_trg(tgt, PAD))

            optim.zero_grad()
            pred = model(src_seq, trg_seq)
            pred = pred.view(-1, pred.shape[-1])

            # backward and update parameters
            loss, n_correct, n_word = cal_performance(
                pred, gold, PAD, smoothing=False) 
            loss.backward()
            optim.step()

            acc = np.round(n_correct/n_word, 4)

            writer.add_scalars('train', {'loss': loss.item()}, count)
            writer.add_scalars('train', {'acc': acc}, count)

            count +=1 

        # validate
        model.eval()
        total_loss, n_word_total, n_word_correct = 0, 0, 0
        preds = []
        with torch.no_grad():
            for batch in tqdm(valid_loader, total=len(valid_loader), desc='  - (Validate)   '):
                tgt, src = batch
                src_seq = src.to(device)
            
                trg_seq, gold = map(lambda x: x.to(device), patch_trg(tgt, PAD))

                optim.zero_grad()
                pred = model(src_seq, trg_seq)
                pred = pred.view(-1, pred.shape[-1])
                for i in pred.max(1)[1]:    
                    preds.append(i.item())
                loss, n_correct, n_word = cal_performance(
                    pred, gold, PAD_index, smoothing=False)

                # note keeping
                n_word_total += n_word
                n_word_correct += n_correct

        accuracy = np.round(n_word_correct / n_word_total, 4)
        msg = 'Part{}/{}, Validate: {}'.format(part, epoch, accuracy)
        logging.info(msg)
        print(msg)

        writer.add_scalars('valid', {'acc': accuracy}, epoch)

        checkpoint = {'part': part, 'epoch': epoch, 'score': accuracy, 'model': model.state_dict()}
        model_name = model_dir + '%d_%d_%.4f.ckept' % (part, epoch, accuracy)
        torch.save(checkpoint, model_name)
                
    writer.close()

def load_model(model_name='0_3_0.9657.ckept'):
    model = TransformerModel(model_args).to(device) 
    model.load_state_dict(torch.load(model_dir + model_name)['model'])
    return model

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)

try:
    subsequent_mask_type = torch.bool
except:
    subsequent_mask_type = torch.uint8

def get_subsequent_mask(seq):
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=subsequent_mask_type),
                    diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1) 
    return mask

def decode_sentence(pred_seq):
    pred = []
    if isinstance(pred_seq, torch.Tensor):
        pred_seq = pred_seq.detach().cpu().numpy()
    for i in pred_seq:    
        if i == BOS_index:
            continue
        elif i != EOS_index and i != PAD_index:
            pred.append(vocab_inverse.get(i, '[UNK]'))
        else:
            break
    pred_line = ''.join(pred)
    return pred_line

def pad_seq(hypotheses):
    pad_tensor = torch.zeros((hypotheses.shape[0], 66 - hypotheses.shape[1])).to(hypotheses)
    return torch.cat([hypotheses, pad_tensor], dim=1)

def test(model_name='0_1_0.9416.ckept'):
    model = load_model(model_name=model_name)
    valid_set = joblib.load(open(input_dir + "valid_set.0.torch", 'rb'))

    model.eval()
    PAD = torch.tensor(PAD_index).to(device)
    BOS = torch.tensor(BOS_index).to(device)
    EOS = torch.tensor(EOS_index).to(device)

    result = []
    beam_size = 16
    vocab_size = 4000
    length_norm_coefficient = 0.5

    with torch.no_grad():
        for batch in tqdm(valid_set, mininterval=2, desc='  - (Test)', leave=False):
            tgt_seq, src_seq = batch

            src_seq = src_seq.unsqueeze(dim=0).to(device)
            src_seq = src_seq.unsqueeze(-1).transpose(1, 3)
            enc_keys = model.encoder(src_seq) # torch.Size([1, 400, 300])

            hypotheses = torch.LongTensor([[BOS]]).to(device)  # (1, 1)
            hypotheses_lengths = torch.LongTensor([hypotheses.size(1)]).to(device)
            hypotheses_scores = torch.zeros(1).to(device)
            completed_hypotheses = list()
            completed_hypotheses_scores = list()
            n_completed_hypotheses = beam_size
            
            step = 1

            while True:
                s = hypotheses.size(0)
                trg_mask = get_pad_mask(hypotheses, PAD_index) & get_subsequent_mask(hypotheses).to(hypotheses.device)
                padded_hypotheses = pad_seq(hypotheses)
                input_mask = padded_hypotheses.ne(PAD_index)
                dec_output = model.decoder(padded_hypotheses, keys=enc_keys.repeat(s, 1, 1), input_mask=input_mask) # # (s, max_len, vocab_size)
                dec_output = dec_output[:, :hypotheses.size(1), :] # (s, step, vocab_size)

                scores = dec_output[:, -1, :] # (s, vocab_size)
                scores = F.log_softmax(scores, dim=-1) # (s, vocab_size)
                scores = hypotheses_scores.unsqueeze(1) + scores # (s, vocab_size)
                
                top_k_hypotheses_scores, unrolled_indices = scores.view(-1).topk(beam_size, 0, True, True) # (k)

                # Convert unrolled indices to actual indices of the scores tensor which yielded the best scores
                prev_word_indices = unrolled_indices // vocab_size  # (k)
                next_word_indices = unrolled_indices % vocab_size  # (k)

                # Construct the the new top k hypotheses from these indices
                top_k_hypotheses = torch.cat([hypotheses[prev_word_indices], next_word_indices.unsqueeze(1)], dim=1)  # (k, step + 1)
                # Which of these new hypotheses are complete (reached <EOS>)?
                complete = next_word_indices == EOS  # (k), bool

                # Set aside completed hypotheses and their scores normalized by their lengths
                # For the length normalization formula, see
                # "Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation"
                completed_hypotheses.extend(top_k_hypotheses[complete].tolist())
                norm = np.power(((5 + step) / (5 + 1)), length_norm_coefficient)
                completed_hypotheses_scores.extend((top_k_hypotheses_scores[complete] / norm).tolist())

                # Stop if we have completed enough hypotheses
                if len(completed_hypotheses) >= n_completed_hypotheses:
                    break

                # Else, continue with incomplete hypotheses
                hypotheses = top_k_hypotheses[~complete]  # (s, step + 1)
                hypotheses_scores = top_k_hypotheses_scores[~complete]  # (s)
                hypotheses_lengths = torch.LongTensor(hypotheses.size(0) * [hypotheses.size(1)]).to(device)  # (s)

                # Stop if things have been going on for too long
                if step > 66:
                    break
                step += 1
            if len(completed_hypotheses) == 0:
                completed_hypotheses = hypotheses.tolist()
                completed_hypotheses_scores = hypotheses_scores.tolist()

            # Decode the hypotheses
            all_hypotheses = list()
            for i, h in enumerate(completed_hypotheses):
                all_hypotheses.append({"hypothesis": decode_sentence(h), "score": completed_hypotheses_scores[i]})

            # Find the best scoring completed hypothesis
            i = completed_hypotheses_scores.index(max(completed_hypotheses_scores))
            best_hypothesis = all_hypotheses[i]["hypothesis"]

            result.append([decode_sentence(tgt_seq), best_hypothesis])
    return result


def show():
    batches = torch.load(input_dir + 'demo.torch')
    batch = batches[0]
    models = ['0_0_0.2645.ckept', '0_1_0.2932.ckept', '0_2_0.3239.ckept', '0_3_0.3594.ckept', '0_4_0.4878.ckept', '0_5_0.8006.ckept', '0_6_0.8941.ckept', '0_7_0.9254.ckept', '0_8_0.9354.ckept', '0_9_0.9360.ckept', '0_10_0.9464.ckept', '0_11_0.9447.ckept', '0_15_0.9529.ckept', '0_19_0.9564.ckept', '0_39_0.9745.ckept', '0_59_0.9778.ckept', '0_99_0.9814.ckept']

    PAD = torch.tensor(PAD_index).to(device)
    BOS = torch.tensor(BOS_index).to(device)
    EOS = torch.tensor(EOS_index).to(device)
    beam_size = 16
    vocab_size = 4000
    length_norm_coefficient = 0.5

    tgt_seq, src_seq = batch
    src_seq = src_seq.unsqueeze(dim=0).to(device)
    src_seq = src_seq.unsqueeze(-1).transpose(1, 3)

    for model_name in models:
        model = load_model(model_name=model_name)
        model.eval()

        with torch.no_grad():
            enc_keys = model.encoder(src_seq) # torch.Size([1, 400, 300])

            hypotheses = torch.LongTensor([[BOS]]).to(device)  # (1, 1)
            hypotheses_lengths = torch.LongTensor([hypotheses.size(1)]).to(device)
            hypotheses_scores = torch.zeros(1).to(device)
            completed_hypotheses = list()
            completed_hypotheses_scores = list()
            n_completed_hypotheses = beam_size
            
            step = 1

            while True:
                s = hypotheses.size(0)
                trg_mask = get_pad_mask(hypotheses, PAD_index) & get_subsequent_mask(hypotheses).to(hypotheses.device)
                padded_hypotheses = pad_seq(hypotheses)
                input_mask = padded_hypotheses.ne(PAD_index)
                dec_output = model.decoder(padded_hypotheses, keys=enc_keys.repeat(s, 1, 1), input_mask=input_mask) # # (s, max_len, vocab_size)
                dec_output = dec_output[:, :hypotheses.size(1), :] # (s, step, vocab_size)

                scores = dec_output[:, -1, :] # (s, vocab_size)
                scores = F.log_softmax(scores, dim=-1) # (s, vocab_size)
                scores = hypotheses_scores.unsqueeze(1) + scores # (s, vocab_size)
                
                top_k_hypotheses_scores, unrolled_indices = scores.view(-1).topk(beam_size, 0, True, True) # (k)

                # Convert unrolled indices to actual indices of the scores tensor which yielded the best scores
                prev_word_indices = unrolled_indices // vocab_size  # (k)
                next_word_indices = unrolled_indices % vocab_size  # (k)

                # Construct the the new top k hypotheses from these indices
                top_k_hypotheses = torch.cat([hypotheses[prev_word_indices], next_word_indices.unsqueeze(1)], dim=1)  # (k, step + 1)
                # Which of these new hypotheses are complete (reached <EOS>)?
                complete = next_word_indices == EOS  # (k), bool
                completed_hypotheses.extend(top_k_hypotheses[complete].tolist())
                norm = np.power(((5 + step) / (5 + 1)), length_norm_coefficient)
                completed_hypotheses_scores.extend((top_k_hypotheses_scores[complete] / norm).tolist())

                # Stop if we have completed enough hypotheses
                if len(completed_hypotheses) >= n_completed_hypotheses:
                    break

                # Else, continue with incomplete hypotheses
                hypotheses = top_k_hypotheses[~complete]  # (s, step + 1)
                hypotheses_scores = top_k_hypotheses_scores[~complete]  # (s)
                hypotheses_lengths = torch.LongTensor(hypotheses.size(0) * [hypotheses.size(1)]).to(device)  # (s)

                hypotheses_hypotheses_list = hypotheses.tolist()
                hypotheses_hypotheses_scores_list = hypotheses_scores.tolist()
                i = hypotheses_hypotheses_scores_list.index(max(hypotheses_hypotheses_scores_list))
                best = hypotheses_hypotheses_list[i]
                print(step, decode_sentence(best))

                # Stop if things have been going on for too long
                if step > 66:
                    break
                step += 1
            
            if len(completed_hypotheses) == 0:
                completed_hypotheses = hypotheses.tolist()
                completed_hypotheses_scores = hypotheses_scores.tolist()

            # Decode the hypotheses
            all_hypotheses = list()
            for i, h in enumerate(completed_hypotheses):
                all_hypotheses.append({"hypothesis": decode_sentence(h), "score": completed_hypotheses_scores[i]})

            # Find the best scoring completed hypothesis
            i = completed_hypotheses_scores.index(max(completed_hypotheses_scores))
            best_hypothesis = all_hypotheses[i]["hypothesis"]

            print("model_name: ", model_name, ", decode: ", best_hypothesis)

if __name__ == '__main__':
    if not os.path.exists(input_dir + 'dataset.torch'):
        dataset = ReviewDataset('dataset.jsonl', encoder_max_len=1604)
        joblib.dump(dataset, open(input_dir + 'dataset.torch', 'wb'))
    else:
        dataset = joblib.load(open(input_dir + 'dataset.torch', 'rb'))

    train(dataset, part=0)
    show()

