#dataset from http://www.ie110704.net/2017/09/23/cnn%E3%80%81rnn%E3%81%A7%E6%96%87%E7%AB%A0%E5%88%86%E9%A1%9E%E3%82%92%E5%AE%9F%E8%A3%85%E3%81%97%E3%81%A6%E3%81%BF%E3%81%9F/
data = [
    ["Could I exchange business cards, if you don’t mind?", 1],
    ["I'm calling regarding the position advertised in the newspaper.", 0],
    ["I'd like to apply for the programmer position.", 0],
    ["Could you tell me what an applicant needs to submit?", 1],
    ["Could you tell me what skills are required?", 1],
    ["We will assist employees with training and skill development.", 0],
    ["What kind of in-house training system do you have for your new recruits?", 1],
    ["For office equipment I think rental is better.", 0],
    ["Is promotion based on the seniority system?", 1],
    ["What's still pending from February?", 1],
    ["Which is better, rental or outright purchase?", 1],
    ["General Administration should do all the preparations for stockholder meetings.", 0],
    ["One of the elevators is out of order. When do you think you can have it fixed?", 1],
    ["General Administration is in charge of office building maintenance.", 0],
    ["Receptionists at the entrance hall belong to General Administration.", 0],
    ["Who is managing the office supplies inventory?", 1],
    ["Is there any difference in pay between males and females?", 1],
    ["The General Administration Dept. is in charge of office maintenance.", 0],
    ["Have you issued the meeting notice to shareholders?", 1],
    ["What is an average annual income in Japan?", 1],
    ["Many Japanese companies introduced the early retirement system.", 0],
    ["How much did you pay for the office equipment?", 1],
    ["Is the employee training very popular here?", 1],
    ["What kind of amount do you have in mind?", 1],
    ["We must prepare our financial statement by next Monday.", 0],
    ["Would it be possible if we check the draft?", 1],
    ["The depreciation of fixed assets amounts to $5 million this year.", 0],
    ["Please expedite the completion of the balance sheet.", 0],
    ["Could you increase the maximum lending limit for us?", 1],
    ["We should cut down on unnecessary expenses to improve our profit ratio.", 0],
    ["What percentage of revenue are we spending for ads?", 1],
    ["One of the objectives of internal auditing is to improve business efficiency.", 0],
    ["Did you have any problems finding us?", 1],
    ["How is your business going?", 1],
    ["Not really well. I might just sell the business.", 0],
    ["What line of business are you in?", 1],
    ["He has been a valued client of our bank for many years.", 0],
    ["Would you like for me to show you around our office?", 1],
    ["It's the second door on your left down this hall.", 0],
    ["This is the … I was telling you about earlier.", 0],
    ["We would like to take you out to dinner tonight.", 0],
    ["Could you reschedule my appointment for next Wednesday?", 1],
    ["Would you like Japanese, Chinese, Italian, French or American?", 1],
    ["Is there anything you prefer not to have?", 1],
    ["Please give my regards to the staff back in San Francisco.", 0],
    ["This is a little expression of our thanks.", 0],
    ["Why don’t you come along with us to the party this evening?", 1],
    ["Unfortunately, I have a prior engagement on that day.", 0],
    ["I am very happy to see all of you today.", 0],
    ["It is a great honor to be given this opportunity to present here.", 0],
    ["The purpose of this presentation is to show you the new direction our business is taking in 2009.", 0],
    ["Could you please elaborate on that?", 1],
    ["What's your proposal?", 1],
    ["That's exactly the point at issue here.", 0],
    ["What happens if our goods arrive after the delivery dates?", 1],
    ["I'm afraid that's not accpetable to us.", 0],
    ["Does that mean you can deliver the parts within three months?", 1],
    ["We can deliver parts in as little as 5 to 10 business days.", 0],
    ["We've considered all the points you've put forward and our final offer is $900.", 0],
    ["Excuse me but, could I have your name again, please?", 1],
    ["It's interesting that you'd say that.", 0],
    ["The pleasure's all ours. Thank you for coimng today.", 0],
    ["Could you spare me a little of your time？", 1],
    ["That's more your area of expertise than mine, so I'd like to hear more.", 0],
    ["I'd like to talk to you about the new project.", 0],
    ["What time is convenient for you?", 1],
    ["How’s 3:30 on Tuesday the 25th?", 1],
    ["Could you inform us of the most convenient dates for our visit?", 1],
    ["Fortunately, I was able to return to my office in time for the appointment.", 0],
    ["I am sorry, but we have to postpone our appointment until next month.", 0],
    ["Great, see you tomorrow then.", 0],
    ["Great, see you tomorrow then.", 1],
    ["I would like to call on you sometime in the morning.", 0],
    ["I'm terribly sorry for being late for the appointment.", 0],
    ["Could we reschedule it for next week?", 1],
    ["I have to fly to New York tomorrow, can we reschedule our meeting when I get back?", 1],
    ["I'm looking forward to seeing you then.", 0],
    ["Would you mind writing down your name and contact information?", 1],
    ["I'm sorry for keeping you waiting.", 0],
    ["Did you find your way to our office wit no problem?", 1],
    ["I need to discuss this with my superior. I'll get back to you with our answer next week.", 0],
    ["I'll get back to you with our answer next week.", 0],
    ["Thank you for your time seeing me.", 0],
    ["What does your company do?", 1],
    ["Could I ask you to make three more copies of this?", 1],
    ["We have appreciated your business.", 0],
    ["When can I have the contract signed?", 1],
    ["His secretary is coming down now.", 0],
    ["Please take the elevator on your right to the 10th floor.", 0],
    ["Would you like to leave a message?", 1],
    ["It's downstairs in the basement.", 0],
    ["Your meeting will be held at the main conference room on the 15th floor of the next building.", 0],
    ["Actually, it is a bit higher than expected. Could you lower it?", 1],
    ["We offer the best price anywhere.", 0],
    ["All products come with a 10-year warranty.", 0],
    ["It sounds good, however, is made to still think; seem to have a problem.", 0],
    ["Why do you need to change the unit price?", 1],
    ["Could you please tell me the gist of the article you are writing?", 1],
    ["Would you mind sending or faxing your request to me?", 1],
    ["About when are you publishing this book?", 1],
    ["May I record the interview?", 1]
]
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import unicodedata
import string
import dill
import yaml
from datetime import datetime
from itertools import chain
import copy
from logging import getLogger, StreamHandler, INFO, DEBUG
from collections import Counter
import re
import random
import sys
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)


class char_word_embeding(nn.Module):
    def __init__(self,word_vocab_size,char_vocab_size,word_embed_size,char_embed_size,char_hidden_dim,char_lstm_layers):
        super().__init__()
        self.word_vocab_size = word_vocab_size
        self.char_vocab_size = char_vocab_size
        self.word_embed_size = word_embed_size
        self.char_embed_size = char_embed_size
        self.word_embed = nn.Embedding(self.word_vocab_size, self.word_embed_size,padding_idx=PAD_TAG[1])
        self.char_embed = nn.Embedding(self.char_vocab_size,self.char_embed_size, padding_idx = PAD_TAG[1])
        self.char_hidden_dim = char_hidden_dim
        self.char_lstm_layers = char_lstm_layers
        self.char_lstm = nn.LSTM(char_embed_size, char_hidden_dim//2,num_layers=char_lstm_layers, bidirectional=True,batch_first = True)
        self.liner = nn.Linear(self.char_hidden_dim, self.char_embed_size)
    
    def init_char_hidden(self,size):
        return (torch.randn(2,size,self.char_hidden_dim // 2),
                torch.randn(2,size,self.char_hidden_dim // 2))

    def forward(self,word_sentences,char_sentences):
        # word part
        word_embeds = self.word_embed(word_sentences) #(batch,max_sentence_size,word_embed_size)
        #char-cnn part
        self.char_hidden = self.init_char_hidden(char_sentences.size(0)*char_sentences.size(1))
        char_embeds = self.char_embed(char_sentences) #(batch,max_sentence_size,max_char_size,char_embed_size)
        char_embeds = char_embeds.view(-1,char_embeds.size(2),self.char_embed_size) #(batch*max_sentence_size,max_char_size,char_embed_size)
        lstm_out, self.char_hidden = self.char_lstm(char_embeds, self.char_hidden)
        lstm_out = lstm_out[:,-1,:].view(char_sentences.size(0),char_sentences.size(1),-1) #(batch,max_sentence_size,char_hidden_dim)
        lstm_out = self.liner(lstm_out) #(batch,max_sentence_size,char_embed_size)
        word_char_embed = torch.cat((word_embeds,lstm_out),dim=2) ##(batch,max_sentence_size,char_embed_size+word_embed_size)
        return word_char_embed

class textCNN(nn.Module):
    def __init__(self,word_vocab_size,char_vocab_size,word_embed_size,char_embed_size,word_filter_height_list,word_kernel_dim_size,char_hidden_dim,char_lstm_layers,dropout,out_size):
        super(textCNN, self).__init__()
        self.word_vocab_size = word_vocab_size
        self.char_vocab_size = char_vocab_size
        self.word_embed_size = word_embed_size
        self.char_embed_size = char_embed_size
        self.word_filter_height_list = word_filter_height_list #[3,4,5]
        self.word_kernel_dim_size = word_kernel_dim_size
        self.char_hidden_dim = char_hidden_dim
        self.char_lstm_layers = char_lstm_layers
        self.out_size = out_size
        self.dropout = dropout
        self.embed = char_word_embeding(self.word_vocab_size,self.char_vocab_size,self.word_embed_size,self.char_embed_size,self.char_hidden_dim,self.char_lstm_layers)
        self.conv13 = nn.Conv2d(1, self.word_kernel_dim_size, (3, self.word_embed_size+self.char_embed_size))
        self.conv14 = nn.Conv2d(1, self.word_kernel_dim_size, (4, self.word_embed_size+self.char_embed_size))
        self.conv15 = nn.Conv2d(1, self.word_kernel_dim_size, (5, self.word_embed_size+self.char_embed_size))
        #self.allconvs = nn.ModuleList([nn.Conv2d(1, self.word_kernel_dim_size, (filter_height, self.word_embed_size+self.char_embed_size)) for filter_height in word_filter_height_list]) //まとめたやつ
        self.dropout = nn.Dropout(self.dropout)
        self.liner = nn.Linear(len(word_filter_height_list)*self.word_kernel_dim_size, self.out_size)
    
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)# (batch, word_kernel_dim_size, 畳み込んだ後の次元数)
        x = F.max_pool1d(x, x.size(2)).squeeze(2) # (batch, word_kernel_dim_size)
        return x
    
    def forward(self,word_sentences,char_sentences):
        embed_sentence = self.embed(word_sentences,char_sentences)# (batch, max_sentence_size, word_embed_size+char_embed_size)
        embed_sentence_plus = embed_sentence.unsqueeze(1) #(batch,1,max_sentence_size,word_embed_size+char_embed_size)
        #print(embed_sentence_plus.size())
        #print(self.char_embed_size)
        x1 = self.conv_and_pool(embed_sentence_plus,self.conv13) #(batch,word_kernel_dim_size)
        x2 = self.conv_and_pool(embed_sentence_plus,self.conv14) #(batch,word_kernel_dim_size)
        x3 = self.conv_and_pool(embed_sentence_plus,self.conv15) #(batch,word_kernel_dim_size)
        x = torch.cat((x1, x2, x3), 1) # (batch,len(word_filter_height_list)*word_kernel_dim_size)
        """
        まとめたやつ
        x = [F.relu(conv(x)).squeeze(3) for conv in self.allconvs]  // [(batch,kernel_dim_size, 畳み込んだ後の要素数), ...]*len(word_filter_height_list)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  // [(batch,kernel_dim_size), ...]*len(word_filter_height_list)
        x = torch.cat(x, 1)
        """
        x = self.dropout(x)
        out = self.liner(x) #(batch, outsize)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batch_padding(word_sentences,char_sentences,word_to_id,char_to_id):
    word_max_length = max([len(seq) for seq in word_sentences])
    char_max_length = max([max([len(char) for char in word]) for word in char_sentences])
    word_pad = [[word_to_id[PAD_TAG[0]]]*char_max_length]
    word_sentences = [x+[word_to_id[PAD_TAG[0]]]*(word_max_length - len(x)) for x in word_sentences]
    char_sentences = [[w + [word_to_id[PAD_TAG[0]]] * (char_max_length - len(w)) for w in x] for x in char_sentences]
    char_sentences = [x + (word_pad * (word_max_length - len(x))) for x in char_sentences]
    return word_sentences,char_sentences

def padding_mask(word_sentences,word_to_id):
    mask = word_sentences.data.eq(word_to_id[PAD_TAG[0]])
    return (mask, word_sentences.size(1) - mask.sum(1))

def train(train_data,word_to_id,id_to_word,char_to_id,model_path):
    logger.info("========= WORD_SIZE={} ==========".format(len(word_to_id)))
    logger.info("========= TRAIN_SIZE={} =========".format(len(train_data)))
    logger.info("========= START_TRAIN ==========")
    CNN = textCNN(len(word_to_id),len(char_to_id),word_embed_size,char_embed_size,word_filter_height_list,word_kernel_dim_size,char_hidden_dim,char_lstm_layers,dropout,out_size).to(device)
    CNN_optimizer = optim.Adam(CNN.parameters(), lr=0.01,weight_decay=1e-4)
    all_EPOCH_LOSS = []
    for epoch in range(epoch_num):
        total_loss = 0
        logger.info("=============== EPOCH {} {} ===============".format(epoch + 1, datetime.now()))
        batch_training_data = makeminibatch(train_data)
        for count,batch_data in enumerate(batch_training_data):
            logger.debug("===== {} / {} =====".format(count, len(batch_training_data)))
            batch_data.sort(key=lambda batch_data:len(batch_data[0]),reverse=True)
            word_sentences = [data[0] for data in batch_data]
            char_sentences = [data[1] for data in batch_data]
            word_sentences,char_sentences = batch_padding(word_sentences,char_sentences,word_to_id,char_to_id)
            labelslist = [data[2] for data in batch_data]
            labels = [int(label[0]) for label in labelslist]
            word_sentences = torch.tensor(word_sentences,dtype=torch.long,device=device)
            word_length = padding_mask(word_sentences,word_to_id)
            char_sentences = torch.tensor(char_sentences,dtype=torch.long,device=device)
            y = torch.tensor(labels,dtype=torch.long,device=device)
            #training
            CNN_optimizer.zero_grad()
            CNN_out = CNN(word_sentences,char_sentences)
            loss = F.cross_entropy(CNN_out, y)
            loss.backward()
            CNN_optimizer.step()
            total_loss += loss
            logger.info("=============== loss: %s ===============" % loss)
        total_loss = total_loss/len(batch_training_data)
        logger.info("=============== total_loss: %s ===============" % total_loss)
        all_EPOCH_LOSS.append(total_loss)
    [logger.info("================ batchnumber: {}---loss: {}=======================".format(batchnumber,loss)) for batchnumber,loss in enumerate(all_EPOCH_LOSS)]
    torch.save(CNN.state_dict(), model_path)

def makeminibatch(training_data):
    n = len(training_data)
    mini_batch_size = int(n/batch_size)
    random.shuffle(training_data)
    batch_training_data = []
    for i in range(0,n,mini_batch_size):
        if i+batch_size>n:
            batch_training_data.append(training_data[i:])
        else:
            batch_training_data.append(training_data[i:i+batch_size])
    return batch_training_data

def get_train_data(TRAIN_TOKEN_RABEL_FILE,max_vocab_size):
    logger.info("========= START TO GET TOKEN ==========")
    with open(file=TRAIN_TOKEN_RABEL_FILE[0], encoding="utf-8") as text_file, open(file=TRAIN_TOKEN_RABEL_FILE[1],
                                                                                    encoding="utf-8") as label_file:
        text_lines = text_file.readlines()
        label_lines = label_file.readlines()
        word_vocab = []
        char_vocab = []
    for line1 in text_lines:
        words = line1.replace("\t", " ").split()
        word_vocab.extend(words)
        for word in words:
            char_vocab.extend(list(word))
    #word part
    logger.debug(char_vocab)
    word_vocab_counter = Counter(word_vocab)
    word_vocab = [v[0] for v in word_vocab_counter.most_common(max_vocab_size)]
    word_to_id = {v: i + 2 for i, v in enumerate(word_vocab)}
    word_to_id[UNKNOWN_TAG[0]] = UNKNOWN_TAG[1]
    word_to_id[PAD_TAG[0]] = PAD_TAG[1]
    id_to_word = {i: v for v, i in word_to_id.items()}
    logger.debug(id_to_word)
    #char part
    logger.debug(char_vocab)
    char_vocab_counter = Counter(char_vocab)
    char_to_id = {v: i + 2 for i, v in enumerate(char_vocab_counter)}
    char_to_id[UNKNOWN_TAG[0]] = UNKNOWN_TAG[1]
    char_to_id[PAD_TAG[0]] = PAD_TAG[1]
    id_to_char = {i: v for v, i in char_to_id.items()}
    logger.debug(id_to_char)
    train_data = []
    for text_line, label_line in zip(text_lines, label_lines):
        if len(text_line) < 1:
            continue
        text_words = [word_to_id[word] if word in word_to_id.keys() else word_to_id[UNKNOWN_TAG[0]] for word in text_line.split()]
        text_chars = [[char_to_id[char] if char in char_to_id.keys() else char_to_id[UNKNOWN_TAG[0]] for char in word] for word in text_line.split()]
        labels = [label_line.replace("\n","")]
        if len(text_words) > 0:
            train_data.append([text_words,text_chars,labels])
    return train_data, word_to_id, id_to_word,char_to_id

def tokenize():
    logger.info("========= START TO TOKENIZE ==========")
    data_tokens = []
    data_labels = []
    for d in data:
        data_tokens.append(" ".join(sentence2words(d[0])))
        data_labels.append(d[1])
    with open(TRAIN_TOKEN_RABEL_FILE[0], "wt", encoding="utf-8") as f1, open(TRAIN_TOKEN_RABEL_FILE[1], "wt",encoding="utf-8") as f2:
        for (line1, line2) in zip(data_tokens,data_labels):
            f1.write(line1 + "\r\n")
            f2.write(str(line2) + "\r\n")

def sentence2words(sentence):
    sentence = sentence.lower()
    sentence = sentence.replace("\n", "")
    sentence = re.sub(re.compile(r"[!-\/:-@[-`{-~]"), " ", sentence)
    sentence = sentence.split(" ")
    sentence_words = []
    for word in sentence:
        if (re.compile(r"^.*[0-9]+.*$").fullmatch(word) is not None):
            continue
        sentence_words.append(word)     
    return sentence_words

"""
config
"""
config = yaml.load(open("config.yml", encoding="utf-8"))
TRAIN_TOKEN_RABEL_FILE = (config["train_token_label_file"]["tokens"],config["train_token_label_file"]["labels"])
MODEL_FILE = config["cnn-charlstm-model"]["model_path"]
epoch_num = int(config["cnn-charlstm-model"]["epoch"])
batch_size = int(config["cnn-charlstm-model"]["batch"])
word_embed_size = int(config["cnn-charlstm-model"]["word_embed"])
char_embed_size = int(config["cnn-charlstm-model"]["char_embed"])
out_size = int(config["cnn-charlstm-model"]["out_size"])
word_filter_height_list = config["cnn-charlstm-model"]["word_filter_height_list"]
word_kernel_dim_size = int(config["cnn-charlstm-model"]["word_kernel_dim_size"])
char_hidden_dim = int(config["cnn-charlstm-model"]["char_hidden"])
char_lstm_layers = int(config["cnn-charlstm-model"]["char_lstm_layers"])
max_vocab_size = int(config["cnn-charlstm-model"]["max_vocab_size"])
dropout = float(config["cnn-charlstm-model"]["dropout"])
UNKNOWN_TAG = ("<UNK>", 0)
PAD_TAG = ("<PAD>",1)

def main():
    traindata,word_to_id,id_to_word,char_to_id=get_train_data(TRAIN_TOKEN_RABEL_FILE,max_vocab_size)
    train(traindata,word_to_id,id_to_word,char_to_id,MODEL_FILE)
if __name__ == '__main__':
    print(torch.cuda.is_available())
    main()