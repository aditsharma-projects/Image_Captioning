import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(self.vocab_size,self.embed_size)
        self.lstm = nn.LSTM(self.embed_size,self.hidden_size,batch_first=True)
        self.fc = nn.Linear(self.hidden_size,self.vocab_size)
        
    
    def forward(self, features, captions):
        caption_embeds = self.embedding(captions[:,:-1])
        features = features.unsqueeze(1)
        embeds = torch.cat((features,caption_embeds),1)
        lstm_out, _ = self.lstm(embeds)
        output = self.fc(lstm_out)
        return output

    
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        out = []
        k = 0
        #hidden = (torch.randn(1, 1, 512).to(inputs.device),
        #          torch.randn(1, 1, 512).to(inputs.device))
        lstm_out, hidden = self.lstm(inputs.view(inputs.shape[1], inputs.shape[0], inputs.shape[2]))
        print(h.shape)
        nxt = torch.zeros(0)
        
        while(True):
                
            #print(lstm_out.shape)
            output = self.fc(lstm_out[0])
            #output_scores = F.log_softmax(output, dim=1)
            #print(output_scores[0])
            values, indices = torch.max(output,1)
            out.append(indices.item())
            nxt = indices
            k = k + 1
            if indices == 1:
                break;
            if k == 20:
                break;
            embeds = self.embedding(nxt)
            #print(embeds.shape)
            embeds = embeds.unsqueeze(0)
            lstm_out, hidden = self.lstm(embeds.view(embeds.shape[1], embeds.shape[0], embeds.shape[2]),hidden)
        return out
    
    
    def sample1(self, inputs, states=None, max_len=20):
        caption = []
        # initialize the hidden state and send it to the same device as the inputs
        hidden = (torch.randn(1, 1, self.hidden_size).to(inputs.device),
                  torch.randn(1, 1, self.hidden_size).to(inputs.device))
        # Now we feed the LSTM output and hidden states back into itself to get the caption
        for i in range(max_len):
            lstm_out, hidden = self.lstm(inputs, hidden) # batch_size=1, sequence length=1 ->1,1,embedsize
            outputs = self.fc(lstm_out)        # 1,1,vocab_size
            outputs = outputs.squeeze(1)                 # 1,vocab_size
            wordid  = outputs.argmax(dim=1)              # 1
            caption.append(wordid.item())
            
            # prepare input for next iteration
            inputs = self.embedding(wordid.unsqueeze(0))  # 1,1->1,1,embed_size
          
        return caption
