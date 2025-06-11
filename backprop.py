#will be updated but as a referece
from torchvision import models
from torch import nn
import torch
import clip
from PIL import Image
from torch.nn import functional as F
import torch.optim as optim

class BaseNet(nn.Module):
    def __init__(self,name='resnet50'):
        super().__init__()
        backbones={'resnet50':models.resnet50}
        self.name=name
        self.basenet=nn.Sequential(*list(backbones[self.name](weights="IMAGENET1K_V1").children())[:-1])    
    def forward(self,x):
        print(x.shape)
        x=self.basenet(x).reshape(1,-1)
        return x


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("RN50", device=device)
    for name, param in model.named_parameters():
        param.requires_grad = False

    cnn_model=BaseNet()
    for param in cnn_model.parameters():
        param.requires_grad = False

    #LSTM
    hidden_size=256
    num_layer=1
    rnn = nn.LSTM(2048,hidden_size,num_layer,batch_first=True,proj_size=10)#10 digit output
    for name, param in rnn.named_parameters():
        print(f"Parameter name: {name}, Trainable: {param.requires_grad}")

    steps=100
    for item in range(steps):#several backpropagation training
        image = preprocess(Image.open("./img/vincent-van-gogh_self-portrait-with-straw-hat-1887.jpg")).unsqueeze(0).to(device)
        image_feat=cnn_model(image)

        #rnn txt generation
        o1,(h1,c1)= rnn(image_feat)#1digit
        o2,(h2,c2)= rnn(image_feat,(h1,c1))#2digit
        o3,(h3,c3)= rnn(image_feat,(h2,c2))#2digit
        o4,(h4,c4)= rnn(image_feat,(h3,c3))#2digit
    
        d1=F.gumbel_softmax(o1, tau=1, hard=True)
        d2=F.gumbel_softmax(o2, tau=1, hard=True)
        d3=F.gumbel_softmax(o3, tau=1, hard=True)
        d4=F.gumbel_softmax(o4, tau=1, hard=True)

        #print("last gumbel",d4)
        #print("last gumbel-shape",d4.shape)
        d=torch.cat((d1,d2,d3,d4),dim=0)
        i1 = torch.nonzero(d1 == 1)
        i2 = torch.nonzero(d2 == 1)
        i3 = torch.nonzero(d3 == 1)
        i4 = torch.nonzero(d4 == 1)

        s1 = str(i1.tolist()[0][1])
        s2 = str(i2.tolist()[0][1])
        s3 = str(i3.tolist()[0][1])
        s4 = str(i4.tolist()[0][1])#maybe without tokenization we may need to directly input the 
        year=s1+s2+s3+s4
        txt=[year]
        print("year:",year)

        hot1 = F.one_hot(torch.tensor([49406]), num_classes=49408)
        hot5 = F.one_hot(torch.tensor([49407]), num_classes=49408)
        hot0 = F.one_hot(torch.tensor([0]), num_classes=49408)        
        hot0=hot0.repeat(71,1)
        
        image_features = model.encode_image(image)#clip
        text_features = model.encode_text(d)
        
        image_= image_features.norm(dim=-1, keepdim=True)
        image_features=image_features/image_
        text_= text_features.norm(dim=-1, keepdim=True)
        text_features=text_features/text_
        similarity = -1.0*image_features@text_features.T


        
        print("similarity",similarity)

        optimizer = optim.SGD(model.parameters(), lr=0.005)

        optimizer.zero_grad()
        similarity.backward()
        optimizer.step()
      
    
if __name__ == "__main__":
    main()


    


