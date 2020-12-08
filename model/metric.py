import torch
import pandas as pd

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)



def utt_accuracy(model_output, val_imgs):
    
    assert len(model_output[0]) == len(val_imgs)
    
    num_classes = model_output[0].shape[1]
    pred_columns = list(range(num_classes))
    model_output = tuple(zip(model_output[0], model_output[1]))
    model_output = pd.DataFrame(model_output, columns=['pred','label'])
    model_output[pred_columns] = pd.DataFrame(model_output.pred.tolist(), index= model_output.index)
    #print(model_output.shape)
    #print(model_output.head(1))

    val_imgs = pd.DataFrame(val_imgs, columns=['utt','label'])
    val_imgs['utt'] = val_imgs['utt'].str.split(pat="/")
    val_imgs['utt'] = val_imgs['utt'].apply(lambda x:x[-1])
    val_imgs['utt'] = val_imgs['utt'].str.split(pat=".")
    val_imgs['utt'] = val_imgs['utt'].apply(lambda x:x[0])
    val_imgs['utt'] = val_imgs['utt'].str.split(pat="_")
    val_imgs['utt'] = val_imgs['utt'].apply(lambda x:x[0])
    #print(val_imgs.shape)
    #print(val_imgs.head(1))
    val_imgs['same'] = val_imgs['label'] == model_output['label']
    print(f'same label: {val_imgs.same.value_counts()}')    

    
    model_output['utt'] = val_imgs['utt']
    pred = model_output.groupby(['utt']).mean()
    pred['score'] = pred[pred_columns].values.tolist()
    pred = pred['score']
    lab = model_output.drop_duplicates(subset=['utt'], keep='last').label
    
    pred = pred.tolist()
    pred = torch.tensor(pred)
    lab = lab.tolist()
    lab = torch.tensor(lab)

    #print(f"pred = {pred}") 
    #print(f"lab  = {lab}")   

    utt_acc = accuracy(pred,lab)
    
    return utt_acc

    
