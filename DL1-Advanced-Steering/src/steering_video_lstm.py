import cv2,torch,pandas as pd
from model_lstm import ConvNeXtLSTM
from dataset_lstm import UdacitySequenceDataset

def main():
 d="cuda" if torch.cuda.is_available() else "cpu"
 ds=UdacitySequenceDataset("../data/driving_log.csv","../data/IMG",seq_len=5)
 m=ConvNeXtLSTM(seq_len=5, hidden=256).to(d)
 m.load_state_dict(torch.load("../outputs/checkpoints/convnext_lstm.pth"))
 m.eval()

 fourcc=cv2.VideoWriter_fourcc(*"mp4v")
 vw=cv2.VideoWriter("../outputs/demo_videos/lstm_demo.mp4",fourcc,20,(320,160))

 buf=[]
 for i in range(len(ds)):
  seq,tgt=ds[i]
  row=ds.data.iloc[i+ds.seq_len-1]
  img=cv2.imread(ds.fix_path(row[0]))
  img=cv2.resize(img,(320,160))
  buf.append(seq[-1])
  if len(buf)<ds.seq_len:
   vw.write(img);continue
  x=torch.stack(buf[-ds.seq_len:]).unsqueeze(0).to(d)
  with torch.no_grad():
   pred=m(x).item()
  txt=f"GT:{tgt[0].item():.2f} P:{pred:.2f}"
  cv2.putText(img,txt,(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
  vw.write(img)

 vw.release()

if __name__=="__main__":
 main()
