# MFFM

## files
├── data  
│   ├── df_drug.csv //ddie dataset  
│   ├── event.db //ddie dataset unzip event.zip  
│   ├── processed  
│   │   ├── ccatp_feature.pt // chemistry feature concatenate pretrained feature generated by c&p.ipynb  
│   │   ├── ddi_drug1.pt // graph feature generated by create_data.py  
│   │   └── ddi_drug2.pt // graph feature generated by create_data.py  
│   ├── tr572.pt // transformer prtrained feature by char-level  
│   └── tr_word_572.pt // transformer prtrained feature by word-level  
├── ddi.csv //generated by c&p.ipynb  
├── smiles.csv // generated by c&p.ipynb  
├── graph.py   
├── c&p.ipynb  
├── create_data.py  
├── readme.md  
└── train.py  

## step
1. run c&p.ipynb get ddi.csv smiles.csv ccatp_feature.pt
2. run create_data.py get ddi_drug1.pt ddi_drug2.pt
3. run train.py

## references
https://github.com/YifanDengWHU/DDIMDL  
https://github.com/Sinwang404/DeepDDs/tree/master     
