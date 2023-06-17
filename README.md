# Neural-Poetry-LSTM-Pytorch  
训练 `python main.py train `  
参数可以选择指定，默认参数为
```
--epoch 50  
--batch-size 128  
--use-gpu True  
--model-path None  
```  

推理可以采用**streamlit**构建的APP  
激活虚拟环境，在命令行中运行`streamlit run app.py`   
**例：**  
<div  align="center">
  <img src="./temp/屏幕截图%202023-06-16%20154916.jpg" alt="img" width="100%" />
</div>
输入prompt, 然后点击 submit   

## 依赖  
`pip install -r requirement.txt -i https://pypi.douban.com/simple`  
## 权重文件  
提供已经训练过的权重文件，位置在:   
`./models/[2023-06-16-14_19_58]_49.pth`