# Neural-Poetry-LSTM-Pytorch  
训练 `python main train `  
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
![result](temp/屏幕截图%202023-06-16%20154916.jpg)  
输入prompt, 然后点击 submit   

## 依赖  
`pip install -r requirement.txt -i https://pypi.douban.com/simple`