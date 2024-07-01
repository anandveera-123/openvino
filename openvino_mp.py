from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np
import openvino as ov
import torchvision.transforms as transforms
import os

def openvino(model_onnx,imagepath,outpath,idx):
    
    compiled_model_onnx = core.compile_model(model=model_onnx, device_name='CPU')
    image=Image.open(imagepath)
    
    to_tensor=transforms.ToTensor()(image).unsqueeze_(0)
  
    res_onnx = compiled_model_onnx([to_tensor])[0]
    
    gg=np.squeeze(np.argmax(res_onnx, axis=1)).astype(np.uint8)

    return gg

def preproces(img_list,outpath):
    ir_path = 'D:\\veera\\Openvino\\latest\\model.xml'
    core = ov.Core()
    model_onnx = core.read_model(model=ir_path)

     with ThreadPoolExecutor() as executor:
         
         predict=list(executor.map(openvino,[model_onnx] * len(img_list),img_list))

    if predict is not None:
        for idx,img in enumerate(predict):
            im=Image.fromarray(img)
            filename=f'{outpath}NSB{idx:03d}.png'
            im.save(filename)
                    

    
    
    





if __name__ == "__main__":
    st=time.time()
    imgpath='D:/veera/Onnx_jeron/Images/input/'
    outpath='D:/veera/Onnx_jeron/Images/op/'
    img_list=[os.path.join(imgpath,x) for x in os.listdir(imgpath) if x.endswith((".tif",".png"))]
    preproces(img_list,outpath)
 
    et=time.time()
    print(f'total time taken for :{et-st}')
    
    

