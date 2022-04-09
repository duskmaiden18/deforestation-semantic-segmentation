from predict import Prediction
from model import U_net_model

def main_func():
    model = U_net_model()
    model_path = r'D:\studying\8\diplom_nn\diplom\modeladam.h5'
    image_path = r'D:\studying\8\diplom_nn\1.jpg'
    save_res_path = r'D:\studying\8\diplom_nn\res.jpg'
    predict_obj = Prediction(model,model_path,image_path,save_res_path)
    predict_obj.predict()

if __name__ == "__main__":
    main_func()
