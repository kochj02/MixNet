import numpy
import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image
from network.textnet import TextNet
from cfglib.config import config as cfg, update_config
from cfglib.option import BaseOptions
from util.augmentation import BaseTransform
from util.misc import to_device, rescale_result
from util.visualize import visualize_detection, visualize_gt

# Load the model
cfg.net = "FSNet_M"
cfg["mid"] = True
cfg["is_training"] = False
cfg["num_points"] = 20

IP = "0.0.0.0"
PORT = 7860

model = TextNet(is_training=False, backbone=cfg.net)
model_path = '/topaz/models/MixNet_FSNet_M_622.pth'
model.load_model(model_path)
model.to(cfg.device)
model.eval()

# Preprocessing function
def preprocess_image(image):
    image = numpy.array(image)
    transform = BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
    img = transform(image)[0]
    img = torch.from_numpy(img) 
    img = img.unsqueeze(0) if img.ndim < 4 else img
    img = img.permute(0, 3, 1, 2).contiguous()
    return img

# Inference function
def inference(image):
    # Preprocess the image
    input_tensor = preprocess_image(image)
    
    # Run inference
    with torch.no_grad():
        input_dict = {'img': to_device(input_tensor)}
        output_dict = model(input_dict)
    
    # Post-process the results
    img_show = input_tensor[0].permute(1, 2, 0).cpu().numpy()
    img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)
    
    contours = output_dict["py_preds"][-1].int().cpu().numpy()
    H, W = image.shape[:2]
    print("Input image visi shape:", img_show.shape)
    
    # Visualize the results
    show_boundary, _ = visualize_detection(img_show, output_dict, meta= {
        "image_id": ["input_image"]
    })
    
    img_show, contours = rescale_result(img_show, contours, H, W)
    
    return show_boundary

# Gradio interface
def text_detection(input_image, num_points, dis_threshold, cls_threshold, approx_factor):
    # Update config with user inputs
    cfg["num_points"] = int(num_points)
    cfg["dis_threshold"] = dis_threshold
    cfg["cls_threshold"] = cls_threshold
    cfg["approx_factor"] = approx_factor

    # Convert Gradio image to numpy array
    input_image = np.array(input_image)
    
    # Run inference
    result_image = inference(input_image)
    
    # Convert result back to PIL Image for Gradio
    return Image.fromarray(result_image)

# Main function to create and launch the Gradio interface
def main():
    iface = gr.Interface(
        fn=text_detection,
        inputs=[
            gr.Image(type="pil"),
            gr.Number(label="Number of Points", value=20),
            gr.Number(label="Distance Threshold", value=0.3),
            gr.Number(label="Classification Threshold", value=0.85),
            gr.Number(label="Approximation Factor", value=0.004),
        ],
        outputs=gr.Image(type="pil"),
        title="MixNet Text Detection",
        description="Upload an image to detect text regions.",
        examples=[
            # ["path/to/example/image1.jpg"],
            # ["path/to/example/image2.jpg"],
        ]
    )
    
    cfg["mid"] = True
    cfg["is_training"] = False
    
    iface.launch(server_name=IP, server_port=PORT, share=True)

# Launch the interface
if __name__ == "__main__":
    options = BaseOptions()
    args = options.initialize()
    update_config(cfg, args)
    
    main()
